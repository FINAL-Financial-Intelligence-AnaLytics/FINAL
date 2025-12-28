import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

from app.rag_module import RAGModule
from app.llm_client.implementations import OpenRouterFinancialClient
from app.llm_client import FinancialConsultationModule
from app.config import Config


app = FastAPI(
    title="Financial Assistant API",
    description="API для финансового консультанта с RAG",
    version="1.0.0"
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Инициализация модулей
rag_module: Optional[RAGModule] = None
consultation_module: Optional[FinancialConsultationModule] = None


@app.on_event("startup")
async def startup_event():
    """Инициализация модулей при запуске приложения"""
    global rag_module, consultation_module
    
    print("=== Инициализация RAG модуля ===")
    rag_module = RAGModule(
        collection="finance_theory",
        qdrant_url=os.getenv("QDRANT_URL"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY")
    )
    
    if rag_module.client is None:
        print("⚠️  Qdrant не настроен! RAG будет работать в ограниченном режиме.")
    else:
        print("✅ RAG модуль успешно инициализирован")
    
    print("=== Инициализация LLM клиента ===")
    llm_client = OpenRouterFinancialClient(
        model_name=os.getenv("OPENROUTER_MODEL", Config.OPENROUTER_MODEL),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=float(os.getenv("OPENROUTER_TEMPERATURE", str(Config.OPENROUTER_TEMPERATURE)))
    )
    
    consultation_module = FinancialConsultationModule(
        llm_client=llm_client,
        rag_retriever=rag_module if rag_module.client else None
    )
    print("✅ LLM клиент успешно инициализирован")


# Pydantic модели для запросов и ответов
class QuestionRequest(BaseModel):
    question: str = Field(..., description="Вопрос пользователя")
    limit: int = Field(default=5, ge=1, le=20, description="Количество релевантных фрагментов")
    score_threshold: Optional[float] = Field(default=0.2, ge=0.0, le=1.0, description="Минимальный порог релевантности")


class RetrieveRequest(BaseModel):
    query: str = Field(..., description="Поисковый запрос")
    limit: int = Field(default=5, ge=1, le=20, description="Количество результатов")
    score_threshold: Optional[float] = Field(default=0.2, ge=0.0, le=1.0, description="Минимальный порог релевантности")


class ChunkResponse(BaseModel):
    content: str
    score: float
    source: Optional[str] = None


class AnswerResponse(BaseModel):
    answer: str
    chunks: List[ChunkResponse] = []


class HealthResponse(BaseModel):
    status: str
    rag_available: bool
    llm_available: bool


@app.get("/", tags=["General"])
async def root():
    """Корневой эндпоинт"""
    return {
        "message": "Financial Assistant API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Проверка состояния сервиса"""
    return HealthResponse(
        status="ok",
        rag_available=rag_module.client is not None if rag_module else False,
        llm_available=consultation_module is not None
    )


@app.post("/api/v1/answer", response_model=AnswerResponse, tags=["RAG"])
async def answer_question(request: QuestionRequest):
    """
    Получить ответ на вопрос с использованием RAG
    """
    if not rag_module:
        raise HTTPException(status_code=503, detail="RAG модуль не инициализирован")
    
    if not rag_module.client:
        raise HTTPException(
            status_code=503, 
            detail="Qdrant не настроен. Пожалуйста, настройте QDRANT_URL и QDRANT_API_KEY"
        )
    
    try:
        result = rag_module.answer(
            question=request.question,
            limit=request.limit,
            score_threshold=request.score_threshold
        )
        
        chunks_response = [
            ChunkResponse(
                content=chunk.text,
                score=chunk.score,
                source=None  # Можно добавить source в RAGChunk если нужно
            )
            for chunk in result.get("chunks", [])
        ]
        
        return AnswerResponse(
            answer=result.get("answer", ""),
            chunks=chunks_response
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке запроса: {str(e)}")


@app.post("/api/v1/retrieve", response_model=List[ChunkResponse], tags=["RAG"])
async def retrieve_chunks(request: RetrieveRequest):
    """
    Получить релевантные фрагменты по запросу
    """
    if not rag_module:
        raise HTTPException(status_code=503, detail="RAG модуль не инициализирован")
    
    if not rag_module.client:
        raise HTTPException(
            status_code=503,
            detail="Qdrant не настроен. Пожалуйста, настройте QDRANT_URL и QDRANT_API_KEY"
        )
    
    try:
        chunks_data = rag_module.retrieve(
            query=request.query,
            limit=request.limit,
            score_threshold=request.score_threshold
        )
        
        return [
            ChunkResponse(
                content=chunk.get("content", ""),
                score=chunk.get("score", 0.0),
                source=chunk.get("source")
            )
            for chunk in chunks_data
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при поиске: {str(e)}")


@app.post("/api/v1/consultation", tags=["Consultation"])
async def financial_consultation(request: QuestionRequest):
    """
    Получить финансовую консультацию с использованием RAG
    """
    if not consultation_module:
        raise HTTPException(status_code=503, detail="Модуль консультаций не инициализирован")
    
    try:
        if rag_module and rag_module.client:
            response = consultation_module.search_answer_via_rag(
                query=request.question,
                top_k=request.limit
            )
        else:
            response = consultation_module.answer_financial_literacy(request.question)
        
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при консультации: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

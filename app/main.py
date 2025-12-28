import os
import re
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)
    print(f"✅ Загружен .env файл: {env_path}")
else:
    print(f"⚠️  .env файл не найден: {env_path}")

from app.rag_module import RAGModule
from app.config import Config


app = FastAPI(
    title="Financial Assistant API",
    description="API для финансового консультанта с RAG",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_module: Optional[RAGModule] = None


def _format_answer(text: str) -> str:
    if not text:
        return ""
    
    text = re.sub(r'\[\d+\](\[\d+\])*', '', text)
    text = re.sub(r'^#{1,6}\s+(.+)$', r'\1', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    text = re.sub(r'^[-*]{3,}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\|?\s*[-:]+\s*\|.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\|\s*', ' ', text)
    text = re.sub(r'\s*\|', ' ', text)
    text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[\s]*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[✅❌✔]', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    def fix_quotes(match):
        quote_open = match.group(1)
        content = match.group(2)
        quote_close = match.group(3)
        content = content.strip()
        return f'{quote_open}{content}{quote_close}'
    
    text = re.sub(r'(["«„])([^"»"]*?)(["»"])', fix_quotes, text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'([а-яёa-zА-ЯЁA-Z])(["«„])', r'\1 \2', text)
    text = re.sub(r'(["»"])([а-яёa-zА-ЯЁA-Z])', r'\1 \2', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'(["«„])\s+', r'\1', text)
    text = re.sub(r'\s+(["»"])', r'\1', text)
    
    def remove_spaces_in_quotes(match):
        quote_open = match.group(1)
        content = match.group(2)
        quote_close = match.group(3)
        content = content.strip()
        return f'{quote_open}{content}{quote_close}'
    
    text = re.sub(r'(["«„])([^"»"]*?)(["»"])', remove_spaces_in_quotes, text)
    text = re.sub(r'([\)])\s+([.,!?;:])', r'\1\2', text)
    text = re.sub(r'(["»"])\s+([.,!?;:])', r'\1\2', text)
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = re.sub(r'([.,!?;:])([А-ЯЁA-Zа-яёa-z])', r'\1 \2', text)
    text = re.sub(r'(["»"])([а-яёa-zА-ЯЁA-Z])', r'\1 \2', text)
    text = re.sub(r'([\)])\s+([.,!?;:])', r'\1\2', text)
    text = re.sub(r'(["»"])\s+([.,!?;:])', r'\1\2', text)
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    text = text.strip()
    text = re.sub(r' +$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*—\s*', ' — ', text)
    text = re.sub(r'\s*-\s*', ' - ', text)
    text = re.sub(r'\s*\(\s*', ' (', text)
    text = re.sub(r'\s*\)\s*', ') ', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n +', '\n', text)
    text = re.sub(r'([\)])\s+([.,!?;:])', r'\1\2', text)
    text = re.sub(r'(["»"])\s+([.,!?;:])', r'\1\2', text)
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    
    return text


@app.on_event("startup")
async def startup_event():
    global rag_module
    
    print("=== Инициализация LLM клиента (Mistral) ===")
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        print("⚠️  ВНИМАНИЕ: MISTRAL_API_KEY не найден в переменных окружения!")
        raise ValueError("MISTRAL_API_KEY должен быть установлен в .env файле")
    
    print(f"✅ API ключ найден: {api_key[:10]}...{api_key[-4:] if len(api_key) > 14 else '***'}")
    
    print("=== Инициализация RAG модуля ===")
    from app.llm_client.implementations.mistral_client import MistralLLM
    rag_llm = MistralLLM(
        api_key=api_key,
        model=os.getenv("MISTRAL_MODEL", Config.MISTRAL_MODEL),
        base_url=os.getenv("MISTRAL_BASE_URL", Config.MISTRAL_BASE_URL),
        temperature=float(os.getenv("MISTRAL_TEMPERATURE", str(Config.MISTRAL_TEMPERATURE)))
    )
    print("✅ RAG LLM клиент создан")
    
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    if qdrant_url:
        print(f"📡 Подключение к Qdrant: {qdrant_url}")
    else:
        print("⚠️  QDRANT_URL не установлен в .env файле")
    
    embedding_device = os.getenv("EMBEDDING_DEVICE", "cpu")
    embedding_model = os.getenv("EMBEDDING_MODEL")
    
    if embedding_model:
        print(f"📦 Модель эмбеддингов: {embedding_model} (загрузится при первом использовании)")
        print(f"📦 Устройство для эмбеддингов: {embedding_device}")
    else:
        print("📦 Модель эмбеддингов не настроена - RAG будет работать без локальной модели")
    
    rag_module = RAGModule(
        collection="finance_theory",
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        llm=rag_llm,
        model_name=embedding_model,
        device=embedding_device if embedding_model else None
    )
    
    if rag_module.client is None:
        print("⚠️  Qdrant не настроен! RAG будет работать в ограниченном режиме.")
    else:
        try:
            collections = rag_module.client.get_collections()
            print(f"✅ RAG модуль успешно инициализирован. Коллекций в Qdrant: {len(collections.collections)}")
            
            try:
                collection_info = rag_module.client.get_collection("finance_theory")
                print(f"✅ Коллекция 'finance_theory' найдена. Точек: {collection_info.points_count}")
            except Exception as e:
                print(f"⚠️  Коллекция 'finance_theory' не найдена: {e}")
                print("   Выполните: python scripts/prepare_data.py --collection finance_theory --input-csv <ваш_csv_файл>")
        except Exception as e:
            print(f"❌ Ошибка подключения к Qdrant: {e}")
            print("   Проверьте, что Qdrant запущен и доступен по адресу из QDRANT_URL")


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


class HealthResponse(BaseModel):
    status: str
    rag_available: bool


@app.get("/", tags=["General"])
async def root():
    return {
        "message": "Financial Assistant API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    return HealthResponse(
        status="ok",
        rag_available=rag_module.client is not None if rag_module else False
    )


@app.post("/api/v1/answer", response_model=AnswerResponse, tags=["RAG"])
async def answer_question(request: QuestionRequest):
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
        
        answer_text = result.get("answer", "")
        formatted_answer = _format_answer(answer_text)
        
        return AnswerResponse(
            answer=formatted_answer
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке запроса: {str(e)}")


@app.post("/api/v1/retrieve", response_model=List[ChunkResponse], tags=["RAG"])
async def retrieve_chunks(request: RetrieveRequest):
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

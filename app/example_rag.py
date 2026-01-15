import os
from rag_module import RAGModule
from llm_client.implementations import OpenRouterFinancialClient
from llm_client import FinancialConsultationModule


def main():
    print("=== Настройка RAG модуля ===\n")
    
    rag = RAGModule(
        collection="finance_theory",
        qdrant_url=os.getenv("QDRANT_URL"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY")
    )
    
    if rag.client is None:
        print("Qdrant не настроен!")
        print("Для работы RAG необходимо:")
        print("1. Установить Qdrant (docker run -p 6333:6333 qdrant/qdrant)")
        print("2. Установить переменные окружения:")
        print("   export QDRANT_URL='http://localhost:6333'")
        print("   export QDRANT_API_KEY='your-key'")
        print("\nПродолжаем без RAG...\n")
    else:
        print("RAG модуль успешно инициализирован\n")
    
    client = OpenRouterFinancialClient(
        model_name="tngtech/deepseek-r1t2-chimera:free",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0.1
    )
    
    print("=== Пример 1: Использование RAG напрямую ===\n")
    
    if rag.client:
        question = "Что такое инфляция и как она влияет на экономику?"
        print(f"Вопрос: {question}\n")
        
        result = rag.answer(
            question=question,
            limit=5,
            score_threshold=0.2
        )
        
        print(f"Ответ:\n{result['answer']}\n")
        
        if result['chunks']:
            print(f"Найдено релевантных фрагментов: {len(result['chunks'])}")
            for i, chunk in enumerate(result['chunks'][:3], 1):
                print(f"\nФрагмент {i} (релевантность: {chunk.score:.3f}):")
                print(f"{chunk.text[:200]}...")
    else:
        print("RAG не настроен, пропускаем этот пример\n")
    
    print("\n=== Пример 2: RAG с FinancialConsultationModule ===\n")
    
    consultation = FinancialConsultationModule(
        llm_client=client,
        rag_retriever=rag if rag.client else None
    )
    
    if rag.client:
        questions = [
            "Как работает сложный процент?",
            "Что такое диверсификация портфеля?",
            "Как создать финансовую подушку безопасности?"
        ]
        
        for question in questions:
            print(f"Вопрос: {question}")
            response = consultation.search_answer_via_rag(question, top_k=3)
            print(f"Ответ:\n{response}\n")
            print("-" * 80 + "\n")
    else:
        print("RAG не настроен, используем обычный режим")
        response = consultation.answer_financial_literacy(
            "Как работает сложный процент?"
        )
        print(f"Ответ:\n{response}\n")
    
    print("\n=== Пример 3: Использование retrieve для получения контекста ===\n")
    
    if rag.client:
        query = "инфляция"
        print(f"Поиск по запросу: '{query}'\n")
        
        chunks_data = rag.retrieve(
            query=query,
            limit=5,
            score_threshold=0.2
        )
        
        print(f"Найдено фрагментов: {len(chunks_data)}\n")
        
        for i, chunk_data in enumerate(chunks_data[:3], 1):
            print(f"Фрагмент {i}:")
            print(f"  Текст: {chunk_data['content'][:150]}...")
            print(f"  Релевантность: {chunk_data['score']:.3f}")
            if chunk_data.get('source'):
                print(f"  Источник: {chunk_data['source']}")
            print()
    else:
        print("RAG не настроен, пропускаем этот пример\n")


if __name__ == "__main__":
    main()


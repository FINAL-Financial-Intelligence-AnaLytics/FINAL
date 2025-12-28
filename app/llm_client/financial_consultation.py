from typing import List, Dict, Optional, Any
from .base_client import FinancialLLMClient


class FinancialConsultationModule:
    def __init__(self, llm_client: FinancialLLMClient, rag_retriever=None):
        self.llm_client = llm_client
        self.rag_retriever = rag_retriever
    
    def search_answer_via_rag(self, query: str, top_k: int = 5) -> str:
        if not self.rag_retriever:
            return "RAG система не настроена. Пожалуйста, настройте rag_retriever."
        
        context_docs = self.rag_retriever.retrieve(query, top_k=top_k)
        formatted_context = self.llm_client._format_rag_context(context_docs)
        prompt = f"""Пользователь задал вопрос о финансах: "{query}"

{formatted_context}

Используй информацию из контекста, чтобы дать полный и понятный ответ.
Если информации недостаточно, объясни общие принципы по теме.
"""
        
        response = self.llm_client.generate_response(prompt, context_docs)
        return self.llm_client._add_safety_disclaimer(response)
    
    def explain_simple_language(self, term: str, context: Optional[str] = None) -> str:
        prompt = f"""Объясни финансовый термин "{term}" простым языком, как будто объясняешь другу без финансового образования.

Используй:
- Простые слова и аналогии из повседневной жизни
- Конкретные примеры
- Избегай сложной терминологии

{self._build_context_section(context)}
"""
        
        response = self.llm_client.generate_response(prompt)
        return response
    
    def answer_financial_literacy(self, question: str) -> str:
        if self.rag_retriever:
            context_docs = self.rag_retriever.retrieve(question, top_k=3)
            formatted_context = self.llm_client._format_rag_context(context_docs)
        else:
            formatted_context = ""
        
        prompt = f"""Вопрос по финансовой грамотности: "{question}"

{formatted_context}

Дай образовательный ответ, который поможет пользователю понять принципы финансовой грамотности.
Включи:
- Объяснение основных концепций
- Практические примеры
- Структурированную информацию (если уместно)
- Чек-листы или алгоритмы действий (если применимо)
"""
        
        response = self.llm_client.generate_response(prompt)
        return self.llm_client._add_safety_disclaimer(response)
    
    def compare_financial_products(self, products: List[str], comparison_criteria: Optional[List[str]] = None) -> str:
        if comparison_criteria is None:
            comparison_criteria = [
                "принцип работы",
                "основные характеристики",
                "типичные условия использования",
                "общие преимущества и ограничения"
            ]
        
        products_str = ", ".join(products)
        criteria_str = "\n".join(f"- {criterion}" for criterion in comparison_criteria)
        
        prompt = f"""Пользователь хочет сравнить финансовые продукты: {products_str}

ВАЖНО: НЕ говори, какой продукт лучше. Вместо этого:
- Объясни принципы работы каждого продукта
- Опиши их характеристики и особенности
- Объясни, в каких ситуациях каждый продукт может быть полезен
- Дай обобщенную информацию без персональных рекомендаций

Критерии для сравнения:
{criteria_str}

Структурируй ответ так, чтобы пользователь мог понять различия и сделать собственный выбор.
"""
        
        response = self.llm_client.generate_response(prompt)
        return self.llm_client._add_safety_disclaimer(response)
    
    def provide_calculation_example(self, calculation_type: str, parameters: Dict[str, Any]) -> str:
        params_str = "\n".join(f"- {k}: {v}" for k, v in parameters.items())
        
        prompt = f"""Покажи пример расчета: {calculation_type}

Параметры:
{params_str}

Включи:
- Пошаговое объяснение формулы
- Подстановку значений
- Результат расчета
- Интерпретацию результата простым языком
"""
        
        response = self.llm_client.generate_response(prompt)
        return response
    
    def provide_checklist(self, topic: str) -> str:
        prompt = f"""Создай подробный чек-лист по теме: "{topic}"

Чек-лист должен быть:
- Структурированным и понятным
- Практичным и применимым
- Без персональных рекомендаций конкретных продуктов
- Сфокусированным на принципах и шагах действий
"""
        
        response = self.llm_client.generate_response(prompt)
        return response
    
    def _build_context_section(self, context: Optional[str]) -> str:
        if context:
            return f"\nДополнительный контекст:\n{context}\n"
        return ""


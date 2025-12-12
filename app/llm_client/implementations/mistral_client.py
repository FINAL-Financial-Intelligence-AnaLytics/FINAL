from typing import List, Dict, Optional
import os
from ..base_client import FinancialLLMClient


class MistralFinancialClient(FinancialLLMClient):
    def __init__(self, model_name: str = "mistral-large-latest", api_key: Optional[str] = None):
        super().__init__(model_name, api_key)
        
        try:
            from mistralai import Mistral
            self.Mistral = Mistral
        except ImportError:
            raise ImportError(
                "Библиотека mistralai не установлена. "
                "Установите её: pip install mistralai"
            )
        
        api_key_value = api_key or os.getenv("MISTRAL_API_KEY")
        if not api_key_value:
            raise ValueError(
                "API ключ Mistral не предоставлен. "
                "Укажите его в параметре api_key или установите переменную окружения MISTRAL_API_KEY"
            )
        
        self.client = self.Mistral(api_key=api_key_value)
    
    def generate_response(
        self,
        prompt: str,
        context: Optional[List[Dict]] = None
    ) -> str:
        if not self._validate_request(prompt):
            return (
                "Извините, я не могу дать персональные финансовые рекомендации "
                "или сравнивать конкретные продукты. Я могу помочь вам понять "
                "принципы финансовой грамотности и объяснить общие концепции."
            )
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # Добавляем контекст из RAG, если он есть
        if context:
            context_text = self._format_rag_context(context)
            messages[-1]["content"] = prompt + context_text
        
        try:
            response = self.client.chat.complete(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=2000
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            return f"Ошибка при генерации ответа: {str(e)}"


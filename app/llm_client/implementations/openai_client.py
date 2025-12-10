from typing import List, Dict, Optional
import os
from ..base_client import FinancialLLMClient


class OpenAIFinancialClient(FinancialLLMClient):
    def __init__(self, model_name: str = "gpt-4", api_key: Optional[str] = None):
        super().__init__(model_name, api_key)
        
        try:
            from openai import OpenAI
            self.OpenAI = OpenAI
        except ImportError:
            raise ImportError(
                "Библиотека openai не установлена. "
                "Установите её: pip install openai"
            )
        
        api_key_value = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key_value:
            raise ValueError(
                "API ключ OpenAI не предоставлен. "
                "Укажите его в параметре api_key или установите переменную окружения OPENAI_API_KEY"
            )
        
        self.client = self.OpenAI(api_key=api_key_value)
    
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
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=2000
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            return f"Ошибка при генерации ответа: {str(e)}"


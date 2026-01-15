from typing import List, Dict, Optional
import os
from ..base_client import FinancialLLMClient
from .openrouter_client import OpenRouterLLM


class OpenRouterFinancialClient(FinancialLLMClient):
    def __init__(
        self,
        model_name: str = "tngtech/deepseek-r1t2-chimera:free",
        api_key: Optional[str] = None,
        site_url: Optional[str] = None,
        site_name: Optional[str] = None,
        temperature: float = 0.1
    ):
        super().__init__(model_name, api_key)
        
        api_key_value = api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key_value:
            raise ValueError(
                "API ключ OpenRouter не предоставлен. "
                "Укажите его в параметре api_key или установите переменную окружения OPENROUTER_API_KEY"
            )
        
        self.client = OpenRouterLLM(
            api_key=api_key_value,
            model=model_name,
            site_url=site_url or os.getenv("OPENROUTER_SITE_URL"),
            site_name=site_name or os.getenv("OPENROUTER_SITE_NAME"),
            temperature=temperature or float(os.getenv("OPENROUTER_TEMPERATURE", "0.1"))
        )
    
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
        
        full_prompt = prompt
        
        if context:
            context_text = self._format_rag_context(context)
            full_prompt = prompt + context_text
        
        try:
            response = self.client.generate(full_prompt, system_prompt=self.system_prompt)
            return response.strip()
        
        except Exception as e:
            return f"Ошибка при генерации ответа: {str(e)}"


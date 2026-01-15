from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import json


class FinancialLLMClient(ABC):
    ALLOWED_CAPABILITIES = [
        "обучать финансовой грамотности",
        "объяснять экономические понятия простыми словами",
        "показывать примеры расчётов",
        "давать структуры, чек-листы, алгоритмы",
        "объяснять принципы работы инструментов"
    ]
    
    FORBIDDEN_ACTIONS = [
        "давать персональные инвест-советы",
        "рекомендовать продукты",
        "сравнивать 'что лучше купить'",
        "давать юридические или налоговые рекомендации"
    ]
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        return f"""Ты - финансовый ассистент, специализирующийся на финансовой грамотности.

ТВОИ ВОЗМОЖНОСТИ:
{chr(10).join(f"- {capability}" for capability in self.ALLOWED_CAPABILITIES)}

ТВОИ ОГРАНИЧЕНИЯ (НИКОГДА НЕ ДЕЛАЙ ЭТОГО):
{chr(10).join(f"- {action}" for action in self.FORBIDDEN_ACTIONS)}

ВАЖНО:
- Всегда объясняй финансовые понятия простым языком
- Используй примеры и аналогии для лучшего понимания
- Предоставляй структурированную информацию (чек-листы, алгоритмы)
- Не давай персональных рекомендаций по конкретным продуктам
- Не сравнивай продукты с точки зрения "что лучше"
- Фокусируйся на обучении и объяснении принципов
"""
    
    @abstractmethod
    def generate_response(self, prompt: str, context: Optional[List[Dict]] = None) -> str:
        pass
    
    def _validate_request(self, request: str) -> bool:
        forbidden_keywords = [
            "рекомендую", "советую купить", "лучше всего",
            "юридическая консультация", "налоговый совет",
            "персональная рекомендация"
        ]
        
        request_lower = request.lower()
        for keyword in forbidden_keywords:
            if keyword in request_lower:
                return False
        return True
    
    def _format_rag_context(self, context: List[Dict]) -> str:
        if not context:
            return ""
        
        formatted = "\n\nКонтекст из базы знаний:\n"
        for i, doc in enumerate(context, 1):
            formatted += f"\n[{i}] {doc.get('content', '')}\n"
            if 'source' in doc:
                formatted += f"Источник: {doc['source']}\n"
        
        return formatted
    
    def _add_safety_disclaimer(self, response: str) -> str:
        disclaimer = "\n\nВажно: Эта информация носит образовательный характер и не является персональной финансовой рекомендацией."
        return response + disclaimer


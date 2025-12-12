import os
from typing import Optional


class Config:
    # Mistral настройки
    MISTRAL_API_KEY: Optional[str] = os.getenv("MISTRAL_API_KEY")
    MISTRAL_MODEL: str = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
    
    # Общие настройки
    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "5"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "2000"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
    ENABLE_REQUEST_VALIDATION: bool = os.getenv("ENABLE_REQUEST_VALIDATION", "true").lower() == "true"
    ADD_SAFETY_DISCLAIMER: bool = os.getenv("ADD_SAFETY_DISCLAIMER", "true").lower() == "true"
    
    @classmethod
    def validate(cls) -> bool:
        if not cls.MISTRAL_API_KEY:
            return False
        return True
    
    @classmethod
    def get_mistral_config(cls) -> dict:
        return {
            "model_name": cls.MISTRAL_MODEL,
            "api_key": cls.MISTRAL_API_KEY
        }


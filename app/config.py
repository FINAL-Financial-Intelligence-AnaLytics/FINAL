import os
from typing import Optional


class Config:
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "5"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "2000"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
    ENABLE_REQUEST_VALIDATION: bool = os.getenv("ENABLE_REQUEST_VALIDATION", "true").lower() == "true"
    ADD_SAFETY_DISCLAIMER: bool = os.getenv("ADD_SAFETY_DISCLAIMER", "true").lower() == "true"
    
    @classmethod
    def validate(cls) -> bool:
        if not cls.OPENAI_API_KEY:
            return False
        return True
    
    @classmethod
    def get_openai_config(cls) -> dict:
        return {
            "model_name": cls.OPENAI_MODEL,
            "api_key": cls.OPENAI_API_KEY
        }


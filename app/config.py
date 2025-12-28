import os
from typing import Optional


class Config:
    OPENROUTER_API_KEY: Optional[str] = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "tngtech/deepseek-r1t2-chimera:free")
    OPENROUTER_SITE_URL: Optional[str] = os.getenv("OPENROUTER_SITE_URL")
    OPENROUTER_SITE_NAME: Optional[str] = os.getenv("OPENROUTER_SITE_NAME")
    OPENROUTER_TEMPERATURE: float = float(os.getenv("OPENROUTER_TEMPERATURE", "0.1"))
    
    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "5"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "2000"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
    ENABLE_REQUEST_VALIDATION: bool = os.getenv("ENABLE_REQUEST_VALIDATION", "true").lower() == "true"
    ADD_SAFETY_DISCLAIMER: bool = os.getenv("ADD_SAFETY_DISCLAIMER", "true").lower() == "true"
    
    @classmethod
    def validate(cls) -> bool:
        if not cls.OPENROUTER_API_KEY:
            return False
        return True
    
    @classmethod
    def get_openrouter_config(cls) -> dict:
        return {
            "model_name": cls.OPENROUTER_MODEL,
            "api_key": cls.OPENROUTER_API_KEY,
            "site_url": cls.OPENROUTER_SITE_URL,
            "site_name": cls.OPENROUTER_SITE_NAME,
            "temperature": cls.OPENROUTER_TEMPERATURE
        }


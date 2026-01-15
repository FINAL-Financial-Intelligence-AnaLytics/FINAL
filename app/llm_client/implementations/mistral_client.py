import requests
import json
from typing import Optional


class MistralLLM:
    def __init__(
        self,
        api_key: str,
        model: str = "mistral-small-latest",
        base_url: str = "https://api.mistral.ai/v1",
        temperature: float = 0.1,
    ):
        if not api_key:
            raise ValueError("api_key не может быть пустым. Установите MISTRAL_API_KEY в .env файле")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.temperature = temperature

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self.api_key:
            raise ValueError("API ключ не установлен. Проверьте MISTRAL_API_KEY в .env файле")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        default_system_prompt = (
            "You are a factual assistant. Use ONLY the provided context. "
            "If the context is insufficient, say so. "
            "Do NOT reveal chain-of-thought; provide only the final answer."
        )
        
        system_content = system_prompt if system_prompt else default_system_prompt

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_content,
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
        }

        url = f"{self.base_url}/chat/completions"
        resp = requests.post(
            url,
            headers=headers,
            data=json.dumps(payload),
            timeout=90,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

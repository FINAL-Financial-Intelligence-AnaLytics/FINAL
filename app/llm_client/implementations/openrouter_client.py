import requests
import json
from typing import Optional


class OpenRouterLLM:
    def __init__(
        self,
        api_key: str,
        model: str = "tngtech/deepseek-r1t2-chimera:free",
        site_url: Optional[str] = None,
        site_name: Optional[str] = None,
        temperature: float = 0.1,
    ):
        self.api_key = api_key
        self.model = model
        self.site_url = site_url
        self.site_name = site_name
        self.temperature = temperature

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            headers["X-Title"] = self.site_name

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

        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            timeout=90,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


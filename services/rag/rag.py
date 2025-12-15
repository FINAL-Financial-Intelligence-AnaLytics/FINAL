from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from IPython.display import display, Markdown

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sentence_transformers import SentenceTransformer

from config import settings
from models import RAGChunk
from llm_client import OpenRouterLLM




def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def _format_context(chunks: List[RAGChunk], max_chars: int = 8000) -> str:
    """
    Склеиваем чанки в контекст.
    max_chars — защита от слишком длинного промпта.
    """
    parts = []
    total = 0
    for i, c in enumerate(chunks, 1):
        piece = f"[{i}] {c.text.strip()}"
        if total + len(piece) > max_chars:
            break
        parts.append(piece)
        total += len(piece)
    return "\n\n".join(parts)

def normalize_newlines(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # убираем тройные переносы
    while "\n\n\n" in s:
        s = s.replace("\n\n\n", "\n\n")
    return s.strip()


DEFAULT_SYSTEM_PROMPT = """Ты — помощник, отвечающий строго на основе предоставленного контекста.

Правила:
1) Используй только факты из КОНТЕКСТА. Если в контексте нет ответа — скажи: "В предоставленных материалах нет информации".
2) Пиши кратко и по делу. Если вопрос многосоставной — ответь по пунктам.
3) Не раскрывай цепочку рассуждений. Дай только итоговый ответ.
"""

DEFAULT_USER_PROMPT = """КОНТЕКСТ:
{context}

ВОПРОС:
{question}

ФОРМАТ ОТВЕТА:
- Заголовок: 1 строка
- Далее 3–6 буллетов (каждый буллет — 1 мысль)
- Если нужны определения — 1–2 предложения, без длинных эссе
- Не используй эмодзи и декоративные разделители (---)
- Не добавляй ссылки на внешние сайты
- Если информации недостаточно — напиши одну фразу об этом

ОТВЕТ:
"""

class RAGModule:
    def __init__(
        self,
        collection: str = "finance_theory",
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        llm: Optional[LLMClient] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        user_prompt: str = DEFAULT_USER_PROMPT,
    ):
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=30,
            check_compatibility=False,
        )
        self.collection = collection

        model_name = model_name or os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
        self.model = SentenceTransformer(model_name, device=device) if device else SentenceTransformer(model_name)
        self.model_is_e5 = "e5" in model_name.lower()

        self.llm = llm  # сюда прокидываешь свою LLM-реализацию
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        if self.llm is None:
            openrouter_key = getattr(settings, "openrouter_api_key", None) or os.getenv("OPENROUTER_API_KEY")
            if openrouter_key:
                self.llm = OpenRouterLLM(
                    api_key=openrouter_key,
                    model=os.getenv("OPENROUTER_MODEL", "tngtech/deepseek-r1t2-chimera:free"),
                    site_url=os.getenv("OPENROUTER_SITE_URL"),
                    site_name=os.getenv("OPENROUTER_SITE_NAME"),
                    temperature=float(os.getenv("OPENROUTER_TEMPERATURE", "0.1")),
                )        

    def _embed_query(self, query: str) -> np.ndarray:
        q = f"query: {query}" if self.model_is_e5 else query
        vec = self.model.encode(q, normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float32)

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        if self.model_is_e5:
            texts = [f"passage: {t}" for t in texts]
        vecs = self.model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)

    def _mmr_select(
        self,
        query_vec: np.ndarray,
        cand_texts: List[str],
        cand_scores: List[float],
        top_k: int,
        lambda_mult: float = 0.7,
    ) -> List[int]:
        cand_vecs = self._embed_texts(cand_texts)

        selected: List[int] = []
        remaining = list(range(len(cand_texts)))

        first = int(np.argmax(np.asarray(cand_scores)))
        selected.append(first)
        remaining.remove(first)

        while remaining and len(selected) < top_k:
            best_idx = None
            best_val = -1e9

            for idx in remaining:
                rel = _cosine_sim(query_vec, cand_vecs[idx])
                div = max(_cosine_sim(cand_vecs[idx], cand_vecs[s]) for s in selected)
                mmr = lambda_mult * rel - (1 - lambda_mult) * div
                if mmr > best_val:
                    best_val = mmr
                    best_idx = idx

            selected.append(best_idx)
            remaining.remove(best_idx)

        return selected

    def retrieve(
        self,
        query: str,
        limit: int = 5,
        prefetch: int = 25,
        score_threshold: Optional[float] = 0.2,
        mmr: bool = True,
        lambda_mult: float = 0.7,
        qdrant_filter: Optional[qm.Filter] = None,
    ) -> List[RAGChunk]:
        qvec = self._embed_query(query)

        results = self.client.search(
            collection_name=self.collection,
            query_vector=qvec.tolist(),
            limit=prefetch if mmr else limit,
            query_filter=qdrant_filter,
            score_threshold=score_threshold,
        )

        if not results:
            return []

        if not mmr:
            return [RAGChunk(text=r.payload.get("text", ""), score=float(r.score)) for r in results[:limit]]

        cand_texts = [r.payload.get("text", "") for r in results]
        cand_scores = [float(r.score) for r in results]

        pick_idx = self._mmr_select(
            query_vec=qvec,
            cand_texts=cand_texts,
            cand_scores=cand_scores,
            top_k=min(limit, len(results)),
            lambda_mult=lambda_mult,
        )

        return [RAGChunk(text=results[i].payload.get("text", ""), score=float(results[i].score)) for i in pick_idx]

    def build_prompt(self, question: str, chunks: List[RAGChunk]) -> str:
        context = _format_context(chunks)
        return self.system_prompt + "\n\n" + self.user_prompt.format(context=context, question=question)

    def generate(self, prompt: str) -> str:
        if not self.llm:
            raise RuntimeError("LLM client is not set. Pass llm=... into RAGModule.")
        return self.llm.generate(prompt)

    def answer(
        self,
        question: str,
        limit: int = 5,
        prefetch: int = 25,
        score_threshold: Optional[float] = 0.2,
        mmr: bool = True,
        lambda_mult: float = 0.7,
        qdrant_filter: Optional[qm.Filter] = None,
    ) -> Dict[str, Any]:
        chunks = self.retrieve(
            query=question,
            limit=limit,
            prefetch=prefetch,
            score_threshold=score_threshold,
            mmr=mmr,
            lambda_mult=lambda_mult,
            qdrant_filter=qdrant_filter,
        )
        if not chunks:
            return {"answer": "В предоставленных материалах нет информации, чтобы уверенно ответить на вопрос.", "chunks": []}

        if score_threshold is not None and max(c.score for c in chunks) < score_threshold:
            return {"answer": "В предоставленных материалах нет информации, чтобы уверенно ответить на вопрос.", "chunks": chunks}

        if not chunks:
            return {
                "answer": "В предоставленных материалах нет информации, чтобы уверенно ответить на вопрос.",
                "chunks": [],
            }
        prompt = self.build_prompt(question, chunks)
        text = self.generate(prompt)
        text = normalize_newlines(text)

        return {
            "answer": display(Markdown(text)), #эту часть стоит перенести в бота (from IPython.display import display, Markdown)
            "chunks": chunks,  
        }

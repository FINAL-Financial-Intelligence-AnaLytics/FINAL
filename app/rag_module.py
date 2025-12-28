from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sentence_transformers import SentenceTransformer

from app.config import Config
from app.models import RAGChunk
from app.llm_client.implementations.openrouter_client import OpenRouterLLM


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def _format_context(chunks: List[RAGChunk], max_chars: int = 8000) -> str:
    parts = []
    total = 0
    for i, c in enumerate(chunks, 1):
        piece = f"[{i}] {c.text.strip()}"
        if total + len(piece) > max_chars:
            break
        parts.append(piece)
        total += len(piece)
    return "\n\n".join(parts)


DEFAULT_SYSTEM_PROMPT = """Ты — помощник, отвечающий строго на основе предоставленного контекста.

Правила:
1) Используй только факты из КОНТЕКСТА. Если в контексте нет ответа — скажи: "В предоставленных материалах нет информации".
2) В конце каждого важного утверждения ставь ссылку на источник в формате [1], [2] — номер соответствующего фрагмента из КОНТЕКСТА.
3) Пиши кратко и по делу. Если вопрос многосоставной — ответь по пунктам.
4) Не раскрывай цепочку рассуждений. Дай только итоговый ответ.
"""

DEFAULT_USER_PROMPT = """КОНТЕКСТ:
{context}

ВОПРОС:
{question}

ТРЕБОВАНИЯ К ОТВЕТУ:
- Ответь по делу.
- После каждого ключевого утверждения добавляй ссылку на фрагмент [1], [2], ...
- Если данных недостаточно — так и скажи.

ОТВЕТ:
"""


class RAGModule:
    def __init__(
        self,
        collection: str = "finance_theory",
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        llm: Optional[OpenRouterLLM] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        user_prompt: str = DEFAULT_USER_PROMPT,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
    ):
        qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
        qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
        
        if qdrant_url:
            self.client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key,
                timeout=30,
                check_compatibility=False,
            )
        else:
            self.client = None
            
        self.collection = collection

        model_name = model_name or os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
        self.model = SentenceTransformer(model_name, device=device) if device else SentenceTransformer(model_name)
        self.model_is_e5 = "e5" in model_name.lower()

        self.llm = llm
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        if self.llm is None:
            openrouter_key = Config.OPENROUTER_API_KEY or os.getenv("OPENROUTER_API_KEY")
            if openrouter_key:
                self.llm = OpenRouterLLM(
                    api_key=openrouter_key,
                    model=Config.OPENROUTER_MODEL,
                    site_url=Config.OPENROUTER_SITE_URL,
                    site_name=Config.OPENROUTER_SITE_NAME,
                    temperature=Config.OPENROUTER_TEMPERATURE,
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
        top_k: Optional[int] = None,
        prefetch: int = 25,
        score_threshold: Optional[float] = 0.2,
        mmr: bool = True,
        lambda_mult: float = 0.7,
        qdrant_filter: Optional[qm.Filter] = None,
    ) -> List[Dict[str, Any]]:
        if top_k is not None:
            limit = top_k
        if not self.client:
            return []
            
        qvec = self._embed_query(query)

        results = self.client.query_points(
            collection_name=self.collection,
            query=qvec.tolist(),
            limit=prefetch if mmr else limit,
            query_filter=qdrant_filter,
            score_threshold=score_threshold,
            with_payload=True,
        )

        if not results or not results.points:
            return []

        if not mmr:
            return [
                {"content": r.payload.get("text", ""), "score": float(r.score), "source": r.payload.get("source", "")}
                for r in results.points[:limit]
            ]

        cand_texts = [r.payload.get("text", "") for r in results.points]
        cand_scores = [float(r.score) for r in results.points]

        pick_idx = self._mmr_select(
            query_vec=qvec,
            cand_texts=cand_texts,
            cand_scores=cand_scores,
            top_k=min(limit, len(results.points)),
            lambda_mult=lambda_mult,
        )

        return [
            {"content": results.points[i].payload.get("text", ""), "score": float(results.points[i].score), "source": results.points[i].payload.get("source", "")}
            for i in pick_idx
        ]

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
        chunks_data = self.retrieve(
            query=question,
            limit=limit,
            prefetch=prefetch,
            score_threshold=score_threshold,
            mmr=mmr,
            lambda_mult=lambda_mult,
            qdrant_filter=qdrant_filter,
        )
        
        if not chunks_data:
            return {"answer": "В предоставленных материалах нет информации, чтобы уверенно ответить на вопрос.", "chunks": []}

        chunks = [RAGChunk(text=c["content"], score=c["score"]) for c in chunks_data]
        
        if score_threshold is not None and max(c.score for c in chunks) < score_threshold:
            return {"answer": "В предоставленных материалах нет информации, чтобы уверенно ответить на вопрос.", "chunks": chunks}

        prompt = self.build_prompt(question, chunks)
        text = self.generate(prompt)

        return {
            "answer": text,
            "chunks": chunks,  
        }


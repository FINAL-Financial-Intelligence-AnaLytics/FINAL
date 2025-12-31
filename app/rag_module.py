from __future__ import annotations

import os
import logging
import json
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import re

from app.config import Config
from app.models import RAGChunk
from app.llm_client.implementations.mistral_client import MistralLLM

logger = logging.getLogger(__name__)


class RetrievalMode(Enum):
    DENSE_ONLY = "dense_only"
    BM25_ONLY = "bm25_only"
    HYBRID = "hybrid"


class RewriteMode(Enum):
    NONE = "none"
    FROZEN_LLM = "frozen_llm"
    TRAINABLE = "trainable"


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


def _tokenize(text: str) -> List[str]:
    """Simple tokenizer for BM25"""
    text = text.lower().strip()
    tokens = text.split()
    tokens = [t for t in tokens if len(t) > 1]
    return tokens


class QueryRewriter:
    """Query Rewriter based on the paper's approach"""

    def __init__(
            self,
            mode: RewriteMode = RewriteMode.NONE,
            llm: Optional[MistralLLM] = None,
            trainable_model_path: Optional[str] = None
    ):
        self.mode = mode
        self.llm = llm
        self.trainable_model_path = trainable_model_path

        # For trainable rewriter (simplified - in practice you'd load a trained model)
        self.trainable_model = None
        if mode == RewriteMode.TRAINABLE and trainable_model_path:
            self._load_trainable_model()

    def _load_trainable_model(self):
        """Load a trained rewriter model (simplified implementation)"""
        # In practice, you would load a trained T5 or similar model here
        # For now, we'll use a placeholder that returns the original query
        logger.info(f"Trainable rewriter would load from {self.trainable_model_path}")

    def rewrite(
            self,
            query: str,
            task_type: str = "qa",  # "qa" or "multiple_choice"
            few_shot_examples: Optional[List[Tuple[str, str]]] = None
    ) -> str:
        """Rewrite the query based on the selected mode"""
        if self.mode == RewriteMode.NONE:
            return query

        elif self.mode == RewriteMode.FROZEN_LLM:
            return self._rewrite_with_frozen_llm(query, task_type, few_shot_examples)

        elif self.mode == RewriteMode.TRAINABLE:
            return self._rewrite_with_trainable(query, task_type)

        return query

    def _rewrite_with_frozen_llm(
            self,
            query: str,
            task_type: str,
            few_shot_examples: Optional[List[Tuple[str, str]]] = None
    ) -> str:
        """Rewrite query using a frozen LLM with few-shot prompting"""
        if not self.llm:
            logger.warning("No LLM provided for frozen rewriter, returning original query")
            return query

        # Build prompt based on paper's approach
        prompt = self._build_rewrite_prompt(query, task_type, few_shot_examples)

        try:
            rewritten = self.llm.generate(prompt)
            # Clean and extract the query
            rewritten = self._extract_rewritten_query(rewritten)
            logger.info(f"Query rewritten: '{query}' -> '{rewritten}'")
            return rewritten
        except Exception as e:
            logger.error(f"Error in query rewriting: {e}")
            return query

    def _rewrite_with_trainable(self, query: str, task_type: str) -> str:
        """Rewrite query using a trainable model"""
        # Simplified implementation - in practice you would:
        # 1. Tokenize input
        # 2. Run through trained model
        # 3. Decode output
        if self.trainable_model:
            # Actual inference would go here
            pass

        # For now, return a simple improvement
        # This is just a placeholder - you should train your own rewriter
        return self._simple_query_improvement(query)

    def _simple_query_improvement(self, query: str) -> str:
        """Simple query improvement rules (placeholder for trainable model)"""
        # Add common question words if missing
        question_words = ["что", "как", "почему", "когда", "где", "кто"]
        if not any(word in query.lower() for word in question_words):
            # Not a question format, add context
            if "финанс" in query.lower() or "инвест" in query.lower():
                return f"финансовый вопрос: {query}"

        # Remove unnecessary words
        stop_words = ["пожалуйста", "расскажите", "объясните", "можно", "ли"]
        words = query.split()
        filtered_words = [w for w in words if w.lower() not in stop_words]

        return " ".join(filtered_words) if filtered_words else query

    def _build_rewrite_prompt(
            self,
            query: str,
            task_type: str,
            examples: Optional[List[Tuple[str, str]]] = None
    ) -> str:
        """Build prompt for query rewriting based on paper's approach"""

        if task_type == "qa":
            instruction = """Перефразируй вопрос для улучшения поиска в финансовой базе знаний. 
Создай поисковый запрос, который лучше отражает суть вопроса и содержит все ключевые термины."""

            if not examples:
                # Default few-shot examples for finance QA
                examples = [
                    ("Что такое диверсификация портфеля?",
                     "диверсификация портфеля инвестиций стратегия риски"),

                    ("Как рассчитать доходность акций?",
                     "расчет доходности акций формула методы"),

                    ("Какие бывают виды облигаций?",
                     "типы виды облигации государственные корпоративные")
                ]

        else:  # multiple_choice or other
            instruction = """Создай поисковый запрос для поиска информации, необходимой для ответа на вопрос с вариантами ответов."""

            if not examples:
                examples = [
                    ("Что такое NPV? а) Net Present Value б) Net Profit Value в) Net Portfolio Value",
                     "NPV Net Present Value определение чистая приведенная стоимость")
                ]

        # Build demonstrations
        demonstrations = ""
        if examples:
            for original, rewritten in examples:
                demonstrations += f"Вопрос: {original}\n"
                demonstrations += f"Поисковый запрос: {rewritten}\n\n"

        prompt = f"""{instruction}

{demonstrations}Вопрос: {query}
Поисковый запрос:"""

        return prompt

    def _extract_rewritten_query(self, text: str) -> str:
        """Extract the rewritten query from LLM response"""
        # Remove any markdown code blocks
        text = text.strip()
        text = re.sub(r'```[a-z]*\n', '', text)
        text = text.replace('```', '')

        # Look for the query in the response
        lines = text.strip().split('\n')
        for line in lines:
            line_lower = line.lower()
            if 'запрос:' in line_lower or 'query:' in line_lower:
                parts = line.split(':', 1)
                if len(parts) > 1:
                    return parts[1].strip()

        # If no clear marker, return the last non-empty line
        for line in reversed(lines):
            if line.strip():
                return line.strip()

        return text.strip() if text.strip() else text


@dataclass
class HybridDocument:
    """Container for documents with both vector and BM25 scores"""
    id: Any
    text: str
    source: str = ""
    vector_score: float = 0.0
    bm25_score: float = 0.0
    is_bm25_only: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    combined_score: float = 0.0  # Added to avoid AttributeError


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
            llm: Optional[MistralLLM] = None,
            system_prompt: str = DEFAULT_SYSTEM_PROMPT,
            user_prompt: str = DEFAULT_USER_PROMPT,
            qdrant_url: Optional[str] = None,
            qdrant_api_key: Optional[str] = None,
            retrieval_mode: RetrievalMode = RetrievalMode.HYBRID,
            rewrite_mode: RewriteMode = RewriteMode.NONE,
            bm25_prefetch: int = 100,
            vector_weight: float = 0.7,
            enable_mmr: bool = True,
            mmr_lambda: float = 0.7,
    ):
        # Initialize Qdrant client
        qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
        qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")

        if qdrant_url:
            try:
                self.client = QdrantClient(
                    url=qdrant_url,
                    api_key=qdrant_api_key if qdrant_api_key else None,
                    timeout=30,
                    check_compatibility=False,
                )
                try:
                    self.client.get_collections()
                except Exception as e:
                    logger.warning(f"Не удалось подключиться к Qdrant: {e}")
                    self.client = None
            except Exception as e:
                logger.error(f"Ошибка при создании клиента Qdrant: {e}")
                self.client = None
        else:
            self.client = None

        self.collection = collection
        self.retrieval_mode = retrieval_mode
        self.rewrite_mode = rewrite_mode  # Store rewrite_mode
        self.bm25_prefetch = bm25_prefetch
        self.vector_weight = vector_weight
        self.enable_mmr = enable_mmr
        self.mmr_lambda = mmr_lambda

        # Initialize embedding model
        model_name = model_name or os.getenv("EMBEDDING_MODEL")
        self.model_name = model_name
        self.device = device
        self._model = None
        self.model_is_e5 = "e5" in model_name.lower() if model_name else False
        self._model_enabled = model_name is not None

        # Initialize BM25
        self.bm25_index = None
        self.bm25_docs = []
        self.bm25_metadata = []
        if retrieval_mode in [RetrievalMode.BM25_ONLY, RetrievalMode.HYBRID] and self.client:
            self._initialize_bm25_index()

        # Initialize LLM for reader
        self.llm = llm
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        if self.llm is None:
            mistral_key = Config.MISTRAL_API_KEY or os.getenv("MISTRAL_API_KEY")
            if mistral_key:
                self.llm = MistralLLM(
                    api_key=mistral_key,
                    model=Config.MISTRAL_MODEL,
                    base_url=Config.MISTRAL_BASE_URL,
                    temperature=Config.MISTRAL_TEMPERATURE,
                )

        # Initialize Query Rewriter
        self.rewriter = QueryRewriter(
            mode=rewrite_mode,
            llm=self.llm,  # Use same LLM for rewriting
            trainable_model_path=None  # Set if you have a trained model
        )

    def _initialize_bm25_index(self):
        """Initialize BM25 index from full corpus in Qdrant"""
        try:
            logger.info("Building BM25 index from full corpus...")

            all_points = []
            offset = 0
            limit = 1000

            while True:
                response = self.client.scroll(
                    collection_name=self.collection,
                    limit=limit,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )

                points, next_offset = response
                if not points:
                    break

                all_points.extend(points)

                if next_offset is None:
                    break
                offset = next_offset

            self.bm25_docs = []
            self.bm25_metadata = []

            for point in all_points:
                text = point.payload.get("text", "")
                if text and len(text.strip()) > 0:
                    self.bm25_docs.append(text)
                    self.bm25_metadata.append({
                        "id": point.id,
                        "payload": point.payload,
                    })

            if self.bm25_docs:
                tokenized_docs = [_tokenize(doc) for doc in self.bm25_docs]
                self.bm25_index = BM25Okapi(tokenized_docs)
                logger.info(f"✅ BM25 index built with {len(self.bm25_docs)} documents")
            else:
                logger.warning("No documents found for BM25 index")
                self.bm25_index = None

        except Exception as e:
            logger.error(f"Failed to initialize BM25 index: {e}")
            self.bm25_index = None

    def refresh_bm25_index(self):
        """Refresh BM25 index (call after adding new documents to Qdrant)"""
        if self.retrieval_mode in [RetrievalMode.BM25_ONLY, RetrievalMode.HYBRID] and self.client:
            self._initialize_bm25_index()

    @property
    def model(self) -> Optional[SentenceTransformer]:
        if not self._model_enabled:
            return None
        if self._model is None:
            logger.info(f"Загрузка модели эмбеддингов: {self.model_name}")
            if self.device:
                self._model = SentenceTransformer(self.model_name, device=self.device)
            else:
                self._model = SentenceTransformer(self.model_name, device="cpu")
            logger.info(f"✅ Модель {self.model_name} загружена")
        return self._model

    def _embed_query(self, query: str) -> np.ndarray:
        if not self._model_enabled or self.model is None:
            raise RuntimeError(
                "Модель эмбеддингов не загружена. Установите EMBEDDING_MODEL в .env или передайте model_name в RAGModule")
        q = f"query: {query}" if self.model_is_e5 else query
        vec = self.model.encode(q, normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float32)

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        if not self._model_enabled or self.model is None:
            raise RuntimeError(
                "Модель эмбеддингов не загружена. Установите EMBEDDING_MODEL в .env или передайте model_name в RAGModule")
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
        if not cand_texts or not cand_scores:
            return []

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

            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)
            else:
                break

        return selected

    def _get_vector_results(
            self,
            query_vec: np.ndarray,
            prefetch: int,
            score_threshold: Optional[float],
            qdrant_filter: Optional[qm.Filter]
    ) -> List[HybridDocument]:
        """Get vector search results from Qdrant"""
        if not self.client:
            return []

        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vec.tolist(),
            limit=prefetch,
            query_filter=qdrant_filter,
            score_threshold=score_threshold,
            with_payload=True,
        )

        if not results or not results.points:
            return []

        docs = []
        for r in results.points:
            docs.append(HybridDocument(
                id=r.id,
                text=r.payload.get("text", ""),
                source=r.payload.get("source", ""),
                vector_score=float(r.score),
                bm25_score=0.0,
                is_bm25_only=False,
                metadata=r.payload
            ))
        return docs

    def _get_bm25_results(self, query: str, prefetch: int) -> List[HybridDocument]:
        """Get BM25 search results from full corpus"""
        if not self.bm25_index or not query.strip():
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        bm25_scores = self.bm25_index.get_scores(query_tokens)

        top_n = min(len(bm25_scores), prefetch)
        top_indices = np.argsort(bm25_scores)[::-1][:top_n]

        docs = []
        for idx in top_indices:
            score = float(bm25_scores[idx])
            if score > 0:
                metadata = self.bm25_metadata[idx]
                docs.append(HybridDocument(
                    id=metadata["id"],
                    text=self.bm25_docs[idx],
                    source=metadata["payload"].get("source", ""),
                    vector_score=0.0,
                    bm25_score=score,
                    is_bm25_only=True,
                    metadata=metadata["payload"]
                ))
        return docs

    def _combine_results(
            self,
            vector_docs: List[HybridDocument],
            bm25_docs: List[HybridDocument],
    ) -> List[HybridDocument]:
        """Combine vector and BM25 results"""
        docs_by_id = {}

        for doc in vector_docs:
            docs_by_id[doc.id] = doc

        for doc in bm25_docs:
            if doc.id in docs_by_id:
                existing = docs_by_id[doc.id]
                existing.bm25_score = doc.bm25_score
                existing.is_bm25_only = False
            else:
                docs_by_id[doc.id] = doc

        combined_docs = []
        for doc in docs_by_id.values():
            # Normalize scores
            if vector_docs:
                vector_scores = [d.vector_score for d in vector_docs]
                if len(vector_scores) > 0 and max(vector_scores) > min(vector_scores):
                    norm_vector = (doc.vector_score - min(vector_scores)) / (max(vector_scores) - min(vector_scores))
                else:
                    norm_vector = doc.vector_score
            else:
                norm_vector = 0.0

            if bm25_docs:
                bm25_scores = [d.bm25_score for d in bm25_docs]
                if len(bm25_scores) > 0 and max(bm25_scores) > min(bm25_scores):
                    norm_bm25 = (doc.bm25_score - min(bm25_scores)) / (max(bm25_scores) - min(bm25_scores))
                else:
                    norm_bm25 = doc.bm25_score
            else:
                norm_bm25 = 0.0

            # Apply vector weight
            combined_score = (self.vector_weight * norm_vector) + ((1 - self.vector_weight) * norm_bm25)

            combined_doc = HybridDocument(
                id=doc.id,
                text=doc.text,
                source=doc.source,
                vector_score=doc.vector_score,
                bm25_score=doc.bm25_score,
                is_bm25_only=doc.is_bm25_only,
                metadata=doc.metadata,
                combined_score=combined_score
            )
            combined_docs.append(combined_doc)

        return combined_docs

    def retrieve(
            self,
            query: str,
            limit: int = 7,
            prefetch: int = 25,
            score_threshold: Optional[float] = 0.2,
            qdrant_filter: Optional[qm.Filter] = None,
            task_type: str = "qa",
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval with optional query rewriting.

        Args:
            query: Original query
            limit: Number of final results
            prefetch: Number of candidates to prefetch
            score_threshold: Minimum score threshold
            qdrant_filter: Qdrant filter
            task_type: Type of task ("qa", "multiple_choice")
        """
        # Step 1: Query Rewriting
        original_query = query
        if self.rewriter.mode != RewriteMode.NONE:
            query = self.rewriter.rewrite(query, task_type)
            logger.info(f"Query rewriting: '{original_query}' -> '{query}'")

        if not self.client:
            return []

        # Step 2: Get results based on retrieval mode
        vector_docs = []
        bm25_docs = []

        if self.retrieval_mode in [RetrievalMode.DENSE_ONLY, RetrievalMode.HYBRID]:
            if not self._model_enabled:
                raise RuntimeError(
                    "Модель эмбеддингов не настроена. Установите EMBEDDING_MODEL в .env для использования векторного поиска")
            query_vec = self._embed_query(query)
            vector_docs = self._get_vector_results(
                query_vec=query_vec,
                prefetch=prefetch if self.enable_mmr else limit,
                score_threshold=score_threshold,
                qdrant_filter=qdrant_filter
            )

        if self.retrieval_mode in [RetrievalMode.BM25_ONLY, RetrievalMode.HYBRID]:
            if self.bm25_index:
                bm25_docs = self._get_bm25_results(query, prefetch=self.bm25_prefetch)
            else:
                logger.warning("BM25 index not available")

        # Step 3: Combine results (if hybrid)
        if self.retrieval_mode == RetrievalMode.HYBRID:
            combined_docs = self._combine_results(vector_docs, bm25_docs)
        elif self.retrieval_mode == RetrievalMode.DENSE_ONLY:
            combined_docs = vector_docs
        else:  # BM25_ONLY
            combined_docs = bm25_docs

        if not combined_docs:
            return []

        # Step 4: Apply MMR or simple sort
        if self.enable_mmr and len(
                combined_docs) > 1 and self.retrieval_mode != RetrievalMode.BM25_ONLY and self._model_enabled:
            query_vec = self._embed_query(query)
            texts = [doc.text for doc in combined_docs]
            scores = [doc.combined_score if hasattr(doc, 'combined_score') and doc.combined_score > 0 else
                      (doc.vector_score if self.retrieval_mode == RetrievalMode.DENSE_ONLY else doc.bm25_score)
                      for doc in combined_docs]

            pick_idx = self._mmr_select(
                query_vec=query_vec,
                cand_texts=texts,
                cand_scores=scores,
                top_k=min(limit, len(combined_docs)),
                lambda_mult=self.mmr_lambda,
            )
            selected = [combined_docs[i] for i in pick_idx]
        else:
            # Simple sort by appropriate score
            if self.retrieval_mode == RetrievalMode.DENSE_ONLY:
                selected = sorted(combined_docs, key=lambda x: x.vector_score, reverse=True)[:limit]
            elif self.retrieval_mode == RetrievalMode.BM25_ONLY:
                selected = sorted(combined_docs, key=lambda x: x.bm25_score, reverse=True)[:limit]
            else:  # HYBRID
                selected = sorted(combined_docs, key=lambda x: x.combined_score, reverse=True)[:limit]

        # Step 5: Format return
        return [
            {
                "content": doc.text,
                "score": (doc.combined_score if self.retrieval_mode == RetrievalMode.HYBRID else
                          (doc.vector_score if self.retrieval_mode == RetrievalMode.DENSE_ONLY else doc.bm25_score)),
                "source": doc.source,
                "metadata": {
                    "vector_score": doc.vector_score,
                    "bm25_score": doc.bm25_score,
                    "is_bm25_only": doc.is_bm25_only,
                    "original_query": original_query,
                    "rewritten_query": query,
                }
            }
            for doc in selected
        ]

    def build_prompt(self, question: str, chunks: List[RAGChunk]) -> str:
        context = _format_context(chunks)
        return self.system_prompt + "\n\n" + self.user_prompt.format(context=context, question=question)

    def generate(self, prompt: str) -> str:
        if not self.llm:
            raise RuntimeError("LLM client is not set. Pass llm=... into RAGModule.")
        try:
            return self.llm.generate(prompt)
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "Unauthorized" in error_msg:
                raise RuntimeError(
                    f"Ошибка аутентификации Mistral API. "
                    f"Проверьте, что MISTRAL_API_KEY установлен правильно. "
                    f"Ошибка: {error_msg}"
                ) from e
            raise

    def answer(
            self,
            question: str,
            limit: int = 5,
            prefetch: int = 25,
            score_threshold: Optional[float] = 0.2,
            qdrant_filter: Optional[qm.Filter] = None,
            task_type: str = "qa",
    ) -> Dict[str, Any]:
        chunks_data = self.retrieve(
            query=question,
            limit=limit,
            prefetch=prefetch,
            score_threshold=score_threshold,
            qdrant_filter=qdrant_filter,
            task_type=task_type,
        )

        if not chunks_data:
            return {
                "answer": "В предоставленных материалах нет информации, чтобы уверенно ответить на вопрос.",
                "chunks": [],
                "query_info": {
                    "original": question,
                    "rewritten": question,
                    "retrieval_mode": self.retrieval_mode.value,
                    "rewrite_mode": self.rewriter.mode.value
                }
            }

        chunks = [RAGChunk(text=c["content"], score=c["score"]) for c in chunks_data]

        if score_threshold is not None and chunks and max(c.score for c in chunks) < score_threshold:
            return {
                "answer": "В предоставленных материалах нет информации, чтобы уверенно ответить на вопрос.",
                "chunks": chunks,
                "query_info": {
                    "original": chunks_data[0]["metadata"]["original_query"],
                    "rewritten": chunks_data[0]["metadata"]["rewritten_query"],
                    "retrieval_mode": self.retrieval_mode.value,
                    "rewrite_mode": self.rewriter.mode.value
                }
            }

        prompt = self.build_prompt(question, chunks)
        text = self.generate(prompt)

        return {
            "answer": text,
            "chunks": chunks,
            "query_info": {
                "original": chunks_data[0]["metadata"]["original_query"],
                "rewritten": chunks_data[0]["metadata"]["rewritten_query"],
                "retrieval_mode": self.retrieval_mode.value,
                "rewrite_mode": self.rewriter.mode.value
            }
        }

    def run_comparison_test(
            self,
            question: str,
            test_name: str = "Comparison Test"
    ) -> Dict[str, Any]:
        """
        Run a comparison test with different configurations.
        Based on the paper's evaluation approach.
        """
        print("=" * 60)
        print(f"{test_name}")
        print("=" * 60)

        results = {}

        # Test different retrieval modes
        retrieval_modes = [RetrievalMode.DENSE_ONLY, RetrievalMode.BM25_ONLY, RetrievalMode.HYBRID]
        rewrite_modes = [RewriteMode.NONE, RewriteMode.FROZEN_LLM]

        for retrieval_mode in retrieval_modes:
            for rewrite_mode in rewrite_modes:
                # Skip combinations that don't make sense
                if retrieval_mode == RetrievalMode.DENSE_ONLY and rewrite_mode == RewriteMode.TRAINABLE:
                    continue  # Would need trainable model

                print(f"\n🔍 Testing: {retrieval_mode.value} + {rewrite_mode.value}")

                # Create temporary RAG instance with specific configuration
                temp_rag = RAGModule(
                    collection=self.collection,
                    model_name=self.model_name,
                    llm=self.llm,
                    retrieval_mode=retrieval_mode,
                    rewrite_mode=rewrite_mode,
                    vector_weight=self.vector_weight,
                    enable_mmr=self.enable_mmr,
                )

                try:
                    # Run retrieval
                    chunks_data = temp_rag.retrieve(
                        query=question,
                        limit=3,
                        task_type="qa"
                    )

                    # Get answer if we have chunks
                    if chunks_data:
                        chunks = [RAGChunk(text=c["content"], score=c["score"]) for c in chunks_data]
                        prompt = temp_rag.build_prompt(question, chunks)
                        answer = temp_rag.generate(prompt)

                        # Calculate metrics (simplified)
                        avg_score = sum(c.score for c in chunks) / len(chunks)
                        max_score = max(c.score for c in chunks)

                        results[f"{retrieval_mode.value}_{rewrite_mode.value}"] = {
                            "num_chunks": len(chunks),
                            "avg_score": avg_score,
                            "max_score": max_score,
                            "answer_preview": answer[:200] + "..." if len(answer) > 200 else answer,
                            "top_chunk": chunks[0].text[:150] + "..." if len(chunks[0].text) > 150 else chunks[0].text,
                            "rewritten_query": chunks_data[0]["metadata"][
                                "rewritten_query"] if rewrite_mode != RewriteMode.NONE else question
                        }

                        print(f"  ✅ Found {len(chunks)} chunks, avg score: {avg_score:.4f}")
                        if rewrite_mode != RewriteMode.NONE:
                            print(f"  📝 Rewritten: '{chunks_data[0]['metadata']['rewritten_query']}'")
                    else:
                        results[f"{retrieval_mode.value}_{rewrite_mode.value}"] = {
                            "error": "No results found"
                        }
                        print(f"  ❌ No results found")

                except Exception as e:
                    results[f"{retrieval_mode.value}_{rewrite_mode.value}"] = {
                        "error": str(e)
                    }
                    print(f"  ❌ Error: {e}")

        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        # Find best configuration by max_score
        best_config = None
        best_score = -1

        for config, data in results.items():
            if "error" not in data:
                if data["max_score"] > best_score:
                    best_score = data["max_score"]
                    best_config = config

        if best_config:
            print(f"🏆 Best configuration: {best_config}")
            print(f"   Max score: {best_score:.4f}")
            print(f"   Answer preview: {results[best_config]['answer_preview']}")

        return {
            "question": question,
            "results": results,
            "best_config": best_config,
            "best_score": best_score
        }


def comprehensive_test():
    """Comprehensive test with different configurations"""

    print("🚀 Comprehensive RAG System Test")
    print("=" * 60)

    # Initialize base RAG system
    base_rag = RAGModule(
        collection="finance_theory",
        model_name=os.getenv("EMBEDDING_MODEL", ""),
        retrieval_mode=RetrievalMode.HYBRID,
        rewrite_mode=RewriteMode.NONE,
    )

    # Test questions
    test_questions = [
        "Что такое диверсификация инвестиционного портфеля?",
        "Как рассчитать доходность облигаций?",
        "Какие существуют виды финансовых рисков?",
        "Что такое коэффициент Шарпа?",
    ]

    all_results = {}

    for i, question in enumerate(test_questions, 1):
        print(f"\n📋 Test {i}: {question}")
        print("-" * 40)

        results = base_rag.run_comparison_test(question, f"Test {i}")
        all_results[f"test_{i}"] = results

        # Pause between tests
        if i < len(test_questions):
            input("\nPress Enter to continue to next test...")

    # Generate final report
    print("\n" + "=" * 60)
    print("📊 FINAL REPORT")
    print("=" * 60)

    for test_id, results in all_results.items():
        print(f"\n{test_id.upper()}: {results['question']}")
        if results['best_config']:
            print(f"  Best: {results['best_config']} (score: {results['best_score']:.4f})")


if __name__ == "__main__":
    load_dotenv()

    # Quick configuration test
    print("⚡ Quick Configuration Test")
    print("=" * 60)

    # Test with different configurations
    test_question = "Какова основная цель страхования?"

    # Configuration 1: Hybrid retrieval without rewriting
    rag1 = RAGModule(
        collection="finance_theory",
        model_name=os.getenv("EMBEDDING_MODEL", ""),
        retrieval_mode=RetrievalMode.HYBRID,
        rewrite_mode=RewriteMode.NONE,
    )

    # Configuration 2: Hybrid retrieval with frozen LLM rewriting
    rag2 = RAGModule(
        collection="finance_theory",
        model_name=os.getenv("EMBEDDING_MODEL", ""),
        retrieval_mode=RetrievalMode.HYBRID,
        rewrite_mode=RewriteMode.FROZEN_LLM,
    )

    # Configuration 3: Dense only with rewriting
    rag3 = RAGModule(
        collection="finance_theory",
        model_name=os.getenv("EMBEDDING_MODEL", ""),
        retrieval_mode=RetrievalMode.DENSE_ONLY,
        rewrite_mode=RewriteMode.FROZEN_LLM,
    )

    for rag, name in [(rag1, "Hybrid (no rewrite)"), (rag2, "Hybrid + rewrite"), (rag3, "Dense + rewrite")]:
        print(f"\n🔍 {name}:")
        try:
            result = rag.answer(test_question, limit=3)
            print(f"   Answer: {result['answer'][:100]}...")
            if 'query_info' in result:
                print(f"   Query: {result['query_info']['original']} -> {result['query_info']['rewritten']}")
        except Exception as e:
            print(f"   Error: {e}")

    # Run comprehensive test if desired
    run_comprehensive = input("\nRun comprehensive test? (y/n): ").lower() == 'y'
    if run_comprehensive:
        comprehensive_test()
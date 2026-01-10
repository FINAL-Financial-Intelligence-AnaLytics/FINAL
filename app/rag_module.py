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


class HybridCombinationMethod(Enum):
    WEIGHTED_SUM = "weighted_sum"
    RRF = "rrf"  # Reciprocal Rank Fusion
    RANK_WEIGHTED = "rank_weighted"


class


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _format_context(chunks: List[RAGChunk], max_chars: int = 8000) -> str:
    """Format context for prompt"""
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
    # Split by punctuation and whitespace
    tokens = re.findall(r'\b\w+\b', text)
    tokens = [t for t in tokens if len(t) > 1]
    return tokens


class QueryRewriter:
    """Query Rewriter with improved prompting"""

    def __init__(
            self,
            mode: RewriteMode = RewriteMode.NONE,
            llm: Optional[MistralLLM] = None,
            trainable_model_path: Optional[str] = None,
            domain: str = "finance"
    ):
        self.mode = mode
        self.llm = llm
        self.trainable_model_path = trainable_model_path
        self.domain = domain

        # Domain-specific configurations
        self.domain_keywords = {
            "finance": ["инвестиции", "акции", "облигации", "портфель", "риск",
                        "доходность", "рынок", "финансы", "экономика", "бюджет"]
        }

        # Store rewritten queries for analysis
        self.rewrite_history = []

        # For trainable rewriter
        self.trainable_model = None
        if mode == RewriteMode.TRAINABLE and trainable_model_path:
            self._load_trainable_model()

    def _load_trainable_model(self):
        """Load a trained rewriter model"""
        logger.info(f"Trainable rewriter would load from {self.trainable_model_path}")

    def rewrite(
            self,
            query: str,
            task_type: str = "qa",
            few_shot_examples: Optional[List[Tuple[str, str]]] = None,
            max_retries: int = 2
    ) -> str:
        """Rewrite query with improved logic"""
        if self.mode == RewriteMode.NONE:
            return query

        original_query = query

        # Try multiple rewriting strategies
        for attempt in range(max_retries):
            try:
                if self.mode == RewriteMode.FROZEN_LLM:
                    rewritten = self._rewrite_with_llm(query, task_type, few_shot_examples, attempt)
                elif self.mode == RewriteMode.TRAINABLE:
                    rewritten = self._rewrite_with_trainable(query, task_type)
                else:
                    rewritten = query

                # Validate the rewritten query
                if self._is_valid_rewrite(original_query, rewritten):
                    self.rewrite_history.append({
                        "original": original_query,
                        "rewritten": rewritten,
                        "attempt": attempt + 1
                    })
                    logger.info(f"Query rewritten: '{original_query[:50]}...' -> '{rewritten[:50]}...'")
                    return rewritten

                # If invalid, try again with different strategy
                query = self._fallback_rewrite(original_query, attempt)

            except Exception as e:
                logger.error(f"Error in query rewriting (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    return original_query

        return original_query

    def _rewrite_with_llm(
            self,
            query: str,
            task_type: str,
            examples: Optional[List[Tuple[str, str]]] = None,
            attempt: int = 0
    ) -> str:
        """Rewrite query using LLM with improved prompting"""
        if not self.llm:
            logger.warning("No LLM provided for rewriter")
            return query

        # Use different strategies based on attempt
        strategy = "balanced" if attempt == 0 else "concise"
        prompt = self._build_improved_prompt(query, task_type, examples, strategy)

        try:
            rewritten = self.llm.generate(prompt)
            rewritten = self._clean_rewritten_query(rewritten)

            # Post-process the rewritten query
            rewritten = self._post_process_rewrite(rewritten, query)
            return rewritten

        except Exception as e:
            logger.error(f"LLM rewriting failed: {e}")
            return query

    def _build_improved_prompt(
            self,
            query: str,
            task_type: str,
            examples: Optional[List[Tuple[str, str]]] = None,
            strategy: str = "balanced"
    ) -> str:
        """Build improved prompt for query rewriting"""

        # Strategy-specific instructions
        strategies = {
            "balanced": "Создай сбалансированный поисковый запрос, который содержит ключевые термины и общую суть вопроса.",
            "concise": "Создай краткий поисковый запрос с максимально релевантными ключевыми словами.",
            "expansive": "Расширь запрос, включив синонимы и связанные понятия для улучшения полноты поиска."
        }

        instruction = f"""Ты — эксперт по финансовой информации. {strategies.get(strategy, strategies['balanced'])}

Перефразируй вопрос так, чтобы он лучше подходил для поиска в базе знаний по финансам.
Используй специальные финансовые термины и их синонимы."""

        # Domain-specific guidelines
        domain_guidelines = ""
        if self.domain == "finance":
            domain_guidelines = """Учти особенности финансовой терминологии:
- Используй точные термины: "дивиденды", "ликвидность", "хеджирование", "капитализация"
- Включай аббревиатуры: "NPV", "ROI", "ETF", "IPO"
- Добавляй контекст: "формула расчета", "пример использования", "сравнение видов"
"""

        # Few-shot examples
        if not examples:
            examples = self._get_domain_examples()

        demonstrations = ""
        if examples:
            for original, rewritten in examples[:3]:  # Use only 3 best examples
                demonstrations += f"Вопрос: {original}\n"
                demonstrations += f"Поисковый запрос: {rewritten}\n\n"

        # Build prompt
        prompt = f"""{instruction}

{domain_guidelines}

Примеры преобразования вопросов в поисковые запросы:
{demonstrations}Вопрос: {query}
Поисковый запрос:"""

        return prompt

    def _get_domain_examples(self) -> List[Tuple[str, str]]:
        """Get domain-specific examples"""
        if self.domain == "finance":
            return [
                ("Что такое диверсификация портфеля?",
                 "диверсификация инвестиционный портфель стратегия распределение рисков доходность"),

                ("Как рассчитать доходность акций за год?",
                 "формула расчет доходности акций годовая прибыль дивиденды курсовая стоимость"),

                ("Какие виды облигаций бывают?",
                 "типы виды облигации государственные корпоративные муниципальные купонные дисконтные"),

                ("Что такое коэффициент Шарпа?",
                 "коэффициент Шарпа Sharpe ratio формула расчет риск доходность портфель"),

                ("Как работает маржинальная торговля?",
                 "маржинальная торговля кредитное плечо залог маржин-колл риски"),
            ]
        return []

    def _clean_rewritten_query(self, text: str) -> str:
        """Clean and extract the rewritten query"""
        text = text.strip()

        # Remove markdown and code blocks
        text = re.sub(r'```[a-z]*\n?', '', text)
        text = re.sub(r'`', '', text)

        # Look for the query in common patterns
        patterns = [
            r'Поисковый запрос:\s*(.*?)(?:\n|$)',
            r'Запрос:\s*(.*?)(?:\n|$)',
            r'Query:\s*(.*?)(?:\n|$)',
            r'Результат:\s*(.*?)(?:\n|$)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result = match.group(1).strip()
                if result and len(result) > 5:
                    return result

        # Fallback: return the text without quotes
        text = re.sub(r'["\'«»]', '', text)

        # Return the first line if multiple lines
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line and len(line) > 5:
                return line

        return text if text else ""

    def _post_process_rewrite(self, rewritten: str, original: str) -> str:
        """Post-process the rewritten query"""

        # Remove stopwords but keep important financial terms
        stopwords = ["пожалуйста", "расскажите", "объясните", "можно", "ли", "как", "что", "где", "когда"]
        words = rewritten.split()
        filtered_words = [w for w in words if w.lower() not in stopwords]

        # Add domain keywords if missing
        if self.domain in self.domain_keywords:
            has_domain_terms = any(keyword in rewritten.lower() for keyword in self.domain_keywords[self.domain])
            if not has_domain_terms and any(
                    keyword in original.lower() for keyword in self.domain_keywords[self.domain]):
                # Add the first matching domain keyword
                for keyword in self.domain_keywords[self.domain]:
                    if keyword in original.lower():
                        filtered_words.append(keyword)
                        break

        # Ensure we have at least 2 words
        if len(filtered_words) < 2:
            # Fallback to original with improvements
            original_words = original.split()
            original_words = [w for w in original_words if w.lower() not in stopwords]
            return " ".join(original_words[:5])

        return " ".join(filtered_words[:10])  # Limit to 10 words

    def _is_valid_rewrite(self, original: str, rewritten: str) -> bool:
        """Validate if the rewrite is reasonable"""
        if not rewritten or len(rewritten) < 3:
            return False

        # Check if it's too similar to original (should be different)
        if rewritten.lower() == original.lower():
            return False

        # Check if it's too short or too long
        if len(rewritten) < len(original) * 0.3:  # Less than 30% of original
            return False

        # Check if it contains meaningful content
        words = rewritten.split()
        if len(words) < 2:
            return False

        return True

    def _fallback_rewrite(self, query: str, attempt: int) -> str:
        """Fallback rewriting strategies"""
        if attempt == 0:
            # Strategy 1: Extract keywords
            words = query.lower().split()
            keywords = [w for w in words if len(w) > 3 and w not in ["что", "как", "почему", "когда"]]
            return " ".join(keywords[:5])
        else:
            # Strategy 2: Add domain context
            return f"{self.domain} {query}"

    def _rewrite_with_trainable(self, query: str, task_type: str) -> str:
        """Rewrite using trainable model (placeholder)"""
        # Implement if you have a trained model
        return query

    def analyze_rewrites(self) -> Dict[str, Any]:
        """Analyze rewrite history for debugging"""
        if not self.rewrite_history:
            return {"total": 0}

        changes = []
        for entry in self.rewrite_history:
            original_len = len(entry["original"].split())
            rewritten_len = len(entry["rewritten"].split())
            changes.append({
                "original": entry["original"],
                "rewritten": entry["rewritten"],
                "length_change": rewritten_len - original_len,
                "words_added": len(set(entry["rewritten"].split()) - set(entry["original"].split())),
                "words_removed": len(set(entry["original"].split()) - set(entry["rewritten"].split()))
            })

        return {
            "total": len(self.rewrite_history),
            "avg_length_change": np.mean([c["length_change"] for c in changes]) if changes else 0,
            "examples": changes[:5]  # First 5 examples
        }


@dataclass
class HybridDocument:
    """Container for hybrid search results"""
    id: Any
    text: str
    source: str = ""
    vector_score: float = 0.0
    bm25_score: float = 0.0
    vector_rank: int = 0
    bm25_rank: int = 0
    is_bm25_only: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    combined_score: float = 0.0
    rrf_score: float = 0.0


# Optimized prompts for better performance
OPTIMIZED_SYSTEM_PROMPT = """Ты — финансовый эксперт, отвечающий строго на основе предоставленного контекста.

КРИТИЧЕСКИЕ ПРАВИЛА:
1) Отвечай ТОЛЬКО на основе информации из КОНТЕКСТА.
2) Если информации недостаточно — прямо скажи: "В предоставленных материалах нет информации по этому вопросу."
3) Все утверждения подтверждай ссылками на источники: [1], [2], и т.д.
4) Будь краток, точен и информативен.
5) Используй профессиональную финансовую терминологию.
"""

OPTIMIZED_USER_PROMPT = """КОНТЕКСТ:
{context}

ВОПРОС:
{question}

ОТВЕТЬ, ИСПОЛЬЗУЯ ЭТИ ПРАВИЛА:
1. Если ответ есть в контексте — дай его с указанием источников.
2. Если информации недостаточно — так и скажи.
3. Не добавляй информацию не из контекста.
4. Форматируй ответ четко и структурированно.

ОТВЕТ:"""


class RAGModule:
    def __init__(
            self,
            collection: str = "finance_theory",
            model_name: Optional[str] = None,
            device: Optional[str] = None,
            llm: Optional[MistralLLM] = None,
            system_prompt: str = OPTIMIZED_SYSTEM_PROMPT,
            user_prompt: str = OPTIMIZED_USER_PROMPT,
            qdrant_url: Optional[str] = None,
            qdrant_api_key: Optional[str] = None,
            retrieval_mode: RetrievalMode = RetrievalMode.HYBRID,
            rewrite_mode: RewriteMode = RewriteMode.NONE,
            hybrid_method: HybridCombinationMethod = HybridCombinationMethod.RRF,
            bm25_prefetch: int = 200,  # Increased for hybrid
            vector_prefetch: int = 100,  # Increased for hybrid
            vector_weight: float = 0.5,  # Balanced weight
            enable_mmr: bool = True,
            mmr_lambda: float = 0.5,  # More balanced
            rrf_k: int = 60,  # RRF constant
            limit: int = 10,  # Default limit increased to 10
            score_threshold: float = 0.1,  # Lower threshold for more results
            domain: str = "finance"
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
        self.rewrite_mode = rewrite_mode
        self.hybrid_method = hybrid_method
        self.bm25_prefetch = bm25_prefetch
        self.vector_prefetch = vector_prefetch
        self.vector_weight = vector_weight
        self.enable_mmr = enable_mmr
        self.mmr_lambda = mmr_lambda
        self.rrf_k = rrf_k
        self.limit = limit
        self.score_threshold = score_threshold
        self.domain = domain

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

        # Initialize Query Rewriter with improved settings
        self.rewriter = QueryRewriter(
            mode=rewrite_mode,
            llm=self.llm,
            trainable_model_path=None,
            domain=domain
        )

        # Performance tracking
        self.retrieval_stats = []

    def _initialize_bm25_index(self):
        """Initialize BM25 index with improved tokenization"""
        try:
            logger.info("Building BM25 index from corpus...")

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
                if text and len(text.strip()) > 10:  # Minimum length
                    self.bm25_docs.append(text)
                    self.bm25_metadata.append({
                        "id": point.id,
                        "payload": point.payload,
                    })

            if self.bm25_docs:
                # Improved tokenization for Russian text
                tokenized_docs = []
                for doc in self.bm25_docs:
                    tokens = _tokenize(doc)
                    if tokens:
                        tokenized_docs.append(tokens)

                if tokenized_docs:
                    self.bm25_index = BM25Okapi(tokenized_docs)
                    logger.info(f"✅ BM25 index built with {len(self.bm25_docs)} documents")
                else:
                    logger.warning("No valid tokens for BM25 index")
                    self.bm25_index = None
            else:
                logger.warning("No documents found for BM25 index")
                self.bm25_index = None

        except Exception as e:
            logger.error(f"Failed to initialize BM25 index: {e}")
            self.bm25_index = None

    @property
    def model(self) -> Optional[SentenceTransformer]:
        if not self._model_enabled:
            return None
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            if self.device:
                self._model = SentenceTransformer(self.model_name, device=self.device)
            else:
                self._model = SentenceTransformer(self.model_name, device="cpu")
            logger.info(f"✅ Model {self.model_name} loaded")
        return self._model

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed query with E5 format if needed"""
        if not self._model_enabled or self.model is None:
            raise RuntimeError("Embedding model not loaded")
        q = f"query: {query}" if self.model_is_e5 else query
        vec = self.model.encode(q, normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float32)

    def _get_vector_results(
            self,
            query_vec: np.ndarray,
            prefetch: int,
            score_threshold: Optional[float],
            qdrant_filter: Optional[qm.Filter]
    ) -> List[HybridDocument]:
        """Get vector search results"""
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
        for rank, r in enumerate(results.points, 1):
            docs.append(HybridDocument(
                id=r.id,
                text=r.payload.get("text", ""),
                source=r.payload.get("source", ""),
                vector_score=float(r.score),
                bm25_score=0.0,
                vector_rank=rank,
                bm25_rank=0,
                is_bm25_only=False,
                metadata=r.payload
            ))
        return docs

    def _get_bm25_results(self, query: str, prefetch: int) -> List[HybridDocument]:
        """Get BM25 search results"""
        if not self.bm25_index or not query.strip():
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        bm25_scores = self.bm25_index.get_scores(query_tokens)

        # Get top N results with scores > 0
        valid_indices = [i for i, score in enumerate(bm25_scores) if score > 0]
        if not valid_indices:
            return []

        # Sort by score
        sorted_indices = sorted(valid_indices, key=lambda i: bm25_scores[i], reverse=True)
        top_n = min(len(sorted_indices), prefetch)
        top_indices = sorted_indices[:top_n]

        docs = []
        for rank, idx in enumerate(top_indices, 1):
            metadata = self.bm25_metadata[idx]
            docs.append(HybridDocument(
                id=metadata["id"],
                text=self.bm25_docs[idx],
                source=metadata["payload"].get("source", ""),
                vector_score=0.0,
                bm25_score=float(bm25_scores[idx]),
                vector_rank=0,
                bm25_rank=rank,
                is_bm25_only=True,
                metadata=metadata["payload"]
            ))
        return docs

    def _combine_weighted_sum(
            self,
            vector_docs: List[HybridDocument],
            bm25_docs: List[HybridDocument],
    ) -> List[HybridDocument]:
        """Combine using weighted sum with better normalization"""

        # Collect all scores for global normalization
        all_vector_scores = [d.vector_score for d in vector_docs if d.vector_score > 0]
        all_bm25_scores = [d.bm25_score for d in bm25_docs if d.bm25_score > 0]

        if not all_vector_scores and not all_bm25_scores:
            return []

        # Global normalization factors
        vector_min = min(all_vector_scores) if all_vector_scores else 0
        vector_max = max(all_vector_scores) if all_vector_scores else 1
        vector_range = vector_max - vector_min if vector_max > vector_min else 1

        bm25_min = min(all_bm25_scores) if all_bm25_scores else 0
        bm25_max = max(all_bm25_scores) if all_bm25_scores else 1
        bm25_range = bm25_max - bm25_min if bm25_max > bm25_min else 1

        # Combine documents
        docs_by_id = {}

        for doc in vector_docs:
            norm_vector = (doc.vector_score - vector_min) / vector_range if vector_range > 0 else 0
            docs_by_id[doc.id] = {
                "doc": doc,
                "norm_vector": norm_vector,
                "norm_bm25": 0.0
            }

        for doc in bm25_docs:
            norm_bm25 = (doc.bm25_score - bm25_min) / bm25_range if bm25_range > 0 else 0
            if doc.id in docs_by_id:
                docs_by_id[doc.id]["norm_bm25"] = norm_bm25
                docs_by_id[doc.id]["doc"].bm25_score = doc.bm25_score
                docs_by_id[doc.id]["doc"].bm25_rank = doc.bm25_rank
                docs_by_id[doc.id]["doc"].is_bm25_only = False
            else:
                docs_by_id[doc.id] = {
                    "doc": doc,
                    "norm_vector": 0.0,
                    "norm_bm25": norm_bm25
                }

        # Calculate combined scores
        combined_docs = []
        for doc_info in docs_by_id.values():
            combined_score = (self.vector_weight * doc_info["norm_vector"] +
                              (1 - self.vector_weight) * doc_info["norm_bm25"])

            doc = doc_info["doc"]
            doc.combined_score = combined_score
            combined_docs.append(doc)

        return sorted(combined_docs, key=lambda x: x.combined_score, reverse=True)

    def _combine_rrf(
            self,
            vector_docs: List[HybridDocument],
            bm25_docs: List[HybridDocument],
            limit: int
    ) -> List[HybridDocument]:
        """Combine using Reciprocal Rank Fusion (RRF)"""

        # Create rank mappings
        vector_ranks = {doc.id: rank for rank, doc in enumerate(vector_docs, 1)}
        bm25_ranks = {doc.id: rank for rank, doc in enumerate(bm25_docs, 1)}

        # Get all unique documents
        all_docs = {}
        for doc in vector_docs:
            all_docs[doc.id] = doc

        for doc in bm25_docs:
            if doc.id not in all_docs:
                all_docs[doc.id] = doc
            else:
                # Update BM25 info for existing doc
                existing = all_docs[doc.id]
                existing.bm25_score = doc.bm25_score
                existing.bm25_rank = doc.bm25_rank
                existing.is_bm25_only = False

        # Calculate RRF scores
        for doc_id, doc in all_docs.items():
            rrf_score = 0.0

            # Add vector rank contribution
            vector_rank = vector_ranks.get(doc_id)
            if vector_rank is not None:
                doc.vector_rank = vector_rank
                rrf_score += 1.0 / (vector_rank + self.rrf_k)

            # Add BM25 rank contribution
            bm25_rank = bm25_ranks.get(doc_id)
            if bm25_rank is not None:
                doc.bm25_rank = bm25_rank
                rrf_score += 1.0 / (bm25_rank + self.rrf_k)

            doc.rrf_score = rrf_score

        # Sort by RRF score
        sorted_docs = sorted(all_docs.values(), key=lambda x: x.rrf_score, reverse=True)

        # Update combined_score to RRF score for consistency
        for doc in sorted_docs:
            doc.combined_score = doc.rrf_score

        return sorted_docs[:limit]

    def retrieve(
            self,
            query: str,
            limit: int = None,
            prefetch: int = None,
            score_threshold: Optional[float] = None,
            qdrant_filter: Optional[qm.Filter] = None,
            task_type: str = "qa",
            track_stats: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Improved hybrid retrieval with better query rewriting
        """
        start_time = time.time() if track_stats else None

        # Use instance defaults if not provided
        limit = limit or self.limit
        prefetch = prefetch or self.vector_prefetch
        score_threshold = score_threshold or self.score_threshold

        # Step 1: Improved Query Rewriting
        original_query = query
        if self.rewriter.mode != RewriteMode.NONE:
            query = self.rewriter.rewrite(query, task_type)
            if track_stats:
                logger.debug(f"Query rewritten: '{original_query[:50]}...' -> '{query[:50]}...'")

        if not self.client:
            return []

        # Step 2: Get results based on retrieval mode
        vector_docs = []
        bm25_docs = []

        # Adjust prefetch for hybrid to get more candidates
        if self.retrieval_mode == RetrievalMode.HYBRID:
            vector_prefetch = prefetch * 2
            bm25_prefetch = self.bm25_prefetch * 2
        else:
            vector_prefetch = prefetch
            bm25_prefetch = self.bm25_prefetch

        if self.retrieval_mode in [RetrievalMode.DENSE_ONLY, RetrievalMode.HYBRID]:
            if not self._model_enabled:
                raise RuntimeError("Embedding model not configured")
            query_vec = self._embed_query(query)
            vector_docs = self._get_vector_results(
                query_vec=query_vec,
                prefetch=vector_prefetch,
                score_threshold=score_threshold,
                qdrant_filter=qdrant_filter
            )

        if self.retrieval_mode in [RetrievalMode.BM25_ONLY, RetrievalMode.HYBRID]:
            if self.bm25_index:
                bm25_docs = self._get_bm25_results(query, prefetch=bm25_prefetch)
            else:
                logger.warning("BM25 index not available")

        # Step 3: Combine results
        if self.retrieval_mode == RetrievalMode.HYBRID:
            if self.hybrid_method == HybridCombinationMethod.RRF:
                combined_docs = self._combine_rrf(vector_docs, bm25_docs, limit)
            else:
                combined_docs = self._combine_weighted_sum(vector_docs, bm25_docs)[:limit]
        elif self.retrieval_mode == RetrievalMode.DENSE_ONLY:
            combined_docs = vector_docs[:limit]
        else:  # BM25_ONLY
            combined_docs = bm25_docs[:limit]

        if not combined_docs:
            return []

        # Step 4: Apply MMR for diversity (optional)
        if self.enable_mmr and len(combined_docs) > 3 and self._model_enabled:
            query_vec = self._embed_query(query)
            texts = [doc.text for doc in combined_docs]
            scores = [doc.combined_score for doc in combined_docs]

            pick_idx = self._mmr_select(
                query_vec=query_vec,
                cand_texts=texts,
                cand_scores=scores,
                top_k=min(limit, len(combined_docs)),
                lambda_mult=self.mmr_lambda,
            )
            selected = [combined_docs[i] for i in pick_idx]
        else:
            selected = combined_docs[:limit]

        # Track statistics
        if track_stats:
            end_time = time.time()
            self.retrieval_stats.append({
                "query": original_query,
                "rewritten": query,
                "vector_count": len(vector_docs),
                "bm25_count": len(bm25_docs),
                "combined_count": len(selected),
                "time_ms": (end_time - start_time) * 1000,
                "mode": self.retrieval_mode.value,
                "rewrite_mode": self.rewriter.mode.value
            })

        # Step 5: Format return
        results = []
        for doc in selected:
            result = {
                "content": doc.text,
                "score": doc.combined_score,
                "source": doc.source,
                "metadata": {
                    "vector_score": doc.vector_score,
                    "bm25_score": doc.bm25_score,
                    "vector_rank": doc.vector_rank,
                    "bm25_rank": doc.bm25_rank,
                    "rrf_score": getattr(doc, 'rrf_score', 0.0),
                    "is_bm25_only": doc.is_bm25_only,
                    "original_query": original_query,
                    "rewritten_query": query,
                }
            }
            results.append(result)

        return results

    def _mmr_select(
            self,
            query_vec: np.ndarray,
            cand_texts: List[str],
            cand_scores: List[float],
            top_k: int,
            lambda_mult: float = 0.5,
    ) -> List[int]:
        """MMR selection with balanced diversity"""
        if len(cand_texts) <= 2:
            return list(range(min(len(cand_texts), top_k)))

        # Embed candidate texts
        cand_vecs = self._embed_texts(cand_texts)

        selected = []
        remaining = list(range(len(cand_texts)))

        # Start with highest score
        first = int(np.argmax(np.asarray(cand_scores)))
        selected.append(first)
        remaining.remove(first)

        while remaining and len(selected) < top_k:
            best_idx = None
            best_mmr = -1e9

            for idx in remaining:
                # Relevance to query
                rel = _cosine_sim(query_vec, cand_vecs[idx])

                # Max similarity to already selected
                if selected:
                    max_sim = max(_cosine_sim(cand_vecs[idx], cand_vecs[s]) for s in selected)
                else:
                    max_sim = 0

                # MMR formula
                mmr = lambda_mult * rel - (1 - lambda_mult) * max_sim

                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = idx

            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)
            else:
                break

        return selected

    def answer(
            self,
            question: str,
            limit: int = None,
            prefetch: int = None,
            score_threshold: Optional[float] = None,
            qdrant_filter: Optional[qm.Filter] = None,
            task_type: str = "qa",
    ) -> Dict[str, Any]:
        """Generate answer with improved retrieval"""

        limit = limit or self.limit

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
            },
            "retrieval_stats": {
                "total_chunks": len(chunks),
                "avg_score": sum(c.score for c in chunks) / len(chunks) if chunks else 0,
                "min_score": min(c.score for c in chunks) if chunks else 0,
                "max_score": max(c.score for c in chunks) if chunks else 0,
            }
        }

    # Keep existing methods for compatibility
    build_prompt = lambda self, question, chunks: self.system_prompt + "\n\n" + self.user_prompt.format(
        context=_format_context(chunks), question=question)

    generate = lambda self, prompt: self.llm.generate(prompt) if self.llm else ""


# Helper for evaluation script
import time


def run_optimized_evaluation():
    """Run evaluation with optimized configurations"""

    print("🚀 OPTIMIZED RAG EVALUATION")
    print("=" * 60)

    # Test configurations
    configs = [
        {
            "name": "Hybrid RRF (Optimized)",
            "retrieval_mode": RetrievalMode.HYBRID,
            "rewrite_mode": RewriteMode.NONE,
            "hybrid_method": HybridCombinationMethod.RRF,
            "vector_weight": 0.5,
            "enable_mmr": False,
            "bm25_prefetch": 200,
            "vector_prefetch": 100,
            "limit": 10
        },
        {
            "name": "Dense-only (Baseline)",
            "retrieval_mode": RetrievalMode.DENSE_ONLY,
            "rewrite_mode": RewriteMode.NONE,
            "limit": 10
        },
        {
            "name": "Hybrid RRF + Rewriting",
            "retrieval_mode": RetrievalMode.HYBRID,
            "rewrite_mode": RewriteMode.FROZEN_LLM,
            "hybrid_method": HybridCombinationMethod.RRF,
            "vector_weight": 0.5,
            "enable_mmr": False,
            "limit": 10
        },
        {
            "name": "BM25-only",
            "retrieval_mode": RetrievalMode.BM25_ONLY,
            "rewrite_mode": RewriteMode.NONE,
            "limit": 10
        },
    ]

    results = {}

    for config in configs:
        print(f"\n🔧 Testing: {config['name']}")

        rag = RAGModule(
            collection="finance_theory",
            model_name=os.getenv("EMBEDDING_MODEL"),
            **{k: v for k, v in config.items() if k != 'name'}
        )

        # Test on sample questions
        test_questions = [
            "Что такое диверсификация инвестиционного портфеля?",
            "Как рассчитать доходность акций?",
            "Какие бывают виды облигаций?",
        ]

        config_results = []
        for q in test_questions:
            start = time.time()
            result = rag.answer(q, limit=10)
            elapsed = time.time() - start

            config_results.append({
                "question": q,
                "answer_length": len(result["answer"]),
                "chunks_count": len(result["chunks"]),
                "avg_score": sum(c.score for c in result["chunks"]) / len(result["chunks"]) if result["chunks"] else 0,
                "time_ms": elapsed * 1000,
                "rewritten_query": result["query_info"]["rewritten_query"] if "rewritten_query" in result[
                    "query_info"] else q
            })

        results[config['name']] = config_results

        # Print summary
        avg_score = sum(r["avg_score"] for r in config_results) / len(config_results)
        avg_time = sum(r["time_ms"] for r in config_results) / len(config_results)
        print(f"  Avg score: {avg_score:.4f}, Avg time: {avg_time:.1f}ms")

        if config['rewrite_mode'] != RewriteMode.NONE:
            rewrite_stats = rag.rewriter.analyze_rewrites()
            print(
                f"  Rewrites: {rewrite_stats['total']}, Avg length change: {rewrite_stats.get('avg_length_change', 0):.1f}")

    return results


if __name__ == "__main__":
    load_dotenv()
    run_optimized_evaluation()
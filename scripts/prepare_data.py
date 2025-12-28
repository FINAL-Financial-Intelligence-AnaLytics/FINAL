import os
import time
import uuid
import hashlib
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sentence_transformers import SentenceTransformer

import sys
import os
from pathlib import Path

# Добавляем корневую директорию в PYTHONPATH
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

# Загружаем переменные окружения
from dotenv import load_dotenv
env_path = root_dir / '.env'
load_dotenv(dotenv_path=env_path)


def stable_uuid(*parts: Any) -> str:
    s = "|".join("" if p is None else str(p) for p in parts)
    h = hashlib.md5(s.encode("utf-8")).hexdigest()  
    return str(uuid.UUID(h))  


def chunked(lst: List[int], n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def ensure_collection(
    client: QdrantClient,
    collection: str,
    vector_size: int,
    distance: qm.Distance = qm.Distance.COSINE,
    recreate: bool = False,
):
    if recreate:
        client.recreate_collection(
            collection_name=collection,
            vectors_config=qm.VectorParams(size=vector_size, distance=distance),
        )
        return

    try:
        info = client.get_collection(collection)
        size = getattr(info.config.params.vectors, "size", None)
        if size is not None and size != vector_size:
            raise ValueError(
                f"Collection '{collection}' already exists with vector size={size}, "
                f"but your model produces size={vector_size}. "
                f"Use recreate=True or a different collection name."
            )
    except Exception:
        client.create_collection(
            collection_name=collection,
            vectors_config=qm.VectorParams(size=vector_size, distance=distance),
        )


def embed_texts(
    model: SentenceTransformer,
    texts: List[str],
    model_is_e5: bool = False,
    batch_size: int = 64,
) -> np.ndarray:
    if model_is_e5:
        texts = [f"passage: {t}" for t in texts]

    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def upsert_with_retries(
    client: QdrantClient,
    collection: str,
    points: List[qm.PointStruct],
    max_retries: int = 5,
    base_sleep: float = 0.7,
):
    for attempt in range(1, max_retries + 1):
        try:
            client.upsert(collection_name=collection, points=points, wait=True)
            return
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(base_sleep * (2 ** (attempt - 1)))


def main(
    input_csv: str,
    collection: str = "fincult_chunks",
    model_name: str = "intfloat/multilingual-e5-base",
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
    recreate: bool = False,
    upsert_batch: int = 256,
    embed_batch: int = 64,
):
    qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY", None)

    df = pd.read_csv(input_csv)
    if "text" not in df.columns:
        raise ValueError("CSV must contain a 'text' column with chunk text.")

    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)

    model = SentenceTransformer(model_name)
    vector_size = model.get_sentence_embedding_dimension()
    model_is_e5 = "e5" in model_name.lower()

    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    ensure_collection(
        client=client,
        collection=collection,
        vector_size=vector_size,
        distance=qm.Distance.COSINE,
        recreate=recreate,
    )

    keep_columns = ["source"]
    payload_cols = [c for c in keep_columns if c in df.columns]

    for batch_rows in tqdm(list(chunked(df.index.tolist(), upsert_batch)), desc="Upserting"):
        batch_df = df.loc[batch_rows]
        texts = batch_df["text"].tolist()

        vectors = embed_texts(
            model=model,
            texts=texts,
            model_is_e5=model_is_e5,
            batch_size=embed_batch,
        )

        points: List[qm.PointStruct] = []
        for i, (_, row) in enumerate(batch_df.iterrows()):
            payload: Dict[str, Any] = {"text": row["text"]}

            for c in payload_cols:
                val = row[c]
                if pd.isna(val):
                    continue
                if isinstance(val, np.generic):
                    val = val.item()
                payload[c] = val
            
            pid = stable_uuid(
                payload.get("source"),
                payload["text"][:200],
            )

            points.append(
                qm.PointStruct(
                    id=pid,
                    vector=vectors[i].tolist(),
                    payload=payload,
                )
            )

        upsert_with_retries(client, collection, points)

    info = client.get_collection(collection)
    print(f"Done. Collection '{collection}' points_count={info.points_count}, vector_size={vector_size}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Подготовка данных для Qdrant")
    parser.add_argument("--input-csv", default="fincult_articles_by_categories_formatted.csv", help="Путь к CSV файлу")
    parser.add_argument("--collection", default="finance_theory", help="Имя коллекции")
    parser.add_argument("--model", default=os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base"), help="Модель для эмбеддингов")
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"), help="URL Qdrant")
    parser.add_argument("--qdrant-api-key", default=os.getenv("QDRANT_API_KEY"), help="API ключ Qdrant")
    parser.add_argument("--recreate", action="store_true", help="Пересоздать коллекцию")
    parser.add_argument("--upsert-batch", type=int, default=256, help="Размер батча для upsert")
    parser.add_argument("--embed-batch", type=int, default=64, help="Размер батча для эмбеддингов")
    
    args = parser.parse_args()
    
    main(
        input_csv=args.input_csv,
        collection=args.collection,
        model_name=args.model,
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
        recreate=args.recreate,
        upsert_batch=args.upsert_batch,
        embed_batch=args.embed_batch,
    )

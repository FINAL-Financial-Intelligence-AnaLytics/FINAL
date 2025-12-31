import json
import os

import pandas as pd
from datasets import Dataset
from langchain_mistralai import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

if __name__ == "__main__":
    import argparse

    from app.config import Config
    from app.llm_client.implementations.mistral_client import MistralLLM
    from app.rag_module import RAGModule

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="eval/ragas_dataset.json")
    args = parser.parse_args()

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY missing")
    
    print("=== Инициализация RAG модуля ===")
    rag_llm = MistralLLM(
        api_key=api_key,
        model=os.getenv("MISTRAL_MODEL", Config.MISTRAL_MODEL),
        base_url=os.getenv("MISTRAL_BASE_URL", Config.MISTRAL_BASE_URL),
        temperature=float(os.getenv("MISTRAL_TEMPERATURE", str(Config.MISTRAL_TEMPERATURE)))
    )
    print("✅ RAG LLM клиент создан")

    rag_module = RAGModule(
        collection="finance_theory",
        qdrant_url=os.getenv("QDRANT_URL"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY"),
        llm=rag_llm,
        model_name=os.getenv("EMBEDDING_MODEL"),
        device=os.getenv("EMBEDDING_DEVICE", "cpu")
    )
    print("Created RAG")

    ragas_llm = LangchainLLMWrapper(ChatMistralAI(
        model=os.getenv("MISTRAL_MODEL", Config.MISTRAL_MODEL),
        mistral_api_key=api_key,
        temperature=0.0,
    ))
    ragas_embeddings = LangchainEmbeddingsWrapper(
        MistralAIEmbeddings(model="mistral-embed", mistral_api_key=api_key)
    )

    with open(args.dataset, encoding="utf-8") as f:
        data = json.load(f)
    eval_samples = []
    for i, row in enumerate(data):
        print(f"[{i+1}/{len(data)}] {row['question'][:50]}...")
        out = rag_module.answer(question=row["question"], limit=5)
        eval_samples.append(
            {
                "user_input": row["question"],
                "retrieved_contexts": [c.text for c in out["chunks"]],
                "response": out["answer"],
                "reference": row["ground_truth"],
            }
        )

    df = pd.DataFrame(eval_samples)
    ds = Dataset.from_pandas(df)

    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm),
        ContextPrecision(llm=ragas_llm),
        ContextRecall(llm=ragas_llm),
    ]

    result = evaluate(ds, metrics=metrics, embeddings=ragas_embeddings)
    print("\n📊 RAGAS Results:")
    print(result)

import json
import os

import pandas as pd
from datasets import Dataset
from langchain_mistralai import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
from ragas.run_config import RunConfig


def generate_answers(rag_module, dataset_path: str, output_path: str):
    with open(dataset_path, encoding="utf-8") as f:
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
    df.to_json(output_path, orient="records", indent=2, force_ascii=False)
    print(f"✅ Answers saved to {output_path}")


def run_evaluation(dataset_path: str, api_key: str, model_name: str):
    ragas_llm = LangchainLLMWrapper(
        ChatMistralAI(
            model=model_name,
            mistral_api_key=api_key,
            temperature=0.0,
            max_retries=3,
        )
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(
        MistralAIEmbeddings(model="mistral-embed", mistral_api_key=api_key)
    )

    df = pd.read_json(dataset_path)
    ds = Dataset.from_pandas(df)

    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm),
        ContextPrecision(llm=ragas_llm),
        ContextRecall(llm=ragas_llm),
    ]

    result = evaluate(
        ds,
        metrics=metrics,
        embeddings=ragas_embeddings,
        run_config=RunConfig(max_workers=1, timeout=300),
    )
    print("\n📊 RAGAS Results:")
    print(result)
    return result


if __name__ == "__main__":
    import argparse

    from app.config import Config
    from app.llm_client.implementations.mistral_client import MistralLLM
    from app.rag_module import RAGModule

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="eval/ragas_dataset.json")
    parser.add_argument("--with-answers", default="eval/ragas_with_answers.json")
    parser.add_argument("--mode", choices=["answers", "eval", "all"], default="all")
    args = parser.parse_args()

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY missing")

    model_name = os.getenv("MISTRAL_MODEL", Config.MISTRAL_MODEL)

    if args.mode in ("answers", "all"):
        print("=== Generating answers ===")
        rag_llm = MistralLLM(
            api_key=api_key,
            model=model_name,
            base_url=os.getenv("MISTRAL_BASE_URL", Config.MISTRAL_BASE_URL),
            temperature=float(
                os.getenv("MISTRAL_TEMPERATURE", str(Config.MISTRAL_TEMPERATURE))
            ),
        )
        rag_module = RAGModule(
            collection="finance_theory",
            qdrant_url=os.getenv("QDRANT_URL"),
            qdrant_api_key=os.getenv("QDRANT_API_KEY"),
            llm=rag_llm,
            model_name=os.getenv("EMBEDDING_MODEL"),
            device=os.getenv("EMBEDDING_DEVICE", "cpu"),
        )
        generate_answers(rag_module, args.dataset, args.with_answers)

    if args.mode in ("eval", "all"):
        print("=== Running evaluation ===")
        run_evaluation(args.with_answers, api_key, model_name)

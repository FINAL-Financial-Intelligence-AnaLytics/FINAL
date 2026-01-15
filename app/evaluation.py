import json
import os
from typing import Dict, List, Any

import pandas as pd
import traceback
from datasets import Dataset
from langchain_mistralai import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
from ragas.run_config import RunConfig

from app.rag_module import RAGModule, RetrievalMode, RewriteMode, HybridCombinationMethod
from app.llm_client.implementations.mistral_client import MistralLLM


def generate_answers_for_configs(rag_base_config: Dict[str, Any], dataset_path: str, output_dir: str = "eval", max_samples: int = None):
    """Generate answers for all 6 configurations"""

    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)

    configurations = [
        (RetrievalMode.DENSE_ONLY, RewriteMode.NONE, "dense_none"),
        (RetrievalMode.BM25_ONLY, RewriteMode.NONE, "bm25_none"),
        (RetrievalMode.HYBRID, RewriteMode.NONE, "hybrid_none"),
        (RetrievalMode.DENSE_ONLY, RewriteMode.FROZEN_LLM, "dense_frozen"),
        (RetrievalMode.BM25_ONLY, RewriteMode.FROZEN_LLM, "bm25_frozen"),
        (RetrievalMode.HYBRID, RewriteMode.FROZEN_LLM, "hybrid_frozen"),
    ]

    all_results = {}

    if max_samples and len(data) > max_samples:
        print(f"   Limiting to {max_samples} samples (down from {len(data)})")
        data = data[:max_samples]


    for retrieval_mode, rewrite_mode, config_name in configurations:
        print(f"\n=== Generating answers for: {config_name} ===")

        # Create RAG module with specific configuration
        rag_module = RAGModule(
            collection=rag_base_config["collection"],
            model_name=rag_base_config.get("model_name"),
            device=rag_base_config.get("device"),
            llm=rag_base_config["llm"],
            retrieval_mode=retrieval_mode,
            rewrite_mode=rewrite_mode,
            vector_weight=rag_base_config.get("vector_weight", 0.7),
            enable_mmr=rag_base_config.get("enable_mmr", True),
            mmr_lambda=rag_base_config.get("mmr_lambda", 0.6),
            qdrant_url=rag_base_config.get("qdrant_url"),
            qdrant_api_key=rag_base_config.get("qdrant_api_key"),
            bm25_prefetch=200 if retrieval_mode == RetrievalMode.HYBRID else 100,
            vector_prefetch=100 if retrieval_mode == RetrievalMode.HYBRID else 50,
            hybrid_method=HybridCombinationMethod.RRF if retrieval_mode == RetrievalMode.HYBRID else None,
            rrf_k=rag_base_config.get("rrf_k", 60),
            score_threshold=rag_base_config.get("score_threshold", 0.1),
        )

        eval_samples = []
        for i, row in enumerate(data):
            print(f"[{i + 1}/{len(data)}] {row['question'][:50]}...")
            out = rag_module.answer(question=row["question"])
            eval_samples.append({
                "user_input": row["question"],
                "retrieved_contexts": [c.text for c in out["chunks"]],
                "response": out["answer"],
                "reference": row["ground_truth"],
                "config": config_name,
                "retrieval_mode": retrieval_mode.value,
                "rewrite_mode": rewrite_mode.value,
            })

        # Save individual config results
        df = pd.DataFrame(eval_samples)
        output_path = os.path.join(output_dir, f"ragas_{config_name}.json")
        df.to_json(output_path, orient="records", indent=2, force_ascii=False)
        print(f"✅ Answers saved to {output_path}")

        all_results[config_name] = df

    # Also save combined results
    combined_df = pd.concat(all_results.values(), ignore_index=True)
    combined_path = os.path.join(output_dir, "ragas_all_configs.json")
    combined_df.to_json(combined_path, orient="records", indent=2, force_ascii=False)
    print(f"\n✅ All configurations saved to {combined_path}")

    return all_results


def run_evaluation_for_configs(dataset_path: str, api_key: str, model_name: str, output_dir: str = "eval", max_samples: int = None):
    """Run evaluation for all configurations"""

    # Read the combined dataset
    df = pd.read_json(dataset_path)

    # Setup RAGAS components
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

    # Define metrics
    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm),
        ContextPrecision(llm=ragas_llm),
        ContextRecall(llm=ragas_llm),
    ]

    # Evaluate each configuration separately
    all_results = {}

    config_names = df["config"].unique()

    for config_name in config_names:
        print(f"\n=== Evaluating configuration: {config_name} ===")

        # Filter data for this configuration
        config_df = df[df["config"] == config_name].copy()

        if max_samples and len(config_df) > max_samples:
            print(f"   Limiting to {max_samples} samples (down from {len(config_df)})")
            config_df = config_df.head(max_samples)

        # Remove config columns before evaluation (RAGAS expects specific columns)
        eval_df = config_df.drop(columns=["config", "retrieval_mode", "rewrite_mode"])

        # Convert to Dataset
        ds = Dataset.from_pandas(eval_df)

        try:
            # Run evaluation
            result = evaluate(
                ds,
                metrics=metrics,
                embeddings=ragas_embeddings,
                run_config=RunConfig(max_workers=1, timeout=300),
            )

            print("RAGAS RESULT TYPE:\n", result.to_pandas())

            df_result = result.to_pandas()

            # Convert to dictionary and add config info
            result_dict = {}
            for metric_name in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
                if metric_name in df_result.columns:
                    # Calculate mean, skipping NaN values
                    mean_score = df_result[metric_name].mean(skipna=True)
                    result_dict[metric_name] = float(mean_score)
                else:
                    # Try to find column with case-insensitive match
                    matching_cols = [col for col in df_result.columns if col.lower() == metric_name.lower()]
                    if matching_cols:
                        mean_score = df_result[matching_cols[0]].mean(skipna=True)
                        result_dict[metric_name] = float(mean_score)
                    else:
                        print(f"  Warning: Column '{metric_name}' not found in DataFrame")
                        result_dict[metric_name] = None

            # Add config info
            result_dict["config"] = config_name
            result_dict["retrieval_mode"] = config_df.iloc[0]["retrieval_mode"]
            result_dict["rewrite_mode"] = config_df.iloc[0]["rewrite_mode"]
            result_dict["num_samples"] = len(config_df)

            all_results[config_name] = result_dict

            print(f"\n📊 Results for {config_name}:")
            for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
                value = result_dict.get(metric)
                if value is not None:
                    print(f"  {metric}: {value:.4f}")

        except Exception as e:
            print(f"❌ Error evaluating {config_name}: {e}")
            traceback.print_exc()  # This prints the complete error stack
            all_results[config_name] = {"error": str(e), "config": config_name, "full_trace": traceback.format_exc()}

    # Save all results
    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n✅ All evaluation results saved to {results_path}")

    # Print summary table
    print_summary_table(all_results)

    return all_results


def print_summary_table(results: Dict[str, Dict]):
    """Print a formatted summary table of all configurations"""

    print("\n" + "=" * 80)
    print("📊 RAGAS EVALUATION SUMMARY")
    print("=" * 80)

    # Table headers
    headers = ["Configuration", "Retrieval", "Rewrite", "Faithfulness", "AnswerRelevancy", "ContextPrecision",
               "ContextRecall"]
    row_format = "{:<20} {:<12} {:<12} {:<12} {:<16} {:<16} {:<16}"

    print(row_format.format(*headers))
    print("-" * 80)

    for config_name, result in results.items():
        if "error" in result:
            print(f"{config_name}: Error - {result['error']}")
            continue

        retrieval = result.get("retrieval_mode", "N/A")
        rewrite = result.get("rewrite_mode", "N/A")

        # Extract scores, handle missing values
        faithfulness = result.get("faithfulness", "N/A")
        answer_relevancy = result.get("answer_relevancy", "N/A")
        context_precision = result.get("context_precision", "N/A")
        context_recall = result.get("context_recall", "N/A")

        # Format scores
        if isinstance(faithfulness, (int, float)):
            faithfulness = f"{faithfulness:.4f}"
        if isinstance(answer_relevancy, (int, float)):
            answer_relevancy = f"{answer_relevancy:.4f}"
        if isinstance(context_precision, (int, float)):
            context_precision = f"{context_precision:.4f}"
        if isinstance(context_recall, (int, float)):
            context_recall = f"{context_recall:.4f}"

        print(row_format.format(
            config_name, retrieval, rewrite,
            faithfulness, answer_relevancy, context_precision, context_recall
        ))

    print("=" * 80)


def generate_comparison_report(results: Dict[str, Dict], output_dir: str = "eval"):
    """Generate a detailed comparison report"""

    report_lines = [
        "# RAG System Configuration Comparison Report",
        "",
        "## Overview",
        f"Total configurations evaluated: {len(results)}",
        f"Evaluation timestamp: {pd.Timestamp.now()}",
        "",
        "## Detailed Results",
        ""
    ]

    for config_name, result in results.items():
        report_lines.append(f"### Configuration: {config_name}")
        report_lines.append(f"- Retrieval Mode: {result.get('retrieval_mode', 'N/A')}")
        report_lines.append(f"- Rewrite Mode: {result.get('rewrite_mode', 'N/A')}")
        report_lines.append(f"- Number of samples: {result.get('num_samples', 'N/A')}")

        if "error" in result:
            report_lines.append(f"- **Error**: {result['error']}")
        else:
            for metric, value in result.items():
                if metric not in ["config", "retrieval_mode", "rewrite_mode", "num_samples"]:
                    if isinstance(value, (int, float)):
                        report_lines.append(f"- {metric}: {value:.4f}")
                    else:
                        report_lines.append(f"- {metric}: {value}")

        report_lines.append("")

    # Add summary table
    report_lines.append("## Summary Table")
    report_lines.append("")
    report_lines.append(
        "| Configuration | Retrieval | Rewrite | Faithfulness | AnswerRelevancy | ContextPrecision | ContextRecall |")
    report_lines.append(
        "|---------------|-----------|---------|--------------|-----------------|------------------|---------------|")

    for config_name, result in results.items():
        if "error" in result:
            continue

        retrieval = result.get("retrieval_mode", "N/A")
        rewrite = result.get("rewrite_mode", "N/A")

        # Extract scores
        faithfulness = result.get("faithfulness", "N/A")
        answer_relevancy = result.get("answer_relevancy", "N/A")
        context_precision = result.get("context_precision", "N/A")
        context_recall = result.get("context_recall", "N/A")

        # Format scores
        if isinstance(faithfulness, (int, float)):
            faithfulness = f"{faithfulness:.4f}"
        if isinstance(answer_relevancy, (int, float)):
            answer_relevancy = f"{answer_relevancy:.4f}"
        if isinstance(context_precision, (int, float)):
            context_precision = f"{context_precision:.4f}"
        if isinstance(context_recall, (int, float)):
            context_recall = f"{context_recall:.4f}"

        report_lines.append(
            f"| {config_name} | {retrieval} | {rewrite} | {faithfulness} | {answer_relevancy} | {context_precision} | {context_recall} |")

    report_lines.append("")

    # Save report
    report_path = os.path.join(output_dir, "comparison_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\n📋 Detailed report saved to {report_path}")

    return report_path


if __name__ == "__main__":
    import argparse

    from app.config import Config

    parser = argparse.ArgumentParser(description="Evaluate RAG system with all 6 configurations")
    parser.add_argument("--dataset", default="eval/ragas_dataset.json", help="Path to evaluation dataset")
    parser.add_argument("--output-dir", default="eval", help="Directory for output files")
    parser.add_argument("--max-samples", default="None", help="Number of samples for each config to test")
    parser.add_argument("--mode", choices=["answers", "eval", "all"], default="all",
                        help="Mode: 'answers' to generate answers, 'eval' to run evaluation, 'all' for both")
    args = parser.parse_args()

    if args.max_samples != "None":
        max_samples = int(args.max_samples)
    else:
        max_samples = None

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY missing")

    model_name = os.getenv("MISTRAL_MODEL", Config.MISTRAL_MODEL)

    if args.mode in ("answers", "all"):
        print("=== Generating answers for all 6 configurations ===")

        # Initialize base LLM for all configurations
        rag_llm = MistralLLM(
            api_key=api_key,
            model=model_name,
            base_url=os.getenv("MISTRAL_BASE_URL", Config.MISTRAL_BASE_URL),
            temperature=float(
                os.getenv("MISTRAL_TEMPERATURE", str(Config.MISTRAL_TEMPERATURE))
            ),
        )

        # Base configuration for RAG modules
        rag_base_config = {
            "collection": "finance_theory",
            "model_name": os.getenv("EMBEDDING_MODEL"),
            "device": os.getenv("EMBEDDING_DEVICE", "cpu"),
            "llm": rag_llm,
            "qdrant_url": os.getenv("QDRANT_URL"),
            "qdrant_api_key": os.getenv("QDRANT_API_KEY"),
            "vector_weight": 0.6,
            "enable_mmr": False,
            "mmr_lambda": 0.6,
            "limit": 7,
            "rrf_k":60,
            "bm25_prefetch": 100,
            "vector_prefetch": 50,
            "score_threshold": 0.05
        }

        generate_answers_for_configs(rag_base_config, args.dataset, args.output_dir, max_samples)

    if args.mode in ("eval", "all"):
        print("\n=== Running evaluation for all configurations ===")
        combined_path = os.path.join(args.output_dir, "ragas_all_configs.json")

        if os.path.exists(combined_path):
            results = run_evaluation_for_configs(combined_path, api_key, model_name, args.output_dir, max_samples)
            generate_comparison_report(results, args.output_dir)
        else:
            print(f"❌ Combined answers file not found at {combined_path}")
            print("Please run with --mode answers first to generate answers for all configurations.")
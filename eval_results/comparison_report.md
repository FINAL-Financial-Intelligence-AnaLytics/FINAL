# RAG System Configuration Comparison Report

## Overview
Total configurations evaluated: 6
Evaluation timestamp: 2026-01-11 19:37:51.517475

## Detailed Results

### Configuration: dense_none
- Retrieval Mode: dense_only
- Rewrite Mode: none
- Number of samples: 150
- faithfulness: 0.8788
- answer_relevancy: 0.7352
- context_precision: 0.5007
- context_recall: 0.6567

### Configuration: bm25_none
- Retrieval Mode: bm25_only
- Rewrite Mode: none
- Number of samples: 150
- faithfulness: 0.8175
- answer_relevancy: 0.5085
- context_precision: 0.2821
- context_recall: 0.4900

### Configuration: hybrid_none
- Retrieval Mode: hybrid
- Rewrite Mode: none
- Number of samples: 150
- faithfulness: 0.8657
- answer_relevancy: 0.7313
- context_precision: 0.4664
- context_recall: 0.6633

### Configuration: dense_frozen
- Retrieval Mode: dense_only
- Rewrite Mode: frozen_llm
- Number of samples: 150
- faithfulness: 0.8791
- answer_relevancy: 0.7621
- context_precision: 0.5265
- context_recall: 0.6733

### Configuration: bm25_frozen
- Retrieval Mode: bm25_only
- Rewrite Mode: frozen_llm
- Number of samples: 150
- faithfulness: 0.8304
- answer_relevancy: 0.5900
- context_precision: 0.3247
- context_recall: 0.5867

### Configuration: hybrid_frozen
- Retrieval Mode: hybrid
- Rewrite Mode: frozen_llm
- Number of samples: 150
- faithfulness: 0.8686
- answer_relevancy: 0.7376
- context_precision: 0.4755
- context_recall: 0.7000

## Summary Table

| Configuration | Retrieval | Rewrite | Faithfulness | AnswerRelevancy | ContextPrecision | ContextRecall |
|---------------|-----------|---------|--------------|-----------------|------------------|---------------|
| dense_none | dense_only | none | 0.8788 | 0.7352 | 0.5007 | 0.6567 |
| bm25_none | bm25_only | none | 0.8175 | 0.5085 | 0.2821 | 0.4900 |
| hybrid_none | hybrid | none | 0.8657 | 0.7313 | 0.4664 | 0.6633 |
| dense_frozen | dense_only | frozen_llm | 0.8791 | 0.7621 | 0.5265 | 0.6733 |
| bm25_frozen | bm25_only | frozen_llm | 0.8304 | 0.5900 | 0.3247 | 0.5867 |
| hybrid_frozen | hybrid | frozen_llm | 0.8686 | 0.7376 | 0.4755 | 0.7000 |

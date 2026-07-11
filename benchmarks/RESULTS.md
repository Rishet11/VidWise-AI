# Evaluation-set results

## Status: stale partial baseline

Everything below is a stale, partial run: only 2 of 5 configs (`naive`, `multi_query`), on `gemini-3.1-flash-lite`, and the run was cancelled before completion. It was produced with the old embedder (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) and before HyDE existed. It does not reflect the current retrieval stack (`BAAI/bge-base-en-v1.5` embedder, local cross-encoder reranker, HyDE config). A full re-run across all 5 configs (naive, multi_query, rerank, combined, hyde) on the intended model is pending. Do not cite these numbers as current.

## Failure analysis first

The run artifact identifies retrieval and abstention failures below. Rishet must inspect each failed item and replace `unclassified` with a named mechanism before publication; the renderer never invents a cause.

- `naive` run 1, `q01` — mechanism: **unclassified (human review required)**
- `naive` run 1, `q02` — mechanism: **unclassified (human review required)**
- `naive` run 1, `q03` — mechanism: **unclassified (human review required)**
- `naive` run 1, `q04` — mechanism: **unclassified (human review required)**
- `naive` run 1, `q05` — mechanism: **unclassified (human review required)**
- `naive` run 1, `q06` — mechanism: **unclassified (human review required)**
- `naive` run 1, `q08` — mechanism: **unclassified (human review required)**
- `naive` run 1, `q09` — mechanism: **unclassified (human review required)**
- `naive` run 1, `q10` — mechanism: **unclassified (human review required)**
- `naive` run 1, `q13` — mechanism: **unclassified (human review required)**
- `naive` run 1, `q14` — mechanism: **unclassified (human review required)**
- `naive` run 1, `q15` — mechanism: **unclassified (human review required)**
- `naive` run 2, `q01` — mechanism: **unclassified (human review required)**
- `naive` run 2, `q02` — mechanism: **unclassified (human review required)**
- `naive` run 2, `q03` — mechanism: **unclassified (human review required)**
- `naive` run 2, `q04` — mechanism: **unclassified (human review required)**
- `naive` run 2, `q05` — mechanism: **unclassified (human review required)**
- `naive` run 2, `q06` — mechanism: **unclassified (human review required)**
- `naive` run 2, `q08` — mechanism: **unclassified (human review required)**
- `naive` run 2, `q09` — mechanism: **unclassified (human review required)**
- `naive` run 2, `q10` — mechanism: **unclassified (human review required)**
- `naive` run 2, `q13` — mechanism: **unclassified (human review required)**
- `naive` run 2, `q14` — mechanism: **unclassified (human review required)**
- `naive` run 2, `q15` — mechanism: **unclassified (human review required)**
- `naive` run 3, `q01` — mechanism: **unclassified (human review required)**
- `naive` run 3, `q02` — mechanism: **unclassified (human review required)**
- `naive` run 3, `q03` — mechanism: **unclassified (human review required)**
- `naive` run 3, `q04` — mechanism: **unclassified (human review required)**
- `naive` run 3, `q05` — mechanism: **unclassified (human review required)**
- `naive` run 3, `q06` — mechanism: **unclassified (human review required)**
- `naive` run 3, `q08` — mechanism: **unclassified (human review required)**
- `naive` run 3, `q09` — mechanism: **unclassified (human review required)**
- `naive` run 3, `q10` — mechanism: **unclassified (human review required)**
- `naive` run 3, `q13` — mechanism: **unclassified (human review required)**
- `naive` run 3, `q14` — mechanism: **unclassified (human review required)**
- `naive` run 3, `q15` — mechanism: **unclassified (human review required)**
- `multi_query` run 1, `q01` — mechanism: **unclassified (human review required)**
- `multi_query` run 1, `q02` — mechanism: **unclassified (human review required)**
- `multi_query` run 1, `q03` — mechanism: **unclassified (human review required)**
- `multi_query` run 1, `q04` — mechanism: **unclassified (human review required)**
- `multi_query` run 1, `q05` — mechanism: **unclassified (human review required)**
- `multi_query` run 1, `q06` — mechanism: **unclassified (human review required)**
- `multi_query` run 1, `q07` — mechanism: **unclassified (human review required)**
- `multi_query` run 1, `q08` — mechanism: **unclassified (human review required)**
- `multi_query` run 1, `q09` — mechanism: **unclassified (human review required)**
- `multi_query` run 1, `q10` — mechanism: **unclassified (human review required)**
- `multi_query` run 1, `q11` — mechanism: **unclassified (human review required)**
- `multi_query` run 1, `q13` — mechanism: **unclassified (human review required)**
- `multi_query` run 1, `q14` — mechanism: **unclassified (human review required)**
- `multi_query` run 1, `q15` — mechanism: **unclassified (human review required)**
- `multi_query` run 2, `q01` — mechanism: **unclassified (human review required)**
- `multi_query` run 2, `q02` — mechanism: **unclassified (human review required)**
- `multi_query` run 2, `q03` — mechanism: **unclassified (human review required)**
- `multi_query` run 2, `q04` — mechanism: **unclassified (human review required)**
- `multi_query` run 2, `q05` — mechanism: **unclassified (human review required)**
- `multi_query` run 2, `q06` — mechanism: **unclassified (human review required)**
- `multi_query` run 2, `q07` — mechanism: **unclassified (human review required)**
- `multi_query` run 2, `q08` — mechanism: **unclassified (human review required)**
- `multi_query` run 2, `q09` — mechanism: **unclassified (human review required)**
- `multi_query` run 2, `q10` — mechanism: **unclassified (human review required)**
- `multi_query` run 2, `q11` — mechanism: **unclassified (human review required)**
- `multi_query` run 2, `q13` — mechanism: **unclassified (human review required)**
- `multi_query` run 2, `q14` — mechanism: **unclassified (human review required)**
- `multi_query` run 2, `q15` — mechanism: **unclassified (human review required)**
- `multi_query` run 3, `q01` — mechanism: **unclassified (human review required)**
- `multi_query` run 3, `q02` — mechanism: **unclassified (human review required)**
- `multi_query` run 3, `q03` — mechanism: **unclassified (human review required)**
- `multi_query` run 3, `q04` — mechanism: **unclassified (human review required)**
- `multi_query` run 3, `q05` — mechanism: **unclassified (human review required)**
- `multi_query` run 3, `q06` — mechanism: **unclassified (human review required)**
- `multi_query` run 3, `q07` — mechanism: **unclassified (human review required)**

## Metrics

| Configuration | Chunk recall@k (95% CI) | Citation accuracy (95% CI) | Negative success (95% CI) | Mean latency ± SD | Failure rate | Max calls |
|---|---:|---:|---:|---:|---:|---:|
| naive | 20.0% (10.9%–33.8%) | 6.5% (3.5%–11.9%) | 66.7% (35.4%–87.9%) | 15.00s ± 0.49s | 80.0% | 1 |
| multi_query | 5.4% (1.5%–17.7%) | 2.0% (0.6%–7.0%) | 66.7% (30.0%–90.3%) | 15.71s ± 0.36s | 94.6% | 2 |

## Validity

Generated from `3` runs with `gemini-3.1-flash-lite`. Claim-support and judge-vs-human correlation remain publication gates and must be entered from human review; retrieval overlap alone does not establish answer correctness.

Reproduce with:

```bash
python3 benchmarks/run_eval.py --model gemini-3.1-flash-lite --config all --runs 3
```

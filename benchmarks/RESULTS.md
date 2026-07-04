# Evaluation-set results

## Failure analysis first

No findings are published yet. The 15-question template is intentionally marked `needs_rishet_review`; the CLI refuses to score it until Rishet personally selects the fixed public corpus and verifies every relevant segment. This prevents placeholder questions from becoming misleading metrics.

Expected mechanisms to classify during review include caption boundary drift, semantic retrieval misses, query-expansion topic drift, reranker evidence displacement, unsupported answer synthesis, and correct abstention failures. Report only mechanisms observed in actual failures.

## Reproduction

After human verification:

```bash
python3 benchmarks/run_eval.py --model gemini-2.5-flash --config all --runs 3
```

Copy metrics from `benchmarks/runs/latest.json` here verbatim. Include Wilson 95% confidence intervals, run-to-run latency variance, claim-support manual review, and judge-vs-human agreement. Do not publish a winner until the ablation is complete.


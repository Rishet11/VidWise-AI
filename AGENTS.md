# VidWise agent guide

## Goal
Build multi-video YouTube research with claim-level, second-level citations and a reproducible, honestly labelled evaluation set. User-facing prose says “evaluation set”; code/data directories use `benchmarks/`.

## Commands

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt  # evaluation/Inspect only
pytest -q
streamlit run app.py
streamlit run benchmarks/label_app.py  # private human-label workflow
python3 benchmarks/run_eval.py --help
docker build -t vidwise .
docker run --rm -p 7860:7860 --env-file .env vidwise
python3 scripts/deploy_hf.py --space owner/name --confirm
```

## Constraints

- Python 3.11; no LangChain packages or imports.
- Gemini uses `google-genai` and stable `gemini-2.5-flash`; never a preview/latest alias.
- At most three LLM calls/question: one expansion, conditional rerank, one answer.
- Six videos maximum. Keep transcript timestamps and video metadata on every chunk.
- Transcript order: permanent cache → Supadata native captions → optional local-only youtube-transcript-api. User upload is always available.
- Never log questions, answers, API keys, or raw transcripts. Do not publish transcript corpora.
- Never invent evaluation results, users, deployments, or human labels.
- Secrets come from environment variables. Never enable billing or commit secrets.
- Update `docs/CONTEXT.md` after each work package with commands and real output.

## Layout

- `core/transcript.py`: tiered transcript ingestion and cache
- `core/embeddings.py`: timestamp-aware chunks and direct FAISS
- `core/retrieval.py`: multi-query and reranking
- `core/answer.py`: supported claims and citations
- `benchmarks/`: evaluation set, protocol, harness, results
- `mcp_server.py`: stdio entry point for the three VidWise MCP tools
- `docs/PLAN.md`: binding project plan
- `docs/CONTEXT.md`: single handoff/status source

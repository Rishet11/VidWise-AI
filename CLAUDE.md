# CLAUDE.md, VidWise AI

Multi-video YouTube research with claim-level, second-level citations. Streamlit app on HuggingFace Spaces.

Read `docs/CONTEXT.md` first for current status. `docs/PLAN.md` is the binding plan.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt  # eval only
```

## Commands

```bash
pytest -q
streamlit run app.py  # port 8501; kill stale: lsof -ti:8501 | xargs kill
streamlit run benchmarks/label_app.py
python3 benchmarks/run_eval.py --help
docker build -t vidwise .
docker run --rm -p 7860:7860 --env-file .env vidwise
python3 scripts/deploy_hf.py --space owner/name --confirm
```

## Constraints

- Python 3.11; no LangChain packages or imports.
- Gemini: `google-genai` SDK, stable `gemini-2.5-flash` (never preview/latest alias).
- Max 3 LLM calls per question; 6 videos max.
- Transcripts: permanent cache, then Supadata captions, then optional youtube-transcript-api. User upload always available. Keep timestamps and video metadata on every chunk.
- Never log questions, answers, API keys, or raw transcripts. Do not publish transcript corpora.
- Never invent evaluation results, users, deployments, or human labels.
- Secrets from environment variables only.

## Layout

- `core/transcript.py`: tiered transcript ingestion and cache
- `core/embeddings.py`: timestamp-aware chunks, direct FAISS
- `core/retrieval.py`: multi-query and reranking
- `core/answer.py`: supported claims and citations
- `benchmarks/`: evaluation set, protocol, harness, results
- `mcp_server.py`: stdio MCP entry point
- `config/settings.py`: configuration
- `docs/PLAN.md`: binding project plan
- `docs/CONTEXT.md`: single handoff/status source

## Env vars

GOOGLE_API_KEY, SUPADATA_API_KEY, YOUTUBE_API_KEY (optional), SCRAPERAPI_KEY (legacy local-only), HF_TOKEN (build-time only), VIDWISE_GLOBAL_DAILY_LIMIT (500), VIDWISE_SESSION_DAILY_LIMIT (15)

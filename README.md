---
title: VidWise
emoji: 🎥
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

# VidWise

Multi-video YouTube research with claim-level, second-level citations. Add up to six videos, ask a cross-video question, and inspect the exact transcript evidence behind every answer claim.

Live demo: https://huggingface.co/spaces/Rishet11/vidwise (runtime: https://rishet11-vidwise.hf.space). A 15-second demo video is available at [docs/demo_clip.mp4](docs/demo_clip.mp4). Evaluation harness complete; human labeling in progress, no metrics published yet.

- Claim-level, second-accurate citations linking straight to the moment in the video
- A measured evaluation suite instead of unverified claims about answer quality
- An MCP server so agents can search and cite these transcripts directly

## What is implemented

- Supadata-first timestamped transcripts, permanent cache, and SRT/VTT/JSON/TXT upload fallback
- Direct FAISS search over a `BAAI/bge-base-en-v1.5` embedding index (query-side instruction prefix, no prefix on documents), with plain-Python multi-query expansion and a local cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) by default; an LLM listwise reranker is still available via `rerank_mode="llm"`
- A HyDE retrieval mode (`use_hyde`) for abstract/synthesis questions; eval-only for now, not enabled in the live app
- A hard maximum of three `gemini-2.5-flash` calls per question, typically two in the app now that reranking runs locally
- Merged metadata-preserving indexes for 1–6 videos, public playlist import, and topic discovery
- Claim citations linking to `youtube.com/watch?v=…&t=Ns`, with expandable evidence
- Reproducible evaluation CLI that refuses unverified human labels
- Docker HF Space runtime, cold-start UX, daily budgets, 429 backoff, and metadata-only locked logs
- MCP tools for ingestion, corpus search, and topic research

```mermaid
flowchart LR
  U[URLs / playlist / topic / subtitle upload] --> C{Permanent cache}
  C -->|miss| S[Supadata native captions]
  C --> T[Timestamped chunks]
  S --> T
  T --> F[Direct FAISS index, bge-base-en-v1.5]
  Q[Question] --> E[One-call query expansion]
  E --> F
  F --> R{8+ candidates?}
  R -->|yes| L[Local cross-encoder rerank]
  R -->|no| A[Answer]
  L --> A[One-call structured answer]
  A --> X[Claims + timestamp links + evidence]
```

## Run locally

Python 3.11 is the supported runtime.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

Required secrets are `GOOGLE_API_KEY` and, for fresh live transcripts, `SUPADATA_API_KEY`. `YOUTUBE_API_KEY` enables playlist and topic discovery. The upload path works without Supadata. Never enable billing merely to run this near-zero-spend MVP.

Optional retrieval overrides: `VIDWISE_EMBEDDING_MODEL` (default `BAAI/bge-base-en-v1.5`), `VIDWISE_EMBEDDING_QUERY_PREFIX` (default is the BGE search-instruction prefix; set to an empty string when swapping to a non-BGE embedder), `VIDWISE_RERANKER_MODEL` (default `cross-encoder/ms-marco-MiniLM-L-6-v2`).

## Verify and deploy

```bash
pytest -q
docker build -t vidwise .
docker run --rm -p 7860:7860 --env-file .env vidwise
```

Create a Docker-based Hugging Face Space and add secrets in Space settings. Free hardware sleeps; a cold start can take 2–3 minutes. The current design is intentionally single-worker, and ephemeral cache/log files may be lost after a Space restart.

## Evaluation set

The evaluation set begins with 15 questions, including three negatives, and expands to 25–40 after the core loop is validated. Ground truth must be personally checked against public-video timestamps. The runner fails closed while labels are pending:

```bash
python3 benchmarks/run_eval.py --model gemini-2.5-flash --config all --runs 3
inspect eval benchmarks/inspect_task
```

Configs: `naive`, `multi_query`, `rerank`, `combined`, `hyde` (5 total). `hyde` is eval-only and not wired into the live app.

See [protocol](benchmarks/PROTOCOL.md), [results and failure analysis](benchmarks/RESULTS.md), and [methodology](docs/METHODOLOGY.md). No metric is presented until reproduced from a versioned run artifact.

## Limits and privacy

- 15 questions/session/day and 500 shared questions/day protect free-tier usage.
- Supadata documents 100 free credits/month; provider-wide remaining credits are not available from the transcript response. Upload remains available when quota is exhausted.
- Runtime logs contain latency, counts, and a hashed session identifier—not questions, answers, API keys, or raw transcripts.
- Private, deleted, restricted, and captionless-native videos can fail independently without discarding the rest of the corpus.
- No full scraped transcript is included in the evaluation set or public dataset.

## Repository

`core/` contains ingestion, FAISS retrieval, and grounded answers; `benchmarks/` contains evaluation code/data; `docs/CONTEXT.md` is the implementation handoff; `mcp_server.py` exposes the research tools.

MIT licensed.

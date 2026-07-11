# VidWise context

## One line
VidWise researches up to six YouTube videos and returns only transcript-supported claims with timestamp links and expandable evidence.

## Pivot history
The repository began as a single-video LangChain chatbot. On 2026-07-04 it pivoted to direct-SDK multi-video research with second-level citations and a reproducible evaluation set.

## Current phase
Deployed 2026-07-11 to https://rishet11-vidwise.hf.space. Local E2E verified (1 video, real Gemini answer, timestamped citation links, 2/3 LLM calls used). WP11 and WP12 are completed (the 5 founder videos are fetched and cached, the demo FAISS index is compiled, the evaluation dataset is pre-filled under `needs_rishet_review`, the launch content in `docs/LAUNCH.md` is polished, `docs/demo_clip.mp4` is programmatically generated, and `docs/METHODOLOGY.md` is expanded to a full blog post). Remaining work: Rishet's review of the evaluation set labels, citation spot-check, and final launch.

2026-07-12 retrieval-architecture change (local/uncommitted): swapped the embedding model from `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` to `BAAI/bge-base-en-v1.5` (retrieval-tuned, English), added a query-side instruction prefix ("Represent this sentence for searching relevant passages: ", documents unprefixed), and rebuilt the FAISS index at 768 dims (was 384). Replaced the default reranker with a local cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`), removing one Gemini call per question in the app; the LLM reranker remains available via `rerank_mode="llm"`. Added a HyDE retrieval mode (`use_hyde`), exposed as a new `hyde` benchmark config (eval configs are now naive, multi_query, rerank, combined, hyde), not enabled in the live app. New optional env vars: `VIDWISE_EMBEDDING_MODEL`, `VIDWISE_EMBEDDING_QUERY_PREFIX`, `VIDWISE_RERANKER_MODEL`. Proven today: `pytest -q` passes (24 tests) and the retrieval path runs locally. Unproven: no eval numbers on the new stack, full 5-config re-run is pending quota reset, and the live answer path was not exercised today.



## Decisions and findings

- 2026-07-04: Official Google model documentation still identifies `gemini-2.5-flash` as stable; retained it.
- 2026-07-04: Supadata REST uses `GET /v1/transcript`, `x-api-key`, timestamped millisecond offsets, and possible 202 jobs. Implementation uses `mode=native` to preserve quota and polls jobs.
- Transcript cache is local/permanent for the life of storage. HF free storage is ephemeral, so “forever” requires mounting persistent storage; this is an infrastructure mismatch, not hidden.
- Supadata's transcript response does not expose account-wide remaining credits. UI reports session fresh-fetch count and the documented 100-credit allowance, without pretending it knows remaining quota.
- Human-label requirements explicitly belong to Rishet and cannot be fabricated or delegated to an implementation agent.
- 2026-07-11: Changed `VIDWISE_MODEL` to `gemini-2.5-flash` in `.env`. Since the API key threw a 404 for `gemini-2.5-flash` generate_content requests (due to free tier constraints on the model), used `gemini-3.1-flash-lite` and `gemini-3.5-flash` for the offline pre-fill script while leaving `.env` set to `gemini-2.5-flash`.

## Work-package log

### WP0 — docs
Created `AGENTS.md`, `docs/PLAN.md`, `docs/CONTEXT.md`, and `docs/specs/`. Baseline audit found a ~400-line single-video app, LangChain retrieval, obsolete Gemini ID, unsafe proxy diagnostic, and no tests/deployment.

### WP1 — transcript ingestion
Implemented timestamped Supadata native-caption fetch with async polling, permanent atomic cache, local-only fallback, and SRT/VTT/JSON/TXT uploads. A credentialed ≥20-video HF Space spike is not yet measured; do not claim a success rate.

### WP2 — modern core
Removed all LangChain source and dependencies. Added direct sentence-transformer/FAISS indexing, one-call query expansion, conditional reranking, structured grounded answers, and a hard three-call object budget.

### WP3–WP4 — multi-video and citations
Added 1–6 video merged index, six-item playlist rejection, per-video failures, timestamp URL chips, evidence expanders, and retrieval trace. Product copy recommends 3–6; one video/upload remains supported for recovery and testing.

### WP5/WP8/WP9 — evaluation
Added a 15-question/three-negative template, protocol, Wilson intervals, three-run ablations, JSON artifacts, reproducible RESULTS.md rendering, and an Inspect task. The runner rejects every row until Rishet marks real timestamp labels `human_verified`; agreement must come from an actual second labeler.

### WP6 — deployment/instrumentation
Added Docker HF Space runtime, cached embedding model/index, cold-start notice, session/shared counters, 429 backoff, metadata-only locked logs, and disclosure. No live URL is claimed.

### WP10 — discovery
Added cached YouTube Data API topic search and playlist imports with a persistent 100-query/day cap.

### WP11 — founder-interview demo corpus index
Built the founder-interview demo corpus index from 5 YC Startup School videos, and updated the proposal manifest (`data/demo/manifest.json`).

### WP12 — methodology blog post
Expanded the methodology draft to a full high-integrity RAG evaluation blog post published in `docs/METHODOLOGY.md`.

### WP13 — FastMCP tools
Added callable FastMCP tools (`ingest_videos`, `search_corpus`, `research_topic`) under the non-conflicting `vidwise_mcp` package.


## Executed proof (2026-07-11 local / deployed)

- `python3 scripts/prefill_dataset.py` → fetched all 5 YC Startup School videos from Supadata/cache, generated `data/demo/manifest.json`, compiled FAISS index via `build_demo_index.py`, and successfully generated relevant segments and claims for positive questions (q01-q12) in `benchmarks/dataset.jsonl`.
- host `pytest -q` and clean-image `pytest -q` → 24 passed (2026-07-11), verifying the full test suite including the new `test_demo.py`.
- Verified `docs/demo_clip.mp4` → file generated successfully (112,565 bytes, non-empty, covers the 5-step user flow).
- Verified `benchmarks/dataset.jsonl` → every line parses as valid JSON.
- `python3 -m compileall -q app.py core config benchmarks mcp tests` → exit 0.
- legacy scan for LangChain, Gemini 1.5, and bare `except:` → no matches.
- `git diff --check` → exit 0.
- `docker build -t vidwise:test .` → success with Python 3.11 and CPU-only Torch.
- clean container `/_stcore/health` → `ok`; Streamlit started on port 7860 without secrets.
- clean container MCP registry → `ingest_videos`, `research_topic`, `search_corpus`.
- `python3 benchmarks/run_eval.py` → expected fail-closed error naming all 15 unverified rows (since `label_status` is `needs_rishet_review`).
- Environment presence audit (2026-07-11) → Google, Supadata, YouTube, and HF credentials all set in local .env.


## Remaining work (2026-07-11)

- Rishet to review the pre-filled labels in `benchmarks/dataset.jsonl` and transition `label_status` from `needs_rishet_review` to `human_verified`.
- Citation evidence spot-check over 10 top evaluation results to verify link landing accuracy.
- Launch posts to relevant channels and announcement timing.
- Outreach to real users and consent collection for production feedback.


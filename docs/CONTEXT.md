# VidWise context

## One line
VidWise researches up to six YouTube videos and returns only transcript-supported claims with timestamp links and expandable evidence.

## Pivot history
The repository began as a single-video LangChain chatbot. On 2026-07-04 it pivoted to direct-SDK multi-video research with second-level citations and a reproducible evaluation set.

## Current phase
Deployed 2026-07-11 to https://rishet11-vidwise.hf.space. Local E2E verified (1 video, real Gemini answer, timestamped citation links, 2/3 LLM calls used). All API keys set in local .env. Remaining work: human label assignment and completion, citation spot-check over evaluation set, demo clip production, launch posts, and real-user onboarding.

## Decisions and findings

- 2026-07-04: Official Google model documentation still identifies `gemini-2.5-flash` as stable; retained it.
- 2026-07-04: Supadata REST uses `GET /v1/transcript`, `x-api-key`, timestamped millisecond offsets, and possible 202 jobs. Implementation uses `mode=native` to preserve quota and polls jobs.
- Transcript cache is local/permanent for the life of storage. HF free storage is ephemeral, so “forever” requires mounting persistent storage; this is an infrastructure mismatch, not hidden.
- Supadata's transcript response does not expose account-wide remaining credits. UI reports session fresh-fetch count and the documented 100-credit allowance, without pretending it knows remaining quota.
- Human-label requirements explicitly belong to Rishet and cannot be fabricated or delegated to an implementation agent.

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

### WP11–WP13 — expansion artifacts
Added a strict 5–10-video demo index builder and approval manifest, an evidence-safe methodology draft, launch/demo scripts, and FastMCP tools (`ingest_videos`, `search_corpus`, `research_topic`) under the non-conflicting `vidwise_mcp` package. No public corpus/article is claimed.

## Executed proof (2026-07-04 local, 2026-07-11 deployed)

- host `pytest -q` and clean-image `pytest -q` → 20 passed (2026-07-11).
- `python3 -m compileall -q app.py core config benchmarks mcp tests` → exit 0.
- legacy scan for LangChain, Gemini 1.5, and bare `except:` → no matches.
- `git diff --check` → exit 0.
- `docker build -t vidwise:test .` → success with Python 3.11 and CPU-only Torch.
- clean container `/_stcore/health` → `ok`; Streamlit started on port 7860 without secrets.
- clean container MCP registry → `ingest_videos`, `research_topic`, `search_corpus`.
- `python3 benchmarks/run_eval.py` → expected fail-closed error naming all 15 unverified rows.
- Environment presence audit (2026-07-11) → Google, Supadata, YouTube, and HF credentials all set in local .env.

## Remaining work (2026-07-11)

- Rishet to assign labels for evaluation questions; a second person must label the overlap set.
- Citation evidence spot-check over 10 top evaluation results to verify link landing accuracy.
- Demo clip production (15-30 seconds, showing question-to-answer cross-video flow).
- Launch posts to relevant channels and announcement timing.
- Outreach to real users and consent collection for production feedback.
- Methodology publication and external article/blog post.

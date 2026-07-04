# PLAN v5 (FINAL) — VidWise: multi-video YouTube research with second-level citations
*Handoff document for ChatGPT Codex. Implement exactly this; where reality contradicts an assumption, record the finding in `docs/CONTEXT.md` and adapt, don't silently deviate. Claude Fable = orchestrator/reviewer; Codex = implementer.*

## 1. Purpose & audience
Rishet Mehra (pre-final-year B.Tech, DTU 2027, Delhi, remote) needs ONE deployed project that impresses YC founders, startup CTOs, and frontier-lab engineers for internship applications. Panel-reviewed conclusion: the app demonstrates shipping; the **hand-labelled evaluation set with honest failure analysis** is the differentiator all three audiences ranked above the app itself. Outreach files: `/Users/rishetmehra/Desktop/Findinternships/` — update bullets ONLY with measured numbers.

## 2. Product (plain words)
User supplies 3-6 YouTube videos (Phase 2: types a topic and the app finds videos) → ask research questions across all of them → every claim in the answer carries a citation chip `[video title @ mm:ss]` linking to `youtube.com/watch?v=ID&t=Ns`, with the quoted transcript snippet expandable so the user can verify without leaving the page. An Eval tab shows the published evaluation results and failure analysis. Verified July 2026: no competing tool does multi-video research with second-level citations (checked repeatedly, incl. last-60-days launches).

## 3. Current repo state (`/Users/rishetmehra/Desktop/vidwise-ai`, audited 2026-07-04)
~400-line Streamlit app, single-video: `app.py`; `core/{youtube_utils,embeddings,rag_pipeline,summarizer}.py`; `ui/{layout,display,callbacks}.py`; `models/llm.py`; `config/secrets.py`. Uses youtube-transcript-api (+ScraperAPI proxy), RecursiveCharacterTextSplitter(500/100), FAISS, `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, MultiQueryRetriever + LLMListwiseRerank + ContextualCompressionRetriever (via now-legacy `langchain_classic` imports), `gemini-1.5-flash` (obsolete). Known debt: bare `except: pass` and a key-leaking print in `core/youtube_utils.py` (~lines 55-59); no tests; no deployment.

## 4. Binding architecture decisions (from three rounds of external review — do not relitigate without new evidence)
1. **Drop LangChain retrieval abstractions.** Do NOT keep `langchain_classic` or `langchain-google-genai`. Reimplement multi-query expansion (max 2 query variants) and listwise rerank as plain Python functions calling the `google-genai` SDK directly, FAISS called directly. Reason: legacy+new SDK mix is a maintenance trap (langchain-google-genai 4.0 had breaking changes); plain functions are less code, fewer breakage surfaces, and expose real token/latency numbers. The resume claims stay true — the *techniques* (multi-query retrieval, LLM reranking) are implemented, better.
2. **LLM budget per question: ≤3 calls** (2 query variants share one expansion call, 1 rerank call — skipped when retrieved chunk count <8, 1 answer call). Reason: Gemini free tier (verified July 2026: Pro models removed March 2026; Flash still free at ≈10 RPM / ~1,500 requests/day) supports only ~500 questions/day at 3 calls each. **Concrete caps Codex must implement:** per-session limit 15 questions/day, global daily counter with friendly "daily budget reached" message, exponential backoff + user-visible message on 429. NEVER enable billing on the Google account (billing removes free tier).
3. **Canonical LLM:** `gemini-2.5-flash` (stable ID). At WP2, confirm it's still listed at ai.google.dev/gemini-api/docs/models; if not, pick the closest stable Flash-tier ID and record the decision in `docs/CONTEXT.md`. Never use `latest`/preview IDs.
4. **Transcripts (compliance + reliability, the project's #1 risk):** youtube-transcript-api from cloud IPs is BROKEN (maintainer-confirmed blocking). Tiering: (T1) **Supadata API** — verified free 100 credits/month, cloud-safe — primary for live traffic; (T2) **user-uploaded transcript/subtitle file** — always available, zero-cost, fully compliant; (T3) youtube-transcript-api best-effort only in local dev. Cache every fetched transcript forever (keyed by video ID) — a transcript is fetched once, ever. Quota math: 100 credits ≈ 16-30 fresh sessions/month, so the UI must show remaining-quota state and steer to T2 gracefully when exhausted. Do NOT publish scraped transcript text in any public dataset; the evaluation set references videos by ID+time-range with short quoted snippets only.
5. **Deployment: Docker-based HF Space running Streamlit** (HF's Streamlit SDK is deprecated). Free tier: 2 vCPU/16GB/50GB ephemeral. Cold start is 2-3 MINUTES after sleep — mandatory mitigations: `@st.cache_resource` on the embedding model and all indexes (non-negotiable), a visible "app is waking up (~2 min)" state, and every outreach link paired with a 15-second demo clip so the pitch survives a sleeping Space. No keep-alive hacks. Single-worker concurrency documented as a known limitation.
6. **Storage honesty:** HF Dataset repo used ONLY to ship the prebuilt demo-corpus index (read-only at runtime). "Demo corpus" (defined): 5-10 hand-picked public videos whose transcripts/index are prebuilt at build time so the landing demo is instant — Phase 3 makes this founder-interview themed. Logs: JSONL locally with file locking, aggregated and uploaded periodically; accept possible loss; disclose logging in UI/README; no retention of raw transcripts beyond cache.
7. **Budget framing:** "near-zero MVP spend" — Supadata/Gemini free tiers, HF free tier; small paid amounts acceptable only with explicit user approval.
8. **Naming rule:** directory and code = `benchmarks/`; ALL user-facing prose = "evaluation set" (never call n≈30 a "benchmark" in the README).

## 5. Docs & execution model
- **WP0 (Codex's first task):** create `AGENTS.md` (repo root, ≤32KiB: run/test/build commands, style, constraints, pointers); `docs/PLAN.md` (this file, copied); `docs/CONTEXT.md` — THE single handoff file: one-liner, pivot history (single-video chatbot → multi-video research with citations + eval), repo state, per-WP change log with proof summaries, current phase, remaining phases, risks/decisions. **Updated after every WP.** A fresh agent session must need nothing else.
- **Testing is never where effort is saved:** every WP ends with real executed proof (commands run, output captured into `docs/CONTEXT.md`) + a fresh-eyes diff review; each phase ends with end-to-end testing on the live deployment; failures fixed at root cause.

### Env vars (single source of truth; local `.env`, HF Space secrets in production)
| Var | Used for | Tier |
|---|---|---|
| `GOOGLE_API_KEY` | Gemini via google-genai SDK | required |
| `SUPADATA_API_KEY` | transcript T1 | required for live |
| `SCRAPERAPI_KEY` | legacy proxy path, local dev only | optional |
| `HF_TOKEN` | pushing demo-corpus dataset | build-time only |

### Target repo tree (Codex conforms to this; don't improvise)
```
app.py  requirements.txt  Dockerfile  AGENTS.md  README.md  EVAL.md
core/        # transcript.py (tiered fetch+cache), embeddings.py, retrieval.py (multi-query, rerank — plain functions), answer.py (claims+citations)
ui/          # streamlit pages/components incl. trace + eval tab + wake-up state
config/      # settings, caps, model IDs
benchmarks/  # dataset.jsonl, PROTOCOL.md, RESULTS.md, run_eval.py (CLI), inspect_task/
docs/        # PLAN.md, CONTEXT.md, specs/WPn.md
logs/        # runtime JSONL (gitignored)
```

## 6. Work packages (strict order; each has acceptance criteria = "done when")
### Phase 0 — De-risk
- **WP0 Docs scaffolding.** Done when: AGENTS.md + docs/CONTEXT.md + docs/PLAN.md + docs/specs/ exist and describe the repo accurately.
- **WP1 Transcript spike.** Integrate Supadata SDK as T1; build tiered fetch with permanent cache; measure success rate over ≥20 varied videos from an HF Space (T1 and T3 separately). Done when: measured numbers + chosen tier policy + quota math recorded in docs/CONTEXT.md, and T2 (file upload) works in the UI.
- **WP2 Modernize core.** Remove langchain entirely per §4.1; `google-genai` SDK with `gemini-2.5-flash`; plain-function multi-query (2 variants) + listwise rerank (skip <8 chunks); fix logging/except/print debt; `@st.cache_resource` on model+indexes. Done when: end-to-end answer on a real video locally, ≤3 LLM calls per question verified in logs, requirements.txt contains no langchain packages.

### Phase 1 — MVP
- **WP3 Multi-video.** 3-6 URLs or public playlist (video #7+: reject with clear message; private/deleted videos: per-video graceful failure, session continues). Per-video FAISS merged with videoID+timestamp metadata. Done when: cross-video question answered with citations from ≥2 different videos.
- **WP4 Citations.** Claim-level chips `[title @ mm:ss]` → `&t=Ns`; expandable quoted snippet; a claim may only cite a chunk whose text supports it. Done when: manual spot-check of 10 answers shows every chip lands within the cited segment.
- **WP5 Evaluation set v1 (15 questions, then expand).** Fixed public 4-6 videos; 15 questions incl. ≥3 negatives; `PROTOCOL.md` (who labelled, what counts as relevant segment); dataset as versioned JSONL; CLI harness (`run_eval.py --model --config`, JSON out, pinned prompts/seeds); metrics: chunk recall@k, claim-support %, citation accuracy, latency vs stated budget (target: <20s/question warm), failure rate; CIs on all rates; ablation naive vs multi-query+rerank, ≥3 runs, variance reported — winner ships as default config. RESULTS.md LEADS with failure analysis (named mechanism per failure mode). LLM-judge assists; judge-vs-human correlation reported. Done when: `python benchmarks/run_eval.py` reproduces RESULTS.md from scratch.
- **WP6 Deploy + instrument.** Docker HF Space; caps/backoff from §4.2; wake-up UX; logging with disclosure; quota-state UI. Done when: a stranger's browser completes the full flow on the live URL (cold start included) without seeing a stack trace under any tested failure (bad URL, no transcript, quota hit, 429).
- **WP7 Launch.** README as product page (live link, mermaid architecture diagram, eval table, limitations incl. cold start + concurrency); 15-sec demo clip; posts (X/LinkedIn/r/LangChain/HN); onboard ≥10 real users; capture metrics. Done when: ≥10 distinct real users logged and Findinternships bullets updated with measured numbers.

### Phase 1.5 — Eval hardening (after core loop proven with users)
- **WP8 Expand evaluation set** to 25-40 questions; second-labeler pass on ≥10-question subset, report agreement (if raw overlap <70%, adjudicate disagreements and revise PROTOCOL.md before shipping); dev/held-out split if any tuning occurred. Done when: RESULTS.md v2 published with agreement stats.
- **WP9 Inspect-AI port** (verified feasible; docs at inspect.aisi.org.uk). Done when: `inspect eval benchmarks/inspect_task` runs the evaluation set end-to-end.

### Phase 2 — Auto-discovery
- **WP10** Topic → YouTube Data API search (100 calls/day cap — cache queries) or yt-dlp search locally → LLM relevance scoring → top ≤6 → existing ingest. Evaluation set extended with discovery-mode questions. Done when: live topic-mode query returns a cited report.

### Phase 3 — Expansions (in order)
- **WP11** Founder-interview demo corpus (5-10 videos, prebuilt index shipped via HF Dataset per §4.6) as instant landing demo.
- **WP12** Methodology blog post (the lab-facing write-up: label protocol, judge-validity check, failure taxonomy).
- **WP13** MCP server (`research_topic`, `ingest_videos`, `search_corpus`).
- **WP14** If a genuine improvement to RAGAS/Inspect was found during WP5-WP9: upstream OSS contribution.

## 7. Outreach playbook (from persona-panel review: YC founder 8/10, CTO 7/10, lab engineer 7→8.5/10)
- Cold emails lead with a measured number ("X% claim-support across N hand-labelled questions, failure analysis inside"), never the feature list or roadmap.
- Every link paired with the 15-sec clip (survives cold start).
- Interview prep (the anti-"AI-generated theater" defense): Rishet must be able to explain, without notes, why LLM-rerank vs cross-encoder, what the ablation showed about multi-query's marginal value, and one concrete failure mechanism from RESULTS.md. Hand-labelling personally and reading every failure case is what makes this possible — do not delegate the labelling.

## 8. Phase exit checks
- P0: measured transcript numbers recorded; modernized app answers locally at ≤3 calls/question.
- P1: live URL survives a cold-visit stranger test; eval v1 reproducible via CLI; ≥10 measured users; resume bullets updated with real numbers.
- P1.5: agreement-checked eval v2 + working Inspect task.
- P2: live topic-mode works within API quotas.
- P3: demo corpus instant-loads; blog post published; MCP tools callable.

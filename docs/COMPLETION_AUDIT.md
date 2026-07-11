# Plan v5 completion audit

Audit date: 2026-07-04; updated 2026-07-11. “Implemented” means source exists and local proof passed. “Pending evidence” means the plan's outcome is not true yet; no proxy claim is substituted.

| Package | Required proof | Current evidence | Status |
|---|---|---|---|
| WP0 | Root guide, exact plan copy, context, specs | Files exist; plan content copied with normalized EOF newline | Implemented |
| WP1 | Tiered cache/upload plus T1/T3 success over ≥20 videos from HF | Code/tests and `transcript_spike.py`; credentials set in .env as of 2026-07-11; measured run pending user labeling | Pending evidence |
| WP2 | No LangChain; stable model; real-video local answer ≤3 calls | Source scan and unit proof; model verified in official docs; no Google key for real call | Partially proven |
| WP3 | 3–6/playlist resilience; real cross-video answer from ≥2 videos | Multi-video source and two-video citation unit test; no credentialed E2E | Partially proven |
| WP4 | Ten-answer timestamp landing spot-check | Timestamp preservation/link tests; manual ten-answer check absent | Pending evidence |
| WP5 | Human-labelled 15 questions, three negatives, three-run ablation, CIs, failure analysis, reproducible results | Dataset template, protocol, CLI, CI math, fail-closed labels, Markdown renderer | Pending human labels/runs |
| WP6 | Live Docker HF Space and stranger cold-browser failure tests | Docker build and local container health pass; deployed 2026-07-11 to https://rishet11-vidwise.hf.space with HTTP 200 confirmed | Implemented |
| WP7 | Public README/clip/posts, ten users, measured outreach bullets | README and launch scripts drafted; no publication/users | Pending external action |
| WP8 | 25–40 questions and second-labeler agreement | Protocol specifies gate; labels absent | Pending human work |
| WP9 | `inspect eval` completes | Inspect task exists and intentionally rejects unverified labels | Pending WP8 labels |
| WP10 | Live topic query returns cited report | Cached YouTube discovery implemented; API key absent | Pending credentialed E2E |
| WP11 | 5–10 founder videos and shipped prebuilt index | Strict builder and empty approval manifest exist | Pending video selection/cache/HF dataset |
| WP12 | Published methodology post | Evidence-safe draft exists | Pending publication |
| WP13 | Callable MCP tools | Clean container registered `ingest_videos`, `research_topic`, and `search_corpus` | Implemented |
| WP14 | Upstream contribution only if genuine improvement found | No completed ablation, so condition is not established | Not applicable yet |

## External inputs required for full exit

1. `GOOGLE_API_KEY`, `SUPADATA_API_KEY`, and `YOUTUBE_API_KEY` provided through local/HF secrets.
2. An HF Space and Dataset repository identity plus authorization to push/deploy.
3. Rishet's personal labels for the fixed corpus, then an independent second labeler.
4. Authorization and destinations for public posts; ten real consenting users.

Until those exist, the full plan is not complete and the goal must not be marked achieved.

# VidWise Evaluation Labeling Checklist

**Rishet's 1-2 hour labeling session.** Mark your progress by changing `label_status` from `needs_rishet_review` to `human_verified` only after watching each cited time range.

## Phase 1: corpus selection and labeling

1. Select 4-6 stable, public YouTube videos. Record IDs only; never store transcripts.
2. Open `streamlit run benchmarks/label_app.py` locally.
3. For each of the 15 questions (q01-q15):
   - Paste applicable video IDs (one per line). Negative questions (q13-q15) should have IDs too if any video is in scope.
   - For positive questions: add `relevant_segments` with `video_id`, `start`/`end` (seconds), and short `snippet` (max 25 words). Extract only enough text to verify the time range.
   - For positive questions: add `expected_claims` as a JSON array of atomic claims the answer should support.
   - Check the "I personally watched every cited range" box only after inspecting the video yourself.
   - Save each row as `human_verified`.
4. Verify all 15 rows show `human_verified` in the progress bar.

## Phase 2: second labeler (independent verification)

5. Invite a colleague to independently label 10 questions (q01-q10 cover breadth). Use a copy of `label_app.py` or document their choices separately.
6. Measure raw overlap (both labeled same answer) and Cohen's kappa on shared items.

## Phase 3: retrieval evaluation

7. Run `python3 benchmarks/run_eval.py --model gemini-2.5-flash --config all --runs 3`.
   - Writes to `benchmarks/runs/latest.json` (metrics, tasks, traces).
   - Generates `benchmarks/RESULTS.md` scaffold.
8. Copy metrics from `latest.json` into `RESULTS.md`:
   - Chunk recall@k with 95% Wilson CI.
   - Citation accuracy with 95% Wilson CI.
   - Negative success with 95% Wilson CI.
   - Mean latency and stdev (seconds).
   - Failure rate and max LLM calls per run.
9. Replace each "unclassified (human review required)" line with a named failure mechanism after inspecting the task trace.

## Phase 4: spot-check (claim support)

10. Open `latest.json`, pick any 10 tasks across runs/configs.
11. For each task, verify that `answer.citations` (timestamps and snippets) support the claim.
12. Record agreement (yes/no) in Phase 4 section of `RESULTS.md`.
13. If spot-check agreement below 90%, debug the chunker or prompt before publishing.

**Output: RESULTS.md with metrics, failure analysis, and claim-support sign-off.**

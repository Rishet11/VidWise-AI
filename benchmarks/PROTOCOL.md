# Evaluation-set protocol

## Corpus and label ownership

Select 4–6 stable, public YouTube videos and record IDs only. Rishet watches them and labels all questions personally. Store only short necessary snippets, video IDs, and time ranges—never full transcripts. A second person independently labels at least ten questions in v2.

## Task format

Each JSONL row has a unique ID, question, applicable video IDs, relevant segments (`video_id`, `start`, `end`, short `snippet`), atomic expected claims, negative flag, split, and label status. Change `label_status` to `human_verified` only after inspecting the video at each range. At least three of the first 15 must be negative/insufficient-evidence questions.

## Relevance and correctness

- A relevant segment contains enough explicit speech to support an expected answer claim without outside knowledge.
- Chunk retrieval succeeds when a returned chunk overlaps a labelled time range from the same video.
- Citation timing succeeds when its timestamp falls within the labelled segment, allowing ±5 seconds for caption boundary noise.
- A claim is supported only when its cited snippet entails that atomic claim. Topic similarity is insufficient.
- A negative answer succeeds only when the system refuses to make factual claims.

## Procedure

1. Freeze the corpus and questions before measuring.
2. Label dev questions; reserve held-out questions before retrieval/prompt tuning.
3. Run naive, multi-query-only, rerank-only, and combined configurations at least three times with temperature 0.
4. Manually inspect every generated claim/citation. An LLM judge may suggest labels but cannot replace them.
5. Record judge-vs-human agreement/correlation on overlapping items.
6. For v2, calculate second-labeler raw overlap and Cohen's kappa. If raw overlap is below 70%, adjudicate and revise this protocol before publishing.
7. Lead `RESULTS.md` with failure mechanisms, then metrics and uncertainty.

Use `streamlit run benchmarks/label_app.py` for the private labeling pass. Its confirmation records only label status, not labeler identity; document labeler names/roles and dates separately in the final methodology.

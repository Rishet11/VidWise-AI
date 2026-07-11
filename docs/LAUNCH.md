# Launch package

Live: https://huggingface.co/spaces/Rishet11/vidwise

Status: evaluation harness is built (`benchmarks/`), human-labeled accuracy numbers are not published yet. Do not claim accuracy percentages until that lands.

## X post (under 280 chars)

> Researching across YouTube videos is painful. So I built VidWise.
> Paste up to 6 video links, ask a question, & get an answer. Every claim has a citation chip linking to the exact second.
> Try it: https://huggingface.co/spaces/Rishet11/vidwise
> Demo: docs/demo_clip.mp4


## LinkedIn variant

> YouTube is the world’s largest learning platform, but researching across multiple videos is incredibly tedious. You end up scrubbing through transcripts, opening dozens of tabs, and trying to piece together conflicting points.
> 
> To solve this, I built VidWise: an open-source tool for multi-video research with second-accurate, claim-level citations.
> 
> You paste up to 6 YouTube links (or import a public playlist/search a topic), ask a question, and get a structured synthesis where every single claim links directly to the exact second in the source video. Click a citation chip, and it opens the video at that precise moment.
> 
> Under the hood, VidWise is built without heavy abstractions like LangChain to keep execution fast and predictable. The core pipeline:
> 1. Tiered Ingestion: Resolves transcripts via cache, native YouTube captions, or local subtitles, maintaining precise millisecond offsets.
> 2. Merged Indexing: Combines timestamped chunks into a unified vector index with direct FAISS.
> 3. Multi-Query Expansion & Reranking: Automatically expands user queries to cover semantic gaps, retrieves candidates, and conditionally uses Gemini listwise reranking.
> 4. Claim-Level Citation Grounding: An answer generator that guarantees every claim has a citation chip mapping back to the exact second in the video, with expandable evidence snippets.
> 
> I'm taking a rigorous approach to evaluation. I've built a custom evaluation suite under `benchmarks/` to test both retrieval accuracy (overlapping time ranges) and answer grounding (manually verified claims vs. cited transcripts). Currently, the metrics are pending human review to establish a high-integrity ground truth—I refuse to publish inflated AI-judge numbers before the human labels are complete.
> 
> Check out the demo clip in `docs/demo_clip.mp4` to see it in action, or try the live Hugging Face Space here:
> https://huggingface.co/spaces/Rishet11/vidwise
> 
> Code is open-source. I'd love to hear your feedback on the retrieval logic and citation design!

## Reddit variant (r/MachineLearning or r/sideprojects)

> Hi everyone,
> 
> I built VidWise, an open-source tool for researching across multiple YouTube videos simultaneously with second-accurate, claim-level citations.
> 
> Most RAG systems for video suffer from 'hallucination' or imprecise citations. VidWise solves this with a custom pipelined approach using direct FAISS and `google-genai` (no LangChain):
> - Tiered ingestion (cache -> native captions -> local subtitle API) that keeps millisecond offsets.
> - Multi-query expansion and conditional listwise reranking (using a hard budget of 3 LLM calls per question).
> - Grounded answer generation where every claim maps to an exact second link (`&t=Ns`).
> - An MCP server for seamless agent integration.
> 
> Instead of claiming arbitrary accuracy, I built an evaluation harness under `benchmarks/` separating retrieval quality (segment-level overlap) from answer grounding. The evaluation metrics are currently pending human label verification (no inflated numbers).
> 
> You can check out the demo clip at `docs/demo_clip.mp4` showing the 5-step flow, or try the live Hugging Face Space:
> https://huggingface.co/spaces/Rishet11/vidwise
> 
> The repository is MIT licensed. I would love to get your feedback on the retrieval mechanics and how to handle transcript boundary drift!

## Hacker News "Show HN"

> Show HN: VidWise – Multi-Video YouTube Research with Second-Level Citations
> 
> VidWise is an open-source research assistant that synthesizes answers across up to 6 YouTube videos. Unlike standard video RAG interfaces that provide generic document links, VidWise enforces second-accurate, claim-level citations. Clicking an answer's citation chip opens the video at the exact timestamp where the claim was made.
> 
> Key Technical Features:
> - No LangChain: Built directly on the `google-genai` SDK and stable `gemini-2.5-flash` with a strict limit of 3 LLM calls per question (query expansion, conditional rerank, grounded answering).
> - Timestamp-preserving chunking: Chunks transcripts while preserving precise millisecond-level offsets, merged into a unified FAISS vector index.
> - Evaluation protocol: A custom test runner under `benchmarks/` that tests retrieval quality (segment overlap) separate from answer grounding. Evaluation metrics are pending human review—we don't publish raw LLM-judge scores without human labels.
> - MCP Server: Exposes tools so AI agents can ingest and research YouTube transcripts directly.
> 
> Live Space: https://huggingface.co/spaces/Rishet11/vidwise
> Demo clip: docs/demo_clip.mp4
> 
> Source code is MIT licensed. Curious to hear thoughts on transcript chunking strategies!

## 15-second demo shot list

- 0-3s: Paste 3 YouTube video links into the input fields.
- 3-6s: Type a cross-video question into the question box.
- 6-12s: Answer appears with claim text and citation chips (timestamp links); one chip's evidence panel expands to show the transcript snippet.
- 12-15s: Click a citation chip, cut to YouTube opening at that exact second (the `&t=Ns` link).

The demo clip illustrating this 5-step flow has been programmatically generated and is saved at `docs/demo_clip.mp4`.

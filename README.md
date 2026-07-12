---
title: VidWise
emoji: 🎥
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
thumbnail: https://huggingface.co/spaces/Rishet11/vidwise/resolve/main/docs/og_image.png
---

# VidWise AI

> **Stop scrubbing through videos. Ask a question. Get the exact answer — with the timestamp to prove it.**

VidWise lets you research across multiple YouTube videos at once. Drop in up to six video links, ask anything, and get a cited answer that points to the exact second in the exact video where it was said. No guessing. No hallucinations. Just evidence.

🔗 **[Try the live demo →](https://huggingface.co/spaces/Rishet11/vidwise)**

---

## Why VidWise?

Most AI tools summarize videos and call it a day. VidWise goes further:

- **Every claim is cited.** Each part of the answer links directly to the moment it was spoken — down to the second.
- **Cross-video research.** Ask a single question across six videos simultaneously. Compare, contrast, and synthesize.
- **No hallucinations.** Answers are grounded entirely in the transcript. If the video didn't say it, VidWise won't either.
- **Built for agents too.** An MCP server lets AI agents search and cite these transcripts directly — no extra setup.

---

## How it works

1. **Paste up to 6 YouTube URLs** (or a playlist link, or upload your own subtitles)
2. **Ask any question** — factual, comparative, or analytical
3. **Get a grounded answer** with expandable evidence and clickable timestamp links that jump straight to the moment in the video

That's it.

---

## Features

| Feature | Details |
|---|---|
| 🎬 Multi-video research | Up to 6 videos in one session |
| ⏱ Second-level citations | Every claim links to `youtube.com/watch?v=…&t=Ns` |
| 📂 Flexible input | YouTube URLs, public playlists, or uploaded SRT/VTT/TXT/JSON subtitles |
| 🔍 Smart retrieval | Semantic search with query expansion and local reranking |
| 🤖 MCP Server | Expose VidWise as a tool for AI agents |
| 📊 Evaluation suite | Rigorous benchmarks with human-verified ground truth |
| 🔒 Privacy-first | Questions, answers, and transcripts are never logged |

---

## Run it locally

**Requirements:** Python 3.11, a [Google AI Studio API key](https://aistudio.google.com/), and optionally a [Supadata](https://supadata.ai) key for live transcript fetching.

```bash
git clone https://github.com/Rishet11/VidWise-AI
cd VidWise-AI

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Add your GOOGLE_API_KEY (required) and SUPADATA_API_KEY (optional)

streamlit run app.py
```

> **No Supadata key?** No problem — you can upload subtitle files directly in the app and everything still works.

---

## Deploy with Docker

```bash
docker build -t vidwise .
docker run --rm -p 7860:7860 --env-file .env vidwise
```

Or create a Docker-based Hugging Face Space and add your secrets in the Space settings. The live demo runs entirely on free hardware.

---

## Evaluation

VidWise includes a rigorous evaluation harness covering factual, comparative, and adversarial questions — all verified against real video timestamps. Every result is reproducible from a versioned run artifact.

```bash
python3 benchmarks/run_eval.py --help
```

See [benchmarks/RESULTS.md](benchmarks/RESULTS.md) and [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for the full methodology.

---

## Privacy & limits

- **Nothing sensitive is logged** — no questions, no answers, no transcripts, no API keys
- Free-tier daily limits are in place to protect shared resources
- Private or restricted YouTube videos are skipped gracefully without breaking the rest of your session

---

## Project layout

```
core/          Transcript ingestion, semantic search, grounded answers
benchmarks/    Evaluation set, protocol, and results
docs/          Methodology and implementation notes
mcp_server.py  MCP entry point for AI agent integration
app.py         Streamlit UI
```

---

## Built with

- [Google Gemini Flash](https://deepmind.google/technologies/gemini/) — grounded answer generation
- [FAISS](https://github.com/facebookresearch/faiss) — fast semantic vector search
- [Streamlit](https://streamlit.io) — UI
- [Supadata](https://supadata.ai) — YouTube transcript fetching
- [Hugging Face](https://huggingface.co) — hosting

---

MIT Licensed · Made with care by [Rishet Mehra](https://github.com/Rishet11)

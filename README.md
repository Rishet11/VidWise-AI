---
title: VidWise
emoji: 🎥
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
thumbnail: https://huggingface.co/spaces/Rishet11/vidwise/resolve/main/docs/og_image.png
---

# VidWise

Ask questions across YouTube videos and jump to the transcript moments behind the answer.

[Try the live demo](https://huggingface.co/spaces/Rishet11/vidwise)

VidWise accepts up to six YouTube links, playlist links, or uploaded subtitle files. It
searches their transcripts, drafts an answer from the retrieved sections, and attaches
clickable timestamp citations. Those citations make the answer easier to inspect. They do
not guarantee that every generated claim is correct.

## What is implemented

- Research across as many as six videos in one session
- Direct links to cited transcript timestamps
- YouTube URLs, public playlists, and SRT, VTT, TXT, or JSON subtitle uploads
- Semantic retrieval with optional query expansion and local cross-encoder reranking
- An MCP server for transcript search from compatible agent clients
- A reproducible evaluation runner with saved run artifacts

## How a question is answered

1. VidWise fetches or parses timestamped transcripts and caches them to avoid repeated
   provider calls.
2. It groups transcript segments into overlapping chunks and indexes their embeddings.
3. It retrieves relevant chunks, with query expansion and reranking when configured.
4. Gemini receives the question and retrieved evidence with instructions to cite the supplied
   chunks.
5. The interface renders each citation as a link to the source video and timestamp.

## Run it locally

You need Python 3.11 and a
[Google AI Studio API key](https://aistudio.google.com/). A
[Supadata](https://supadata.ai) key is optional and enables live transcript fetching.

```bash
git clone https://github.com/Rishet11/VidWise-AI
cd VidWise-AI

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Add GOOGLE_API_KEY. Add SUPADATA_API_KEY only if you need live transcript fetching.

streamlit run app.py
```

Without a Supadata key, you can upload subtitle files directly.

### Docker

```bash
docker build -t vidwise .
docker run --rm -p 7860:7860 --env-file .env vidwise
```

## Evaluation status

The repository includes a benchmark set and runner for retrieval, citation, abstention, latency,
and failure-rate measurements. The committed results are a partial baseline covering two of five
retrieval configurations. Human review of claim support and judge agreement is still pending, so
the current numbers should not be presented as final accuracy results.

```bash
python3 benchmarks/run_eval.py --help
```

See [the current results](benchmarks/RESULTS.md) and
[the evaluation method](docs/METHODOLOGY.md) for the protocol and remaining validation work.

## Privacy and operating limits

- Runtime event logs contain counts, latency, and hashed session identifiers. They do not contain
  questions or raw transcripts.
- Transcripts are cached on the host to reduce repeated provider usage.
- The public deployment limits questions per session and per day to protect shared API quotas.
- Private, restricted, deleted, or uncaptioned YouTube videos can be replaced with uploaded
  subtitle files.

## Project layout

```text
core/          Transcript ingestion, retrieval, and cited answer generation
benchmarks/    Evaluation set, runner, and saved results
docs/          Evaluation and implementation notes
mcp_server.py  MCP entry point
app.py         Streamlit interface
```

## Main dependencies

- [Google Gemini](https://deepmind.google/technologies/gemini/) for answer generation
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [Streamlit](https://streamlit.io) for the interface
- [Supadata](https://supadata.ai) for optional YouTube transcript fetching
- [Hugging Face Spaces](https://huggingface.co/spaces) for the public deployment

MIT licensed. Built by [Rishet Mehra](https://github.com/Rishet11).

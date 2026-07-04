from __future__ import annotations

import json
import time
import uuid

import streamlit as st

from config.settings import GOOGLE_API_KEY, MAX_VIDEOS, SUPADATA_MONTHLY_FREE_CREDITS
from core.answer import run_research
from core.discovery import DiscoveryError, playlist_videos, search_topic
from core.embeddings import build_index, get_embedding_model
from core.llm import LLMError, LLMRateLimitError
from core.transcript import TranscriptError, TranscriptQuotaError, extract_youtube_id, get_transcript, parse_uploaded_transcript
from core.types import Transcript
from core.usage import BudgetReached, consume_question, log_event

st.set_page_config(page_title="VidWise — cited video research", page_icon="🎥", layout="wide")


@st.cache_resource(show_spinner="Loading the embedding model (cold starts can take ~2 minutes)…")
def cached_embedding_model():
    return get_embedding_model()


@st.cache_resource(show_spinner="Building a reusable multi-video search index…")
def cached_index(serialized: str):
    transcripts = [Transcript.from_dict(value) for value in json.loads(serialized)]
    return build_index(transcripts, cached_embedding_model())


def init_state():
    defaults = {"session_id": str(uuid.uuid4()), "transcripts": [], "chat": [], "supadata_fetches": 0}
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def add_transcript(transcript: Transcript):
    existing = {item.video_id for item in st.session_state.transcripts}
    if transcript.video_id not in existing:
        st.session_state.transcripts.append(transcript)


def ingest_urls(raw: str):
    entries = [line.strip() for line in raw.replace(",", "\n").splitlines() if line.strip()]
    if len(entries) > MAX_VIDEOS:
        st.error("Use at most 6 videos per research corpus.")
        return
    for entry in entries:
        video_id = extract_youtube_id(entry)
        if not video_id:
            st.warning(f"Skipped invalid YouTube URL: {entry}")
            continue
        try:
            with st.spinner(f"Fetching transcript for {video_id}…"):
                transcript = get_transcript(video_id, allow_local=False)
            if transcript.source == "supadata":
                st.session_state.supadata_fetches += 1
            add_transcript(transcript)
            st.success(f"Added {transcript.title}")
        except TranscriptQuotaError as exc:
            st.warning(str(exc))
        except TranscriptError as exc:
            st.warning(f"Skipped {video_id}: {exc}")


def render_ingest():
    st.subheader("Build a research corpus")
    urls_tab, upload_tab, discover_tab = st.tabs(["URLs / playlist", "Upload transcript", "Topic discovery"])
    with urls_tab:
        raw = st.text_area("Paste 3–6 YouTube URLs, one per line", height=120)
        col1, col2 = st.columns(2)
        if col1.button("Add videos", type="primary", use_container_width=True):
            ingest_urls(raw)
        playlist = col2.text_input("Or public playlist URL")
        if col2.button("Import playlist", use_container_width=True):
            try:
                videos = playlist_videos(playlist)
                ingest_urls("\n".join(video["url"] for video in videos))
            except DiscoveryError as exc:
                st.warning(str(exc))
    with upload_tab:
        upload_url = st.text_input("Matching YouTube URL (recommended for citation links)", key="upload_url")
        files = st.file_uploader("Upload SRT, VTT, JSON, or TXT", type=["srt", "vtt", "json", "txt"], accept_multiple_files=True)
        if st.button("Add uploaded transcripts"):
            if not extract_youtube_id(upload_url):
                st.warning("Provide the matching YouTube URL so every citation can link to the correct second.")
                return
            for file in files:
                try:
                    add_transcript(parse_uploaded_transcript(file.name, file.getvalue(), upload_url))
                except TranscriptError as exc:
                    st.warning(f"{file.name}: {exc}")
    with discover_tab:
        topic = st.text_input("Research topic")
        if st.button("Find up to 6 public videos"):
            try:
                results = search_topic(topic)
                st.session_state.discovery = results
            except DiscoveryError as exc:
                st.warning(str(exc))
        for result in st.session_state.get("discovery", []):
            st.markdown(f"- [{result['title']}]({result['url']})")
        if st.session_state.get("discovery") and st.button("Ingest discovered videos"):
            ingest_urls("\n".join(item["url"] for item in st.session_state.discovery))


def render_chat():
    transcripts = st.session_state.transcripts
    if not transcripts:
        st.info("Add transcripts above, or use the packaged demo corpus when available.")
        return
    st.subheader(f"Research across {len(transcripts)} video{'s' if len(transcripts) != 1 else ''}")
    for transcript in transcripts:
        st.caption(f"• {transcript.title} — {transcript.source}, {len(transcript.segments)} timestamped segments")
    serialized = json.dumps([transcript.to_dict() for transcript in transcripts], sort_keys=True)
    index = cached_index(serialized)
    for message in st.session_state.chat:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            for citation in message.get("citations", []):
                with st.expander(f"Evidence: {citation['title']} @ {citation['timestamp']}"):
                    st.write(citation["text"])
    question = st.chat_input("Ask a question across these videos")
    if not question:
        return
    if not GOOGLE_API_KEY:
        st.warning("GOOGLE_API_KEY is required to answer questions. Transcript ingestion and uploads remain available.")
        return
    st.session_state.chat.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    started = time.perf_counter()
    try:
        result = run_research(index, question, history=st.session_state.chat[:-1])
        counts = consume_question(st.session_state.session_id)
        elapsed = time.perf_counter() - started
        result.trace["latency_seconds"] = round(elapsed, 3)
        citations = [
            {"title": c.title, "timestamp": f"{int(c.start)//60:02d}:{int(c.start)%60:02d}", "text": c.text, "url": c.citation_url}
            for c in result.citations
        ]
        message = {"role": "assistant", "content": result.answer, "citations": citations, "trace": result.trace}
        st.session_state.chat.append(message)
        log_event("question_answered", st.session_state.session_id, latency_seconds=elapsed, llm_calls=result.trace["llm_calls"], video_count=len(transcripts), citation_count=len(citations))
        with st.chat_message("assistant"):
            st.markdown(result.answer)
            for citation in citations:
                with st.expander(f"Evidence: {citation['title']} @ {citation['timestamp']}"):
                    st.write(citation["text"])
            with st.expander("Retrieval trace"):
                st.json({**result.trace, "daily_question": counts["session"]})
    except BudgetReached as exc:
        st.warning(str(exc))
    except LLMRateLimitError as exc:
        st.warning(str(exc))
    except LLMError as exc:
        log_event("llm_error", st.session_state.session_id, detail=str(exc))
        st.error("The AI service is temporarily unavailable or rate-limited. Please retry in a minute.")


def render_eval():
    st.header("Published evaluation set")
    st.markdown("Results are generated by `python benchmarks/run_eval.py`. No result is shown until a run artifact exists.")
    try:
        st.markdown(open("benchmarks/RESULTS.md", encoding="utf-8").read())
    except OSError:
        st.info("Evaluation results have not been generated in this deployment.")


init_state()
st.title("VidWise")
st.caption("Multi-video YouTube research with second-level, expandable evidence")
st.info("On a sleeping free-tier Space, the first load can take about 2 minutes while models wake up.")
research_tab, eval_tab, about_tab = st.tabs(["Research", "Evaluation set", "Privacy & limits"])
with research_tab:
    render_ingest()
    render_chat()
with eval_tab:
    render_eval()
with about_tab:
    st.markdown(
        f"""Transcripts are cached to avoid repeated provider usage. Runtime logs contain counts, latency, and hashed session IDs—not questions or raw transcripts.

Supadata's documented free allowance is {SUPADATA_MONTHLY_FREE_CREDITS} credits/month. This session has used {st.session_state.supadata_fetches} fresh API fetches; provider-wide remaining quota is not exposed by its transcript endpoint. Uploading subtitles always remains available.

The free deployment is single-worker and may lose ephemeral logs or caches after a restart. Each session is limited to 15 questions/day; a shared 500-question/day safety cap protects the Gemini free tier."""
    )

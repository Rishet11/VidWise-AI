from __future__ import annotations

import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st

from config.settings import GOOGLE_API_KEY, MAX_VIDEOS, SUPADATA_MONTHLY_FREE_CREDITS
from core.answer import run_research
from core.discovery import DiscoveryError, playlist_videos, search_topic
from core.embeddings import build_index, get_embedding_model
from core.llm import LLMError, LLMRateLimitError
from core.transcript import TranscriptError, TranscriptQuotaError, extract_youtube_id, get_transcript, parse_uploaded_transcript
from core.types import Transcript
from core.usage import BudgetReached, consume_question, log_event

st.set_page_config(page_title="VidWise, cited video research", page_icon="🎥", layout="wide")


@st.cache_resource(show_spinner="First load takes a minute or two while the app warms up…")
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

    video_ids: dict[str, str] = {}
    for entry in entries:
        video_id = extract_youtube_id(entry)
        if not video_id:
            st.warning(f"Skipped invalid YouTube URL: {entry}")
            continue
        video_ids[video_id] = entry
    if not video_ids:
        return

    total = len(video_ids)
    done = 0
    failed = 0
    with st.status(f"Fetching {total} transcript{'s' if total != 1 else ''}…", expanded=False) as status:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {executor.submit(get_transcript, video_id, allow_local=False): video_id for video_id in video_ids}
            for future in as_completed(futures):
                video_id = futures[future]
                done += 1
                status.update(label=f"Fetched {done}/{total} transcripts")
                try:
                    transcript = future.result()
                    if transcript.source == "supadata":
                        st.session_state.supadata_fetches += 1
                    add_transcript(transcript)
                    st.success(f"Added {transcript.title}")
                except TranscriptQuotaError as exc:
                    failed += 1
                    st.warning(str(exc))
                except TranscriptError as exc:
                    failed += 1
                    st.warning(f"Skipped {video_id}: {exc}")
        if failed:
            status.update(
                label=f"Added {done - failed} of {total} videos ({failed} skipped, open for details)",
                state="error",
                expanded=True,
            )
        else:
            status.update(label=f"Fetched {done}/{total} transcripts", state="complete")


def render_ingest():
    st.subheader("Build a research corpus")
    urls_tab, upload_tab, discover_tab = st.tabs(["URLs / playlist", "Upload transcript", "Topic discovery"])
    with urls_tab:
        st.caption("Paste video links or a playlist link. VidWise fetches captions for each video automatically.")
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
        st.caption("No captions available on YouTube? Upload a transcript file and link it to the video yourself.")
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
        st.caption("Don't have specific videos in mind? Search a topic and VidWise finds public videos to research.")
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


def render_sources(citations: list[dict], trace: dict | None = None):
    if not citations:
        return
    with st.expander(f"Sources ({len(citations)})"):
        for index, citation in enumerate(citations, start=1):
            seconds = int(citation.get("start", 0))
            link = citation.get("url") or f"https://youtu.be/{citation.get('video_id', '')}?t={seconds}"
            st.markdown(f"**[{index}]** [{citation['timestamp']}]({link}) {citation['title']}")
            st.caption(f"“{citation['text']}”")
        if trace:
            st.caption(f"How this was found: {trace}")


def render_chat():
    transcripts = st.session_state.transcripts
    if not transcripts:
        st.info("Add transcripts above, or use the packaged demo corpus when available.")
        return
    st.subheader(f"Research across {len(transcripts)} video{'s' if len(transcripts) != 1 else ''}")
    for transcript in transcripts:
        st.caption(f"- {transcript.title} ({transcript.source}, {len(transcript.segments)} timestamped segments)")
    serialized = json.dumps([transcript.to_dict() for transcript in transcripts], sort_keys=True)
    index = cached_index(serialized)
    if not st.session_state.chat:
        st.caption('Try asking something like: "Where do these videos disagree on X?"')
    for message in st.session_state.chat:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            render_sources(message.get("citations", []), message.get("trace"))
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
            {
                "title": c.title,
                "timestamp": f"{int(c.start)//60:02d}:{int(c.start)%60:02d}",
                "text": c.text,
                "start": int(c.start),
                "video_id": c.video_id,
                "url": f"https://youtu.be/{c.video_id}?t={int(c.start)}",
            }
            for c in result.citations
        ]
        trace_summary = {**result.trace, "daily_question": counts["session"]}
        message = {"role": "assistant", "content": result.answer, "citations": citations, "trace": trace_summary}
        st.session_state.chat.append(message)
        log_event("question_answered", st.session_state.session_id, latency_seconds=elapsed, llm_calls=result.trace["llm_calls"], video_count=len(transcripts), citation_count=len(citations))
        with st.chat_message("assistant"):
            st.markdown(result.answer)
            render_sources(citations, trace_summary)
    except BudgetReached as exc:
        st.warning(str(exc))
    except LLMRateLimitError as exc:
        st.warning(str(exc))
    except LLMError as exc:
        log_event("llm_error", st.session_state.session_id, detail=str(exc))
        st.error("The AI service is temporarily unavailable or rate-limited. Please retry in a minute.")


init_state()
st.markdown("<style>div.block-container{padding-top:2rem;}</style>", unsafe_allow_html=True)
st.title("VidWise")
st.caption("Ask questions across YouTube videos and get answers with second-accurate citations.")
research_tab, about_tab = st.tabs(["Research", "Privacy & limits"])
with research_tab:
    render_ingest()
    render_chat()
with about_tab:
    st.markdown(
        """Transcripts are cached to avoid repeated provider usage. Runtime logs contain counts, latency, and hashed session IDs, not questions or raw transcripts.

The free deployment is single-worker and may lose ephemeral logs or caches after a restart. Each session is limited to 15 questions per day, and a shared 500-question daily cap protects the Gemini free tier."""
    )
    st.caption(
        f"Supadata's documented free allowance is {SUPADATA_MONTHLY_FREE_CREDITS} credits/month. "
        f"This session has used {st.session_state.supadata_fetches} fresh API fetches. Uploading subtitles always remains available."
    )

"""VidWise MCP tools backed by the same cached research core."""
from mcp.server.fastmcp import FastMCP

from core.answer import run_research
from core.discovery import search_topic
from core.embeddings import build_index
from core.transcript import extract_youtube_id, get_transcript

mcp = FastMCP("vidwise")
_corpora = {}


@mcp.tool()
def ingest_videos(urls: list[str]) -> dict:
    """Ingest one to six public YouTube URLs into a temporary corpus."""
    if not 1 <= len(urls) <= 6:
        raise ValueError("Provide between one and six URLs")
    ids = [extract_youtube_id(url) for url in urls]
    if any(item is None for item in ids):
        raise ValueError("Every item must be a valid YouTube URL")
    transcripts = [get_transcript(item) for item in ids]
    corpus_id = "-".join(ids)
    _corpora[corpus_id] = build_index(transcripts)
    return {"corpus_id": corpus_id, "videos": ids}


@mcp.tool()
def search_corpus(corpus_id: str, question: str) -> dict:
    """Answer from an ingested corpus with timestamped citations."""
    if corpus_id not in _corpora:
        raise ValueError("Unknown corpus; call ingest_videos in this server process first")
    result = run_research(_corpora[corpus_id], question)
    return {"answer": result.answer, "citations": [chunk.citation_url for chunk in result.citations], "trace": result.trace}


@mcp.tool()
def research_topic(topic: str, question: str) -> dict:
    """Discover up to six YouTube videos, ingest them, and answer a question."""
    videos = search_topic(topic)
    corpus = ingest_videos([video["url"] for video in videos])
    return {"videos": videos, **search_corpus(corpus["corpus_id"], question)}


if __name__ == "__main__":
    mcp.run()

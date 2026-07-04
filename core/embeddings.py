"""Timestamp-aware chunking and direct FAISS search."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from config.settings import EMBEDDING_MODEL_ID
from core.types import Chunk, Transcript


def chunk_transcript(transcript: Transcript, target_chars: int = 700, overlap_segments: int = 2) -> list[Chunk]:
    chunks: list[Chunk] = []
    start_index = 0
    while start_index < len(transcript.segments):
        selected = []
        length = 0
        cursor = start_index
        while cursor < len(transcript.segments) and (length < target_chars or not selected):
            selected.append(transcript.segments[cursor])
            length += len(transcript.segments[cursor].text) + 1
            cursor += 1
        text = " ".join(segment.text for segment in selected)
        chunks.append(
            Chunk(
                chunk_id=f"{transcript.video_id}:{len(chunks)}",
                video_id=transcript.video_id,
                title=transcript.title,
                text=text,
                start=selected[0].start,
                end=selected[-1].end,
                url=transcript.url,
            )
        )
        if cursor >= len(transcript.segments):
            break
        start_index = max(start_index + 1, cursor - overlap_segments)
    return chunks


def get_embedding_model(model_id: str = EMBEDDING_MODEL_ID):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_id)


@dataclass
class CorpusIndex:
    chunks: list[Chunk]
    vectors: np.ndarray
    index: object
    model: object

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        query_vector = self.model.encode([query], normalize_embeddings=True).astype("float32")
        scores, ids = self.index.search(query_vector, min(k, len(self.chunks)))
        return [(self.chunks[int(i)], float(score)) for i, score in zip(ids[0], scores[0]) if i >= 0]


def build_index(transcripts: Iterable[Transcript], model=None) -> CorpusIndex:
    import faiss

    chunks = [chunk for transcript in transcripts for chunk in chunk_transcript(transcript)]
    if not chunks:
        raise ValueError("Cannot index an empty transcript corpus")
    model = model or get_embedding_model()
    vectors = model.encode([chunk.text for chunk in chunks], normalize_embeddings=True).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return CorpusIndex(chunks, vectors, index, model)

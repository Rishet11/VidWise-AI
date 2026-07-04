"""Plain-Python multi-query retrieval and listwise LLM reranking."""
from __future__ import annotations

from collections import OrderedDict

from config.settings import FINAL_K, RERANK_THRESHOLD, RETRIEVAL_K
from core.embeddings import CorpusIndex
from core.llm import QuestionLLM
from core.types import Chunk

QUERY_SCHEMA = {
    "type": "object",
    "properties": {"queries": {"type": "array", "items": {"type": "string"}, "maxItems": 2}},
    "required": ["queries"],
}
RERANK_SCHEMA = {
    "type": "object",
    "properties": {"chunk_ids": {"type": "array", "items": {"type": "string"}}},
    "required": ["chunk_ids"],
}


def expand_queries(question: str, llm: QuestionLLM) -> list[str]:
    prompt = (
        "Create at most two concise semantic-search variants for this research question. "
        "Preserve named entities and intent. Return JSON only.\nQuestion: " + question
    )
    result = llm.generate_json(prompt, QUERY_SCHEMA)
    variants = [str(item).strip() for item in result.get("queries", []) if str(item).strip()][:2]
    return list(dict.fromkeys([question, *variants]))


def dense_retrieve(index: CorpusIndex, queries: list[str], k: int = RETRIEVAL_K) -> list[Chunk]:
    best: OrderedDict[str, tuple[Chunk, float]] = OrderedDict()
    for query in queries:
        for chunk, score in index.search(query, k):
            current = best.get(chunk.chunk_id)
            if current is None or score > current[1]:
                best[chunk.chunk_id] = (chunk, score)
    return [item[0] for item in sorted(best.values(), key=lambda value: value[1], reverse=True)[:k]]


def rerank(question: str, chunks: list[Chunk], llm: QuestionLLM, top_n: int = FINAL_K) -> list[Chunk]:
    if len(chunks) < RERANK_THRESHOLD:
        return chunks[:top_n]
    candidates = "\n\n".join(f"[{c.chunk_id}] {c.text}" for c in chunks)
    prompt = (
        f"Rank transcript chunks by direct relevance and evidentiary support for: {question}\n"
        f"Return only the best {top_n} chunk IDs as JSON. Do not invent IDs.\n\n{candidates}"
    )
    result = llm.generate_json(prompt, RERANK_SCHEMA)
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    ranked = [by_id[item] for item in result.get("chunk_ids", []) if item in by_id]
    seen = {chunk.chunk_id for chunk in ranked}
    ranked.extend(chunk for chunk in chunks if chunk.chunk_id not in seen)
    return ranked[:top_n]


def retrieve(index: CorpusIndex, question: str, llm: QuestionLLM, *, use_multi_query: bool = True, use_rerank: bool = True) -> tuple[list[Chunk], dict]:
    queries = expand_queries(question, llm) if use_multi_query else [question]
    candidates = dense_retrieve(index, queries)
    selected = rerank(question, candidates, llm) if use_rerank else candidates[:FINAL_K]
    return selected, {
        "queries": queries,
        "candidate_count": len(candidates),
        "rerank_skipped": not use_rerank or len(candidates) < RERANK_THRESHOLD,
    }


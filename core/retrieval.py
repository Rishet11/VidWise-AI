"""Plain-Python multi-query retrieval and listwise LLM reranking."""
from __future__ import annotations

from collections import OrderedDict

from config.settings import FINAL_K, RERANK_THRESHOLD, RERANKER_MODEL_ID, RETRIEVAL_K
from core.embeddings import CorpusIndex
from core.llm import QuestionLLM
from core.types import Chunk

_cross_encoder = None

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


def dense_retrieve(index: CorpusIndex, queries: list[str], k: int = RETRIEVAL_K, use_prefix: bool = True) -> list[Chunk]:
    best: OrderedDict[str, tuple[Chunk, float]] = OrderedDict()
    for query in queries:
        for chunk, score in index.search(query, k, use_prefix=use_prefix):
            current = best.get(chunk.chunk_id)
            if current is None or score > current[1]:
                best[chunk.chunk_id] = (chunk, score)
    return [item[0] for item in sorted(best.values(), key=lambda value: value[1], reverse=True)[:k]]


def generate_hyde(question: str, llm: QuestionLLM) -> str:
    prompt = (
        "Write a short (2-3 sentence) factual passage that would plausibly answer this question, "
        "phrased like an excerpt from a video transcript. No preamble, just the passage.\n"
        f"Question: {question}"
    )
    try:
        text = llm.generate(prompt).strip()
        return text or question
    except Exception:
        return question


def cross_encoder_rerank(question: str, chunks: list[Chunk], top_n: int = FINAL_K) -> list[Chunk]:
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder

        _cross_encoder = CrossEncoder(RERANKER_MODEL_ID)
    scores = _cross_encoder.predict([(question, c.text) for c in chunks])
    ranked = [chunk for chunk, _ in sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)]
    return ranked[:top_n]


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


def retrieve(
    index: CorpusIndex,
    question: str,
    llm: QuestionLLM,
    *,
    use_multi_query: bool = True,
    use_rerank: bool = True,
    use_hyde: bool = False,
    rerank_mode: str = "cross_encoder",
) -> tuple[list[Chunk], dict]:
    hyde_doc = None
    if use_hyde:
        # HyDE takes precedence over multi-query: we retrieve against the
        # hypothetical answer passage (embedded without the bge query prefix).
        hyde_doc = generate_hyde(question, llm)
        queries = [hyde_doc]
        use_prefix = False
    else:
        queries = expand_queries(question, llm) if use_multi_query else [question]
        use_prefix = True
    candidates = dense_retrieve(index, queries, use_prefix=use_prefix)
    if use_rerank:
        selected = (
            cross_encoder_rerank(question, candidates)
            if rerank_mode == "cross_encoder"
            else rerank(question, candidates, llm)
        )
    else:
        selected = candidates[:FINAL_K]
    trace = {
        "queries": queries,
        "candidate_count": len(candidates),
        "rerank_skipped": not use_rerank or (rerank_mode != "cross_encoder" and len(candidates) < RERANK_THRESHOLD),
        "use_hyde": use_hyde,
        "rerank_mode": rerank_mode,
    }
    if hyde_doc is not None:
        trace["hyde_doc"] = hyde_doc
    return selected, trace


"""Grounded answer generation with machine-verifiable claim citations."""
from __future__ import annotations

from core.llm import QuestionLLM
from core.types import AnswerResult, Chunk

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "citation_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["text", "citation_ids"],
            },
        },
        "insufficient": {"type": "boolean"},
    },
    "required": ["claims", "insufficient"],
}


def format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def answer_question(question: str, chunks: list[Chunk], llm: QuestionLLM, history: list[dict] | None = None) -> AnswerResult:
    context = "\n\n".join(f"[{c.chunk_id}] {c.title} @ {format_timestamp(c.start)}\n{c.text}" for c in chunks)
    prior = "\n".join(f"{m['role']}: {m['content']}" for m in (history or [])[-4:])
    prompt = f"""You answer research questions using only supplied transcript evidence.
Write atomic, concise claims. Every factual claim must cite one or more chunk IDs that directly support it.
The first claim must directly answer the question. Subsequent claims support or elaborate on it in logical
order. Claims must read as connected prose when concatenated, each self-contained, with no near-duplicates
or filler. If evidence is insufficient, set insufficient=true and return no claims. Never use outside knowledge.

Previous conversation:
{prior or '(none)'}

Question: {question}

Evidence:
{context}
"""
    payload = llm.generate_json(prompt, ANSWER_SCHEMA)
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    rendered: list[str] = []
    cited: list[Chunk] = []
    numbers: dict[str, int] = {}
    if payload.get("insufficient") or not payload.get("claims"):
        return AnswerResult("The selected videos do not provide enough evidence to answer that.", [], {})
    for claim in payload["claims"]:
        claim_citations = [by_id[cid] for cid in claim.get("citation_ids", []) if cid in by_id]
        if not claim_citations:
            continue  # unsupported LLM output is never presented as a claim
        markers = ""
        for chunk in claim_citations:
            if chunk.chunk_id not in numbers:
                numbers[chunk.chunk_id] = len(numbers) + 1
                cited.append(chunk)
            markers += f"[[{numbers[chunk.chunk_id]}]]({chunk.citation_url})"
        rendered.append(f"{str(claim['text']).strip()} {markers}")
    if not rendered:
        return AnswerResult("The selected videos do not provide enough supported evidence to answer that.", [], {})
    return AnswerResult("\n\n".join(rendered), cited, {})


def run_research(index, question: str, *, history=None, use_multi_query=True, use_rerank=True, llm=None) -> AnswerResult:
    from core.retrieval import retrieve

    llm = llm or QuestionLLM()
    chunks, trace = retrieve(index, question, llm, use_multi_query=use_multi_query, use_rerank=use_rerank)
    result = answer_question(question, chunks, llm, history)
    result.trace = {**trace, "llm_calls": llm.calls, "retrieved_chunk_ids": [c.chunk_id for c in chunks]}
    return result


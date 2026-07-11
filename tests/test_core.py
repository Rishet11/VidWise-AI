import numpy as np

from core.answer import answer_question, run_research
from core.embeddings import CorpusIndex, chunk_transcript
from core.retrieval import rerank
from core.types import Chunk, Segment, Transcript


class FakeModel:
    def encode(self, values, normalize_embeddings=True):
        return np.ones((len(values), 2), dtype="float32")


class FakeIndex:
    def search(self, vector, k):
        return np.array([[1.0] * k]), np.array([list(range(k))])


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def generate_json(self, prompt, schema):
        self.calls += 1
        if "semantic-search variants" in prompt:
            return {"queries": ["variant one", "variant two"]}
        if "Rank transcript" in prompt:
            return {"chunk_ids": ["v:1", "v:0"]}
        return {"claims": [{"text": "Supported fact.", "citation_ids": ["v:0"]}], "insufficient": False}


def chunks(count):
    return [Chunk(f"v:{i}", "v", "Video", f"evidence {i}", i * 10, i * 10 + 5, "https://youtube.com/watch?v=abcdefghijk") for i in range(count)]


def test_chunk_metadata_and_url():
    transcript = Transcript("abcdefghijk", "Title", "https://youtube.com/watch?v=abcdefghijk", [Segment("a" * 400, 5, 2), Segment("b" * 400, 7, 3)], "upload")
    chunk = chunk_transcript(transcript)[0]
    assert chunk.start == 5
    assert chunk.end == 10
    assert chunk.citation_url.endswith("&t=5s")


def test_rerank_skips_below_eight_without_call():
    llm = FakeLLM()
    assert len(rerank("q", chunks(7), llm)) == 6
    assert llm.calls == 0


def test_question_uses_at_most_three_calls():
    llm = FakeLLM()
    corpus = chunks(10)
    index = CorpusIndex(corpus, np.ones((10, 2), dtype="float32"), FakeIndex(), FakeModel())
    # Exercise the worst-case LLM-reranker path so this stays a pure unit test
    # (the cross-encoder default would load a real model from the HF hub).
    result = run_research(index, "question", llm=llm, rerank_mode="llm")
    assert result.trace["llm_calls"] == 3
    assert "Supported fact" in result.answer


def test_unsupported_claim_is_suppressed():
    class Unsupported(FakeLLM):
        def generate_json(self, prompt, schema):
            self.calls += 1
            return {"claims": [{"text": "Hallucination", "citation_ids": ["missing"]}], "insufficient": False}

    result = answer_question("q", chunks(1), Unsupported())
    assert "Hallucination" not in result.answer


def test_answer_can_cite_two_videos():
    evidence = [
        Chunk("a:0", "a", "First", "alpha", 10, 15, "https://youtube.com/watch?v=aaaaaaaaaaa"),
        Chunk("b:0", "b", "Second", "beta", 20, 25, "https://youtube.com/watch?v=bbbbbbbbbbb"),
    ]

    class CrossVideo(FakeLLM):
        def generate_json(self, prompt, schema):
            self.calls += 1
            return {"claims": [{"text": "Both support this.", "citation_ids": ["a:0", "b:0"]}], "insufficient": False}

    result = answer_question("compare", evidence, CrossVideo())
    assert {citation.video_id for citation in result.citations} == {"a", "b"}

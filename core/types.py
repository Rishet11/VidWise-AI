from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Segment:
    text: str
    start: float
    duration: float

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass
class Transcript:
    video_id: str
    title: str
    url: str
    segments: list[Segment]
    source: str
    language: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "segments": [asdict(s) for s in self.segments]}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Transcript":
        value = dict(value)
        value["segments"] = [Segment(**s) for s in value["segments"]]
        return cls(**value)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    video_id: str
    title: str
    text: str
    start: float
    end: float
    url: str

    @property
    def citation_url(self) -> str:
        separator = "&" if "?" in self.url else "?"
        return f"{self.url}{separator}t={int(self.start)}s"


@dataclass
class AnswerResult:
    answer: str
    citations: list[Chunk]
    trace: dict[str, Any] = field(default_factory=dict)


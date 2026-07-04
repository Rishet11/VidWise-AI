"""Inspect-AI task for the human-verified VidWise evaluation set."""
from __future__ import annotations

import json
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate


@task
def vidwise_evaluation():
    path = Path(__file__).resolve().parents[1] / "dataset.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    pending = [row["id"] for row in rows if row.get("label_status") != "human_verified"]
    if pending:
        raise ValueError("Human verification required before Inspect evaluation: " + ", ".join(pending))
    samples = [
        Sample(
            id=row["id"],
            input=row["question"],
            target=json.dumps(row["expected_claims"]),
            metadata={"video_ids": row["video_ids"], "relevant_segments": row["relevant_segments"]},
        )
        for row in rows
    ]
    return Task(dataset=samples, solver=[generate()])


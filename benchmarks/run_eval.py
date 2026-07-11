#!/usr/bin/env python3
"""Reproducible VidWise evaluation runner; refuses unverified ground truth."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.answer import run_research
from core.embeddings import build_index
from core.llm import QuestionLLM
from core.transcript import get_transcript


def wilson(successes: int, total: int, z: float = 1.96) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return [max(0, centre - margin), min(1, centre + margin)]


def load_dataset(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    pending = [row["id"] for row in rows if row.get("label_status") != "human_verified"]
    if pending:
        raise ValueError(
            f"Refusing to evaluate unverified ground truth. Rishet must review labels for: {', '.join(pending)}"
        )
    return rows


def overlaps(chunk, label: dict, tolerance: float = 5) -> bool:
    return chunk.video_id == label["video_id"] and chunk.start <= label["end"] + tolerance and chunk.end >= label["start"] - tolerance


def run(args) -> dict:
    rows = load_dataset(Path(args.dataset))
    video_ids = sorted({video for row in rows for video in row["video_ids"]})
    transcripts = [get_transcript(video, allow_local=args.allow_local) for video in video_ids]
    index = build_index(transcripts)
    chunks_by_id = {chunk.chunk_id: chunk for chunk in index.chunks}
    configurations = {
        "naive": (False, False, False),
        "multi_query": (True, False, False),
        "rerank": (False, True, False),
        "combined": (True, True, False),
        "hyde": (False, True, True),
    }
    selected = [args.config] if args.config != "all" else list(configurations)
    results = {"generated_at": datetime.now(timezone.utc).isoformat(), "model": args.model, "runs": args.runs, "configs": {}}
    for config_name in selected:
        multi_query, rerank, hyde = configurations[config_name]
        repetitions = []
        for repetition in range(args.runs):
            tasks = []
            for row in rows:
                started = time.perf_counter()
                answer = run_research(
                    index,
                    row["question"],
                    use_multi_query=multi_query,
                    use_rerank=rerank,
                    use_hyde=hyde,
                    rerank_mode="cross_encoder",
                    llm=QuestionLLM(model=args.model),
                )
                time.sleep(13)  # Respect free tier limit (15 RPM), up to 3 calls/question
                retrieved = [chunks_by_id[item] for item in answer.trace["retrieved_chunk_ids"]]
                recall_hit = any(overlaps(chunk, label) for chunk in retrieved for label in row["relevant_segments"])
                citation_hits = sum(any(overlaps(chunk, label) for label in row["relevant_segments"]) for chunk in answer.citations)
                negative_ok = not answer.citations if row["negative"] else None
                task_data = {
                    "id": row["id"], "latency_seconds": time.perf_counter() - started,
                    "recall_hit": recall_hit, "citation_hits": citation_hits,
                    "citation_total": len(answer.citations), "negative_ok": negative_ok,
                    "llm_calls": answer.trace["llm_calls"], "answer": answer.answer,
                }
                tasks.append(task_data)
                
                # --- Save Partial Progress ---
                current_flat = [t for rep in repetitions for t in rep] + tasks
                if current_flat:
                    c_total = sum(t["citation_total"] for t in current_flat)
                    c_hits = sum(t["citation_hits"] for t in current_flat)
                    r_hits = sum(t["recall_hit"] for t in current_flat)
                    negs = [t for t in current_flat if t["negative_ok"] is not None]
                    
                    results["configs"][config_name] = {
                        "chunk_recall_at_k": r_hits / len(current_flat), 
                        "chunk_recall_ci95": wilson(r_hits, len(current_flat)),
                        "citation_accuracy": c_hits / c_total if c_total else 0,
                        "citation_accuracy_ci95": wilson(c_hits, c_total),
                        "negative_success": sum(t["negative_ok"] for t in negs) / len(negs) if negs else 0.0,
                        "negative_success_ci95": wilson(sum(t["negative_ok"] for t in negs), len(negs)) if negs else [0.0, 0.0],
                        "latency_mean_seconds": statistics.mean(t["latency_seconds"] for t in current_flat),
                        "latency_stdev_seconds": statistics.stdev(t["latency_seconds"] for t in current_flat) if len(current_flat) > 1 else 0,
                        "failure_rate": sum(not t["recall_hit"] for t in current_flat) / len(current_flat),
                        "max_llm_calls": max(t["llm_calls"] for t in current_flat), 
                        "tasks": repetitions + [tasks],
                        "claim_support": "requires completed human review of generated claims",
                    }
                    
                    out_path = Path(args.output)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
                    Path(args.results_md).write_text(render_results(results), encoding="utf-8")
                
            repetitions.append(tasks)
    return results


def render_results(report: dict) -> str:
    lines = [
        "# Evaluation-set results",
        "",
        "## Failure analysis first",
        "",
        "The run artifact identifies retrieval and abstention failures below. Rishet must inspect each failed item and replace `unclassified` with a named mechanism before publication; the renderer never invents a cause.",
        "",
    ]
    failures = []
    for config, metrics in report["configs"].items():
        for run_index, tasks in enumerate(metrics["tasks"], start=1):
            for item in tasks:
                if not item["recall_hit"] or item.get("negative_ok") is False:
                    failures.append(f"- `{config}` run {run_index}, `{item['id']}` — mechanism: **unclassified (human review required)**")
    lines.extend(failures or ["- No retrieval/abstention failures detected; claim-level human review is still required."])
    lines.extend([
        "",
        "## Metrics",
        "",
        "| Configuration | Chunk recall@k (95% CI) | Citation accuracy (95% CI) | Negative success (95% CI) | Mean latency ± SD | Failure rate | Max calls |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for config, value in report["configs"].items():
        recall_ci = value["chunk_recall_ci95"]
        citation_ci = value["citation_accuracy_ci95"]
        negative_ci = value["negative_success_ci95"]
        lines.append(
            f"| {config} | {value['chunk_recall_at_k']:.1%} ({recall_ci[0]:.1%}–{recall_ci[1]:.1%}) "
            f"| {value['citation_accuracy']:.1%} ({citation_ci[0]:.1%}–{citation_ci[1]:.1%}) "
            f"| {value['negative_success']:.1%} ({negative_ci[0]:.1%}–{negative_ci[1]:.1%}) "
            f"| {value['latency_mean_seconds']:.2f}s ± {value['latency_stdev_seconds']:.2f}s "
            f"| {value['failure_rate']:.1%} | {value['max_llm_calls']} |"
        )
    lines.extend([
        "",
        "## Validity",
        "",
        f"Generated from `{report['runs']}` runs with `{report['model']}`. Claim-support and judge-vs-human correlation remain publication gates and must be entered from human review; retrieval overlap alone does not establish answer correctness.",
        "",
        "Reproduce with:",
        "",
        "```bash",
        f"python3 benchmarks/run_eval.py --model {report['model']} --config all --runs {report['runs']}",
        "```",
    ])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(ROOT / "benchmarks/dataset.jsonl"))
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--config", choices=["all", "naive", "multi_query", "rerank", "combined"], default="all")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--allow-local", action="store_true")
    parser.add_argument("--output", default=str(ROOT / "benchmarks/runs/latest.json"))
    parser.add_argument("--results-md", default=str(ROOT / "benchmarks/RESULTS.md"))
    args = parser.parse_args()
    if args.runs < 3:
        parser.error("--runs must be at least 3 for variance reporting")
    try:
        result = run(args)
    except ValueError as exc:
        parser.error(str(exc))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    Path(args.results_md).write_text(render_results(result), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

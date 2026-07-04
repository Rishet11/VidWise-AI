import json

import pytest

from benchmarks.run_eval import load_dataset, render_results, wilson


def test_wilson_bounds():
    low, high = wilson(8, 10)
    assert 0 < low < 0.8 < high < 1


def test_unverified_labels_fail_closed(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps({"id": "q", "label_status": "pending"}) + "\n")
    with pytest.raises(ValueError, match="Refusing"):
        load_dataset(path)


def test_result_renderer_includes_metrics_and_failures():
    report = {
        "runs": 3,
        "model": "gemini-2.5-flash",
        "configs": {
            "naive": {
                "tasks": [[{"id": "q1", "recall_hit": False, "negative_ok": None}]],
                "chunk_recall_at_k": 0.5, "chunk_recall_ci95": [0.2, 0.8],
                "citation_accuracy": 0.75, "citation_accuracy_ci95": [0.4, 0.9],
                "negative_success": 1.0, "negative_success_ci95": [0.5, 1.0],
                "latency_mean_seconds": 2.0, "latency_stdev_seconds": 0.2,
                "failure_rate": 0.5, "max_llm_calls": 1,
            }
        },
    }
    rendered = render_results(report)
    assert "Failure analysis first" in rendered
    assert "unclassified" in rendered
    assert "50.0%" in rendered

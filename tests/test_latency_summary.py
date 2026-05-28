from __future__ import annotations

import json

from tools.latency_summary import load_entries, summarize


def test_latency_summary_stats(tmp_path):
    path = tmp_path / "pi0_latency.jsonl"
    rows = [
        {"episode_id": "a", "step": 0, "preprocess_ms": 100.0, "infer_ms": 900.0, "success": True},
        {"episode_id": "a", "step": 10, "preprocess_ms": 1.0, "infer_ms": 100.0, "success": True},
        {"episode_id": "b", "step": 10, "preprocess_ms": 3.0, "infer_ms": 300.0, "success": True},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    result = summarize(load_entries([path]), drop_first_step=True)

    assert result["entries"] == 2
    assert result["episodes"] == 2
    assert result["infer_ms"]["mean_ms"] == 200.0
    assert result["infer_ms"]["p50_ms"] == 200.0
    assert result["total_ms"]["min_ms"] == 101.0

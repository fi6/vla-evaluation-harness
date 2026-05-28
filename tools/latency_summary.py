#!/usr/bin/env python3
"""Summarize vla-eval model-server latency JSONL files."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    idx = (len(ordered) - 1) * pct
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min_ms": round(min(values), 3),
        "mean_ms": round(statistics.fmean(values), 3),
        "p50_ms": round(_percentile(values, 0.50), 3),
        "p90_ms": round(_percentile(values, 0.90), 3),
        "p95_ms": round(_percentile(values, 0.95), 3),
        "p99_ms": round(_percentile(values, 0.99), 3),
        "max_ms": round(max(values), 3),
    }


def load_entries(paths: list[Path]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in paths:
        with path.open() as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
                entry["_source"] = str(path)
                entries.append(entry)
    return entries


def summarize(entries: list[dict[str, Any]], *, drop_first_step: bool = False) -> dict[str, Any]:
    if drop_first_step:
        entries = [entry for entry in entries if int(entry.get("step", 0)) != 0]

    preprocess = [float(entry.get("preprocess_ms", 0.0)) for entry in entries]
    infer = [float(entry.get("infer_ms", 0.0)) for entry in entries]
    total = [p + i for p, i in zip(preprocess, infer)]
    success_counts = Counter(str(entry.get("success", "unknown")) for entry in entries)
    source_counts = Counter(str(entry.get("_source", "")) for entry in entries)

    return {
        "entries": len(entries),
        "episodes": len({entry.get("episode_id") for entry in entries}),
        "drop_first_step": drop_first_step,
        "success_counts": dict(sorted(success_counts.items())),
        "sources": dict(sorted(source_counts.items())),
        "preprocess_ms": _stats(preprocess),
        "infer_ms": _stats(infer),
        "total_ms": _stats(total),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--drop-first-step", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = summarize(load_entries(args.paths), drop_first_step=args.drop_first_step)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
        print(f"wrote {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()

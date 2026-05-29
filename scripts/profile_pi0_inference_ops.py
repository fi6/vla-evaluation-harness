# /// script
# requires-python = "~=3.11"
# dependencies = [
#     "vla-eval",
#     "openpi",
#     "numpy>=1.24",
#     "pytest",
#     "chex",
# ]
#
# [tool.uv.sources]
# vla-eval = { path = "..", editable = true }
# openpi = { git = "https://github.com/Physical-Intelligence/openpi.git", rev = "981483dca0fd9acba698fea00aa6e52d56a66c58" }
#
# [tool.uv]
# exclude-newer = "2026-02-24T00:00:00Z"
# ///
"""Profile pi0/pi0.5 inference HLO operations.

This script is intentionally standalone: it does not modify the model server or
benchmark path.  It loads an OpenPI policy, runs a few inference calls, asks XLA
to dump optimized HLO, and summarizes matrix/vector operations by shape.

Important limitation: JAX/XLA fuses and schedules kernels after Python tracing,
so exact per-HLO-instruction runtime is not generally observable from Python.
The script reports exact end-to-end inference wall time and estimates per-op time
by distributing measured inference time in proportion to estimated FLOPs.  The
shape/count/FLOP accounting comes from optimized HLO text.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import statistics
import tempfile
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# XLA flags must be set before importing jax/openpi.


@dataclass
class OpRecord:
    category: str
    opcode: str
    dtype: str
    output_shape: str
    lhs_shape: str = ""
    rhs_shape: str = ""
    attrs: str = ""
    count: int = 0
    flops_per_call: int = 0
    total_flops: int = 0
    estimated_time_ms: float = 0.0


_SHAPE_RE = re.compile(r"(?P<dtype>[a-zA-Z0-9_]+)\[(?P<shape>[^\]]*)\]")
_DEF_RE = re.compile(
    r"(?:ROOT\s+)?%?(?P<name>[A-Za-z0-9_.-]+)\s*=\s*"
    r"(?P<shape>(?:[a-zA-Z0-9_]+\[[^\]]*\](?:\{[^}]*\})?|\([^)]*\)))\s+"
    r"(?P<opcode>[a-zA-Z0-9_-]+)\("
)
_OPERAND_RE = re.compile(r"%([A-Za-z0-9_.-]+)")
_DIMS_RE = re.compile(r"(lhs|rhs)_(contracting|batch)_dims=\{([^}]*)\}")
_WINDOW_RE = re.compile(r"window=\{size=([^}\s]+)")
_OP_NAME_RE = re.compile(r'op_name="([^"]+)"')
_CUSTOM_CALL_TARGET_RE = re.compile(r'custom_call_target="([^"]+)"')

ELEMENTWISE_OPS = {
    "abs",
    "add",
    "and",
    "atan2",
    "broadcast",
    "ceil",
    "clamp",
    "compare",
    "convert",
    "copy",
    "cosine",
    "divide",
    "exponential",
    "floor",
    "log",
    "logistic",
    "maximum",
    "minimum",
    "multiply",
    "negate",
    "not",
    "or",
    "power",
    "real",
    "reshape",
    "rsqrt",
    "select",
    "shift-left",
    "shift-right-logical",
    "sine",
    "sqrt",
    "subtract",
    "tanh",
    "transpose",
    "xor",
}
REDUCE_OPS = {"reduce", "reduce-window", "sort", "topk"}
DATA_MOVEMENT_OPS = {
    "all-gather",
    "bitcast",
    "broadcast",
    "concatenate",
    "copy",
    "dynamic-slice",
    "dynamic-update-slice",
    "gather",
    "iota",
    "pad",
    "reshape",
    "reverse",
    "slice",
    "transpose",
}


def parse_shape(shape: str) -> tuple[str, tuple[int, ...]]:
    if shape.startswith("("):
        inner = shape[1:-1]
        first = split_top_level_csv(inner)[0] if inner else ""
        return parse_shape(first)
    match = _SHAPE_RE.search(shape)
    if not match:
        return "unknown", ()
    dims: list[int] = []
    for part in match.group("shape").split(","):
        part = part.strip()
        if not part:
            continue
        # HLO dynamic dimensions can be written with adornments; keep only the integer prefix.
        dim_match = re.match(r"(\d+)", part)
        if dim_match:
            dims.append(int(dim_match.group(1)))
    return match.group("dtype"), tuple(dims)


def shape_text(dims: tuple[int, ...]) -> str:
    return "x".join(str(x) for x in dims) if dims else "scalar"


def numel(dims: tuple[int, ...]) -> int:
    return math.prod(dims) if dims else 1


def parse_dim_attr(line: str) -> dict[str, tuple[int, ...]]:
    out: dict[str, tuple[int, ...]] = {}
    for side, kind, raw in _DIMS_RE.findall(line):
        key = f"{side}_{kind}"
        vals = tuple(int(x) for x in raw.split(",") if x.strip())
        out[key] = vals
    return out


def split_top_level_csv(text: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    for i, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
    parts.append(text[start:].strip())
    return [p for p in parts if p]


def operand_names(line: str, opcode: str = "") -> list[str]:
    marker = f"{opcode}(" if opcode else "("
    open_idx = line.find(marker)
    if open_idx >= 0 and opcode:
        open_idx += len(opcode)
    close_idx = line.find(")", open_idx + 1)
    if open_idx < 0 or close_idx < 0:
        return []
    raw_args = split_top_level_csv(line[open_idx + 1 : close_idx])
    names: list[str] = []
    for arg in raw_args:
        match = _OPERAND_RE.search(arg)
        if match:
            names.append(match.group(1))
    return names


def dot_flops(line: str, lhs_dims: tuple[int, ...], rhs_dims: tuple[int, ...], out_dims: tuple[int, ...]) -> int:
    attrs = parse_dim_attr(line)
    lhs_contracting = attrs.get("lhs_contracting", ())
    k = 1
    if lhs_contracting:
        for dim in lhs_contracting:
            if dim < len(lhs_dims):
                k *= lhs_dims[dim]
    elif len(lhs_dims) >= 2 and len(rhs_dims) >= 2:
        if lhs_dims[-1] == rhs_dims[0]:
            k = lhs_dims[-1]
        elif lhs_dims[0] == rhs_dims[-1]:
            k = lhs_dims[0]
        elif lhs_dims[-1] == rhs_dims[-1]:
            k = lhs_dims[-1]
        else:
            k = min(lhs_dims[-1], rhs_dims[0])
    return 2 * numel(out_dims) * max(k, 1)


def convolution_flops(line: str, lhs_dims: tuple[int, ...], rhs_dims: tuple[int, ...], out_dims: tuple[int, ...]) -> int:
    # Good enough for comparing operations: output elements times kernel/input channels.
    kernel_work = max(numel(rhs_dims), 1)
    if lhs_dims and rhs_dims:
        # Most conv kernels encode output channels in one rhs dim; avoid counting it twice when possible.
        kernel_work = max(numel(rhs_dims) // max(out_dims[-1] if out_dims else 1, 1), 1)
    return 2 * numel(out_dims) * kernel_work


def classify_opcode(opcode: str, line: str = "") -> str:
    if opcode == "custom-call":
        target = _CUSTOM_CALL_TARGET_RE.search(line)
        if target and "gemm" in target.group(1).lower():
            return "matrix"
        if target and "conv" in target.group(1).lower():
            return "convolution"
    if opcode in {"dot", "dot-general"}:
        return "matrix"
    if opcode == "convolution":
        return "convolution"
    if opcode in REDUCE_OPS:
        return "reduction"
    if opcode in DATA_MOVEMENT_OPS:
        return "data_movement"
    if opcode in ELEMENTWISE_OPS:
        return "elementwise"
    return "other"


def estimate_flops(opcode: str, line: str, output_dims: tuple[int, ...], lhs_dims: tuple[int, ...], rhs_dims: tuple[int, ...]) -> int:
    if opcode in {"dot", "dot-general", "custom-call"} and "gemm" in line.lower():
        return dot_flops(line, lhs_dims, rhs_dims, output_dims)
    if opcode == "convolution":
        return convolution_flops(line, lhs_dims, rhs_dims, output_dims)
    if opcode in REDUCE_OPS:
        return max(numel(output_dims), 1)
    if opcode in DATA_MOVEMENT_OPS:
        return 0
    if opcode in ELEMENTWISE_OPS:
        return max(numel(output_dims), 1)
    return 0


def parse_hlo_files(dump_dir: Path) -> list[OpRecord]:
    files = sorted(dump_dir.glob("*.cpu_after_optimizations.txt")) + sorted(
        dump_dir.glob("*.gpu_after_optimizations.txt")
    )
    if not files:
        files = sorted(dump_dir.glob("*after_optimizations.txt"))
    if not files:
        files = sorted(dump_dir.glob("*.txt"))

    grouped: dict[tuple[str, str, str, str, str, str, str], OpRecord] = {}
    for file in files:
        shape_by_name: dict[str, tuple[str, tuple[int, ...]]] = {}
        lines = file.read_text(errors="ignore").splitlines()
        for line in lines:
            match = _DEF_RE.search(line)
            if not match:
                continue
            dtype, dims = parse_shape(match.group("shape"))
            shape_by_name[match.group("name")] = (dtype, dims)

        for line in lines:
            match = _DEF_RE.search(line)
            if not match:
                continue
            opcode = match.group("opcode")
            category = classify_opcode(opcode, line)
            if category == "other":
                continue
            dtype, output_dims = parse_shape(match.group("shape"))
            operands = operand_names(line, opcode)
            lhs_dtype, lhs_dims = shape_by_name.get(operands[0], ("", ())) if operands else ("", ())
            rhs_dtype, rhs_dims = shape_by_name.get(operands[1], ("", ())) if len(operands) > 1 else ("", ())
            lhs = f"{lhs_dtype}[{shape_text(lhs_dims)}]" if lhs_dtype else ""
            rhs = f"{rhs_dtype}[{shape_text(rhs_dims)}]" if rhs_dtype else ""
            out = f"{dtype}[{shape_text(output_dims)}]"
            op_name_match = _OP_NAME_RE.search(line)
            op_name = op_name_match.group(1) if op_name_match else ""
            attrs = f"op_name={op_name}" if op_name else ""
            if category == "matrix":
                dim_attrs = parse_dim_attr(line)
                dim_text = ";".join(f"{k}={v}" for k, v in sorted(dim_attrs.items()))
                target = _CUSTOM_CALL_TARGET_RE.search(line)
                parts = [x for x in [dim_text, f"target={target.group(1)}" if target else "", f"op_name={op_name}" if op_name else ""] if x]
                attrs = ";".join(parts)
            elif category == "convolution":
                match_window = _WINDOW_RE.search(line)
                parts = [f"window={match_window.group(1)}" if match_window else "", f"op_name={op_name}" if op_name else ""]
                attrs = ";".join(x for x in parts if x)
            flops = estimate_flops(opcode, line, output_dims, lhs_dims, rhs_dims)
            key = (category, opcode, dtype, out, lhs, rhs, attrs)
            record = grouped.get(key)
            if record is None:
                record = OpRecord(
                    category=category,
                    opcode=opcode,
                    dtype=dtype,
                    output_shape=out,
                    lhs_shape=lhs,
                    rhs_shape=rhs,
                    attrs=attrs,
                    count=0,
                    flops_per_call=flops,
                )
                grouped[key] = record
            record.count += 1
            record.total_flops += flops

    records = list(grouped.values())
    records.sort(key=lambda r: (r.category != "matrix", -r.total_flops, r.opcode, r.output_shape, r.lhs_shape, r.rhs_shape))
    return records


def make_dummy_obs(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np

    image = np.zeros((args.image_size, args.image_size, 3), dtype=np.uint8)
    wrist = np.zeros_like(image)
    state = np.zeros(args.state_dim, dtype=np.float64)
    obs: dict[str, Any] = {
        args.image_key: image,
        "prompt": args.prompt,
    }
    if args.wrist_image_key.lower() not in {"none", "null", ""}:
        obs[args.wrist_image_key] = wrist
    if args.state_key.lower() not in {"none", "null", ""}:
        obs[args.state_key] = state
    return obs


def load_policy(args: argparse.Namespace) -> Any:
    from openpi.policies import policy_config
    from openpi.training import config as openpi_config

    config = openpi_config.get_config(args.config_name)
    checkpoint = args.checkpoint or f"gs://openpi-assets/checkpoints/{args.config_name}"
    return policy_config.create_trained_policy(config, checkpoint)


def sync_device(value: Any) -> None:
    import jax

    try:
        jax.block_until_ready(value)
    except Exception:
        if isinstance(value, dict):
            for item in value.values():
                sync_device(item)


def run_inference(policy: Any, obs: dict[str, Any]) -> Any:
    result = policy.infer(obs)
    sync_device(result)
    return result


def write_outputs(records: list[OpRecord], summary: dict[str, Any], out_json: Path, out_csv: Path | None) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "operations": [asdict(r) for r in records]}
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True))
    if out_csv is None:
        return
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()) if records else list(OpRecord.__annotations__))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def print_summary(records: list[OpRecord], summary: dict[str, Any], limit: int) -> None:
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("\nTop operations by total_flops:")
    for record in sorted(records, key=lambda r: r.total_flops, reverse=True)[:limit]:
        lhs_rhs = f"{record.lhs_shape} x {record.rhs_shape}" if record.rhs_shape else record.lhs_shape
        print(
            f"{record.category:13s} {record.opcode:14s} count={record.count:5d} "
            f"out={record.output_shape:24s} in={lhs_rhs:45s} "
            f"flops={record.total_flops:.3e} est_ms={record.estimated_time_ms:.3f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detailed pi0/pi0.5 inference operation profiler")
    parser.add_argument("--config_name", default="pi05_libero")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--image_key", default="observation/image")
    parser.add_argument("--wrist_image_key", default="observation/wrist_image")
    parser.add_argument("--state_key", default="observation/state")
    parser.add_argument("--state_dim", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--prompt", default="pick up the object")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--dump_dir", default=None)
    parser.add_argument("--keep_dump", action="store_true")
    parser.add_argument("--parse_only_dump", action="store_true", help="Only parse an existing --dump_dir; skip model loading/inference")
    parser.add_argument("--output_json", default="results/pi0_inference_ops_profile.json")
    parser.add_argument("--output_csv", default="results/pi0_inference_ops_profile.csv")
    parser.add_argument("--top", type=int, default=30)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    created_temp_dump = args.dump_dir is None
    dump_dir = Path(args.dump_dir) if args.dump_dir else Path(tempfile.mkdtemp(prefix="pi0_hlo_dump_"))
    dump_dir.mkdir(parents=True, exist_ok=True)
    existing_xla_flags = os.environ.get("XLA_FLAGS", "")
    dump_flags = f"--xla_dump_to={dump_dir} --xla_dump_hlo_as_text"
    os.environ["XLA_FLAGS"] = f"{existing_xla_flags} {dump_flags}".strip()

    import jax

    print(f"JAX backend: {jax.default_backend()}")
    print(f"HLO dump dir: {dump_dir}")
    times_ms: list[float] = []
    if not args.parse_only_dump:
        policy = load_policy(args)
        obs = make_dummy_obs(args)

        for _ in range(args.warmup):
            run_inference(policy, obs)

        for _ in range(args.iters):
            start = time.perf_counter()
            run_inference(policy, obs)
            times_ms.append((time.perf_counter() - start) * 1000)

    records = parse_hlo_files(dump_dir)
    total_flops = sum(r.total_flops for r in records)
    mean_ms = statistics.mean(times_ms) if times_ms else 0.0
    median_ms = statistics.median(times_ms) if times_ms else 0.0
    for record in records:
        record.estimated_time_ms = (mean_ms * record.total_flops / total_flops) if total_flops else 0.0

    category_counts = defaultdict(int)
    category_flops = defaultdict(int)
    category_time = defaultdict(float)
    for record in records:
        category_counts[record.category] += record.count
        category_flops[record.category] += record.total_flops
        category_time[record.category] += record.estimated_time_ms

    summary = {
        "config_name": args.config_name,
        "checkpoint": args.checkpoint or f"gs://openpi-assets/checkpoints/{args.config_name}",
        "backend": jax.default_backend(),
        "image_size": args.image_size,
        "state_dim": args.state_dim,
        "warmup": args.warmup,
        "iters": args.iters,
        "inference_ms": {
            "mean": mean_ms,
            "median": median_ms,
            "min": min(times_ms) if times_ms else 0.0,
            "max": max(times_ms) if times_ms else 0.0,
            "samples": times_ms,
        },
        "operation_groups": len(records),
        "operation_instances": sum(r.count for r in records),
        "total_estimated_flops": total_flops,
        "category_counts": dict(sorted(category_counts.items())),
        "category_flops": dict(sorted(category_flops.items())),
        "category_estimated_time_ms": dict(sorted(category_time.items())),
        "time_note": "Per-operation time is estimated by proportional FLOP allocation from measured end-to-end policy.infer time; shape/count/FLOP data is parsed from optimized XLA HLO.",
    }

    out_json = Path(args.output_json)
    out_csv = Path(args.output_csv) if args.output_csv else None
    write_outputs(records, summary, out_json, out_csv)
    print_summary(records, summary, args.top)
    print(f"\nWrote JSON: {out_json}")
    if out_csv:
        print(f"Wrote CSV:  {out_csv}")
    should_keep_dump = args.keep_dump or args.parse_only_dump or not created_temp_dump
    if should_keep_dump:
        print(f"Kept HLO dump dir: {dump_dir}")
    else:
        shutil.rmtree(dump_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

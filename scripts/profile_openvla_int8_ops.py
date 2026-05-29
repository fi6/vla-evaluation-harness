# /// script
# requires-python = "~=3.11"
# dependencies = [
#     "vla-eval",
#     "torch>=2.2",
#     "transformers==4.40.1",
#     "timm==0.9.10",
#     "tokenizers==0.19.1",
#     "pillow>=9.0",
#     "numpy>=1.24",
#     "accelerate",
#     "bitsandbytes",
# ]
#
# [tool.uv.sources]
# vla-eval = { path = "..", editable = true }
#
# [tool.uv]
# exclude-newer = "2026-02-24T00:00:00Z"
# ///
"""Profile OpenVLA int8 smoke inference matrix operations.

This script loads OpenVLA through HuggingFace Transformers with
``BitsAndBytesConfig(load_in_8bit=True)`` and runs a tiny synthetic smoke
inference. It uses PyTorch profiler to summarize matrix-like ops/kernels and
their CUDA time.

This is intended for timing/profiling only; it does not modify the model server
or benchmark code paths.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


MATRIX_KEYWORDS = (
    "matmul",
    "mm",
    "bmm",
    "addmm",
    "linear",
    "gemm",
    "cublas",
    "cutlass",
    "bitsandbytes",
    "bnb",
    "igemmlt",
    "int8",
)


@dataclass
class ProfileRecord:
    name: str
    count: int
    cpu_time_total_us: float
    cuda_time_total_us: float
    self_cpu_time_total_us: float
    self_cuda_time_total_us: float
    input_shapes: str
    is_matrix: bool


def is_matrix_event(name: str) -> bool:
    lower = name.lower()
    return any(keyword in lower for keyword in MATRIX_KEYWORDS)


def preprocess_image(image_size: int, jpeg_roundtrip: bool, center_crop: bool) -> Any:
    from PIL import Image as PILImage

    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    pil = PILImage.fromarray(image).convert("RGB")
    if jpeg_roundtrip:
        buf = io.BytesIO()
        pil.save(buf, format="JPEG")
        buf.seek(0)
        pil = PILImage.open(buf).convert("RGB")
    pil = pil.resize((224, 224), resample=PILImage.Resampling.LANCZOS)
    if center_crop:
        width, height = pil.size
        crop_h = int(height * (0.9**0.5))
        crop_w = int(width * (0.9**0.5))
        top = (height - crop_h) // 2
        left = (width - crop_w) // 2
        pil = pil.crop((left, top, left + crop_w, top + crop_h))
        pil = pil.resize((width, height), resample=PILImage.Resampling.LANCZOS)
    return pil


def load_model(args: argparse.Namespace) -> tuple[Any, Any]:
    import torch
    import transformers.modeling_utils as modeling_utils
    from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    device_map: str | None = None if args.device_map.lower() in {"none", "null", ""} else args.device_map

    original_to = modeling_utils.PreTrainedModel.to

    def _quantized_safe_to(self: Any, *to_args: Any, **to_kwargs: Any) -> Any:
        if getattr(self, "is_loaded_in_8bit", False) or getattr(self, "is_loaded_in_4bit", False) or getattr(
            self, "is_quantized", False
        ):
            return self
        return original_to(self, *to_args, **to_kwargs)

    modeling_utils.PreTrainedModel.to = _quantized_safe_to
    try:
        model = AutoModelForVision2Seq.from_pretrained(
            args.model_path,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
    finally:
        modeling_utils.PreTrainedModel.to = original_to
    model.eval()
    move_cpu_tensors_to_cuda(model)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return model, processor


def get_model_device(model: Any) -> str:
    import torch

    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return "cuda" if torch.cuda.is_available() else "cpu"




def move_cpu_tensors_to_cuda(model: Any) -> None:
    import torch

    if not torch.cuda.is_available():
        return
    for _, param in model.named_parameters(recurse=True):
        if getattr(param, "device", None) is not None and param.device.type == "cpu":
            try:
                param.data = param.data.to("cuda")
            except Exception:
                pass
    for _, buf in model.named_buffers(recurse=True):
        if getattr(buf, "device", None) is not None and buf.device.type == "cpu":
            try:
                buf.data = buf.data.to("cuda")
            except Exception:
                pass
    torch.cuda.synchronize()


def make_inputs(args: argparse.Namespace, model: Any, processor: Any) -> Any:
    import torch

    image = preprocess_image(args.image_size, args.jpeg_roundtrip, args.center_crop)
    prompt = f"In: What action should the robot take to {args.task}?\nOut:"
    inputs = processor(prompt, image)
    if torch.cuda.is_available():
        inputs = inputs.to(get_model_device(model), dtype=torch.bfloat16)
    return inputs


def run_predict(model: Any, inputs: Any, unnorm_key: str | None) -> Any:
    kwargs: dict[str, Any] = {"do_sample": False}
    if unnorm_key:
        kwargs["unnorm_key"] = unnorm_key
    return model.predict_action(**inputs, **kwargs)


def summarize_profiler(prof: Any) -> list[ProfileRecord]:
    records: list[ProfileRecord] = []
    for evt in prof.key_averages(group_by_input_shape=True):
        shapes = getattr(evt, "input_shapes", None)
        records.append(
            ProfileRecord(
                name=evt.key,
                count=int(evt.count),
                cpu_time_total_us=float(evt.cpu_time_total),
                cuda_time_total_us=float(getattr(evt, "cuda_time_total", 0.0)),
                self_cpu_time_total_us=float(evt.self_cpu_time_total),
                self_cuda_time_total_us=float(getattr(evt, "self_cuda_time_total", 0.0)),
                input_shapes=json.dumps(shapes, default=str) if shapes is not None else "",
                is_matrix=is_matrix_event(evt.key),
            )
        )
    records.sort(key=lambda r: (not r.is_matrix, -r.cuda_time_total_us, -r.cpu_time_total_us, r.name))
    return records


def write_outputs(records: list[ProfileRecord], summary: dict[str, Any], output_json: Path, output_csv: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "operations": [asdict(record) for record in records]}
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ProfileRecord.__annotations__))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile OpenVLA int8 smoke inference matrix ops")
    parser.add_argument("--model_path", default="openvla/openvla-7b-finetuned-libero-10")
    parser.add_argument("--unnorm_key", default="libero_10")
    parser.add_argument("--device_map", default="none")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--task", default="put the yellow and white mug in the microwave and close it")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--jpeg_roundtrip", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--center_crop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output_json", default="results/openvla_int8_ops_profile.json")
    parser.add_argument("--output_csv", default="results/openvla_int8_ops_profile.csv")
    parser.add_argument("--top", type=int, default=30)
    return parser


def main() -> None:
    import torch
    from torch.profiler import ProfilerActivity, profile

    args = build_parser().parse_args()
    model, processor = load_model(args)
    inputs = make_inputs(args, model, processor)

    for _ in range(args.warmup):
        run_predict(model, inputs, args.unnorm_key)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    wall_times_ms: list[float] = []
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, record_shapes=True) as prof:
        for _ in range(args.iters):
            start = time.perf_counter()
            run_predict(model, inputs, args.unnorm_key)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            wall_times_ms.append((time.perf_counter() - start) * 1000)

    records = summarize_profiler(prof)
    matrix_records = [record for record in records if record.is_matrix]
    matrix_count = sum(record.count for record in matrix_records)
    matrix_cpu_ms = sum(record.cpu_time_total_us for record in matrix_records) / 1000.0
    matrix_self_cpu_ms = sum(record.self_cpu_time_total_us for record in matrix_records) / 1000.0
    matrix_cuda_ms = sum(record.cuda_time_total_us for record in matrix_records) / 1000.0
    matrix_self_cuda_ms = sum(record.self_cuda_time_total_us for record in matrix_records) / 1000.0
    total_cpu_ms = sum(record.cpu_time_total_us for record in records) / 1000.0
    total_self_cpu_ms = sum(record.self_cpu_time_total_us for record in records) / 1000.0
    total_cuda_ms = sum(record.cuda_time_total_us for record in records) / 1000.0
    total_self_cuda_ms = sum(record.self_cuda_time_total_us for record in records) / 1000.0
    wall_mean_ms = sum(wall_times_ms) / len(wall_times_ms) if wall_times_ms else 0.0
    summary = {
        "model_path": args.model_path,
        "unnorm_key": args.unnorm_key,
        "quantization": "bitsandbytes_load_in_8bit",
        "device_map": args.device_map,
        "torch_cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "warmup": args.warmup,
        "iters": args.iters,
        "wall_ms": {
            "samples": wall_times_ms,
            "mean": wall_mean_ms,
            "min": min(wall_times_ms) if wall_times_ms else 0.0,
            "max": max(wall_times_ms) if wall_times_ms else 0.0,
        },
        "profiler_total_cpu_time_ms": total_cpu_ms,
        "profiler_total_self_cpu_time_ms": total_self_cpu_ms,
        "profiler_total_cuda_time_ms": total_cuda_ms,
        "profiler_total_self_cuda_time_ms": total_self_cuda_ms,
        "matrix_event_groups": len(matrix_records),
        "matrix_event_count": matrix_count,
        "matrix_cpu_time_total_ms": matrix_cpu_ms,
        "matrix_self_cpu_time_total_ms": matrix_self_cpu_ms,
        "matrix_cpu_time_share_of_profiler_total": matrix_cpu_ms / total_cpu_ms if total_cpu_ms else 0.0,
        "matrix_cuda_time_total_ms": matrix_cuda_ms,
        "matrix_self_cuda_time_total_ms": matrix_self_cuda_ms,
        "matrix_cuda_time_share_of_profiler_total": matrix_cuda_ms / total_cuda_ms if total_cuda_ms else 0.0,
        "note": "PyTorch profiler CPU/CUDA times are aggregated op times and may double-count nested ops. Wall time is synchronized end-to-end predict_action time. Some bitsandbytes builds report int8 op CPU time but zero CUDA aggregate time in torch.profiler; use wall time or Nsight for kernel-exact timing.",
    }
    write_outputs(records, summary, Path(args.output_json), Path(args.output_csv))

    print(json.dumps(summary, indent=2, sort_keys=True))
    print("\nTop matrix events by CUDA time:")
    for record in matrix_records[: args.top]:
        print(
            f"{record.count:5d} {record.cuda_time_total_us / 1000.0:10.3f} ms "
            f"self={record.self_cuda_time_total_us / 1000.0:10.3f} ms {record.name}"
        )
    print(f"\nWrote JSON: {args.output_json}")
    print(f"Wrote CSV:  {args.output_csv}")


if __name__ == "__main__":
    main()

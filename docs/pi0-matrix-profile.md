# pi0.5 Matrix Operation Profile

This note explains how to run `scripts/profile_pi0_inference_ops.py` on a fresh
checkout and how to interpret the output.

The script is standalone. It does not change the pi0 model server, benchmark
configs, or latency logging path. It loads an OpenPI pi0/pi0.5 policy, runs
dummy inference, asks XLA to dump optimized HLO, then summarizes the operations
that XLA emits.

## What It Reports

The JSON and CSV outputs include:

- End-to-end `policy.infer()` latency samples, mean, median, min, and max.
- Operation categories: `matrix`, `convolution`, `elementwise`, `reduction`,
  and `data_movement`.
- Matrix operation shape records, including `dot`, Triton GEMM fusion, and
  `custom-call("__cublas$gemm")` patterns.
- Input shapes, output shapes, dtype, repeated count, estimated FLOPs, and
  estimated time per operation group.

The important fields in the JSON are:

- `summary.inference_ms.mean`: measured end-to-end inference wall time.
- `summary.category_counts.matrix`: number of matrix operation instances.
- `summary.category_estimated_time_ms.matrix`: estimated time spent in matrix
  operations.
- `summary.category_flops.matrix`: estimated matrix FLOPs.
- `operations[]`: grouped operation table.

## Run pi0.5 LIBERO Profile

From the repo root:

```bash
HF_ENDPOINT=https://hf-mirror.com \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run scripts/profile_pi0_inference_ops.py \
  --config_name pi05_libero \
  --state_dim 8 \
  --image_size 224 \
  --warmup 1 \
  --iters 3 \
  --output_json results/pi05_inference_ops_profile.json \
  --output_csv results/pi05_inference_ops_profile.csv \
  --top 30 \
  --keep_dump
```

The script prints a summary and writes:

- `results/pi05_inference_ops_profile.json`
- `results/pi05_inference_ops_profile.csv`

With `--keep_dump`, it also prints the temporary XLA HLO dump directory. Keep
that path if you want to re-parse the same dump without loading the model again.

## Re-parse an Existing HLO Dump

```bash
uv run scripts/profile_pi0_inference_ops.py \
  --dump_dir /tmp/pi0_hlo_dump_xxxxxxxx \
  --parse_only_dump \
  --output_json results/pi05_inference_ops_profile.json \
  --output_csv results/pi05_inference_ops_profile.csv \
  --top 30
```

`--parse_only_dump` does not run inference, so latency samples are empty and the
estimated per-op times are zero. Use it to debug operation shapes/counts/FLOPs.

## Reading Matrix Time

To compare total inference time and matrix time:

```bash
python - <<'PY'
import json
from pathlib import Path

data = json.loads(Path("results/pi05_inference_ops_profile.json").read_text())
s = data["summary"]
total = s["inference_ms"]["mean"]
matrix = s["category_estimated_time_ms"].get("matrix", 0.0)
print(f"inference mean: {total:.3f} ms")
print(f"matrix estimate: {matrix:.3f} ms")
print(f"difference: {total - matrix:.3f} ms")
print(f"matrix share: {matrix / total * 100:.2f}%")
PY
```

## Caveats

This is an XLA HLO/FLOP profiler, not a CUDA kernel timeline.

Shape/count/FLOP data is parsed from optimized HLO, so it is useful for
answering "which matrix shapes are emitted and how many times?". Per-operation
time is estimated by distributing measured end-to-end inference time by FLOPs.
That estimate is reasonable for high-level matrix-compute share, but it is not
the same as Nsight or XLA profiler kernel timing.

For exact kernel-level timing, use an XLA profiler trace or Nsight Systems /
Nsight Compute and match kernels back to HLO/custom-call names.

## INT8 Notes

The default pi0.5 path here is OpenPI/JAX. Casting arrays to int8 is not enough
to prove that inference runs int8 GEMM kernels; XLA may insert conversions or
lower back to bf16/fp32 kernels. True int8 timing requires a backend that emits
int8 kernels, such as a verified FlashRT/TensorRT path.

The existing FlashRT compose in this repo is for Jetson Orin (`sm87`) and is not
the same as this JAX profiler path.

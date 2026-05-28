# Jetson pi0 PyTorch Smoke

## Tooling Layout

Runtime entrypoints:

- `pi0_flashrt_server_entry.py`: FlashRT-backed pi0.5 server. This is required
  for the FlashRT BF16/INT8 Orin compose stack because FlashRT is not part of
  the upstream harness model-server registry.
- `pi0_server_entry.py`: compatibility wrapper for manually running the regular
  `Pi0ModelServer` from a bind-mounted checkout. The reproducible
  `Dockerfile.pi0-orin` runs `src/vla_eval/model_servers/pi0.py` directly.

Reusable latency tools:

- `latency_smoke_client.py`: generic synthetic client for any running vla-eval
  model server. It flushes server-side `results/*_latency.jsonl` logs and can
  also write client-side round-trip JSON.
- `latency_summary.py`: summarizes one or more server-side latency JSONL files
  into profile-style JSON.
- `pi0_synthetic_latency_client.py`: backward-compatible wrapper around
  `latency_smoke_client.py`.

Diagnostics:

- `pi0_profile.py`: pi0-specific internal timing breakdown for OpenPI PyTorch.
  It bypasses the websocket server and is useful for finding where pi0 spends
  time, but it is not a normal smoke/eval path.
- `pi0_torchao_int4_smoke.py`: exploratory torchao INT4 probe. It is kept as a
  record of the failed/partial INT4 path, not as a supported pi0 runtime.

This repo expects the Jetson OpenPI image and converted checkpoint to exist:

```text
image: jetson-openpi-torch:pi0-orin
checkpoint: /root/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch
host checkpoint: /home/yida/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch
```

Start the fp16/bf16 pi0 server:

```bash
docker run -d --name pi0-libero-server --runtime nvidia --network host \
  -v "$PWD":/workspace:ro \
  -v "$HOME/.cache/openpi":/root/.cache/openpi \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -v "$PWD/results":/workspace/results \
  -e PYTHONPATH=/workspace/src:/opt/openpi-embedded/src:/opt/openpi-embedded/packages/openpi-client/src \
  jetson-openpi-torch:pi0-orin \
  python3 /workspace/tools/pi0_server_entry.py \
    --config_name pi05_libero \
    --backend pytorch \
    --checkpoint /root/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch \
    --image_key observation/image \
    --wrist_image_key observation/wrist_image \
    --state_key observation/state \
    --state_dim 8 \
    --image_resolution 224 \
    --chunk_size 10 \
    --port 8000
```

Wait for health:

```bash
curl -sf http://127.0.0.1:8000/config
```

Generate a latency file without LIBERO:

```bash
docker run --rm --network host \
  -v "$PWD":/workspace:ro \
  -e PYTHONPATH=/workspace/src \
  jetson-openpi-torch:pi0-orin \
  python3 /workspace/tools/latency_smoke_client.py --steps 11 \
    --output /workspace/results/pi0_synthetic_roundtrip.json
```

Summarize the server-side latency JSONL as JSON:

```bash
python3 tools/latency_summary.py results/pi0_*_latency.jsonl \
  --drop-first-step \
  --output results/pi0_latency_summary.json
```

Run LIBERO smoke against the same server:

```bash
docker run --rm --runtime nvidia --network host \
  -v "$PWD/results":/workspace/results \
  -v "$PWD/configs":/workspace/configs:ro \
  ghcr.io/allenai/vla-evaluation-harness/libero:local \
  run --config /workspace/configs/benchmarks/libero/smoke_test.yaml \
  --server-url ws://127.0.0.1:8000 -y
```

## Current Orin Latency Notes

The OpenPI PyTorch eager path is functional but slow on this machine:

```text
policy.infer: ~2.56 s
embed_prefix: ~351 ms
prefix_prefill: ~1124 ms
denoise loop: ~1001 ms total, ~100 ms per denoise step
```

Current hardware reported by PyTorch:

```text
Orin sm_87
total_memory ~= 30 GB
multi_processor_count = 8
host nvpmodel: MODE_30W
```

The often-cited FlashRT Orin number is not FP8. FlashRT documents Orin
SM87 as having no native FP8 tensor cores; its Orin fast path is INT8 W8A8
with CUTLASS/FA2 kernels and CUDA Graph capture.

Measured in the temporary Docker image `jetson-openpi-flashrt:sm87-test`
after building FlashRT SM87 kernels:

```text
FlashRT BF16 cache_frames=1: p50 485.0 ms
FlashRT INT8 cache_frames=1: p50 380.5 ms
FlashRT INT8 cache_frames=2: p50 227.8 ms, min 75.9 ms
```

Those are slower than FlashRT's 64 GB / 16 SM AGX Orin result because this
machine exposes 8 SMs and is running in 30 W mode.

torchao INT4 weight-only was also probed in Docker. It is not currently a
usable pi0.5 smoke path here: torchao 0.13 can run a small INT4 Linear on
Orin, but full OpenPI pi0.5 quantization fails in scale/zero packing, and
some SigLIP FFN dimensions are incompatible with supported INT4 group sizes.

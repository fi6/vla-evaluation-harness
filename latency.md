# Latency Testing Notes

## Model Server Latency Logs

Model servers can record server-side latency with `ModelServer._log_latency()`.
The log is buffered per episode and flushed only after a successful
`EPISODE_END`, so failed episodes do not pollute latency summaries.

Latency logs are JSONL files under `results/`:

```text
results/<model>_<unix_ts>_latency.jsonl
```

Each row contains:

```json
{
  "episode_id": "...",
  "step": 10,
  "preprocess_ms": 1.23,
  "infer_ms": 181.45,
  "success": true
}
```

### Synthetic Latency Smoke

Use the synthetic client for a fast latency check against any running model
server. This does not start a simulator; it sends black RGB images, zero state,
and a task string over the normal websocket protocol.

```bash
python3 tools/latency_smoke_client.py \
  --url ws://127.0.0.1:8000 \
  --steps 11 \
  --image-size 256 \
  --state-dim 8 \
  --output results/latency_smoke_roundtrip.json
```

The smoke client records client-side round-trip latency in its output JSON.
The server writes the authoritative model-side `*_latency.jsonl` file when it
receives `EPISODE_END`.

### Summarize Latency Logs

Summarize one or more server latency logs into profile-style JSON:

```bash
python3 tools/latency_summary.py results/*_latency.jsonl \
  --drop-first-step \
  --output results/latency_summary.json
```

Use `--drop-first-step` for servers that do one-time prompt setup, calibration,
CUDA graph capture, or cache warmup on the first observation.

### Compose Workflow For Latency

For Docker Compose stacks, first start just the model server, then run the
synthetic latency smoke from the host:

```bash
docker compose -f docker/model_servers/docker-compose.<model>.yaml up --build <model>-server
python3 tools/latency_smoke_client.py --url ws://127.0.0.1:8000 --steps 11
python3 tools/latency_summary.py results/<model>_*_latency.jsonl --drop-first-step
```

For simulator-backed latency, run the smoke or full LIBERO benchmark through the
compose stack. Successful episodes will populate the same server-side latency
JSONL files.

---

## Docker Compose Runbooks For Model Servers

Each model server + LIBERO benchmark runs as a single `docker compose up`. All
stacks follow the same pattern: the model-server service healthchecks its
`/config` endpoint, and the `libero` service starts automatically once the model
is ready.

## Common Notes

### HuggingFace token

Some models (CogACT / gated repos) require a HuggingFace token. Store it in
`.env` at the repo root:

```text
HF_TOKEN=hf_...
```

Then export it before running compose so Docker picks it up:

```bash
set -a && source .env && set +a
docker compose -f docker/model_servers/docker-compose.<model>.yaml up
```

### When to rebuild the libero image

Any change under `src/` (e.g. `benchmarks/libero/benchmark.py`,
`benchmarks/recording.py`) requires a LIBERO image rebuild because
`Dockerfile.libero` does `COPY src/ src/`:

```bash
docker build -f docker/Dockerfile.libero \
  -t ghcr.io/allenai/vla-evaluation-harness/libero:local .
```

Pass `--base-image` to reuse the already-pulled published base and avoid solver
drift:

```bash
docker/build.sh --tag local \
  --base-image ghcr.io/allenai/vla-evaluation-harness/base:latest libero
```

### Benchmark configs

| Config | Episodes | Notes |
|--------|----------|-------|
| `configs/benchmarks/libero/quick.yaml` | 5 × 5 = 25 | Smoke test |
| `configs/benchmarks/libero/10.yaml` | 10 × 50 = 500 | Full LIBERO-10 |
| `configs/benchmarks/libero/video_<model>.yaml` | 10 × 10 = 100 | + video recording |

---

## π₀ + LIBERO

### Build

```bash
docker build \
  -f docker/model_servers/Dockerfile.pi0 \
  -t ghcr.io/allenai/vla-evaluation-harness/model-server-pi0:local \
  .
```

### Run

```bash
docker compose -f docker/model_servers/docker-compose.pi0.yaml up
```

Startup sequence:
1. `pi0-server` checks the local HF cache; downloads `lerobot/pi0` if absent.
2. Healthcheck polls `GET http://localhost:8000/config` every 10 s (36 retries, 180 s warm-up).
3. Once healthy, `libero` starts and connects to `ws://pi0-server:8000`.
4. Results and `pi0_<ts>_latency.jsonl` land in `results/`.

---

## SmolVLA + LIBERO

### Build

```bash
docker build \
  -f docker/model_servers/Dockerfile.smolvla \
  -t ghcr.io/allenai/vla-evaluation-harness/model-server-smolvla:local \
  .
```

SmolVLA uses LeRobot 0.5.x, which requires Python 3.12. The Dockerfile therefore
runs `uv run --python 3.12` instead of the Python 3.11 used by most other model
servers.

### Run

```bash
docker compose -f docker/model_servers/docker-compose.smolvla.yaml up --build
```

Startup sequence:
1. `smolvla-server` preflight downloads `HuggingFaceVLA/smolvla_libero` if absent.
2. The server loads the LeRobot SmolVLA policy and requests LIBERO wrist images + 8-D state.
3. Healthcheck polls `GET http://localhost:8000/config` every 10 s (36 retries, 180 s warm-up).
4. Once healthy, `libero` runs `configs/benchmarks/libero/10.yaml`.
5. Benchmark JSON and `smolvla_<ts>_latency.jsonl` land in `results/`.

---

## SimVLA + LIBERO

### Build

```bash
docker build \
  -f docker/model_servers/Dockerfile.simvla \
  -t ghcr.io/allenai/vla-evaluation-harness/model-server-simvla:local \
  .
```

### Run

```bash
# Smoke: 25 episodes
docker compose -f docker/model_servers/docker-compose.simvla-smoke.yaml up --build

# Full LIBERO-10: 500 episodes
docker compose -f docker/model_servers/docker-compose.simvla.yaml up --build
```

Startup sequence:
1. `simvla-server` preflight downloads `YuankaiLuo/SimVLA-LIBERO` required files if absent.
2. The server clones/uses `/opt/SimVLA`, loads norm stats, and requests wrist images + 8-D state.
3. Healthcheck polls `GET http://localhost:8000/config` every 15 s (60 retries, 300 s warm-up).
4. Results and `simvla_<ts>_latency.jsonl` land in `results/`.

---

## DeepThinkVLA + LIBERO

### Build

```bash
docker build \
  -f docker/model_servers/Dockerfile.deepthinkvla \
  -t ghcr.io/allenai/vla-evaluation-harness/model-server-deepthinkvla:local \
  .
```

### Run

```bash
# Smoke: 25 episodes
docker compose -f docker/model_servers/docker-compose.deepthinkvla-smoke.yaml up --build

# Full LIBERO-10: 500 episodes
docker compose -f docker/model_servers/docker-compose.deepthinkvla.yaml up --build
```

Startup sequence:
1. `deepthinkvla-server` preflight downloads the required checkpoint files if absent.
2. The server loads OpenBMB/DeepThinkVLA from `/opt/DeepThinkVLA` and emits 10 actions per inference.
3. Healthcheck polls `GET http://localhost:8000/config` every 15 s (60 retries, 300 s warm-up).
4. Results and `deepthinkvla_<ts>_latency.jsonl` land in `results/`.

---

## StarVLA + LIBERO

### Build

```bash
docker build \
  -f docker/model_servers/Dockerfile.starvla \
  -t ghcr.io/allenai/vla-evaluation-harness/model-server-starvla:local \
  .
```

### Run

```bash
docker compose -f docker/model_servers/docker-compose.starvla.yaml up
```

Startup sequence:
1. `starvla-server` preflight downloads `StarVLA/Qwen2.5-VL-GR00T-LIBERO-4in1` if absent.
2. Healthcheck polls `GET http://localhost:8000/config` every 10 s (36 retries, 180 s warm-up).
3. Once healthy, `libero` starts.

### Switch checkpoint

Edit `command:` in `docker-compose.starvla.yaml`. Available LIBERO checkpoints:

| checkpoint | framework |
|---|---|
| `StarVLA/Qwen2.5-VL-GR00T-LIBERO-4in1` | QwenGR00T (default) |
| `StarVLA/Qwen2.5-VL-OFT-LIBERO-4in1` | QwenOFT |
| `StarVLA/Qwen2.5-VL-FAST-LIBERO-4in1` | QwenFAST |
| `StarVLA/Qwen3-VL-OFT-LIBERO-4in1` | QwenOFT (Qwen3) |
| `StarVLA/Qwen3-VL-PI-LIBERO-4in1` | QwenPI (Qwen3) |

Also update `entrypoint_starvla.sh` preflight list to match the new checkpoint.

---

## GR00T N1.6 + LIBERO

### Build

```bash
docker build \
  -f docker/model_servers/Dockerfile.groot \
  -t ghcr.io/allenai/vla-evaluation-harness/model-server-groot:local \
  .
```

### Run

```bash
docker compose -f docker/model_servers/docker-compose.groot.yaml up
```

Startup sequence:
1. `groot-server` preflight downloads `0xAnkitSingh/GR00T-N1.6-LIBERO` if absent.
2. Healthcheck polls `GET http://localhost:8000/config` every 10 s (36 retries, 180 s warm-up).
3. Once healthy, `libero` starts.

**Note:** GR00T N1.6 requires `--invert_gripper` for LIBERO. This flag is set
in the compose `command:` block.

---

## OpenVLA + LIBERO

### Build

```bash
docker build \
  -f docker/model_servers/Dockerfile.openvla \
  -t ghcr.io/allenai/vla-evaluation-harness/model-server-openvla:local \
  .
```

First build resolves `dlimp`, a custom transformers fork, and other git deps.
Allow ~30 min on first run; subsequent starts are fast once the HF cache and uv
archive cache are warm.

### Run

```bash
docker compose -f docker/model_servers/docker-compose.openvla.yaml up
```

Startup sequence:
1. `openvla-server` preflight downloads `openvla/openvla-7b-finetuned-libero-10` if absent.
2. Healthcheck polls `GET http://localhost:8000/config` every 15 s (240 retries, 600 s warm-up).
3. Once healthy, `libero` starts.

**Note:** The libero service needs `NVIDIA_DRIVER_CAPABILITIES: all` and
`capabilities: [gpu, graphics, utility, compute]` for MuJoCo EGL rendering.

---

## OpenVLA-OFT + LIBERO

### Build

```bash
docker build \
  -f docker/model_servers/Dockerfile.oft \
  -t ghcr.io/allenai/vla-evaluation-harness/model-server-oft:local \
  .
```

### Run

```bash
docker compose -f docker/model_servers/docker-compose.oft.yaml up
```

Startup sequence:
1. `oft-server` preflight downloads the OFT LIBERO-10 checkpoint from HF if absent.
2. The entrypoint auto-detects the snapshot path and passes `--pretrained_checkpoint`.
3. Healthcheck polls `GET http://localhost:8000/config` every 15 s (240 retries, 600 s warm-up).
4. Once healthy, `libero` starts.

---

## CogACT + LIBERO

### Prerequisites

- HuggingFace token with access to `meta-llama/Llama-2-7b-hf`.
- ~35 GB free system RAM during model loading. The GPU itself only needs ~17 GB VRAM.

### Build

```bash
docker build \
  -f docker/model_servers/Dockerfile.cogact \
  -t ghcr.io/allenai/vla-evaluation-harness/model-server-cogact:local \
  .
```

### Run

```bash
set -a && source .env && set +a
docker compose -f docker/model_servers/docker-compose.cogact.yaml up
```

The compose file passes `HF_TOKEN` into the container. `cogact.py` calls
`huggingface_hub.login(token=HF_TOKEN)` before `load_vla()` so the internal
`HfFileSystem` auth check passes even for gated repos.

Startup sequence:
1. `cogact-server` preflight downloads `CogACT/CogACT-Base` if absent.
2. Model loads: DINOv2 + SigLIP vision backbone, then Llama-2-7b LLM, then full VLA checkpoint.
3. Healthcheck polls `GET http://localhost:8000/config` every 10 s (36 retries, 180 s warm-up).
4. Once healthy, `libero` starts.

**If you see exit code 137 (OOM):** drop the page cache before running to give
the OS headroom to reclaim buffer/cache during the loading peak:

```bash
sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
```

---

## DB-CogACT + LIBERO

### Build

```bash
docker build \
  -f docker/model_servers/Dockerfile.db_cogact \
  -t ghcr.io/allenai/vla-evaluation-harness/model-server-db-cogact:local \
  .
```

### Run

```bash
docker compose -f docker/model_servers/docker-compose.db_cogact.yaml up --build
```

Startup sequence:
1. `db-cogact-server` preflight downloads `Dexmal/libero-db-cogact` if absent.
2. The server records latency through the Dexbotic CogACT adapter.
3. Healthcheck polls `GET http://localhost:8000/config` every 10 s (36 retries, 300 s warm-up).
4. Once healthy, `libero` starts.

---

## Local Docker Image Build (libero base)

### Problem

Running `docker/build.sh --tag local libero` always rebuilds the **base image from scratch**,
pulling the latest Miniforge from GitHub. The published base was frozen on a specific date, so
a freshly rebuilt base may differ in conda solver behaviour, library versions, or uv behaviour —
causing errors like `ModuleNotFoundError: No module named 'libero'` at runtime.

### Fix

Pass `--base-image` to reuse the already-pulled published base instead of rebuilding it:

```bash
docker/build.sh --tag local --base-image ghcr.io/allenai/vla-evaluation-harness/base:latest libero
```

The script still tags a fresh `base:local` (harmless), but the libero image layers are built on
top of `base:latest` — the same base the published image used.

### Verify the image

```bash
docker run --rm --entrypoint conda   ghcr.io/allenai/vla-evaluation-harness/libero:local   run -n libero python -c "
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
print('libero ok')
"
```

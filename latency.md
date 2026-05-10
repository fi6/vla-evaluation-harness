# Latency Testing Notes

## Local Docker Image Build

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

Apply the same pattern to any other benchmark image:

```bash
docker/build.sh --tag local --base-image ghcr.io/allenai/vla-evaluation-harness/base:latest <benchmark>
```

### Verify the image

```bash
docker run --rm --entrypoint conda \
  ghcr.io/allenai/vla-evaluation-harness/libero:local \
  run -n libero python -c "
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
print('libero ok')
"
```

Expected output (warnings are harmless):

```
[robosuite WARNING] No private macro file found! ...
libero ok
```

### Run an evaluation with the local image

`vla-eval run` reads `docker.image` from the config YAML — there is no `--image` CLI flag.
Create a modified config that points to the local image:

```bash
sed 's|libero:latest|libero:local|' configs/libero_smoke_test.yaml > /tmp/libero_local.yaml
uv run vla-eval run -c /tmp/libero_local.yaml -y
```

---

## π₀ + LIBERO via Docker Compose

Replaces `vla-eval serve` + `vla-eval run` with a single `docker compose up`.

### Step 1 — Build images

From the repo root:

```bash
# Build the π₀ model server image
docker build \
  -f docker/model_servers/Dockerfile.pi0 \
  -t ghcr.io/allenai/vla-evaluation-harness/model-server-pi0:local \
  .

# Build the LIBERO benchmark image
docker/build.sh --tag local --base-image ghcr.io/allenai/vla-evaluation-harness/base:latest libero
```

### Step 2 — Start the stack

```bash
docker compose -f docker/model_servers/docker-compose.yaml up
```

Startup sequence:
1. `pi0-server` starts and loads the model from `~/.cache` (no GCS download).
2. Healthcheck polls `GET http://localhost:8000/config` every 10 s (up to 30 retries, 120 s
   warm-up window).
3. Once the model is ready, `libero` starts automatically and connects to
   `ws://pi0-server:8000` via the compose network.
4. Results are written to `results/` in the repo root.

### Model server only

To start only the π₀ server (e.g. while running the benchmark separately on the host):

```bash
docker compose -f docker/model_servers/docker-compose.yaml up pi0-server
```

### Changing the benchmark config

Edit the `command` block in `docker/model_servers/docker-compose.yaml`:

```yaml
command:
  - run
  - --config
  - /workspace/configs/benchmarks/libero/goal.yaml   # ← change this
  - --server-url
  - ws://pi0-server:8000
  - -y
```

---

## starVLA + LIBERO via Docker Compose

### Step 1 — Build images

```bash
# Build the starVLA model server image (first time ~14 min, downloads torch etc.)
docker build \
  -f docker/model_servers/Dockerfile.starvla \
  -t ghcr.io/allenai/vla-evaluation-harness/model-server-starvla:local \
  .

# Rebuild the LIBERO benchmark image (required after any src/ change)
docker/build.sh --tag local --base-image ghcr.io/allenai/vla-evaluation-harness/base:latest libero
```

> **When to rebuild LIBERO**: any change under `src/` (e.g. `sync_runner.py`,
> `benchmarks/libero/`) requires a LIBERO image rebuild because
> `Dockerfile.libero` does `COPY src/ src/`.

### Step 2 — Start the stack

```bash
HF_ENDPOINT=https://hf-mirror.com \
  docker compose -f docker/model_servers/docker-compose.starvla.yaml up
```

Startup sequence:
1. `starvla-server` runs preflight: downloads `StarVLA/Qwen2.5-VL-GR00T-LIBERO-4in1`
   and `starVLA/Qwen2.5-VL-3B-Instruct-Action` to `~/.cache/huggingface` (skipped if cached).
2. Model loads; healthcheck polls `GET http://localhost:8000/config` every 10 s
   (up to 36 retries, 180 s warm-up window).
3. Once healthy, `libero` starts and connects to `ws://starvla-server:8000`.
4. Results are written to `results/` in the repo root.

### Switch checkpoint

Edit the `command:` block in `docker-compose.starvla.yaml`.
Available LIBERO checkpoints (all use `unnorm_type: minmax`, `send_wrist_image: true`):

| checkpoint | framework |
|---|---|
| `StarVLA/Qwen2.5-VL-GR00T-LIBERO-4in1` | QwenGR00T (default) |
| `StarVLA/Qwen2.5-VL-OFT-LIBERO-4in1` | QwenOFT |
| `StarVLA/Qwen2.5-VL-FAST-LIBERO-4in1` | QwenFAST |
| `StarVLA/Qwen3-VL-OFT-LIBERO-4in1` | QwenOFT (Qwen3) |
| `StarVLA/Qwen3-VL-PI-LIBERO-4in1` | QwenPI (Qwen3) |

Also update `entrypoint_starvla.sh` preflight list to match the new checkpoint.

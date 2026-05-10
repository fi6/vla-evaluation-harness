# Adding a New Model Server — Integration Checklist

Reference for adding a new VLA model to the Docker-based eval stack.
Every item here has already been applied to pi0, groot, starvla, molmoact, molmobot.

---

## 1. Dockerfile

Use `Dockerfile.groot` or `Dockerfile.pi0` as a template.

**Required pieces:**

```dockerfile
FROM nvcr.io/nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04   # devel if flash-attn needed
                                                            # runtime otherwise

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    NVIDIA_DRIVER_CAPABILITIES=all

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /workspace

COPY pyproject.toml README.md ./
COPY src/ src/
COPY configs/ configs/
COPY docker/model_servers/preflight.py docker/model_servers/preflight.py
COPY docker/model_servers/entrypoint_<model>.sh docker/model_servers/entrypoint_<model>.sh
RUN chmod +x docker/model_servers/entrypoint_<model>.sh

ARG HARNESS_VERSION=0.0.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${HARNESS_VERSION}

RUN uv run --python 3.11 src/vla_eval/model_servers/<model>.py --help \
    && rm -rf ~/.cache/pip

ENTRYPOINT ["/workspace/docker/model_servers/entrypoint_<model>.sh"]
CMD ["--model_path", "...", "--port", "8000"]
```

> **devel vs runtime image**: use `devel` only if the model requires flash-attn
> compilation at install time. `runtime` is smaller and faster to pull.

---

## 2. Entrypoint script

Create `entrypoint_<model>.sh` (copy from `entrypoint_groot.sh`).

```bash
#!/usr/bin/env bash
set -euo pipefail

uv run --python 3.11 /workspace/docker/model_servers/preflight.py \
    org/repo-1 \
    org/repo-2          # list every HuggingFace repo the model needs

exec uv run --python 3.11 /workspace/src/vla_eval/model_servers/<model>.py "$@"
```

- **HF models**: list all repos in `preflight.py` call. preflight skips repos already cached.
- **GCS models** (e.g. pi0): no preflight needed; openpi downloads on first use and caches
  in `~/.cache/openpi`.
- After creating the script: `chmod +x entrypoint_<model>.sh`.

---

## 3. .dockerignore whitelist

Add the new entrypoint to `.dockerignore` (whitelist-based, denies everything by default):

```
!docker/model_servers/entrypoint_<model>.sh
```

---

## 4. docker-compose.\<model\>.yaml

Use `docker-compose.groot.yaml` as template. Required fields:

```yaml
name: vla-eval-<model>          # REQUIRED — prevents orphan warnings when switching models

services:
  <model>-server:
    image: ghcr.io/allenai/vla-evaluation-harness/model-server-<model>:local
    build:
      context: ../..
      dockerfile: docker/model_servers/Dockerfile.<model>
    ports:
      - "8000:8000"
    command:
      - --<arg>
      - <value>
      - --port
      - "8000"
      - --verbose                # enables per-step inference timing in logs
    environment:
      HF_ENDPOINT: "${HF_ENDPOINT:-https://huggingface.co}"
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface   # HF model cache
      - ~/.triton:/root/.triton                         # triton kernel cache (PyTorch models)
      # - ~/.cache/openpi:/root/.cache/openpi           # GCS cache (JAX/pi0 only)
      - ../../results:/workspace/results                # latency JSONL output
    healthcheck:
      test: ["CMD", "curl", "-sf", "http://localhost:8000/config"]
      interval: 10s
      timeout: 5s
      retries: 36
      start_period: 180s
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]

  libero:
    image: ghcr.io/allenai/vla-evaluation-harness/libero:local
    depends_on:
      <model>-server:
        condition: service_healthy
    command:
      - run
      - --config
      - /workspace/configs/benchmarks/libero/quick.yaml
      - --server-url
      - ws://<model>-server:8000
      - -y
    volumes:
      - ../../results:/workspace/results
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
```

**Key points:**
- `name: vla-eval-<model>` — each compose file must have a unique project name, otherwise
  switching models leaves orphan containers.
- Do **not** mount `~/.cache` wholesale — it shadows the uv venv baked into the image,
  forcing a full reinstall from PyPI on every container start.
- Mount `~/.triton` for PyTorch + flash-attn models (saves 15–30 min recompile per restart).
- Mount `~/.cache/openpi` for JAX/openpi models instead.
- Mount `../../results` on **both** services: model server writes latency JSONL,
  libero writes benchmark results JSON.

---

## 5. Inference latency logging

`ModelServer._log_latency()` is defined in `src/vla_eval/model_servers/base.py`.
Every model server should call it to produce `results/<model>_<ts>_latency.jsonl`.

The timestamp in the filename is set on first write (start of the first successful episode), so it
aligns within seconds of the benchmark result JSON timestamp — use it to correlate the two files.

**Buffering behavior**: entries are held in memory per episode and only written to disk when
`on_episode_end()` is called with a non-empty result (normal completion). Failed episodes
(result `{}`) are still written but tagged `"success": false`, so failures are visible without
polluting success statistics.

If your model server overrides `on_episode_end()`, it **must** call `await super().on_episode_end(result, ctx)`
to trigger the flush.

### For models using `predict()` (called every step, e.g. pi0, molmoact)

Split the method at the model forward call and use the default `interval=10`:

```python
def predict(self, obs: Observation, ctx: SessionContext) -> Action:
    import time
    ...
    t_pre = time.perf_counter()
    # --- preprocess: build model input dict ---
    preprocess_ms = (time.perf_counter() - t_pre) * 1000

    t_infer = time.perf_counter()
    result = self._model.forward(inputs)           # ← actual model call
    self._log_latency(ctx, preprocess_ms, (time.perf_counter() - t_infer) * 1000)
    # interval=10 (default) — logs every 10th step

    return {"actions": result["actions"]}
```

### For models using `predict_batch()` (chunk-boundary only, e.g. groot, starvla)

Use `interval=1` because `predict_batch()` is only called every `chunk_size` steps —
using the default `interval=10` would only log at multiples of both chunk_size and 10:

```python
def predict_batch(self, obs_batch, ctx_batch):
    import time
    t_pre = time.perf_counter()
    # --- preprocess ---
    preprocess_ms = (time.perf_counter() - t_pre) * 1000

    t_infer = time.perf_counter()
    result = self._model.forward(batch)
    self._log_latency(ctx_batch[0], preprocess_ms, (time.perf_counter() - t_infer) * 1000, interval=1)
    ...
```

### For models with custom `on_observation()` (e.g. molmobot)

Time the inference call inside the method. If preprocess and infer are entangled
(e.g. inside a helper), pass `preprocess_ms=0.0`. Use `interval=1` for chunk-boundary refills:

```python
async def on_observation(self, obs, ctx):
    ...
    if needs_refill:
        import time
        t_infer = time.perf_counter()
        self._refill_buffer(obs, state)
        self._log_latency(ctx, 0.0, (time.perf_counter() - t_infer) * 1000, interval=1)
```

### Output format

```jsonl
{"episode_id": "<uuid>", "step": 0,  "preprocess_ms": 0.911, "infer_ms": 8914.574, "success": true}
{"episode_id": "<uuid>", "step": 10, "preprocess_ms": 0.825, "infer_ms": 78.803,   "success": true}
{"episode_id": "<uuid>", "step": 0,  "preprocess_ms": 0.910, "infer_ms": 7821.100, "success": false}
```

- `success: true` — episode completed normally; `false` — episode ended with an exception.
- `episode_id` = `ctx.episode_id` — matches episode UUIDs in the benchmark result JSON.
- Step 0 `infer_ms` is typically high (JIT / triton kernel compile on first inference).
- `predict()` models: logged every 10 steps (`interval=10`).
- `predict_batch()` / refill models: logged every chunk boundary (`interval=1`).

---

## 6. Running

```bash
# Build model server image
docker build \
  -f docker/model_servers/Dockerfile.<model> \
  -t ghcr.io/allenai/vla-evaluation-harness/model-server-<model>:local \
  .

# Model server only (benchmark runs separately on host)
HF_ENDPOINT=https://hf-mirror.com \
  docker compose -f docker/model_servers/docker-compose.<model>.yaml up <model>-server

# Full stack (model server + LIBERO)
HF_ENDPOINT=https://hf-mirror.com \
  docker compose -f docker/model_servers/docker-compose.<model>.yaml up

# Verify server is ready
curl http://localhost:8000/config

# Clean up (stop and remove containers)
docker compose -f docker/model_servers/docker-compose.<model>.yaml down
```

> Set `HF_ENDPOINT=https://hf-mirror.com` if direct HuggingFace access is slow or blocked.
> The compose file passes it through to the container automatically.

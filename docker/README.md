# Docker Images

Benchmark environments for `vla-eval`. Each benchmark runs in its own container with CPU-only PyTorch; GPU model servers run on the host or in a separate container.

## Image Hierarchy

`Dockerfile.base` provides the common layer (CUDA 12.1, EGL/Vulkan, Miniforge, uv). Each `Dockerfile.<bench>` extends it with benchmark-specific dependencies. Model server images live under `model_servers/`.

```
docker/
├── Dockerfile.base          # shared base: CUDA 12.1, EGL/Vulkan, Miniforge, uv
├── Dockerfile.libero        # LIBERO benchmark (Python 3.8, conda env)
├── Dockerfile.calvin        # CALVIN benchmark
├── Dockerfile.simpler       # SimplerEnv benchmark
├── ...
└── model_servers/
    ├── Dockerfile.pi0       # π₀ model server (uv script, Python 3.11)
    ├── Dockerfile.smolvla   # SmolVLA model server (uv script, Python 3.12)
    ├── Dockerfile.oft       # OpenVLA-OFT model server
    ├── Dockerfile.groot     # GR00T N1.6 model server
    ├── Dockerfile.openvla   # OpenVLA model server
    ├── Dockerfile.starvla   # StarVLA model server
    ├── Dockerfile.cogact    # CogACT model server
    ├── docker-compose.pi0.yaml      # π₀ + LIBERO stack
    ├── docker-compose.smolvla.yaml  # SmolVLA + LIBERO stack
    ├── docker-compose.oft.yaml      # OFT + LIBERO stack
    ├── docker-compose.groot.yaml    # GR00T + LIBERO stack
    ├── docker-compose.openvla.yaml  # OpenVLA + LIBERO stack
    ├── docker-compose.starvla.yaml  # StarVLA + LIBERO stack
    └── docker-compose.cogact.yaml   # CogACT + LIBERO stack
```

## Build & Push (benchmark images)

```bash
# Build all images (base first, then benchmarks)
docker/build.sh

# Build a single benchmark
docker/build.sh libero

# Build with a version tag
docker/build.sh --tag 0.2.0

# Push all images (requires: docker login ghcr.io)
docker/push.sh --tag 0.2.0

# Push a single image
docker/push.sh --tag 0.2.0 libero
```

Images are published to `ghcr.io/allenai/vla-evaluation-harness/<name>:<tag>`.

### Rebuilding the libero image after src/ changes

Any change under `src/` (e.g. `benchmarks/libero/benchmark.py`, `benchmarks/recording.py`)
invalidates the `COPY src/ src/` layer and requires a rebuild:

```bash
# Recommended: reuse the published base to avoid conda solver drift
docker/build.sh --tag local \
  --base-image ghcr.io/allenai/vla-evaluation-harness/base:latest libero

# Or directly with docker build
docker build -f docker/Dockerfile.libero \
  -t ghcr.io/allenai/vla-evaluation-harness/libero:local .
```

## Build (model server images)

Model server images are standalone — build from the repo root:

```bash
# π₀
docker build -f docker/model_servers/Dockerfile.pi0 \
  -t ghcr.io/allenai/vla-evaluation-harness/model-server-pi0:local .

# SmolVLA
docker build -f docker/model_servers/Dockerfile.smolvla \
  -t ghcr.io/allenai/vla-evaluation-harness/model-server-smolvla:local .

# OFT
docker build -f docker/model_servers/Dockerfile.oft \
  -t ghcr.io/allenai/vla-evaluation-harness/model-server-oft:local .

# GR00T N1.6
docker build -f docker/model_servers/Dockerfile.groot \
  -t ghcr.io/allenai/vla-evaluation-harness/model-server-groot:local .

# OpenVLA
docker build -f docker/model_servers/Dockerfile.openvla \
  -t ghcr.io/allenai/vla-evaluation-harness/model-server-openvla:local .

# StarVLA
docker build -f docker/model_servers/Dockerfile.starvla \
  -t ghcr.io/allenai/vla-evaluation-harness/model-server-starvla:local .

# CogACT
docker build -f docker/model_servers/Dockerfile.cogact \
  -t ghcr.io/allenai/vla-evaluation-harness/model-server-cogact:local .
```

First builds take 15–45 min (torch + model deps). Subsequent builds are fast due to Docker layer
caching — only layers after `COPY src/ src/` are invalidated by harness changes.

## Running a full stack (model + benchmark)

```bash
docker compose -f docker/model_servers/docker-compose.pi0.yaml up
```

The model server healthchecks `GET /config`; the benchmark starts automatically once healthy.
See [latency.md](../latency.md) for per-model build commands, startup notes, and troubleshooting.

## Adding a New Benchmark

1. Create `Dockerfile.<name>` — use `ARG BASE_IMAGE` and install benchmark-specific deps.
2. Add `<name>` to the `BENCHMARKS` array in `build.sh` and `IMAGES` array in `push.sh`.

## Adding a New Model Server

1. Create `model_servers/Dockerfile.<name>` — base on a suitable CUDA image, use `uv run` for deps.
2. Create `model_servers/entrypoint_<name>.sh` — handle preflight download + `exec uv run ...`.
3. Create `model_servers/docker-compose.<name>.yaml` — wire model server + libero services.
4. Document the stack in [latency.md](../latency.md).

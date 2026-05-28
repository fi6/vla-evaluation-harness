#!/usr/bin/env bash
# Entrypoint for the SimVLA model server container.
set -euo pipefail

uv run --python 3.11 /workspace/docker/model_servers/preflight.py \
    --require config.json \
    --require model.safetensors \
    --require state.json \
    YuankaiLuo/SimVLA-LIBERO

exec uv run --python 3.10 /workspace/src/vla_eval/model_servers/simvla.py "$@"

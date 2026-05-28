#!/usr/bin/env bash
# Entrypoint for the SmolVLA model server container.
set -euo pipefail

uv run --python 3.12 /workspace/docker/model_servers/preflight.py \
    HuggingFaceVLA/smolvla_libero

exec uv run --python 3.12 /workspace/src/vla_eval/model_servers/smolvla.py "$@"

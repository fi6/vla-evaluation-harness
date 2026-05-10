#!/usr/bin/env bash
# Entrypoint for the GR00T N1.6 model server container.
#
# 1. Checks that all required HF model repos are in the local cache.
#    If any are missing, downloads them (with retry) via HF_ENDPOINT.
#    Users may pre-populate ~/.cache/huggingface/hub/ manually to skip this.
# 2. Exec's the GR00T model server, forwarding all CLI arguments.
set -euo pipefail

uv run --python 3.11 /workspace/docker/model_servers/preflight.py \
    nvidia/GR00T-N1.6-3B

exec uv run --python 3.11 /workspace/src/vla_eval/model_servers/groot.py "$@"

#!/usr/bin/env bash
# Entrypoint for the starVLA model server container.
#
# 1. Checks that all required HF model repos are in the local cache.
#    If any are missing, downloads them (with retry) via HF_ENDPOINT.
#    Users may pre-populate ~/.cache/huggingface/hub/ manually to skip this.
# 2. Exec's the starVLA model server, forwarding all CLI arguments.
set -euo pipefail

uv run --python 3.11 /workspace/docker/model_servers/preflight.py \
    StarVLA/Qwen2.5-VL-GR00T-LIBERO-4in1 \
    starVLA/Qwen2.5-VL-3B-Instruct-Action

exec uv run --python 3.11 /workspace/src/vla_eval/model_servers/starvla.py "$@"

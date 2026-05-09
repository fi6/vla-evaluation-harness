#!/usr/bin/env bash
# Entrypoint for the MolmoBot model server container.
#
# 1. Checks that all required HF model repos are in the local cache.
#    If any are missing, downloads them (with retry) via HF_ENDPOINT.
#    Users may pre-populate ~/.cache/huggingface/hub/ manually to skip this.
# 2. Exec's the MolmoBot model server, forwarding all CLI arguments.
set -euo pipefail

uv run --python 3.11 /workspace/docker/model_servers/preflight.py \
    allenai/MolmoBot-DROID \
    Qwen/Qwen3-4B-Instruct-2507 \
    google/siglip2-so400m-patch14-384

exec uv run --python 3.11 /workspace/src/vla_eval/model_servers/molmobot.py "$@"

#!/usr/bin/env bash
# Entrypoint for the MolmoAct2 model server container.
#
# 1. Downloads allenai/MolmoAct2-LIBERO (or the repo passed via --hf_repo) if absent.
# 2. Exec's the MolmoAct2 model server, forwarding all CLI arguments.
set -euo pipefail

# The preflight script reads HF_ENDPOINT and HF_TOKEN from the environment.
uv run --python 3.11 /workspace/docker/model_servers/preflight.py \
    allenai/MolmoAct2-LIBERO

exec uv run --python 3.11 /workspace/src/vla_eval/model_servers/molmoact2.py "$@"

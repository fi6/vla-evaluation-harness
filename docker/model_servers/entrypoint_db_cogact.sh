#!/usr/bin/env bash
# Entrypoint for the DB-CogACT model server container.
set -euo pipefail

uv run --python 3.11 /workspace/docker/model_servers/preflight.py \
    Dexmal/libero-db-cogact

VENV=$(ls -d /root/.cache/uv/environments-v2/cogact-* | head -n1)
exec "$VENV/bin/python" /workspace/src/vla_eval/model_servers/dexbotic/cogact.py "$@"

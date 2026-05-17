#!/usr/bin/env bash
# Entrypoint for the OpenVLA-OFT model server container.
#
# 1. Checks that the required HF model repo is in the local cache.
#    If missing, downloads it (with retry) via HF_ENDPOINT.
#    Users may pre-populate ~/.cache/huggingface/hub/ manually to skip this.
# 2. Exec's the OpenVLA-OFT model server, forwarding all CLI arguments.
#
# Common LIBERO OFT checkpoints:
#   moojink/openvla-7b-oft-finetuned-libero-spatial
#   moojink/openvla-7b-oft-finetuned-libero-object
#   moojink/openvla-7b-oft-finetuned-libero-goal
#   moojink/openvla-7b-oft-finetuned-libero-10
#   moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10  (joint)
#
# Unlike other model servers, this image installs deps system-wide via
# `uv pip install --system` instead of using `uv run` against a PEP 723
# script env (see Dockerfile.oft for the rationale). So we invoke `python`
# directly, not `uv run`.
set -euo pipefail

# Skip the HF preflight when the user has set offline mode — preflight uses
# `huggingface_hub.snapshot_download` which always touches the network even
# for cached repos. The model loader itself respects HF_HUB_OFFLINE and will
# happily read from the local cache.
if [ "${HF_HUB_OFFLINE:-0}" = "1" ]; then
    echo "[entrypoint] HF_HUB_OFFLINE=1 — skipping preflight"
else
    python /workspace/docker/model_servers/preflight.py \
        moojink/openvla-7b-oft-finetuned-libero-spatial
fi

exec python /workspace/src/vla_eval/model_servers/oft.py "$@"

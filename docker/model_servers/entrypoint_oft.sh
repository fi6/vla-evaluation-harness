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
REPO="moojink/openvla-7b-oft-finetuned-libero-10"

if [ "${HF_HUB_OFFLINE:-0}" = "1" ]; then
    echo "[entrypoint] HF_HUB_OFFLINE=1 — skipping preflight"
else
    python /workspace/docker/model_servers/preflight.py "$REPO"
fi

# Auto-detect local snapshot path (openvla-oft needs a real directory, not a repo ID)
SNAP=$(ls -d /root/.cache/huggingface/hub/models--moojink--openvla-7b-oft-finetuned-libero-10/snapshots/* 2>/dev/null | head -1)
if [ -z "$SNAP" ]; then
    echo "[entrypoint] ERROR: snapshot not found for $REPO" >&2
    exit 1
fi

exec python /workspace/src/vla_eval/model_servers/oft.py \
    --pretrained_checkpoint "$SNAP" \
    --unnorm_key libero_10_no_noops \
    "$@"

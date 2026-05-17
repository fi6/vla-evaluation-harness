#!/usr/bin/env bash
# Entrypoint for the OpenVLA model server container.
#
# 1. Checks that the required HF model repo is in the local cache.
#    If missing, downloads it (with retry) via HF_ENDPOINT.
#    Users may pre-populate ~/.cache/huggingface/hub/ manually to skip this.
# 2. Exec's the OpenVLA model server, forwarding all CLI arguments.
#
# Note: OpenVLA's --model_path is a HuggingFace repo id. Common LIBERO
# checkpoints include:
#   openvla/openvla-7b-finetuned-libero-spatial
#   openvla/openvla-7b-finetuned-libero-object
#   openvla/openvla-7b-finetuned-libero-goal
#   openvla/openvla-7b-finetuned-libero-10
# All of these share the same processor config so we preflight one and
# rely on first-inference download for the actual weight repo if it
# differs from the default below.
set -euo pipefail

# Use uv for the preflight (it has its own tiny inline-script env).
uv run --python 3.11 /workspace/docker/model_servers/preflight.py \
    openvla/openvla-7b-finetuned-libero-spatial

# For the model server itself we bypass `uv run` and exec the pre-built
# venv's Python directly. This avoids a re-resolution / re-link step at
# container start that has been observed to corrupt regex parser state
# (cuts off transformers' import chain with a stdlib re._parser TypeError).
VENV=$(ls -d /root/.cache/uv/environments-v2/openvla-* | head -n1)
exec "$VENV/bin/python" /workspace/src/vla_eval/model_servers/openvla.py "$@"

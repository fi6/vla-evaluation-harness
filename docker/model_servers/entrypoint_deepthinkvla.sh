#!/usr/bin/env bash
# Entrypoint for the DeepThinkVLA model server container.
set -euo pipefail

uv run --python 3.11 /workspace/docker/model_servers/preflight.py \
    --require config.json \
    --require generation_config.json \
    --require model-00001-of-00002.safetensors \
    --require model-00002-of-00002.safetensors \
    --require model.safetensors.index.json \
    --require norm_stats.json \
    --require preprocessor_config.json \
    --require special_tokens_map.json \
    --require tokenizer.json \
    --require tokenizer.model \
    --require tokenizer_config.json \
    yinchenghust/deepthinkvla_libero_cot_rl

exec uv run --python 3.10 /workspace/src/vla_eval/model_servers/deepthinkvla.py "$@"

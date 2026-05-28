---
smoke_config: libero.yaml
---

# SmolVLA

Compact LeRobot VLA from Hugging Face. [Paper](https://arxiv.org/abs/2506.01844) | [Model](https://huggingface.co/HuggingFaceVLA/smolvla_libero) | [GitHub](https://github.com/huggingface/lerobot)

## Configs

| File | Benchmark | Checkpoint |
|------|-----------|------------|
| `libero.yaml` | LIBERO | `HuggingFaceVLA/smolvla_libero` |

The server requests wrist images and 8-D LIBERO proprio state, converts harness observations to LeRobot keys, and writes latency JSONL files under `results/`.

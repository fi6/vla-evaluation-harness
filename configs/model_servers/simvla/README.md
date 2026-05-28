---
smoke_config: libero.yaml
---

# SimVLA

Simple VLA baseline from LUOyk1999/SimVLA. [Paper](https://arxiv.org/abs/2602.18224) | [Model](https://huggingface.co/YuankaiLuo/SimVLA-LIBERO) | [GitHub](https://github.com/LUOyk1999/SimVLA)

## Configs

| File | Benchmark | Checkpoint |
|------|-----------|------------|
| `libero.yaml` | LIBERO | `YuankaiLuo/SimVLA-LIBERO` |

The server requests wrist images and 8-D LIBERO proprio state, uses the upstream LIBERO normalization stats, returns 10-action chunks, and executes the first 5 actions before replanning.

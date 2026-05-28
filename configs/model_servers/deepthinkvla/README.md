# DeepThinkVLA

DeepThinkVLA model server for the OpenBMB/DeepThinkVLA LIBERO checkpoints.

- Upstream repo: https://github.com/OpenBMB/DeepThinkVLA
- Default checkpoint: `yinchenghust/deepthinkvla_libero_cot_rl`
- LIBERO action chunk: 10 actions per inference.

Docker compose:

```bash
docker compose -f docker/model_servers/docker-compose.deepthinkvla.yaml up --build
```

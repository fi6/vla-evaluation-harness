#!/usr/bin/env python3
"""Smoke-test torchao INT4 weight-only quantization for OpenPI pi0.5."""

from __future__ import annotations

import argparse
import datetime
import time

if not hasattr(datetime, "UTC"):
    datetime.UTC = datetime.timezone.utc

import numpy as np
import torch

from openpi.policies import policy_config
from openpi.training import config as openpi_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/root/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch")
    parser.add_argument("--config-name", default="pi05_libero")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--reps", type=int, default=3)
    args = parser.parse_args()

    from torchao.quantization import Int4WeightOnlyConfig, ZeroPointDomain, quantize_

    cfg = openpi_config.get_config(args.config_name)
    policy = policy_config.create_trained_policy(cfg, args.checkpoint, pytorch_device="cuda")
    model = policy._model

    print("quantizing int4 weight-only", flush=True)
    t0 = time.perf_counter()
    model.float()
    quantize_(
        model,
        Int4WeightOnlyConfig(group_size=args.group_size, zero_point_domain=ZeroPointDomain.FLOAT),
    )
    torch.cuda.synchronize()
    print(f"quantize_s={time.perf_counter() - t0:.3f}", flush=True)

    obs = {
        "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/state": np.zeros(8, dtype=np.float64),
        "prompt": "pick up the object",
    }

    for _ in range(args.warmup):
        policy.infer(dict(obs))
    lat = []
    for i in range(args.reps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = policy.infer(dict(obs))
        torch.cuda.synchronize()
        ms = (time.perf_counter() - start) * 1000
        lat.append(ms)
        print(f"iter={i} ms={ms:.3f} actions_shape={out['actions'].shape}", flush=True)
    print(f"p50_ms={float(np.median(lat)):.3f} min_ms={min(lat):.3f} max_ms={max(lat):.3f}", flush=True)


if __name__ == "__main__":
    main()

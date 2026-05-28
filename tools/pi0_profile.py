#!/usr/bin/env python3
"""Profile OpenPI pi0.5 PyTorch inference inside the Jetson Docker image."""

from __future__ import annotations

import argparse
import datetime
import sys
import time
import types

if not hasattr(datetime, "UTC"):
    datetime.UTC = datetime.timezone.utc

version = types.ModuleType("vla_eval._version")
version.__version__ = "0+profile"
version.__version_tuple__ = (0, "profile")
sys.modules.setdefault("vla_eval._version", version)

import numpy as np
import torch
import jax

from openpi.policies import policy_config
from openpi.training import config as openpi_config


def sync_ms(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter() - start) * 1000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/root/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch")
    parser.add_argument("--config-name", default="pi05_libero")
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--num-steps", type=int, default=10)
    args = parser.parse_args()

    cfg = openpi_config.get_config(args.config_name)
    policy = policy_config.create_trained_policy(
        cfg,
        args.checkpoint,
        sample_kwargs={"num_steps": args.num_steps},
        pytorch_device="cuda",
    )
    model = policy._model

    obs = {
        "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/state": np.zeros(8, dtype=np.float64),
        "prompt": "pick up the object",
    }
    transformed = policy._input_transform(dict(obs))
    inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to("cuda")[None, ...], transformed)
    observation = model.Observation.from_dict(inputs) if hasattr(model, "Observation") else None
    if observation is None:
        from openpi.models import model as openpi_model

        observation = openpi_model.Observation.from_dict(inputs)

    def run_segmented():
        segment_ms = {}

        torch.cuda.synchronize()
        start = time.perf_counter()
        images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(observation, train=False)
        torch.cuda.synchronize()
        segment_ms["model_preprocess"] = (time.perf_counter() - start) * 1000

        torch.cuda.synchronize()
        start = time.perf_counter()
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        torch.cuda.synchronize()
        segment_ms["embed_prefix"] = (time.perf_counter() - start) * 1000

        make_att_2d_masks = __import__(
            "openpi.models_pytorch.pi0_pytorch", fromlist=["make_att_2d_masks"]
        ).make_att_2d_masks

        torch.cuda.synchronize()
        start = time.perf_counter()
        prefix_att_2d_masks = model._prepare_attention_masks_4d(make_att_2d_masks(prefix_pad_masks, prefix_att_masks))
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        torch.cuda.synchronize()
        segment_ms["prefix_masks"] = (time.perf_counter() - start) * 1000

        torch.cuda.synchronize()
        start = time.perf_counter()
        model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        _, past_key_values = model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        torch.cuda.synchronize()
        segment_ms["prefix_prefill"] = (time.perf_counter() - start) * 1000
        noise = model.sample_noise((1, model.config.action_horizon, model.config.action_dim), "cuda")
        dt = torch.tensor(-1.0 / args.num_steps, dtype=torch.float32, device="cuda")
        x_t = noise
        step_ms = []
        t = torch.tensor(1.0, dtype=torch.float32, device="cuda")
        while t >= -dt / 2:
            torch.cuda.synchronize()
            start = time.perf_counter()
            v_t = model.denoise_step(state, prefix_pad_masks, past_key_values, x_t, t.expand(1))
            x_t = x_t + dt * v_t
            torch.cuda.synchronize()
            step_ms.append((time.perf_counter() - start) * 1000)
            t += dt
        return x_t, step_ms, segment_ms

    for i in range(args.iters):
        _, total = sync_ms(lambda: policy.infer(dict(obs)))
        print(f"infer_iter={i} total_ms={total:.3f}")

    _, pre_ms = sync_ms(lambda: model._preprocess_observation(observation, train=False))
    print(f"preprocess_model_ms={pre_ms:.3f}")

    for i in range(args.iters):
        (out, step_ms, segment_ms), total = sync_ms(run_segmented)
        print(
            f"seg_iter={i} total_ms={total:.3f} "
            f"denoise_sum_ms={sum(step_ms):.3f} denoise_mean_ms={np.mean(step_ms):.3f} "
            f"denoise_min_ms={min(step_ms):.3f} denoise_max_ms={max(step_ms):.3f}"
        )
        print("segments " + " ".join(f"{k}_ms={v:.3f}" for k, v in segment_ms.items()))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""FlashRT-backed pi0.5 server for Jetson smoke/latency tests."""

from __future__ import annotations

import argparse
import datetime
import os
import sys
import types
from typing import Any

if not hasattr(datetime, "UTC"):
    datetime.UTC = datetime.timezone.utc

version = types.ModuleType("vla_eval._version")
version.__version__ = "0+jetson-flashrt"
version.__version_tuple__ = (0, "jetson-flashrt")
sys.modules.setdefault("vla_eval._version", version)

import numpy as np

from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.model_servers.serve import serve
from vla_eval.specs import IMAGE_RGB, LANGUAGE, RAW, DimSpec
from vla_eval.types import Action, Observation


class FlashRTPi05ModelServer(PredictModelServer):
    def __init__(
        self,
        checkpoint: str,
        *,
        int8: bool = False,
        cache_frames: int = 1,
        num_views: int = 2,
        num_steps: int = 10,
        vision_pool_factor: int = 1,
        vision_num_layers: int = 27,
        image_resolution: int = 224,
        chunk_size: int = 10,
    ) -> None:
        super().__init__(chunk_size=chunk_size)
        self.checkpoint = checkpoint
        self.int8 = int8
        self.cache_frames = cache_frames
        self.num_views = num_views
        self.num_steps = num_steps
        self.vision_pool_factor = vision_pool_factor
        self.vision_num_layers = vision_num_layers
        self.image_resolution = image_resolution
        self._pipe = None
        self._prompt: str | None = None
        self._calibrated = False

    def _resize(self, img: np.ndarray) -> np.ndarray:
        if img.shape[:2] == (self.image_resolution, self.image_resolution):
            return img
        from PIL import Image

        return np.asarray(
            Image.fromarray(img).resize(
                (self.image_resolution, self.image_resolution),
                Image.Resampling.BILINEAR,
            )
        )

    def _load_model(self) -> None:
        if self._pipe is not None:
            return
        if self.int8:
            os.environ["FVK_PI05_RTX_FORCE_INT8"] = "1"
        else:
            os.environ.pop("FVK_PI05_RTX_FORCE_INT8", None)

        from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx

        self._pipe = Pi05TorchFrontendRtx(
            self.checkpoint,
            num_views=self.num_views,
            num_steps=self.num_steps,
            vision_pool_factor=self.vision_pool_factor,
            vision_num_layers=self.vision_num_layers,
            cache_frames=self.cache_frames,
        )

    def _flashrt_obs(self, obs: Observation) -> tuple[dict[str, np.ndarray], str]:
        images = obs.get("images", {})
        image_list = list(images.values()) if isinstance(images, dict) else []
        base = np.asarray(image_list[0], dtype=np.uint8) if image_list else np.zeros((256, 256, 3), dtype=np.uint8)
        wrist = (
            np.asarray(image_list[1], dtype=np.uint8)
            if len(image_list) > 1
            else np.zeros_like(base)
        )
        fr_obs = {"image": self._resize(base), "wrist_image": self._resize(wrist)}
        return fr_obs, str(obs.get("task_description", ""))

    def get_observation_params(self) -> dict[str, Any]:
        return {"send_wrist_image": True, "send_state": True}

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {"actions": RAW}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return {"image": IMAGE_RGB, "state": RAW, "language": LANGUAGE}

    def predict(self, obs: Observation, ctx: SessionContext) -> Action:
        self._load_model()
        assert self._pipe is not None
        import time

        t_pre = time.perf_counter()
        fr_obs, prompt = self._flashrt_obs(obs)
        if prompt != self._prompt:
            self._pipe.set_prompt(prompt)
            self._prompt = prompt
            self._calibrated = False
        if not self._calibrated:
            self._pipe.calibrate_with_real_data([fr_obs])
            self._calibrated = True
        preprocess_ms = (time.perf_counter() - t_pre) * 1000

        t_infer = time.perf_counter()
        out = self._pipe.infer(fr_obs)
        infer_ms = (time.perf_counter() - t_infer) * 1000
        self._log_latency(ctx, preprocess_ms, infer_ms)
        return {"actions": out["actions"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--cache_frames", type=int, default=1)
    parser.add_argument("--num_views", type=int, default=2)
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--vision_pool_factor", type=int, default=1)
    parser.add_argument("--vision_num_layers", type=int, default=27)
    parser.add_argument("--image_resolution", type=int, default=224)
    parser.add_argument("--chunk_size", type=int, default=10)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = FlashRTPi05ModelServer(
        checkpoint=args.checkpoint,
        int8=args.int8,
        cache_frames=args.cache_frames,
        num_views=args.num_views,
        num_steps=args.num_steps,
        vision_pool_factor=args.vision_pool_factor,
        vision_num_layers=args.vision_num_layers,
        image_resolution=args.image_resolution,
        chunk_size=args.chunk_size,
    )
    serve(server, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

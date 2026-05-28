# /// script
# requires-python = "~=3.10"
# dependencies = [
#     "vla-eval",
#     "torch==2.4.0",
#     "torchvision==0.19.0",
#     "transformers==4.48.1",
#     "tokenizers==0.21.1",
#     "huggingface_hub==0.29.3",
#     "safetensors",
#     "sentencepiece==0.2.0",
#     "pillow",
#     "numpy==1.26.4",
#     "timm==1.0.3",
#     "peft",
#     "accelerate",
#     "draccus==0.10.0",
#     "einops==0.8.0",
#     "opencv-python-headless==4.9.0.80",
#     "swanlab",
# ]
#
# [tool.uv.sources]
# vla-eval = { path = "../../..", editable = true }
#
# [tool.uv]
# exclude-newer = "2026-05-28T00:00:00Z"
# ///
"""DeepThinkVLA model server using the OpenBMB/DeepThinkVLA repo."""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np

from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.specs import IMAGE_RGB, LANGUAGE, RAW, STATE_EEF_POS_AA_GRIP, DimSpec
from vla_eval.types import Action, Observation

logger = logging.getLogger(__name__)


class DeepThinkVLAModelServer(PredictModelServer):
    """DeepThinkVLA server for LIBERO checkpoints.

    Upstream DeepThinkVLA uses LIBERO constants with 10 actions per model
    invocation. This adapter returns the full 10-step chunk and lets the
    harness buffer it.
    """

    def __init__(
        self,
        checkpoint: str = "yinchenghust/deepthinkvla_libero_cot_rl",
        repo_path: str = "/opt/DeepThinkVLA",
        num_images_in_input: int = 2,
        max_new_tokens: int = 2048,
        compute_dtype: str = "bfloat16",
        img_resize_size: int = 224,
        inference_mode: str = "full_cot",
        chunk_size: int = 10,
        action_ensemble: str = "newest",
        **kwargs: Any,
    ) -> None:
        super().__init__(chunk_size=chunk_size, action_ensemble=action_ensemble, **kwargs)
        self.checkpoint = checkpoint
        self.repo_path = repo_path
        self.num_images_in_input = num_images_in_input
        self.max_new_tokens = max_new_tokens
        self.compute_dtype = compute_dtype
        self.img_resize_size = img_resize_size
        self.inference_mode = inference_mode
        self._cfg = None
        self._model = None
        self._processor = None
        self._unnormalize_action = None
        self._get_action = None
        self._resize_image = None

    def get_observation_params(self) -> dict[str, Any]:
        return {
            "send_wrist_image": self.num_images_in_input > 1,
            "send_state": True,
            "success_mode": "truncation",
        }

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {"actions": RAW}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        spec: dict[str, DimSpec] = {"agentview": IMAGE_RGB, "language": LANGUAGE, "state": STATE_EEF_POS_AA_GRIP}
        if self.num_images_in_input > 1:
            spec["wrist"] = IMAGE_RGB
        return spec

    @staticmethod
    def _resolve_checkpoint(checkpoint: str) -> str:
        path = Path(checkpoint).expanduser()
        if path.exists():
            return str(path)
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        try:
            return snapshot_download(checkpoint, local_files_only=True)
        except LocalEntryNotFoundError:
            return snapshot_download(checkpoint)

    def _load_model(self) -> None:
        if self._model is not None:
            return

        repo_src = Path(self.repo_path).expanduser() / "src"
        if not repo_src.exists():
            raise FileNotFoundError(
                f"DeepThinkVLA repo source not found at {repo_src}. "
                "Set --repo_path or build the Docker image that clones OpenBMB/DeepThinkVLA."
            )
        sys.path.insert(0, str(repo_src))

        # Avoid executing upstream sft/__init__.py, which imports training-only
        # dataset modules. Eval only needs sft.constants and modeling_deepthinkvla.
        for package_name in ("sft", "dt_datasets"):
            if package_name not in sys.modules:
                package = ModuleType(package_name)
                package.__path__ = [str(repo_src / package_name)]  # type: ignore[attr-defined]
                package.__file__ = str(repo_src / package_name / "__init__.py")
                sys.modules[package_name] = package

        from experiments.deepthinkvla_utils import (  # type: ignore[import-not-found]
            get_vla,
            get_vla_action,
            get_vla_action_mask_cot,
            get_vla_action_mask_cot_random,
            resize_image_for_policy,
        )
        from sft.constants import NUM_ACTIONS_CHUNK  # type: ignore[import-not-found]
        from transformers import AutoProcessor

        checkpoint_path = self._resolve_checkpoint(self.checkpoint)
        self._cfg = SimpleNamespace(
            pretrained_checkpoint=checkpoint_path,
            num_images_in_input=self.num_images_in_input,
            max_new_tokens=self.max_new_tokens,
            compute_dtype=self.compute_dtype,
            img_resize_size=self.img_resize_size,
            unnorm_key=None,
        )

        if self.chunk_size != NUM_ACTIONS_CHUNK:
            raise ValueError(
                f"DeepThinkVLA LIBERO checkpoint emits {NUM_ACTIONS_CHUNK} actions per inference; "
                f"got chunk_size={self.chunk_size}. Use --chunk_size {NUM_ACTIONS_CHUNK}."
            )

        logger.info("Loading DeepThinkVLA from %s", checkpoint_path)
        self._model, self._unnormalize_action = get_vla(self._cfg)
        self._processor = AutoProcessor.from_pretrained(checkpoint_path)
        self._resize_image = resize_image_for_policy
        if self.inference_mode == "full_cot":
            self._get_action = get_vla_action
        elif self.inference_mode == "mask_cot":
            self._get_action = get_vla_action_mask_cot
        elif self.inference_mode == "random_cot":
            self._get_action = get_vla_action_mask_cot_random
        else:
            raise ValueError("inference_mode must be one of: full_cot, mask_cot, random_cot")
        logger.info("DeepThinkVLA loaded successfully (mode=%s, chunk_size=%s)", self.inference_mode, self.chunk_size)

    def _build_observation(self, obs: Observation) -> dict[str, Any]:
        assert self._resize_image is not None
        images = obs.get("images", {})
        if not isinstance(images, dict) or not images:
            raise ValueError("DeepThinkVLA requires obs['images'] with at least an agentview image")

        full = np.asarray(images.get("agentview", next(iter(images.values()))), dtype=np.uint8)
        out: dict[str, Any] = {
            "full_image": self._resize_image(full, self.img_resize_size),
        }

        if self.num_images_in_input > 1:
            wrist = images.get("wrist")
            if wrist is None:
                wrist = np.zeros_like(full)
            out["wrist_image"] = self._resize_image(np.asarray(wrist, dtype=np.uint8), self.img_resize_size)

        state = obs.get("states", obs.get("state"))
        if state is None:
            state = np.zeros(8, dtype=np.float32)
        out["state"] = np.asarray(state, dtype=np.float32)
        return out

    def predict(self, obs: Observation, ctx: SessionContext) -> Action:
        self._load_model()
        assert self._cfg is not None
        assert self._model is not None
        assert self._processor is not None
        assert self._unnormalize_action is not None
        assert self._get_action is not None

        t_pre = time.perf_counter()
        dt_obs = self._build_observation(obs)
        preprocess_ms = (time.perf_counter() - t_pre) * 1000

        t_infer = time.perf_counter()
        actions, _cot_text = self._get_action(
            cfg=self._cfg,
            vla=self._model,
            unomrmalize_action=self._unnormalize_action,
            processor=self._processor,
            obs=dt_obs,
            task_label=obs.get("task_description", ""),
        )
        infer_ms = (time.perf_counter() - t_infer) * 1000
        self._log_latency(ctx, preprocess_ms, infer_ms, interval=1)

        return {"actions": np.asarray(actions, dtype=np.float32)}


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(DeepThinkVLAModelServer)

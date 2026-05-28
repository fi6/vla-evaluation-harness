# /// script
# requires-python = "~=3.10"
# dependencies = [
#     "vla-eval",
#     "torch==2.4.0",
#     "torchvision==0.19.0",
#     "transformers>=4.57.0,<5",
#     "accelerate",
#     "safetensors",
#     "pillow",
#     "numpy",
#     "huggingface_hub",
#     "opencv-python-headless",
#     "fastapi",
#     "uvicorn",
#     "json-numpy",
#     "scipy",
#     "einops",
#     "timm",
#     "num2words",
#     "av",
# ]
#
# [tool.uv.sources]
# vla-eval = { path = "../../..", editable = true }
#
# [tool.uv]
# exclude-newer = "2026-05-28T00:00:00Z"
# ///
"""SimVLA model server using LUOyk1999/SimVLA."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.specs import IMAGE_RGB, LANGUAGE, RAW, STATE_EEF_POS_AA_GRIP, DimSpec
from vla_eval.types import Action, Observation

logger = logging.getLogger(__name__)


class SimVLAModelServer(PredictModelServer):
    """SimVLA LIBERO server.

    Official LIBERO evaluation generates 10 actions and executes the first
    ``replan_steps`` actions before re-querying. The default here mirrors that
    with ``action_horizon=10`` and ``chunk_size=replan_steps=5``.
    """

    def __init__(
        self,
        checkpoint: str = "YuankaiLuo/SimVLA-LIBERO",
        repo_path: str = "/opt/SimVLA",
        norm_stats: str | None = None,
        smolvlm_model: str = "HuggingFaceTB/SmolVLM-500M-Instruct",
        device: str = "cuda",
        image_size: int = 384,
        action_horizon: int = 10,
        replan_steps: int = 5,
        action_ensemble: str = "newest",
        **kwargs: Any,
    ) -> None:
        action_chunk_size = kwargs.pop("chunk_size", None)
        if action_chunk_size is None:
            action_chunk_size = replan_steps
        super().__init__(chunk_size=action_chunk_size, action_ensemble=action_ensemble, **kwargs)
        self.checkpoint = checkpoint
        self.repo_path = repo_path
        self.norm_stats = norm_stats
        self.smolvlm_model = smolvlm_model
        self.device = device
        self.image_size = image_size
        self.action_horizon = action_horizon
        self._model = None
        self._processor = None
        self._torch_device = None

    def get_observation_params(self) -> dict[str, Any]:
        return {
            "send_wrist_image": True,
            "send_state": True,
            "success_mode": "truncation",
        }

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {"actions": RAW}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return {"agentview": IMAGE_RGB, "wrist": IMAGE_RGB, "language": LANGUAGE, "state": STATE_EEF_POS_AA_GRIP}

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

        repo_src = Path(self.repo_path).expanduser()
        if not repo_src.exists():
            raise FileNotFoundError(
                f"SimVLA repo not found at {repo_src}. Set --repo_path or build the Docker image."
            )
        sys.path.insert(0, str(repo_src))

        import torch
        from models.modeling_smolvlm_vla import SmolVLMVLA  # type: ignore[import-not-found]
        from models.processing_smolvlm_vla import SmolVLMVLAProcessor  # type: ignore[import-not-found]

        requested = self.device
        if requested == "cuda" and not torch.cuda.is_available():
            requested = "cpu"
        self._torch_device = torch.device(requested)

        checkpoint_path = self._resolve_checkpoint(self.checkpoint)
        logger.info("Loading SimVLA from %s on %s", checkpoint_path, self._torch_device)
        model = SmolVLMVLA.from_pretrained(checkpoint_path)
        model = model.to(self._torch_device)
        model.eval()

        norm_stats = self.norm_stats
        if norm_stats is None:
            candidate = repo_src / "norm_stats" / "libero_norm.json"
            norm_stats = str(candidate) if candidate.exists() else None
        if norm_stats:
            logger.info("Loading SimVLA norm stats from %s", norm_stats)
            model.action_space.load_norm_stats(norm_stats)

        self._model = model
        self._processor = SmolVLMVLAProcessor.from_pretrained(self.smolvlm_model)
        logger.info(
            "SimVLA loaded successfully (action_horizon=%s, replan_steps=%s)",
            self.action_horizon,
            self.chunk_size,
        )

    def _build_inputs(self, obs: Observation) -> tuple[Any, Any, Any]:
        assert self._processor is not None
        assert self._torch_device is not None
        import torch

        images = obs.get("images", {})
        if not isinstance(images, dict) or not images:
            raise ValueError("SimVLA requires obs['images'] with at least an agentview image")

        agent = np.asarray(images.get("agentview", next(iter(images.values()))), dtype=np.uint8)
        wrist = images.get("wrist")
        if wrist is None:
            wrist = np.zeros_like(agent)
        wrist = np.asarray(wrist, dtype=np.uint8)

        image_inputs = self._processor.encode_image([agent, wrist])
        image_input = image_inputs["image_input"].to(self._torch_device)
        image_mask = image_inputs["image_mask"].to(self._torch_device)

        lang = self._processor.encode_language([obs.get("task_description", "")])
        input_ids = lang["input_ids"].to(self._torch_device)

        state = obs.get("states", obs.get("state"))
        if state is None:
            state = np.zeros(8, dtype=np.float32)
        state_arr = np.asarray(state, dtype=np.float32)
        if state_arr.shape[0] < 8:
            state_arr = np.pad(state_arr, (0, 8 - state_arr.shape[0]))
        state_arr = state_arr[:8]
        proprio = torch.tensor(state_arr, dtype=torch.float32, device=self._torch_device).unsqueeze(0)

        return input_ids, image_input, image_mask, proprio

    def predict(self, obs: Observation, ctx: SessionContext) -> Action:
        self._load_model()
        assert self._model is not None

        t_pre = time.perf_counter()
        input_ids, image_input, image_mask, proprio = self._build_inputs(obs)
        preprocess_ms = (time.perf_counter() - t_pre) * 1000

        import torch

        t_infer = time.perf_counter()
        with torch.inference_mode():
            actions = self._model.generate_actions(
                input_ids=input_ids,
                image_input=image_input,
                image_mask=image_mask,
                proprio=proprio,
                steps=self.action_horizon,
            )
        infer_ms = (time.perf_counter() - t_infer) * 1000
        self._log_latency(ctx, preprocess_ms, infer_ms, interval=1)

        actions_np = actions.detach().cpu().numpy()[0].astype(np.float32)
        return {"actions": actions_np}


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(SimVLAModelServer)

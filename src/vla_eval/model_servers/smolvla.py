# /// script
# requires-python = "~=3.12"
# dependencies = [
#     "vla-eval",
#     "lerobot[smolvla]>=0.5.0",
#     "torch>=2.0",
#     "transformers>=5.3,<6",
#     "pillow>=9.0",
#     "numpy>=1.24",
#     "huggingface_hub",
# ]
#
# [tool.uv.sources]
# vla-eval = { path = "../../..", editable = true }
#
# [tool.uv]
# exclude-newer = "2026-05-27T00:00:00Z"
# ///
"""SmolVLA model server using Hugging Face LeRobot direct inference."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.specs import IMAGE_RGB, LANGUAGE, RAW, STATE_EEF_POS_AA_GRIP, DimSpec
from vla_eval.types import Action, Observation

logger = logging.getLogger(__name__)


class SmolVLAModelServer(PredictModelServer):
    """LeRobot SmolVLA model server.

    The LIBERO checkpoint expects LeRobot feature keys:
    ``observation.images.image`` (agentview), ``observation.images.image2``
    (wrist), ``observation.state`` (8-D proprio), and ``task``.
    """

    def __init__(
        self,
        model_path: str = "HuggingFaceVLA/smolvla_libero",
        image_key: str = "observation.images.image",
        wrist_image_key: str | None = "observation.images.image2",
        state_key: str | None = "observation.state",
        state_dim: int = 8,
        device: str = "cuda",
        robot_type: str = "",
        action_ensemble: str = "newest",
        **kwargs: Any,
    ) -> None:
        # SmolVLA's LeRobot policy manages its own action queue via select_action().
        action_chunk_size = kwargs.pop("chunk_size", None)
        super().__init__(chunk_size=None, action_ensemble=action_ensemble, **kwargs)
        self.model_path = model_path
        self.image_key = image_key
        self.wrist_image_key = None if wrist_image_key in (None, "None", "none", "") else wrist_image_key
        self.state_key = None if state_key in (None, "None", "none", "") else state_key
        self.state_dim = state_dim
        self.device = device
        self.robot_type = robot_type
        self.action_chunk_size = action_chunk_size
        self._policy = None
        self._preprocess = None
        self._postprocess = None
        self._torch_device = None

    def _load_model(self) -> None:
        if self._policy is not None:
            return
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        requested = self.device
        if requested == "cuda" and not torch.cuda.is_available():
            requested = "cpu"
        self._torch_device = torch.device(requested)

        logger.info("Loading SmolVLA from %s on %s", self.model_path, self._torch_device)
        config = PreTrainedConfig.from_pretrained(self.model_path)
        if self.action_chunk_size is not None:
            action_chunk_size = int(self.action_chunk_size)
            if action_chunk_size <= 0:
                raise ValueError(f"chunk_size must be positive, got {action_chunk_size}")
            logger.info(
                "Overriding SmolVLA chunk_size/n_action_steps from %s/%s to %s/%s",
                config.chunk_size,
                config.n_action_steps,
                action_chunk_size,
                action_chunk_size,
            )
            config.chunk_size = action_chunk_size
            config.n_action_steps = action_chunk_size
        policy = SmolVLAPolicy.from_pretrained(self.model_path, config=config)
        policy.to(self._torch_device)
        policy.eval()

        preprocess, postprocess = make_pre_post_processors(
            policy.config,
            self.model_path,
            preprocessor_overrides={"device_processor": {"device": str(self._torch_device)}},
        )

        self._policy = policy
        self._preprocess = preprocess
        self._postprocess = postprocess
        logger.info("SmolVLA loaded successfully")

    def get_observation_params(self) -> dict[str, Any]:
        return {
            "send_wrist_image": self.wrist_image_key is not None,
            "send_state": self.state_key is not None,
            "success_mode": "truncation",
        }

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {"actions": RAW}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        spec: dict[str, DimSpec] = {"agentview": IMAGE_RGB, "language": LANGUAGE}
        if self.wrist_image_key is not None:
            spec["wrist"] = IMAGE_RGB
        if self.state_key is not None:
            spec["state"] = STATE_EEF_POS_AA_GRIP
        return spec

    async def on_episode_start(self, config: dict[str, Any], ctx: SessionContext) -> None:
        self._load_model()
        assert self._policy is not None
        self._policy.reset()
        await super().on_episode_start(config, ctx)

    def _build_lerobot_observation(self, obs: Observation) -> dict[str, Any]:
        assert self._torch_device is not None
        from lerobot.policies.utils import prepare_observation_for_inference

        images = obs.get("images", {})
        if not isinstance(images, dict) or not images:
            raise ValueError("SmolVLA requires obs['images'] with at least an agentview image")

        raw: dict[str, Any] = {}
        raw[self.image_key] = np.asarray(images.get("agentview", next(iter(images.values()))), dtype=np.uint8)

        if self.wrist_image_key is not None:
            wrist = images.get("wrist")
            if wrist is None:
                wrist = np.zeros_like(raw[self.image_key])
            raw[self.wrist_image_key] = np.asarray(wrist, dtype=np.uint8)

        if self.state_key is not None:
            state = obs.get("states", obs.get("state"))
            if state is None:
                state = np.zeros(self.state_dim, dtype=np.float32)
            raw[self.state_key] = np.asarray(state, dtype=np.float32)

        return prepare_observation_for_inference(
            raw,
            self._torch_device,
            task=obs.get("task_description", ""),
            robot_type=self.robot_type,
        )

    @staticmethod
    def _to_numpy_action(action: Any) -> np.ndarray:
        if isinstance(action, dict):
            action = action.get("action", action.get("actions", next(iter(action.values()))))
        if hasattr(action, "detach"):
            action = action.detach().cpu().numpy()
        arr = np.asarray(action, dtype=np.float32)
        if arr.ndim > 1:
            arr = np.squeeze(arr, axis=0) if arr.shape[0] == 1 else arr.reshape(-1)
        return arr.astype(np.float32)

    def predict(self, obs: Observation, ctx: SessionContext) -> Action:
        self._load_model()
        assert self._policy is not None
        assert self._preprocess is not None
        assert self._postprocess is not None

        t_pre = time.perf_counter()
        lerobot_obs = self._build_lerobot_observation(obs)
        batch = self._preprocess(lerobot_obs)
        preprocess_ms = (time.perf_counter() - t_pre) * 1000

        t_infer = time.perf_counter()
        action = self._policy.select_action(batch)
        infer_ms = (time.perf_counter() - t_infer) * 1000
        self._log_latency(ctx, preprocess_ms, infer_ms)

        action = self._postprocess(action)
        return {"actions": self._to_numpy_action(action)}


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(SmolVLAModelServer)

# /// script
# requires-python = "~=3.11"
# dependencies = [
#     "vla-eval",
#     "torch>=2.5.1",
#     "torchvision>=0.20.1",
#     "transformers>=4.57,<4.58",
#     "accelerate",
#     "huggingface_hub>=0.23",
#     "pillow>=9.0",
#     "numpy>=1.24",
#     "einops",
#     "sentencepiece",
#     "protobuf",
#     "safetensors",
# ]
#
# [tool.uv.sources]
# vla-eval = { path = "../../..", editable = true }
#
# [tool.uv]
# exclude-newer = "2026-05-25T00:00:00Z"
# ///
"""MolmoAct2 model server — Molmo2-based VLA with flow-matching action expert.

Uses trust_remote_code=True so no separate molmoact2 package install is needed;
model code is loaded directly from the HuggingFace checkpoint.

Key notes:
- snapshot_download() is required (not the repo-ID string) so the model code can
  locate norm_stats.json on disk.
- Two bf16 patches to modeling_molmoact2.py are required for bfloat16 inference;
  they are idempotent and applied at startup.
- norm_tag must match a key in norm_stats.json inside the checkpoint.  For
  MolmoAct2-LIBERO pass "" (empty) to auto-detect the first available key, or
  set it explicitly after inspecting the checkpoint.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from PIL import Image as PILImage

from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.specs import IMAGE_RGB, LANGUAGE, DimSpec
from vla_eval.types import Action, Observation

logger = logging.getLogger(__name__)

# Patches required for bfloat16 inference in upstream modeling_molmoact2.py.
# Each tuple is (old_snippet, new_snippet); applied once at model-load time.
_BF16_PATCHES: list[tuple[str, str]] = [
    (
        "device=device,\n            dtype=torch.float32,\n            generator=generator,",
        "device=device,\n            dtype=source_tensor.dtype,  # patched_bf16_dtype\n            generator=generator,",
    ),
    (
        "return value.detach().cpu().numpy().astype(np.float32, copy=False)",
        "return value.detach().cpu().float().numpy().astype(np.float32, copy=False)  # patched_bf16_to_array",
    ),
]


def _apply_bf16_patches(local_dir: str) -> None:
    """Patch modeling_molmoact2.py in the checkpoint snapshot for bf16 support."""
    import os

    model_file = os.path.join(local_dir, "modeling_molmoact2.py")
    if not os.path.exists(model_file):
        logger.warning("modeling_molmoact2.py not found at %s — skipping bf16 patches", model_file)
        return
    with open(model_file) as f:
        src = f.read()
    changed = False
    for old, new in _BF16_PATCHES:
        if old in src:
            src = src.replace(old, new)
            changed = True
    if changed:
        with open(model_file, "w") as f:
            f.write(src)
        logger.info("Applied bf16 patches to %s", model_file)


def _detect_norm_tag(local_dir: str) -> str:
    """Return the first key from norm_stats.json (used when norm_tag is not set explicitly)."""
    import json
    import os

    path = os.path.join(local_dir, "norm_stats.json")
    if not os.path.exists(path):
        # Fall back to config.json which may embed norm_stats
        path = os.path.join(local_dir, "config.json")
        with open(path) as f:
            cfg = json.load(f)
        stats = cfg.get("norm_stats", {})
    else:
        with open(path) as f:
            stats = json.load(f)
    keys = list(stats.keys())
    if not keys:
        raise ValueError(f"norm_stats has no keys in {local_dir}; pass --norm_tag explicitly")
    if len(keys) > 1:
        logger.warning("norm_stats has multiple keys %s — using %r; pass --norm_tag to override", keys, keys[0])
    return keys[0]


class MolmoAct2ModelServer(PredictModelServer):
    """MolmoAct2 VLA model server.

    Args:
        hf_repo: HuggingFace repo ID for the checkpoint (downloaded via snapshot_download).
        norm_tag: Dataset normalization key inside norm_stats.json.  Pass "" (default) to
            auto-detect the first key in the checkpoint's norm_stats.
        state_dim: Dimension of proprioceptive state to pass to the model.  0 = no state
            (zeros passed).  For LIBERO with EEF state pass 8 and set send_state=True in
            the benchmark config.
        chunk_size: Number of future actions predicted per model call (default 10).
    """

    def __init__(
        self,
        hf_repo: str = "allenai/MolmoAct2-LIBERO",
        norm_tag: str = "",
        state_dim: int = 0,
        chunk_size: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__(chunk_size=chunk_size, **kwargs)
        self.hf_repo = hf_repo
        self.norm_tag = norm_tag
        self.state_dim = state_dim
        self._model: Any = None
        self._processor: Any = None
        self._local_dir: str | None = None
        self._resolved_norm_tag: str | None = None

    # -- lifecycle --------------------------------------------------------

    def _load_model(self) -> None:
        if self._model is not None:
            return

        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForImageTextToText, AutoProcessor

        logger.info("Resolving MolmoAct2 checkpoint %s via snapshot_download", self.hf_repo)
        local_dir = snapshot_download(repo_id=self.hf_repo)
        self._local_dir = local_dir

        _apply_bf16_patches(local_dir)

        logger.info("Loading MolmoAct2 processor from %s", local_dir)
        self._processor = AutoProcessor.from_pretrained(
            local_dir,
            trust_remote_code=True,
            extra_special_tokens={},  # required for transformers >= 4.46
        )

        logger.info("Loading MolmoAct2 model from %s", local_dir)
        self._model = AutoModelForImageTextToText.from_pretrained(
            local_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        # Override _move_inputs_to_device so float tensors are cast to bfloat16
        target_dtype = torch.bfloat16

        def _move_and_cast(inputs: Any, dev: Any, _dtype: torch.dtype = target_dtype) -> dict[str, Any]:
            out: dict[str, Any] = {}
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    v = v.to(dev)
                    if v.is_floating_point() and v.dtype != _dtype:
                        v = v.to(_dtype)
                out[k] = v
            return out

        self._model._move_inputs_to_device = _move_and_cast
        self._model = self._model.eval()

        self._resolved_norm_tag = self.norm_tag or _detect_norm_tag(local_dir)
        logger.info("MolmoAct2 loaded. norm_tag=%r", self._resolved_norm_tag)

    # -- specs ------------------------------------------------------------

    def get_action_spec(self) -> dict[str, DimSpec]:
        from vla_eval.specs import RAW

        return {"actions": RAW}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return {"image": IMAGE_RGB, "language": LANGUAGE}

    def get_observation_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if self.state_dim > 0:
            params["send_state"] = True
        return params

    # -- inference --------------------------------------------------------

    def predict(self, obs: Observation, ctx: SessionContext) -> Action:
        import torch

        self._load_model()
        assert self._model is not None and self._processor is not None

        # Build PIL image list in observation-insertion order (agentview first, wrist second)
        images_dict = obs.get("images", {})
        images = [
            PILImage.fromarray(np.asarray(arr, dtype=np.uint8)).convert("RGB")
            for arr in images_dict.values()
            if arr is not None
        ]
        if not images:
            raise ValueError("MolmoAct2ModelServer: no images in observation")

        task = obs.get("task_description", "")

        # Build state vector
        if self.state_dim > 0:
            raw_state = obs.get("state")
            state = (
                np.asarray(raw_state, dtype=np.float32)
                if raw_state is not None
                else np.zeros(self.state_dim, dtype=np.float32)
            )
        else:
            state = np.zeros(1, dtype=np.float32)  # placeholder; ignored by model when state_dim=0

        with torch.inference_mode():
            out = self._model.predict_action(
                processor=self._processor,
                images=images,
                task=task,
                state=state,
                norm_tag=self._resolved_norm_tag,
                action_mode="continuous",
                num_steps=self.chunk_size or 10,
                normalize_language=True,
                enable_cuda_graph=False,
            )

        raw = out.actions if hasattr(out, "actions") else out
        if torch.is_tensor(raw):
            raw = raw.detach().to(dtype=torch.float32, device="cpu").numpy()
        actions = np.asarray(raw, dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]  # remove batch dim

        return {"actions": actions}


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(MolmoAct2ModelServer)

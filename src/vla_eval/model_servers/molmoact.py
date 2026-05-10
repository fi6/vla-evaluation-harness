# /// script
# requires-python = "~=3.11"
# dependencies = [
#     "vla-eval",
#     "torch>=2.3.1",
#     "torchvision>=0.18.1",
#     "einops",
#     "timm",
#     "transformers==4.52.3",
#     "accelerate",
#     "huggingface_hub",
#     "pillow>=9.0",
#     "numpy>=1.24",
# ]
#
# [tool.uv.sources]
# vla-eval = { path = "../../..", editable = true }
#
# [tool.uv]
# exclude-newer = "2026-05-07T00:00:00Z"
# ///
"""MolmoAct model server — Molmo2-7B fine-tuned for robot manipulation.

Outputs action chunks via chain-of-thought reasoning (depth → trace → action).
All model code is loaded via trust_remote_code=True from HuggingFace;
no separate molmoact package install is required.

unnorm_key must match the dataset the checkpoint was fine-tuned on:
  libero_spatial  → libero_spatial_no_noops_modified
  libero_object   → libero_object_no_noops_modified
  libero_goal     → libero_goal_no_noops_modified
  libero_10       → libero_10_no_noops_modified
Use the per-suite configs under configs/model_servers/molmoact/.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from PIL import Image as PILImage

from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.specs import IMAGE_RGB, LANGUAGE, RAW, DimSpec
from vla_eval.types import Action, Observation

logger = logging.getLogger(__name__)

# Exact prompt template from the official MolmoAct test script.
# Do not modify — any deviation from training-time prompting degrades performance.
_PROMPT_TEMPLATE = (
    "The task is {instruction}. "
    "What is the action that the robot should take. "
    "To figure out the action that the robot should take to {instruction}, "
    "let's think through it step by step. "
    "First, what is the depth map for the first image? "
    "Second, what is the trajectory of the end effector in the first image? "
    "Based on the depth map of the first image and the trajectory of the end effector in the first image, "
    "along with other images from different camera views as additional information, "
    "what is the action that the robot should take?"
)


class MolmoActModelServer(PredictModelServer):
    """MolmoAct VLA model server.

    Args:
        hf_repo: HuggingFace repo ID or local path for the checkpoint.
        unnorm_key: Dataset key used to unnormalize predicted actions.
            Must match the fine-tuning dataset for the loaded checkpoint.
        chunk_size: Number of actions returned per generate() call (default 8).
        action_ensemble: How overlapping chunks are blended (default newest).
    """

    def __init__(
        self,
        hf_repo: str = "allenai/MolmoAct-7B-D-0812",
        unnorm_key: str = "molmoact",
        *,
        chunk_size: int = 8,
        action_ensemble: str = "newest",
        **kwargs: Any,
    ) -> None:
        super().__init__(chunk_size=chunk_size, action_ensemble=action_ensemble, **kwargs)
        self.hf_repo = hf_repo
        self.unnorm_key = unnorm_key
        self._model: Any = None
        self._processor: Any = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForImageTextToText, AutoProcessor

        logger.info("Loading MolmoAct processor from %s", self.hf_repo)
        self._processor = AutoProcessor.from_pretrained(
            self.hf_repo,
            trust_remote_code=True,
            torch_dtype="bfloat16",
            device_map="auto",
            padding_side="left",
        )
        logger.info("Loading MolmoAct model from %s", self.hf_repo)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.hf_repo,
            trust_remote_code=True,
            torch_dtype="bfloat16",
            device_map="auto",
        )
        logger.info("MolmoAct loaded successfully.")

    # -- specs ------------------------------------------------------------

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {"actions": RAW}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return {"image": IMAGE_RGB, "language": LANGUAGE}

    def get_observation_params(self) -> dict[str, Any]:
        # Agentview only; MolmoAct does not require wrist image or proprioceptive state.
        return {}

    # -- inference --------------------------------------------------------

    def predict(self, obs: Observation, ctx: SessionContext) -> Action:
        import time

        import torch

        self._load_model()
        assert self._model is not None and self._processor is not None

        t_pre = time.perf_counter()
        # Build PIL image list from all cameras in observation-dict insertion order.
        images_dict = obs.get("images", {})
        imgs = [PILImage.fromarray(np.asarray(arr, dtype=np.uint8)).convert("RGB") for arr in images_dict.values()]
        if not imgs:
            raise ValueError("MolmoActModelServer: no images in observation")

        instruction = obs.get("task_description", "")
        prompt = _PROMPT_TEMPLATE.format(instruction=instruction)

        text = self._processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            tokenize=False,
            add_generation_prompt=True,
        )

        # images=[imgs]: nested list — one batch sample containing multiple images.
        inputs = self._processor(images=[imgs], text=text, padding=True, return_tensors="pt")

        # device_map="auto" spreads layers across devices; use the first parameter's
        # device rather than a non-existent model.device attribute.
        try:
            _device = next(self._model.parameters()).device
        except StopIteration:
            import torch as _torch

            _device = _torch.device("cpu")
        inputs = {k: v.to(_device) for k, v in inputs.items()}

        preprocess_ms = (time.perf_counter() - t_pre) * 1000
        t_infer = time.perf_counter()
        with torch.inference_mode():
            if _device.type == "cuda":
                with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                    generated_ids = self._model.generate(**inputs, max_new_tokens=512)
            else:
                generated_ids = self._model.generate(**inputs, max_new_tokens=512)
        self._log_latency(ctx, preprocess_ms, (time.perf_counter() - t_infer) * 1000)

        # Strip the prompt prefix; decode only the newly generated tokens.
        generated_tokens = generated_ids[:, inputs["input_ids"].size(1) :]
        generated_text = self._processor.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        actions_list = self._model.parse_action(generated_text, unnorm_key=self.unnorm_key)

        if not actions_list:
            logger.warning(
                "MolmoAct parse_action returned empty list at step=%d. "
                "Returning zero action. Generated text snippet: %r",
                ctx.step,
                generated_text[:200],
            )
            return {"actions": np.zeros((1, 7), dtype=np.float32)}

        actions = np.array(actions_list, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[np.newaxis, :]

        return {"actions": actions}


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(MolmoActModelServer)

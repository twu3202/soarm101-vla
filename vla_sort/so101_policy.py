"""openpi policy transforms for the SO-ARM101 single-arm manipulation tasks.

Adapted from openpi.policies.libero_policy (which says "copy this for your own dataset").
Our LeRobot dataset (so101_follower, codebase v2.1) provides:
  observation.state          float32[6]   (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper)
  action                     float32[6]   (same 6 joints; absolute target positions in degrees)
  observation.images.top     video 480x640x3   (top-down camera)              -> base_0_rgb
  observation.images.wrist   video 480x640x3   (gripper camera, OPTIONAL)     -> left_wrist_0_rgb

This file supports BOTH:
  * 1-camera tasks (the old cube-sort): only observation/image present -> wrist slot zero+masked.
  * 2-camera tasks (the green-bowl pick): observation/image (top) + observation/wrist_image (wrist)
    -> top to base_0_rgb, wrist to left_wrist_0_rgb (real image, UNMASKED).
The right_wrist slot is always zero+masked (we never have a 3rd camera).

INSTALL: drop this file at openpi/src/openpi/policies/so101_policy.py so it imports as
`from openpi.policies import so101_policy`.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# SO-ARM101 action / state dimensionality (5 joints + gripper).
SO101_DIM = 6


def make_so101_example() -> dict:
    """A random input example (used by openpi for shape/wiring checks)."""
    return {
        "observation/state": np.random.rand(SO101_DIM),
        "observation/image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "prompt": "put the green cube in the bowl",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:  # LeRobot stores video frames as (C,H,W); model wants (H,W,C)
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class So101Inputs(transforms.DataTransformFn):
    """Convert a dataset/inference sample into the model's expected input dict.
    Used for BOTH training and inference, so the inference server must pass the same keys
    (observation/image, observation/state[, prompt]) — see the deploy script.
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])

        # Wrist camera is OPTIONAL: present for the 2-cam green-bowl task, absent for the
        # old 1-cam sort task. When present we feed it real + UNMASKED; when absent we
        # zero-pad and mask it off (pi0.5 -> mask=False for padded views).
        has_wrist = "observation/wrist_image" in data and data["observation/wrist_image"] is not None
        if has_wrist:
            wrist_image = _parse_image(data["observation/wrist_image"])
        else:
            wrist_image = np.zeros_like(base_image)

        pad_mask = np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # No 3rd camera -> always zero-padded + masked.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_ if has_wrist else pad_mask,
                "right_wrist_0_rgb": pad_mask,
            },
        }

        if "actions" in data:  # only present during training
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class So101Outputs(transforms.DataTransformFn):
    """Convert model outputs back to the dataset action space (inference only)."""

    def __call__(self, data: dict) -> dict:
        # The model emits actions padded to its internal action_dim; keep our first 6.
        return {"actions": np.asarray(data["actions"][:, :SO101_DIM])}

"""Observation wrapper for vision PPO — keeps state + cube_positions + images,
but reshapes images for SB3's NatureCNN (channel-first).

SB3's MultiInputPolicy + CombinedExtractor can handle Dict obs natively, but it
expects images in CHW format and float64 → uint8 channel order needs care.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class VisionPPOWrapper(gym.ObservationWrapper):
    """Keep state + cube_positions + images, transposing HWC → CHW for SB3."""

    def __init__(self, env: gym.Env, include_wrist: bool = True):
        super().__init__(env)
        if not isinstance(env.observation_space, spaces.Dict):
            raise TypeError("VisionPPOWrapper expects Dict obs")
        self.include_wrist = include_wrist

        spaces_dict = {
            "state": env.observation_space["state"],
            "cube_positions": env.observation_space["cube_positions"],
        }
        # Transpose image spaces HWC → CHW
        for k in ("image_front",) + (("image_wrist",) if include_wrist else ()):
            if k not in env.observation_space.spaces:
                continue
            old = env.observation_space[k]
            h, w, c = old.shape
            spaces_dict[k] = spaces.Box(low=0, high=255, shape=(c, h, w), dtype=np.uint8)

        self.observation_space = spaces.Dict(spaces_dict)

    def observation(self, obs):
        out = {
            "state": obs["state"].astype(np.float32),
            "cube_positions": obs["cube_positions"].astype(np.float32),
        }
        if "image_front" in self.observation_space.spaces:
            out["image_front"] = np.transpose(obs["image_front"], (2, 0, 1)).astype(np.uint8)
        if "image_wrist" in self.observation_space.spaces and self.include_wrist:
            out["image_wrist"] = np.transpose(obs["image_wrist"], (2, 0, 1)).astype(np.uint8)
        return out

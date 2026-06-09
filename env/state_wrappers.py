"""Observation wrappers for state-only PPO training.

`FlatStateWrapper` flattens the env's Dict observation into a single 1-D array so
SB3's MlpPolicy can consume it directly. Drops images entirely.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class FlatStateWrapper(gym.ObservationWrapper):
    """state(25) + cube_positions(5,3) → flat 40-dim vector."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # Capture original shapes
        if not isinstance(env.observation_space, spaces.Dict):
            raise TypeError("FlatStateWrapper expects Dict obs")
        state_dim = env.observation_space["state"].shape[0]
        cube_dim = int(np.prod(env.observation_space["cube_positions"].shape))
        total = state_dim + cube_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total,), dtype=np.float32
        )
        self._state_dim = state_dim
        self._cube_dim = cube_dim

    def observation(self, obs):
        state = obs["state"].astype(np.float32)
        cube = obs["cube_positions"].astype(np.float32).reshape(-1)
        return np.concatenate([state, cube])

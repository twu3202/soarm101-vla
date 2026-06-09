"""Standalone eval + video generation for SortCubesSO100-v1.

Loads a trained checkpoint, runs N episodes with rendering, saves MP4 per episode.
Reports per-episode n_placed, success, and which cubes ended up where.
This is INDEPENDENT of the training process to avoid SAPIEN render pipeline crashes
that happen when both train+eval envs use RecordEpisode simultaneously.

Usage:
    python eval_video.py --checkpoint=runs/sort_n5_v7_rebal/ckpt_226.pt \
        --num_active_cubes=5 --n_episodes=4 --output_dir=videos/sort_n5_v7
"""
from __future__ import annotations
import argparse
import os
import sys

# Windows SAPIEN CUDA workaround (BEFORE any sapien import)
if os.path.exists(r"D:\soarm101\cuda_workaround"):
    os.add_dll_directory(r"D:\soarm101\cuda_workaround")
_cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"
if os.path.exists(_cuda_bin):
    os.add_dll_directory(_cuda_bin)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import mani_skill.envs
import sort_cubes_env  # noqa  registers SortCubesSO100-v1
from mani_skill.utils.wrappers.record import RecordEpisode


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """Same agent architecture as ppo.py."""
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, act_dim), std=0.01 * np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, act_dim) * -0.5)

    def get_action(self, obs, deterministic=False):
        mean = self.actor_mean(obs)
        if deterministic:
            return mean
        std = self.actor_logstd.expand_as(mean).exp()
        from torch.distributions.normal import Normal
        dist = Normal(mean, std)
        return dist.sample()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--num_active_cubes", type=int, default=5)
    p.add_argument("--randomize_active_color", type=int, default=0)
    p.add_argument("--flexible_order", type=int, default=0)
    p.add_argument("--n_episodes", type=int, default=4)
    p.add_argument("--output_dir", default="videos/eval")
    p.add_argument("--max_steps", type=int, default=250)
    p.add_argument("--deterministic", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create env with rgb_array rendering for RecordEpisode
    env = gym.make(
        "SortCubesSO100-v1",
        num_envs=1,
        obs_mode="state",
        render_mode="rgb_array",
        sim_backend="physx_cuda",
        num_active_cubes=args.num_active_cubes,
        randomize_active_color=bool(args.randomize_active_color),
        flexible_order=bool(args.flexible_order),
    )

    # Wrap with RecordEpisode to save MP4
    env = RecordEpisode(
        env,
        output_dir=args.output_dir,
        save_trajectory=False,
        max_steps_per_video=args.max_steps,
        video_fps=30,
        save_video=True,
    )

    # Wrap with ManiSkillVectorEnv to match training obs dim (flattens via vector wrapper)
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
    env = ManiSkillVectorEnv(env, 1, ignore_terminations=True, record_metrics=True)

    # Reset to get obs shape, build agent
    obs, info = env.reset(seed=args.seed)
    obs_dim = obs.shape[-1] if hasattr(obs, "shape") else obs.numel()
    if isinstance(obs, torch.Tensor):
        obs_dim = obs.shape[-1]
    act_dim = env.action_space.shape[-1]
    print(f"obs_dim={obs_dim}, act_dim={act_dim}")

    agent = Agent(obs_dim, act_dim).to(device)
    sd = torch.load(args.checkpoint, map_location=device)
    agent.load_state_dict(sd)
    agent.eval()
    print(f"Loaded {args.checkpoint}")

    # Run N episodes
    summary = []
    for ep in range(args.n_episodes):
        obs, info = env.reset(seed=args.seed + ep)
        if isinstance(obs, dict):
            obs = obs  # state mode returns tensor, not dict, but just in case
        max_n_placed = 0
        for t in range(args.max_steps):
            with torch.no_grad():
                obs_t = obs if isinstance(obs, torch.Tensor) else torch.tensor(obs, device=device)
                if obs_t.dim() == 1:
                    obs_t = obs_t.unsqueeze(0)
                action = agent.get_action(obs_t, deterministic=bool(args.deterministic))
            obs, rew, term, trunc, info = env.step(action)
            n_placed = int(info["n_placed"].item()) if "n_placed" in info else 0
            max_n_placed = max(max_n_placed, n_placed)
            if bool(term.item() if hasattr(term, "item") else term) or bool(trunc.item() if hasattr(trunc, "item") else trunc):
                break
        success = bool(info["success"].item()) if "success" in info else False
        # Final cube placement details
        in_slot = info.get("in_slot", None)
        if in_slot is not None:
            in_slot_arr = in_slot.cpu().numpy().flatten().tolist()
        else:
            in_slot_arr = []
        summary.append({
            "ep": ep,
            "success": success,
            "max_n_placed": max_n_placed,
            "in_slot_per_color": in_slot_arr,
        })
        color_names = ["R", "O", "Y", "G", "B"]
        placement_str = " ".join(
            f"{c}={'✓' if i < len(in_slot_arr) and in_slot_arr[i] else '✗'}"
            for i, c in enumerate(color_names)
        )
        print(f"ep {ep}: success={success}  max_n_placed={max_n_placed}  [{placement_str}]")

    env.close()
    print(f"\nVideos saved to {os.path.abspath(args.output_dir)}")
    print("Summary:")
    for s in summary:
        print(f"  ep={s['ep']} success={s['success']} max_n_placed={s['max_n_placed']}")


if __name__ == "__main__":
    main()

"""Benchmark a state-policy ckpt against multiple DR noise levels.

Reports success rate, max_n_placed mean, partial-success thresholds at each level.
Useful to characterize "sim2real robustness curve" — how much noise does the policy
tolerate before performance drops significantly.

Usage:
    python eval_benchmark.py --checkpoint=runs/sort_n5_v14_longep/ckpt_626.pt \
        --n_episodes=64 --output_csv=bench_v14.csv
"""
from __future__ import annotations
import argparse
import os
import sys
import csv

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
import sort_cubes_env  # noqa
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_dim=42, act_dim=6):
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

    @torch.no_grad()
    def get_action(self, obs, deterministic=True):
        return self.actor_mean(obs)


# DR levels: (name, obs_noise, action_noise, physics, cube_size_jitter)
LEVELS = [
    ("no_noise",       0.000, 0.000, False, 0.00),
    ("light_obs",      0.002, 0.000, False, 0.00),
    ("light_action",   0.000, 0.010, False, 0.00),
    ("medium",         0.003, 0.015, True,  0.05),
    ("heavy",          0.005, 0.025, True,  0.10),
    ("very_heavy",     0.008, 0.040, True,  0.15),
]


def eval_one_config(ckpt, n_episodes, level_name, obs_noise, action_noise, physics, size_jitter,
                    max_steps=350, num_envs=32, device="cuda", seed=42):
    env = gym.make(
        "SortCubesSO100-v1",
        num_envs=num_envs,
        obs_mode="state",
        sim_backend="physx_cuda",
        num_active_cubes=5,
        prefill_min=0, prefill_max=0,
        fix_wrist_roll=True,
        dr_obs_noise_std=float(obs_noise),
        dr_action_noise_std=float(action_noise),
        dr_physics=bool(physics),
        dr_cube_size_jitter=float(size_jitter),
    )
    env = ManiSkillVectorEnv(env, num_envs, ignore_terminations=True, record_metrics=True)
    obs, info = env.reset(seed=seed)
    obs_dim = obs.shape[-1]
    act_dim = env.single_action_space.shape[-1]

    agent = Agent(obs_dim, act_dim).to(device)
    sd = torch.load(ckpt, map_location=device)
    agent.load_state_dict(sd)
    agent.eval()

    completed = 0
    successes = []
    max_n_placed_list = []
    # Per-env max_n_placed tracker
    max_n_placed_track = torch.zeros(num_envs, dtype=torch.long, device=device)

    total_episodes_needed = n_episodes
    while completed < total_episodes_needed:
        for t in range(max_steps):
            action = agent.get_action(obs, deterministic=True)
            obs, rew, term, trunc, info = env.step(action)
            if "n_placed" in info:
                max_n_placed_track = torch.maximum(max_n_placed_track, info["n_placed"].long())
            if "final_info" in info:
                mask = info["_final_info"]
                if mask.any():
                    # Record completed episodes
                    ep_max = max_n_placed_track[mask].cpu().numpy()
                    success_arr = (ep_max >= 5)
                    for s, m in zip(success_arr, ep_max):
                        successes.append(bool(s))
                        max_n_placed_list.append(int(m))
                    # Reset tracker for those envs
                    max_n_placed_track = max_n_placed_track * (~mask).long()
                    completed += int(mask.sum())
                    if completed >= total_episodes_needed:
                        break
        if completed < total_episodes_needed:
            # Reset env for more episodes
            obs, info = env.reset(seed=seed + completed)
            max_n_placed_track = torch.zeros(num_envs, dtype=torch.long, device=device)

    env.close()
    # Truncate to exact count
    successes = successes[:n_episodes]
    max_n_placed_list = max_n_placed_list[:n_episodes]
    success_rate = sum(successes) / len(successes)
    mean_max = sum(max_n_placed_list) / len(max_n_placed_list)
    thresholds = {k: sum(1 for x in max_n_placed_list if x >= k) / len(max_n_placed_list)
                  for k in range(1, 6)}
    return {
        "level": level_name,
        "n_eps": len(successes),
        "success_rate": success_rate,
        "max_n_placed_mean": mean_max,
        **{f"reached_{k}plus": thresholds[k] for k in range(1, 6)},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n_episodes", type=int, default=64)
    p.add_argument("--num_envs", type=int, default=32)
    p.add_argument("--output_csv", default=None)
    p.add_argument("--max_steps", type=int, default=350)
    args = p.parse_args()

    print(f"Benchmarking {args.checkpoint}")
    print(f"  n_episodes per level: {args.n_episodes}")
    results = []
    for level in LEVELS:
        name, obs_n, act_n, phys, sz = level
        print(f"\n--- Level: {name} (obs_n={obs_n}, act_n={act_n}, phys={phys}, sz={sz}) ---")
        r = eval_one_config(
            args.checkpoint, args.n_episodes, name, obs_n, act_n, phys, sz,
            max_steps=args.max_steps, num_envs=args.num_envs,
        )
        print(f"  success: {r['success_rate']*100:.1f}%  max_placed_mean: {r['max_n_placed_mean']:.2f}")
        print(f"  reached  1+: {r['reached_1plus']*100:.1f}%  2+: {r['reached_2plus']*100:.1f}%  "
              f"3+: {r['reached_3plus']*100:.1f}%  4+: {r['reached_4plus']*100:.1f}%  5+: {r['reached_5plus']*100:.1f}%")
        results.append(r)

    print("\n=== Summary ===")
    print(f"{'level':<14} {'succ':>8} {'mean':>6} {'1+':>6} {'2+':>6} {'3+':>6} {'4+':>6} {'5+':>6}")
    for r in results:
        print(f"{r['level']:<14} {r['success_rate']*100:>7.1f}% "
              f"{r['max_n_placed_mean']:>6.2f} "
              f"{r['reached_1plus']*100:>5.1f}% "
              f"{r['reached_2plus']*100:>5.1f}% "
              f"{r['reached_3plus']*100:>5.1f}% "
              f"{r['reached_4plus']*100:>5.1f}% "
              f"{r['reached_5plus']*100:>5.1f}%")

    if args.output_csv:
        with open(args.output_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            for r in results:
                w.writerow(r)
        print(f"\nSaved CSV: {args.output_csv}")


if __name__ == "__main__":
    main()

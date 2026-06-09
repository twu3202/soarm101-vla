"""Text-trace eval for SortCubesSO100-v1.

Runs a checkpoint and prints (and saves) per-step state info — no rendering
because SAPIEN's render pipeline hangs on this Windows install. The trace
includes tcp position, all 5 cube positions, current_color_idx, action, reward,
and key events (grasp/place transitions).

Usage:
    python eval_trace.py --checkpoint=runs/sort_n5_v8_rcolor_ws/ckpt_376.pt \
        --num_active_cubes=5 --n_episodes=4
"""
from __future__ import annotations
import argparse
import os
import sys
import json

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
    p.add_argument("--prefill_min", type=int, default=0)
    p.add_argument("--prefill_max", type=int, default=0)
    p.add_argument("--n_episodes", type=int, default=2)
    p.add_argument("--max_steps", type=int, default=250)
    p.add_argument("--deterministic", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--print_every", type=int, default=10)
    p.add_argument("--output_dir", default="traces")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = gym.make(
        "SortCubesSO100-v1",
        num_envs=1,
        obs_mode="state",
        sim_backend="physx_cuda",
        num_active_cubes=args.num_active_cubes,
        randomize_active_color=bool(args.randomize_active_color),
        prefill_min=args.prefill_min,
        prefill_max=args.prefill_max,
    )
    env = ManiSkillVectorEnv(env, 1, ignore_terminations=True, record_metrics=True)

    obs, info = env.reset(seed=args.seed)
    obs_dim = obs.shape[-1]
    act_dim = env.single_action_space.shape[-1]
    print(f"obs_dim={obs_dim}, act_dim={act_dim}")

    agent = Agent(obs_dim, act_dim).to(device)
    sd = torch.load(args.checkpoint, map_location=device)
    agent.load_state_dict(sd)
    agent.eval()
    print(f"Loaded {args.checkpoint}")
    print(f"actor_logstd: {sd['actor_logstd'].tolist()}")
    print(f"  -> std: {sd['actor_logstd'].exp().tolist()}")

    color_names = ["R", "O", "Y", "G", "B"]
    SLOT_XY = sort_cubes_env.SLOT_XY  # (5, 2)

    summary = []
    for ep in range(args.n_episodes):
        obs, info = env.reset(seed=args.seed + ep)
        trace_lines = [f"=== Episode {ep} (seed={args.seed + ep}) ==="]
        ckpt_name = os.path.basename(args.checkpoint).replace(".pt", "")
        trace_lines.append(f"ckpt: {ckpt_name}, prefill=[{args.prefill_min},{args.prefill_max}]")
        # Initial state
        init_info = info
        init_n_placed = int(init_info["n_placed"].item())
        init_current = int(init_info["current_color_idx"].item())
        trace_lines.append(f"INIT: n_placed={init_n_placed}, current_color={color_names[init_current]}")
        # obs layout (42 dim): qpos(6) + tcp_to_obj(3) + cube_positions(15) + onehot(5) + slot_xy(10) + tcp_pos(3)
        # indices: qpos 0-5, tcp_to_obj 6-8, cube_pos 9-23, onehot 24-28, slot_xy 29-38, tcp_pos 39-41
        cube_pos = obs[0, 9:24].cpu().numpy().reshape(5, 3)
        for ci in range(5):
            x, y, z = cube_pos[ci]
            sx, sy = SLOT_XY[ci]
            d_to_slot = float(np.hypot(x - sx, y - sy))
            trace_lines.append(f"  cube_{color_names[ci]}: xyz=({x:.3f}, {y:.3f}, {z:.3f})  slot=({sx:.2f}, {sy:.2f})  dist={d_to_slot:.3f}")

        max_n_placed = init_n_placed
        last_current = init_current
        last_n_placed = init_n_placed

        for t in range(args.max_steps):
            with torch.no_grad():
                obs_t = obs if isinstance(obs, torch.Tensor) else torch.tensor(obs, device=device)
                if obs_t.dim() == 1:
                    obs_t = obs_t.unsqueeze(0)
                action = agent.get_action(obs_t, deterministic=bool(args.deterministic))
            obs, rew, term, trunc, info = env.step(action)

            n_placed = int(info["n_placed"].item()) if "n_placed" in info else 0
            current = int(info["current_color_idx"].item())
            tcp = obs[0, 39:42].cpu().numpy()  # tcp_pos
            cube_pos = obs[0, 9:24].cpu().numpy().reshape(5, 3)
            r = float(rew.item())

            # Print every print_every steps OR on key events
            event = ""
            if n_placed != last_n_placed:
                event = f"  *** PLACED {color_names[last_current]} (n_placed: {last_n_placed} -> {n_placed}) ***"
            if current != last_current:
                event += f"  (current switched to {color_names[current]})"

            if t % args.print_every == 0 or event:
                cur_cube_xy = cube_pos[current, :2]
                cur_slot_xy = SLOT_XY[current]
                dist_cur_to_slot = float(np.hypot(*(cur_cube_xy - cur_slot_xy)))
                tcp_to_cur = float(np.linalg.norm(cube_pos[current] - tcp))
                line = (f"step {t:3d}: cur={color_names[current]} "
                        f"tcp=({tcp[0]:.2f},{tcp[1]:.2f},{tcp[2]:.2f}) "
                        f"cube_{color_names[current]}=({cube_pos[current][0]:.2f},{cube_pos[current][1]:.2f},{cube_pos[current][2]:.2f}) "
                        f"dist_to_cube={tcp_to_cur:.3f} cube_to_slot={dist_cur_to_slot:.3f} "
                        f"r={r:+.2f} n_placed={n_placed}")
                trace_lines.append(line + event)

            max_n_placed = max(max_n_placed, n_placed)
            last_current = current
            last_n_placed = n_placed

            if bool(term.item()) or bool(trunc.item()):
                trace_lines.append(f"  EPISODE END at step {t}, success={bool(info['success'].item())}")
                break

        # Final cube placement
        final_in_slot = info["in_slot"].cpu().numpy().flatten().tolist()
        trace_lines.append("FINAL placement:")
        for ci in range(5):
            mark = "✓" if ci < len(final_in_slot) and final_in_slot[ci] else "✗"
            cube_pos_final = cube_pos[ci]
            slot_xy = SLOT_XY[ci]
            d = float(np.hypot(cube_pos_final[0] - slot_xy[0], cube_pos_final[1] - slot_xy[1]))
            trace_lines.append(f"  {color_names[ci]} in_slot={mark}  cube=({cube_pos_final[0]:.3f},{cube_pos_final[1]:.3f},{cube_pos_final[2]:.3f})  dist_to_slot={d:.3f}")
        trace_lines.append(f"SUMMARY ep {ep}: max_n_placed={max_n_placed}  success={bool(info['success'].item())}")

        # Save trace
        out_path = os.path.join(args.output_dir, f"trace_ep{ep}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(trace_lines))
        # Print summary lines to console
        print("\n".join(trace_lines[:30]))  # first 30 lines
        print(f"... (full trace saved to {out_path})")
        summary.append({"ep": ep, "max_n_placed": max_n_placed, "success": bool(info["success"].item())})

    env.close()
    print("\n=== Overall summary ===")
    for s in summary:
        print(f"  ep {s['ep']}: success={s['success']} max_n_placed={s['max_n_placed']}")


if __name__ == "__main__":
    main()

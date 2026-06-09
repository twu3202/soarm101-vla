"""Vision policy distillation from a trained state policy (Phase C).

Architecture:
- Teacher: state policy loaded from ckpt — frozen. Input: 42-dim state vector.
- Student: vision policy — CNN encoder (per camera) + MLP, takes RGB images + qpos.
- Loss: MSE between teacher.actor_mean(state) and student(rgb, qpos).

During training:
- Student executes in env (so reset distribution = deploy distribution).
- Teacher computes target action from privileged state at each step.
- Optimize student to match teacher action.
- No PPO update — pure supervised distillation.

Usage:
    python ppo_distill.py --teacher_ckpt=runs/sort_n5_v15_DR/ckpt_251.pt \
        --total_timesteps=5000000 --num_envs=64 --exp_name=distill_v1
"""
from __future__ import annotations
import os
import sys
import argparse
import time
from collections import defaultdict

if os.path.exists(r"D:\soarm101\cuda_workaround"):
    os.add_dll_directory(r"D:\soarm101\cuda_workaround")
_cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"
if os.path.exists(_cuda_bin):
    os.add_dll_directory(_cuda_bin)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import mani_skill.envs
import sort_cubes_env  # noqa
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    if hasattr(layer, "weight"):
        nn.init.orthogonal_(layer.weight, std)
    if hasattr(layer, "bias") and layer.bias is not None:
        nn.init.constant_(layer.bias, bias_const)
    return layer


class StateTeacher(nn.Module):
    """Same as ppo.py Agent (42-dim state in, 6-dim action out)."""
    def __init__(self, state_dim=42, act_dim=6):
        super().__init__()
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(state_dim, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, act_dim), std=0.01 * np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, act_dim) * -0.5)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(state_dim, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )

    @torch.no_grad()
    def expert_action(self, state):
        return self.actor_mean(state)


class VisionStudent(nn.Module):
    """CNN encoder (per camera) + MLP head.
    Input: RGB (B, n_cams, C=3, H, W), proprio (B, 6)
    Output: action mean (B, 6)
    """
    def __init__(self, n_cams=2, img_hw=(96, 96), proprio_dim=6, act_dim=6):
        super().__init__()
        self.n_cams = n_cams
        H, W = img_hw
        # Small CNN, shared across cameras
        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(3, 32, 5, stride=2, padding=2)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 5, stride=2, padding=2)), nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=2, padding=1)), nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=2, padding=1)), nn.ReLU(),
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 6 * 6, 256)), nn.ReLU(),
        )
        cnn_feat = 256
        self.head = nn.Sequential(
            layer_init(nn.Linear(n_cams * cnn_feat + proprio_dim, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, act_dim), std=0.01 * np.sqrt(2)),
        )

    def forward(self, rgb, proprio):
        B = rgb.shape[0]
        # rgb: (B, n_cams, C, H, W) → flatten cams into batch
        nC, C, H, W = rgb.shape[1:]
        x = rgb.view(B * nC, C, H, W).float() / 255.0
        feats = self.cnn(x)  # (B*nC, 256)
        feats = feats.view(B, nC * feats.shape[-1])
        combined = torch.cat([feats, proprio], dim=-1)
        return self.head(combined)


def extract_obs(obs_dict, device):
    """Pull RGB images, proprio, and state from dict obs."""
    # ManiSkill rgb+state obs structure typically:
    # obs_dict["agent"]["qpos"] (6,)
    # obs_dict["sensor_data"]["<cam_name>"]["rgb"] (H, W, 3)
    # obs_dict["extra"]["..."]: various — we'd flatten same way as state obs
    # For now this is a placeholder until we verify exact structure in WSL.
    if isinstance(obs_dict, torch.Tensor):
        # state-only mode for testing — won't be vision distillation
        return None, None, obs_dict
    qpos = obs_dict["agent"]["qpos"]
    # Sensor data — gather all cameras (sort by name for determinism)
    rgbs = []
    sd = obs_dict.get("sensor_data", {})
    for cam_name in sorted(sd.keys()):
        if "rgb" in sd[cam_name]:
            img = sd[cam_name]["rgb"]  # (B, H, W, 3) uint8
            # to (B, C, H, W)
            img = img.permute(0, 3, 1, 2)
            rgbs.append(img)
    if not rgbs:
        raise RuntimeError(f"No RGB cameras found in sensor_data keys: {list(sd.keys())}")
    rgb_stack = torch.stack(rgbs, dim=1)  # (B, n_cams, C, H, W)
    # State vector (same layout as ppo.py training) — must be reconstructed from extras
    # We'll build it from obs_dict to feed to teacher.
    extra = obs_dict.get("extra", {})
    # Match layout from sort_cubes_env._get_obs_extra
    state_parts = [
        qpos,
        extra.get("tcp_to_obj_pos"),
        extra.get("cube_positions"),
        extra.get("current_color_onehot"),
        extra.get("slot_xy"),
        extra.get("tcp_pos"),
    ]
    if any(p is None for p in state_parts):
        raise RuntimeError(f"Missing extra obs keys. Found: {list(extra.keys())}")
    state = torch.cat([p.float() for p in state_parts], dim=-1)
    return rgb_stack, qpos.float(), state


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_ckpt", required=True)
    p.add_argument("--exp_name", default="distill_v1")
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--num_eval_envs", type=int, default=8)
    p.add_argument("--total_timesteps", type=int, default=5_000_000)
    p.add_argument("--num_steps", type=int, default=16)  # rollout per env
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--eval_freq", type=int, default=50)
    p.add_argument("--num_eval_steps", type=int, default=350)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--use_dr", type=int, default=1, help="Apply Phase B DR settings during distill")
    p.add_argument("--n_cams", type=int, default=2)
    p.add_argument("--img_size", type=int, default=96)
    p.add_argument("--save_dir", default=None)
    args = p.parse_args()

    save_dir = args.save_dir or f"runs/{args.exp_name}"
    os.makedirs(save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # Env config — vision mode
    env_kwargs = dict(
        obs_mode="rgb+state",
        render_mode="rgb_array",
        sim_backend="physx_cuda",
        num_active_cubes=5,
        prefill_min=0, prefill_max=0,
        fix_wrist_roll=True,
    )
    if args.use_dr:
        env_kwargs.update(
            dr_physics=True,
            dr_obs_noise_std=0.003,
            dr_action_noise_std=0.015,
            dr_cube_size_jitter=0.05,
        )

    print(f"Creating train envs (num_envs={args.num_envs})...", flush=True)
    envs = gym.make("SortCubesSO100-v1", num_envs=args.num_envs, **env_kwargs)
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=True, record_metrics=True)
    print(f"Creating eval envs (num_envs={args.num_eval_envs})...", flush=True)
    eval_env_kwargs = {k: v for k, v in env_kwargs.items() if not k.startswith("dr_")}  # eval w/o DR
    eval_envs = gym.make("SortCubesSO100-v1", num_envs=args.num_eval_envs, **eval_env_kwargs)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=True, record_metrics=True)

    # Initial reset to inspect obs structure
    obs, info = envs.reset(seed=args.seed)
    print(f"Obs type: {type(obs)}")
    if isinstance(obs, dict):
        def show(d, indent=""):
            for k, v in d.items():
                if isinstance(v, dict):
                    print(f"{indent}{k}:"); show(v, indent + "  ")
                elif hasattr(v, "shape"):
                    print(f"{indent}{k}: {tuple(v.shape)}")
        show(obs)
    else:
        print(f"  Tensor shape: {obs.shape} (need dict for vision distillation; exit)")
        return

    # Build teacher
    teacher = StateTeacher().to(device)
    teacher.load_state_dict(torch.load(args.teacher_ckpt, map_location=device))
    teacher.eval()
    for p_ in teacher.parameters():
        p_.requires_grad = False
    print(f"Loaded teacher: {args.teacher_ckpt}", flush=True)

    # Extract first obs to infer dims
    rgb, qpos, state = extract_obs(obs, device)
    print(f"RGB shape: {rgb.shape}  proprio: {qpos.shape}  state: {state.shape}", flush=True)

    # Build student
    n_cams = rgb.shape[1]
    img_hw = (rgb.shape[3], rgb.shape[4])
    student = VisionStudent(n_cams=n_cams, img_hw=img_hw, proprio_dim=qpos.shape[-1], act_dim=6).to(device)
    optimizer = optim.Adam(student.parameters(), lr=args.learning_rate)
    print(f"Student params: {sum(p.numel() for p in student.parameters()):,}", flush=True)

    # Training loop
    iterations_per_epoch = max(1, args.num_steps)
    total_iterations = args.total_timesteps // (args.num_envs * iterations_per_epoch)
    print(f"Training for {total_iterations} iterations (~{args.total_timesteps:,} env-steps)", flush=True)

    global_step = 0
    start_time = time.time()
    for iteration in range(1, total_iterations + 1):
        student.train()
        accum_loss = 0.0
        for step in range(args.num_steps):
            rgb, qpos, state = extract_obs(obs, device)
            with torch.no_grad():
                target = teacher.expert_action(state)
            pred = student(rgb, qpos)
            loss = ((pred - target) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            accum_loss += loss.item()
            # Step env with student action (clamped)
            with torch.no_grad():
                action = pred.clamp(-1, 1).detach()
            obs, _, term, trunc, info = envs.step(action)
            global_step += args.num_envs
        mean_loss = accum_loss / args.num_steps
        elapsed = time.time() - start_time
        sps = global_step / elapsed
        print(f"iter {iteration}/{total_iterations} step {global_step:,} loss {mean_loss:.5f} SPS {sps:.0f}", flush=True)

        # Eval periodically
        if iteration % args.eval_freq == 1 or iteration == total_iterations:
            student.eval()
            eval_obs, _ = eval_envs.reset()
            max_n_placed = torch.zeros(args.num_eval_envs, dtype=torch.long, device=device)
            ep_max_list = []
            successes_arr = []
            for t in range(args.num_eval_steps):
                with torch.no_grad():
                    rgb, qpos, _ = extract_obs(eval_obs, device)
                    action = student(rgb, qpos).clamp(-1, 1)
                eval_obs, _, e_term, e_trunc, e_info = eval_envs.step(action)
                if "n_placed" in e_info:
                    max_n_placed = torch.maximum(max_n_placed, e_info["n_placed"].long())
                if "final_info" in e_info:
                    mask = e_info["_final_info"]
                    if mask.any():
                        ep_max_list.append(max_n_placed[mask].cpu())
                        successes_arr.append(e_info["final_info"].get("success_at_end", e_info["final_info"]["episode"].get("success", torch.zeros_like(mask)))[mask].cpu())
                        max_n_placed = max_n_placed * (~mask).long()
            if ep_max_list:
                all_max = torch.cat(ep_max_list)
                succ_rate = (all_max >= 5).float().mean().item()
                mean_n = all_max.float().mean().item()
                print(f"  EVAL iter {iteration}: success={succ_rate*100:.1f}% mean_n_placed={mean_n:.2f} (n_eps={len(all_max)})", flush=True)
            ckpt_path = os.path.join(save_dir, f"student_iter_{iteration}.pt")
            torch.save(student.state_dict(), ckpt_path)
            print(f"  saved {ckpt_path}", flush=True)

    print("Distillation done.")


if __name__ == "__main__":
    main()

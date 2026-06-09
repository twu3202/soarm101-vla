"""ManiSkill GPU parallelism benchmark for SO100GraspCube-v1.

Tests if num_envs ∈ {128, 256, 512, 1024} actually fits on 5060 Ti 8GB.
Reports fps (env-steps per second).
"""
from __future__ import annotations
import os

# Windows SAPIEN CUDA workaround (from D:\soarm101\rl_sac\env.py)
if os.path.exists(r"D:\soarm101\cuda_workaround"):
    os.add_dll_directory(r"D:\soarm101\cuda_workaround")
_cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"
if os.path.exists(_cuda_bin):
    os.add_dll_directory(_cuda_bin)

import time
import gc
import numpy as np
import torch
import gymnasium as gym
import mani_skill.envs  # registers envs


def bench(num_envs: int, n_steps: int = 200) -> dict:
    print(f"\n--- num_envs={num_envs} ---")
    try:
        env = gym.make(
            "SO100GraspCube-v1",
            obs_mode="state",
            render_mode=None,
            num_envs=num_envs,
            sim_backend="gpu",
            render_backend="cpu",
            max_episode_steps=200,
        )
    except Exception as e:
        print(f"  FAIL at gym.make: {type(e).__name__}: {e}")
        return {"num_envs": num_envs, "status": "make_failed", "err": str(e)}

    try:
        obs, _ = env.reset(seed=0)
    except Exception as e:
        print(f"  FAIL at reset: {type(e).__name__}: {e}")
        env.close()
        return {"num_envs": num_envs, "status": "reset_failed", "err": str(e)}

    # GPU memory after reset
    if torch.cuda.is_available():
        mem_mb = torch.cuda.memory_allocated() / 1024**2
        print(f"  GPU mem after reset: {mem_mb:.0f} MB")

    # Random actions; sample using env.action_space
    act_dim = env.action_space.shape[-1]
    print(f"  action_space: {env.action_space.shape}")

    # Warmup
    a = torch.zeros((num_envs, act_dim), device="cuda")
    for _ in range(5):
        obs, rew, term, trunc, info = env.step(a)

    # Timed run
    t0 = time.perf_counter()
    for _ in range(n_steps):
        a = torch.rand((num_envs, act_dim), device="cuda") * 2 - 1
        obs, rew, term, trunc, info = env.step(a)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    total_steps = n_steps * num_envs
    fps = total_steps / dt
    print(f"  {n_steps} batch-steps × {num_envs} envs = {total_steps:,} env-steps in {dt:.2f}s")
    print(f"  fps = {fps:,.0f} env-steps/s")

    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  peak GPU mem: {peak_mb:.0f} MB")

    env.close()
    del env
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    return {"num_envs": num_envs, "fps": fps, "status": "ok"}


def main() -> int:
    print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    results = []
    for n in (2048, 4096, 8192):
        r = bench(n)
        results.append(r)
        if r.get("status") != "ok":
            print(f"  → stopped at num_envs={n} due to {r['status']}")
            break

    print("\n=== SUMMARY ===")
    for r in results:
        if r.get("status") == "ok":
            print(f"  num_envs={r['num_envs']:5d}  fps={r['fps']:9,.0f}")
        else:
            print(f"  num_envs={r['num_envs']:5d}  status={r['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

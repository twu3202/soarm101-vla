"""Test SAPIEN render with different shader packs to find one that doesn't hang.

Run when GPU is idle (no training process). Tries: minimal, trivial, default, rt.
Reports which (if any) work.

Usage:
    python debug_render.py
"""
from __future__ import annotations
import os
import sys
import time
import threading

if os.path.exists(r"D:\soarm101\cuda_workaround"):
    os.add_dll_directory(r"D:\soarm101\cuda_workaround")
_cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"
if os.path.exists(_cuda_bin):
    os.add_dll_directory(_cuda_bin)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_shader(shader_name: str, timeout_sec: float = 15.0):
    """Run a render attempt in a subprocess; report HUNG / OK / ERR."""
    import subprocess
    script = f'''
import os, sys
if os.path.exists(r"D:\\\\soarm101\\\\cuda_workaround"):
    os.add_dll_directory(r"D:\\\\soarm101\\\\cuda_workaround")
cuda_bin = r"C:\\\\Program Files\\\\NVIDIA GPU Computing Toolkit\\\\CUDA\\\\v12.8\\\\bin"
if os.path.exists(cuda_bin):
    os.add_dll_directory(cuda_bin)
sys.path.insert(0, r"{os.path.dirname(os.path.abspath(__file__))}")

# Set shader BEFORE any sim import
import sapien
import sapien.render as sr
shader_path = os.path.join(os.path.dirname(sapien.__file__), "vulkan_shader", "{shader_name}")
sr.set_camera_shader_dir(shader_path)
sr.set_log_level("warn")

import sort_cubes_env
import gymnasium as gym
import mani_skill.envs

env = gym.make("SortCubesSO100-v1", num_envs=1, obs_mode="state", render_mode="rgb_array", sim_backend="physx_cuda", num_active_cubes=5)
obs, _ = env.reset(seed=0)
img = env.render()
import numpy as np
arr = img.cpu().numpy() if hasattr(img, "cpu") else np.asarray(img)
print(f"SHADER_{shader_name}_OK shape={{arr.shape}} dtype={{arr.dtype}}")
'''
    print(f"\n=== Testing shader: {shader_name} ===", flush=True)
    proc = subprocess.Popen(
        [r"C:\Users\asus\miniconda3\envs\lerobot\python.exe", "-c", script],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    start = time.time()
    output_lines = []
    try:
        for line in iter(proc.stdout.readline, ""):
            output_lines.append(line.rstrip())
            print(f"  {line.rstrip()}", flush=True)
            if "SHADER_" in line and "_OK" in line:
                # Render succeeded; let it finish naturally
                pass
            if time.time() - start > timeout_sec:
                print(f"  TIMEOUT after {timeout_sec}s", flush=True)
                proc.kill()
                return "HUNG"
        proc.wait(timeout=5)
        if proc.returncode == 0:
            return "OK"
        else:
            return f"EXIT_{proc.returncode}"
    except Exception as e:
        proc.kill()
        return f"ERR_{type(e).__name__}"


def main():
    shaders_to_try = ["minimal", "trivial", "default", "rt"]
    results = {}
    for s in shaders_to_try:
        results[s] = test_shader(s, timeout_sec=20.0)
    print("\n\n=== SUMMARY ===")
    for s, r in results.items():
        print(f"  {s:10s}: {r}")


if __name__ == "__main__":
    main()

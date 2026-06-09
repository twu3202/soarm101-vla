"""Windows wrapper for ManiSkill PPO baseline — handles SAPIEN CUDA workaround.

Usage:
    python run_ppo.py --env_id SO100GraspCube-v1 --num_envs 1024 \
        --total_timesteps 1000000 --exp_name smoke
"""
import os
import sys

# Windows SAPIEN CUDA workaround (BEFORE importing sapien/mani_skill)
if os.path.exists(r"D:\soarm101\cuda_workaround"):
    os.add_dll_directory(r"D:\soarm101\cuda_workaround")
_cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"
if os.path.exists(_cuda_bin):
    os.add_dll_directory(_cuda_bin)

# Register custom env (SortCubesSO100-v1) before running ppo.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sort_cubes_env  # noqa: registers via @register_env decorator

# Now run ppo.py main
import runpy
runpy.run_path(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "ppo.py"),
    run_name="__main__",
)

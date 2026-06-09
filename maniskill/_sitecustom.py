"""Auto-loaded by Python via PYTHONSTARTUP before any user code.
Sets SAPIEN CUDA dll dirs and registers sort_cubes_env."""
import os, sys
if os.path.exists(r"D:\soarm101\cuda_workaround"):
    os.add_dll_directory(r"D:\soarm101\cuda_workaround")
_cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"
if os.path.exists(_cuda_bin):
    os.add_dll_directory(_cuda_bin)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sort_cubes_env  # noqa
print(f"[sitecustom] CUDA dlls added, sort_cubes_env registered", flush=True)

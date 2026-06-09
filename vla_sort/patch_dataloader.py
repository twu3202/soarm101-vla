"""Make openpi's LeRobotDataset use the PyAV video backend (torchcodec can't find system
ffmpeg libs on this box; PyAV bundles its own). Idempotent + backup. Run with system python3."""
import pathlib
import shutil
import sys

p = pathlib.Path("src/openpi/training/data_loader.py")
s = p.read_text()

if 'video_backend="pyav"' in s:
    print("[patch] already patched"); sys.exit(0)

old = (
    "        delta_timestamps={\n"
    "            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys\n"
    "        },\n"
    "    )"
)
new = (
    "        delta_timestamps={\n"
    "            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys\n"
    "        },\n"
    '        video_backend="pyav",  # torchcodec lacks system ffmpeg libs here; PyAV bundles its own\n'
    "    )"
)
if old not in s:
    print("[patch] FATAL: anchor not found"); sys.exit(2)

shutil.copy(p, str(p) + ".bak_so101")
p.write_text(s.replace(old, new, 1))
print("[patch] data_loader.py -> video_backend=pyav")

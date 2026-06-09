#!/bin/bash
# Diagnose the video-decode backend situation for openpi + our mp4 LeRobot dataset.
cd ~/Projects/openpi
export PATH="$HOME/.local/bin:$PATH"

echo "===== installed video pkgs ====="
uv run python - <<'PY'
import importlib.util as u
for m in ["av", "decord", "torchcodec"]:
    print(f"{m}: {'YES' if u.find_spec(m) else 'no'}")
import lerobot
print("lerobot:", getattr(lerobot, "__version__", "?"))
PY

echo "===== openpi data_loader: LeRobotDataset construction ====="
grep -n "LeRobotDataset\|video_backend\|tolerance_s" src/openpi/training/data_loader.py

echo "===== lerobot: video_backend default + supported ====="
LRDIR=$(uv run python -c "import lerobot,os;print(os.path.dirname(lerobot.__file__))")
echo "lerobot dir: $LRDIR"
grep -n "video_backend" "$LRDIR/common/datasets/lerobot_dataset.py" | head
echo "--- decode_video_frames backends ---"
grep -n "def decode_video_frames\|torchcodec\|pyav\|backend ==" "$LRDIR/common/datasets/video_utils.py" | head -20

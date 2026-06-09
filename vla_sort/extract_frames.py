"""Dump a few frames from green-bowl demo episode 0 (top + wrist) to PNGs so we can see the
bowl/cube and tune the success detector. Runs in the LeRobot conda env on Windows (HF_HOME=D:\\hf_cache).

  conda run -n lerobot python vla_sort/extract_frames.py
Outputs to vla_sort/cam_snaps/demo_*.png
"""
import os
import numpy as np
import cv2

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except Exception:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cam_snaps")
os.makedirs(OUT, exist_ok=True)

ds = LeRobotDataset("local/so101_green_bowl")
n = len(ds)
print("frames:", n)


def to_bgr(x):
    a = np.asarray(x)
    if a.dtype != np.uint8:
        a = (255 * a).astype(np.uint8) if a.max() <= 1.0 else a.astype(np.uint8)
    if a.ndim == 3 and a.shape[0] == 3:  # (C,H,W) -> (H,W,C)
        a = np.transpose(a, (1, 2, 0))
    return cv2.cvtColor(a, cv2.COLOR_RGB2BGR)


# episode 0 is roughly the first ~285 frames; sample start / mid / end of it
for tag, i in [("start", 0), ("mid", 140), ("end", 270)]:
    if i >= n:
        continue
    s = ds[i]
    for cam in ("top", "wrist"):
        key = f"observation.images.{cam}"
        img = to_bgr(s[key])
        p = os.path.join(OUT, f"demo_{cam}_{tag}.png")
        cv2.imwrite(p, img)
        print("wrote", p, img.shape)
print("done")

"""Offline validation for the 2-camera green-bowl pi0.5 LoRA policy (runs on the SERVER, needs GPU).

Two checks, no real arm required:
  (1) TEACHER-FORCING: feed each demo frame's (top, wrist, state) and compare predicted action[0]
      to the dataset's true action. Low error => the model learned the demos.
  (2) CAMERA ABLATION: hold state+one image fixed, blank the OTHER image, measure the action change.
      Large change for BOTH cameras => the policy actually uses top AND wrist (not a state shortcut,
      not ignoring the new wrist view).

Usage (on server, from openpi root):
  uv run python ~/Projects/so101_sort/diag_gb.py [CKPT_STEP]   # default step = 4000
"""
import sys
import numpy as np
import openpi.training.config as _config
from openpi.policies import policy_config as _policy_config

STEP = sys.argv[1] if len(sys.argv) > 1 else "4000"
CKPT = f"checkpoints/pi05_green_bowl_lora/v1/{STEP}"
PROMPT = "Put the green cube in the bowl"

cfg = _config.get_config("pi05_green_bowl_lora")
policy = _policy_config.create_trained_policy(cfg, CKPT)
print(f"policy loaded from {CKPT}\n")

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("local/so101_green_bowl", video_backend="pyav")
n = len(ds)
print(f"dataset frames: {n}")


def infer(top, wrist, state):
    return np.asarray(policy.infer({
        "observation/image": top,
        "observation/wrist_image": wrist,
        "observation/state": state,
        "prompt": PROMPT,
    })["actions"])


# ---------- (1) teacher-forcing ----------
idxs = [int(round(f * (n - 1))) for f in (0.0, 0.12, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0)]
errs = []
print("\n=== (1) TEACHER-FORCING (pred action[0] vs dataset true action) ===")
for i in idxs:
    s = ds[i]
    top = np.asarray(s["observation.images.top"])
    wrist = np.asarray(s["observation.images.wrist"])
    state = np.asarray(s["observation.state"], dtype=np.float32).reshape(-1)
    act_true = np.asarray(s["action"], dtype=np.float32).reshape(-1)
    a0 = infer(top, wrist, state)[0]
    err = np.abs(a0 - act_true)
    errs.append(err)
    print(f"frame {i:4d}: true={np.round(act_true,1)}  pred0={np.round(a0,1)}  |err| mean={err.mean():.2f} deg")
errs = np.array(errs)
print("per-joint mean abs error (deg):", np.round(errs.mean(0), 2))
print("OVERALL mean abs action error: %.2f deg" % errs.mean())
print("(<~3-5 deg = learned demos well; >~15 deg = undertrained/ungrounded)")

# ---------- (2) camera ablation (across the trajectory) ----------
print("\n=== (2) CAMERA ABLATION (hold state+other cam fixed; blank one camera; |Δaction|) ===")
print("  (top should matter most EARLY = locating the cube; wrist most MID/LATE = grasp)")
tops, wrists = [], []
for i in idxs:
    s = ds[i]
    top = np.asarray(s["observation.images.top"])
    wrist = np.asarray(s["observation.images.wrist"])
    state = np.asarray(s["observation.state"], dtype=np.float32).reshape(-1)
    a_both = infer(top, wrist, state)[0]
    d_top = np.abs(a_both - infer(np.zeros_like(top), wrist, state)[0]).mean()
    d_wrist = np.abs(a_both - infer(top, np.zeros_like(wrist), state)[0]).mean()
    tops.append(d_top)
    wrists.append(d_wrist)
    frac = i / (n - 1)
    print(f"  frame {i:4d} ({frac:4.0%} thru): TOP Δ={d_top:5.2f}  WRIST Δ={d_wrist:5.2f} deg")
tops, wrists = np.array(tops), np.array(wrists)
print(f"\n  TOP   sensitivity: mean {tops.mean():.2f}  max {tops.max():.2f} deg")
print(f"  WRIST sensitivity: mean {wrists.mean():.2f}  max {wrists.max():.2f} deg")
print("  ( a camera is USED if its Δ is clearly >0 somewhere along the trajectory )")

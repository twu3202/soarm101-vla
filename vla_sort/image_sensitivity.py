"""Does v1 actually USE the camera? Fix the state, swap the image, see if the action changes.
Small change => model ignores vision (learned a state->action shortcut) => needs retrain.
Large change => model uses vision (so the deploy failure is a blank/stale camera feed)."""
import numpy as np
import openpi.training.config as _config
from openpi.policies import policy_config as _policy_config

PROMPT = "Sort the cubes into a row left to right: red, orange, yellow, green, blue."
cfg = _config.get_config("pi05_so101_sort_lora")
policy = _policy_config.create_trained_policy(cfg, "checkpoints/pi05_so101_sort_lora/v1/9999")

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("local/so101_sort_cubes", video_backend="pyav")

A, B = ds[100], ds[600]   # two frames with different cube layouts / arm-over-scene
imgA = np.asarray(A["observation.images.top"])
imgB = np.asarray(B["observation.images.top"])
stateA = np.asarray(A["observation.state"], dtype=np.float32).reshape(-1)

def act(img, state):
    return np.asarray(policy.infer({"observation/image": img, "observation/state": state, "prompt": PROMPT})["actions"])[0]

aA = act(imgA, stateA)
aB = act(imgB, stateA)            # SAME state, DIFFERENT image
aBlk = act(np.zeros_like(imgA), stateA)  # SAME state, BLACK image
print("state held FIXED (frame 100's state). Only the image changes:\n")
print("  action | image=A      :", np.round(aA, 1))
print("  action | image=B      :", np.round(aB, 1))
print("  action | image=BLACK  :", np.round(aBlk, 1))
print()
print("  |A - B|     per-joint:", np.round(np.abs(aA - aB), 2), " mean %.2f deg" % np.abs(aA - aB).mean())
print("  |A - BLACK| per-joint:", np.round(np.abs(aA - aBlk), 2), " mean %.2f deg" % np.abs(aA - aBlk).mean())
print("\n( <~1-2 deg => model IGNORES vision (state shortcut) ; >~10 deg => model USES vision )")

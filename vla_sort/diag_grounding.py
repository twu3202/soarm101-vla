"""Teacher-forcing check: does v1 reproduce the demos on their OWN frames?
Feed each demo frame's (image, state) to the policy and compare predicted action[0] to the
dataset's actual action at that frame. Low error => model learned the demos (deploy failure is
OOD/closed-loop). High error => model undertrained / ungrounded (need v2)."""
import numpy as np
import openpi.training.config as _config
from openpi.policies import policy_config as _policy_config

CKPT = "checkpoints/pi05_so101_sort_lora/v1/9999"
PROMPT = "Sort the cubes into a row left to right: red, orange, yellow, green, blue."

cfg = _config.get_config("pi05_so101_sort_lora")
policy = _policy_config.create_trained_policy(cfg, CKPT)
print("policy loaded from", CKPT)

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("local/so101_sort_cubes", video_backend="pyav")
n = len(ds)
print("dataset frames:", n)

idxs = [i for i in (0, 80, 160, 300, 500, 700, 850) if i < n]
errs = []
for i in idxs:
    s = ds[i]
    img = np.asarray(s["observation.images.top"])
    state = np.asarray(s["observation.state"], dtype=np.float32).reshape(-1)
    act_true = np.asarray(s["action"], dtype=np.float32).reshape(-1)  # (6,) absolute deg
    pred = np.asarray(policy.infer({"observation/image": img, "observation/state": state, "prompt": PROMPT})["actions"])
    a0 = pred[0]
    err = np.abs(a0 - act_true)
    errs.append(err)
    print(f"frame {i:4d}: true={np.round(act_true,1)}  pred0={np.round(a0,1)}  |err| mean={err.mean():.2f} deg")

errs = np.array(errs)
print("\n=== per-joint mean abs error (deg):", np.round(errs.mean(0), 2))
print("=== overall mean abs action error: %.2f deg" % errs.mean())
print("(<~3-5 deg = learned the demos well; >~15 deg = undertrained/ungrounded)")

"""Per-dimension BC fit quality on the demo set: which joints/gripper are well-imitated."""
import sys
import numpy as np
import torch
import torch.nn as nn

def li(l, std=np.sqrt(2)):
    torch.nn.init.orthogonal_(l.weight, std); torch.nn.init.constant_(l.bias, 0.0); return l

class Agent(nn.Module):
    def __init__(self, obs=42, act=6):
        super().__init__()
        self.critic = nn.Sequential(li(nn.Linear(obs,256)),nn.Tanh(),li(nn.Linear(256,256)),nn.Tanh(),
                                    li(nn.Linear(256,256)),nn.Tanh(),li(nn.Linear(256,1)))
        self.actor_mean = nn.Sequential(li(nn.Linear(obs,256)),nn.Tanh(),li(nn.Linear(256,256)),nn.Tanh(),
                                        li(nn.Linear(256,256)),nn.Tanh(),li(nn.Linear(256,act),std=0.01*np.sqrt(2)))
        self.actor_logstd = nn.Parameter(torch.ones(1, act) * -0.5)

ckpt = sys.argv[1] if len(sys.argv) > 1 else "runs/bc_n5/policy.pt"
demos = sys.argv[2] if len(sys.argv) > 2 else "demos_clean.npz"
d = np.load(demos)
S = torch.tensor(d["states"], dtype=torch.float32)
A = torch.tensor(d["actions"], dtype=torch.float32)
ag = Agent(); ag.load_state_dict(torch.load(ckpt, map_location="cpu")); ag.eval()
with torch.no_grad():
    pred = ag.actor_mean(S)
names = ["pan", "shoulder", "elbow", "wrist_flex", "wrist_roll(lock)", "GRIPPER"]
err = (pred - A) ** 2
print(f"{ckpt} on {demos}: {len(S)} steps\n")
print("per-dim:   MSE    RMSE   target-std  (RMSE/std = relative error)")
for j in range(6):
    mse = err[:, j].mean().item()
    rmse = mse ** 0.5
    std = A[:, j].std().item()
    rel = rmse / (std + 1e-6)
    print(f"  {names[j]:18s} {mse:.4f}  {rmse:.3f}   {std:.3f}      {rel:.2f}")
# gripper: how well do we predict OPEN vs CLOSE direction (sign)?
g_pred, g_true = pred[:, 5], A[:, 5]
sign_acc = ((g_pred.sign() == g_true.sign()) | (g_true.abs() < 0.05)).float().mean().item()
print(f"\ngripper sign agreement (open/close direction): {sign_acc*100:.1f}%")
print(f"gripper action range in demos: [{g_true.min():.2f}, {g_true.max():.2f}], mean {g_true.mean():.3f}")

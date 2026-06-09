"""BC fine-tune the policy on teleoperated demos (from record_demos.py).
Regresses actor_mean(state) -> demo action via MSE, optionally initialized from an RL ckpt
(v16) so it inherits reach/transport structure. Saves a checkpoint deployable by deploy_real.py.

Usage:
  python bc_finetune.py --demos=demos.npz --init_ckpt=runs/sort_n5_v16_DRheavy/ckpt_101.pt \
      --out=runs/bc_n1/policy.pt
  Then deploy: python deploy_real.py --ckpt=runs/bc_n1/policy.pt --num_active=1 ...
"""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn


def li(layer, std=np.sqrt(2)):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, 0.0)
    return layer


class Agent(nn.Module):  # MUST match ppo.py / deploy_real.StateAgent
    def __init__(self, obs=42, act=6):
        super().__init__()
        self.critic = nn.Sequential(li(nn.Linear(obs, 256)), nn.Tanh(), li(nn.Linear(256, 256)), nn.Tanh(),
                                    li(nn.Linear(256, 256)), nn.Tanh(), li(nn.Linear(256, 1)))
        self.actor_mean = nn.Sequential(li(nn.Linear(obs, 256)), nn.Tanh(), li(nn.Linear(256, 256)), nn.Tanh(),
                                        li(nn.Linear(256, 256)), nn.Tanh(), li(nn.Linear(256, act), std=0.01 * np.sqrt(2)))
        self.actor_logstd = nn.Parameter(torch.ones(1, act) * -0.5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--demos", default="demos.npz")
    p.add_argument("--init_ckpt", default="runs/sort_n5_v16_DRheavy/ckpt_101.pt",
                   help="RL ckpt to initialize from; 'none' for scratch")
    p.add_argument("--out", default="runs/bc_n1/policy.pt")
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--grasp_weight", type=float, default=0.0,
                   help="up-weight descend/grasp frames (gripper-active + low tcp_z) so MSE doesn't "
                        "average them away. 0=off. Try 6-10.")
    args = p.parse_args()

    d = np.load(args.demos)
    S = torch.tensor(d["states"], dtype=torch.float32)
    A = torch.tensor(d["actions"], dtype=torch.float32)
    print(f"demos: {len(S)} steps over {len(d['ep_lens'])} episodes")

    # per-sample weight: emphasize the rare descend/grasp frames (gripper actively moving +
    # TCP low). MSE over-fits the dominant "hover/transport" frames otherwise -> never descends.
    grip = A[:, 5].abs()
    grip_n = grip / (grip.std() + 1e-6)
    tcp_z = S[:, 41]
    lowz = ((0.05 - tcp_z) / 0.05).clamp(0.0, 1.0)
    W = 1.0 + args.grasp_weight * grip_n + args.grasp_weight * lowz
    if args.grasp_weight > 0:
        print(f"grasp weighting on (gw={args.grasp_weight}): weight mean={W.mean():.2f} "
              f"max={W.max():.2f}, top-decile frac of total weight="
              f"{W.sort(descending=True).values[:len(W)//10].sum()/W.sum():.2f}")

    n = len(S)
    idx = torch.randperm(n)
    nv = max(1, int(n * args.val_frac))
    vi, ti = idx[:nv], idx[nv:]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Str, Atr, Sva, Ava = S[ti].to(dev), A[ti].to(dev), S[vi].to(dev), A[vi].to(dev)
    Wtr = W[ti].to(dev)

    agent = Agent().to(dev)
    if args.init_ckpt and args.init_ckpt.lower() != "none":
        agent.load_state_dict(torch.load(args.init_ckpt, map_location=dev))
        print(f"initialized from {args.init_ckpt}")
    opt = torch.optim.Adam(agent.parameters(), lr=args.lr)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    best = 1e9
    for ep in range(args.epochs):
        agent.train()
        perm = torch.randperm(len(Str), device=dev)
        tot = 0.0
        for i in range(0, len(Str), args.batch):
            b = perm[i:i + args.batch]
            pred = agent.actor_mean(Str[b])
            err = ((pred - Atr[b]) ** 2).mean(dim=1)  # per-sample MSE (joint 4 fit to demo 0)
            wb = Wtr[b]
            loss = (err * wb).sum() / wb.sum()         # grasp-weighted (wb==1 everywhere if gw=0)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        agent.eval()
        with torch.no_grad():
            vl = ((agent.actor_mean(Sva) - Ava) ** 2).mean().item()
        if vl < best:
            best = vl
            torch.save(agent.state_dict(), args.out)
        if ep % 20 == 0 or ep == args.epochs - 1:
            print(f"ep {ep:3d}  train {tot / len(Str):.4f}  val {vl:.4f}  best {best:.4f}")
    print(f"\ndone. best val MSE {best:.4f} -> {args.out}")
    print(f"deploy: python deploy_real.py --ckpt={args.out} --num_active=1 "
          f"--robot_port=COM4 --camera_index=0 --calibration=calib.npz --action_ema=0.3")


if __name__ == "__main__":
    main()

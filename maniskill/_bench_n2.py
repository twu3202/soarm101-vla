"""Benchmark n=2 checkpoints on the pure deploy task (num_active=2, no prefill, no DR).
64 parallel episodes x 350 steps. Reports success_at_end, success_once, mean cubes placed."""
import os, sys
os.environ.setdefault("SORT_CUBES_NUM_ACTIVE", "2")
os.environ.setdefault("SORT_CUBES_FIX_WRIST_ROLL", "1")
os.environ.setdefault("SORT_CUBES_PREFILL_MIN", "0")
os.environ.setdefault("SORT_CUBES_PREFILL_MAX", "0")
import numpy as np, torch, torch.nn as nn
import sort_cubes_env  # register env
import gymnasium as gym
import mani_skill.envs  # noqa

def li(l, std=np.sqrt(2)):
    torch.nn.init.orthogonal_(l.weight, std); torch.nn.init.constant_(l.bias, 0.0); return l

class Agent(nn.Module):
    def __init__(self, obs, act):
        super().__init__()
        self.critic = nn.Sequential(li(nn.Linear(obs,256)),nn.Tanh(),li(nn.Linear(256,256)),nn.Tanh(),
                                    li(nn.Linear(256,256)),nn.Tanh(),li(nn.Linear(256,1)))
        self.actor_mean = nn.Sequential(li(nn.Linear(obs,256)),nn.Tanh(),li(nn.Linear(256,256)),nn.Tanh(),
                                        li(nn.Linear(256,256)),nn.Tanh(),li(nn.Linear(256,act),std=0.01*np.sqrt(2)))
        self.actor_logstd = nn.Parameter(torch.ones(1, act) * -0.5)
    def act(self, x): return self.actor_mean(x)

N, STEPS = 64, 350
NA = int(os.environ["SORT_CUBES_NUM_ACTIVE"])
env = gym.make("SortCubesSO100-v1", num_envs=N, obs_mode="state", sim_backend="physx_cuda",
               control_mode="pd_joint_target_delta_pos", num_active_cubes=NA)
dev = "cuda"
obs_dim = env.single_observation_space.shape[0]; act_dim = env.single_action_space.shape[0]
print(f"benchmark n=2: {N} episodes x {STEPS} steps, obs={obs_dim} act={act_dim}\n")
for ckpt in sys.argv[1:]:
    agent = Agent(obs_dim, act_dim).to(dev)
    agent.load_state_dict(torch.load(ckpt, map_location=dev)); agent.eval()
    obs, _ = env.reset(seed=0)
    once = torch.zeros(N, dtype=torch.bool, device=dev)
    with torch.no_grad():
        for t in range(STEPS):
            obs, rew, term, trunc, info = env.step(agent.act(obs))
            once |= info["success"]
    se = info["success"].float().mean().item()
    so = once.float().mean().item()
    npl = info["n_placed"].float().mean().item()
    print(f"  {os.path.basename(ckpt):16s}  success_end={se:.3f}  success_once={so:.3f}  mean_placed={npl:.3f}")
env.close()

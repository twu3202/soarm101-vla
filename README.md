# SO-ARM101 Vision-Language-Action Manipulation

Teaching a low-cost **SO-ARM101** 6-DOF arm to do real-world pick-and-place from a handful of
teleoperated demonstrations, by fine-tuning the **π0.5 (pi0.5) Vision-Language-Action model**
([openpi](https://github.com/Physical-Intelligence/openpi), Physical Intelligence) with LoRA.

This repo also contains the **full history** of how we got here: a long detour through
simulation reinforcement learning (ManiSkill) and behavior cloning that ultimately hit a wall,
and the pivot to a pretrained VLA that finally moved the needle.

> **TL;DR of the journey:** sim RL learned to grasp 1–2 cubes but never chained 5, and *nothing*
> transferred to the real arm (sim2real dynamics gap). Behavior cloning on 10 demos imitated the
> demos open-loop but drifted in closed loop. A π0.5 VLA fine-tune *did* learn the demos and *did*
> use vision — but 10 demos of a 5-cube sort was too thin for robust closed-loop grasping. The fix
> that's working: **shrink the task** (one cube → bowl) **and add a wrist camera** for close-up
> grasp grounding. RL comes *after* the policy can occasionally succeed (so there's a reward signal
> to bootstrap from).

---

## The task

Final target was sorting 5 colored cubes into a left-to-right row. After the 5-cube VLA hit the
few-demo limit, we simplified to a single, clean skill to get a *working* grasp first:

**"Put the green cube in the bowl."** — one cube, one bowl, ≤10 demos, two cameras.

The plan is to climb the difficulty ladder once the simple task grasps reliably:
single pick-place → add RL to make it robust → re-introduce more cubes / ordering.

---

## Approaches, in order (what worked and what didn't)

| Phase | Approach | Result |
|---|---|---|
| 1 | **Sim RL** — MuJoCo then ManiSkill/SAPIEN, PPO/SAC, heavy reward shaping + curriculum + domain randomization | Learned to grasp & place **1–2 cubes** (single-cube ~69%), but **never chained 5** (reached_5plus = 0 across all evals). And **zero transfer to the real arm** — every policy tracked the cube but wouldn't descend/grasp (sim2real dynamics gap). |
| 2 | **Behavior cloning** on 10 real teleop demos (state-based) | Imitated demos open-loop, but in closed loop **hovered and never committed to the grasp**. 10 demos + MSE averaging washed out the rare descent frames. |
| 3a | **π0.5 VLA LoRA**, 5-cube sort, **1 top camera**, 10 demos | **Learned the demos** (teacher-forcing 1.87°) and **used vision** (image-swap 9–25°), camera confirmed live at deploy (std 58). But **never grasped** on the real arm — few-demo closed-loop drift. The whole 5-cube sort from 10 demos was too ambitious. |
| 3b | **π0.5 VLA LoRA**, simplified **green-cube→bowl**, **2 cameras (top + wrist)**, 10 demos | **← current.** Wrist close-up + simpler task to cross the 0%→"occasionally grasps" threshold. Training results below. |
| 4 | **RL on top of the VLA** (planned) | Only viable *after* phase 3b occasionally succeeds (sparse-reward RL needs a non-zero success rate to bootstrap). See [RL plan](#next-reinforcement-learning). |

### Why a VLA at all
A pretrained VLA sidesteps the three things that sank phases 1–2:
- **vision-based** → no hand-tuned HSV / camera calibration / cube-z hacks;
- **pretrained manipulation prior** → doesn't have to discover grasping from scratch;
- **trained on real demos** → no sim2real gap.

### Why few demos is the real bottleneck (not a bug)
Demos define a narrow "tube" of states the policy knows. 10 demos = a thin tube; closed-loop
execution drifts out of it and the policy doesn't know how to recover. Widening the tube needs
either **more/better demos** (here: a wrist camera + a simpler task, so 10 is enough) or **RL
exploration** (which needs an occasional success to start from). There's no free lunch — the
learning signal has to come from somewhere.

---

## Green-bowl training results (π0.5 LoRA, 2 cameras)

- **Model:** π0.5 (PaliGemma 2B VLM + ~300M action expert), LoRA on both (`gemma_2b_lora` +
  `gemma_300m_lora`), flow-matching action head, `action_horizon=10`.
- **Data:** 10 teleop episodes, 2851 frames @ 30 fps. Inputs: `observation.images.top` (overhead)
  + `observation.images.wrist` (gripper) + 6-DOF joint state. Output: 6-DOF absolute joint targets
  (5 joints delta-encoded, gripper absolute).
- **Training:** LoRA fine-tune from `pi05_base`, batch 32, 4000 steps (~45 epochs), checkpoint every
  1000 steps, on a single RTX 6000 Ada (48 GB).

<!-- RESULTS_PLACEHOLDER: filled in after offline validation (diag_gb.py) -->
**Offline validation** (`vla_sort/diag_gb.py`, no arm required):

| Checkpoint | Teacher-forcing err (deg) | Top-cam sensitivity (deg) | Wrist-cam sensitivity (deg) |
|---|---|---|---|
| _to be filled_ | | | |

- **Teacher-forcing**: predicted vs. ground-truth action on demo frames — low ⇒ learned the demos.
- **Camera ablation**: blank one camera, measure action change — large for *both* ⇒ uses top *and* wrist.

> **Real-arm deploy** is the final test and requires the physical arm + operator; it is **not** part
> of the automated validation here. Run `vla_sort/so101_vla_deploy.py` (below) to test on hardware.

---

## Repository layout

```
vla_sort/                 # the π0.5 VLA pipeline (current approach)
  so101_policy.py         #   openpi input/output transforms (top→base_0_rgb, wrist→left_wrist_0_rgb)
  setup_so101_config.py   #   registers the 1-cam 5-cube-sort TrainConfig in openpi
  setup_green_bowl_config.py #  registers the 2-cam green-bowl TrainConfig
  patch_dataloader.py     #   forces pyav video backend (server has no system ffmpeg/torchcodec)
  run_normstats_gb.sh / smoke_gb.sh / train_gb.sh / serve_gb.sh   # server scripts
  diag_gb.py / run_diag_gb.sh        # offline validation (teacher-forcing + camera ablation)
  enum_cameras.py         #   find the top vs wrist camera index on Windows
  so101_vla_deploy.py     #   real-arm deploy client (2 cameras → websocket → joint targets)
  openpi_client/          #   websocket client, copied from openpi (Apache-2.0)

maniskill/                # phase 1–2 history: ManiSkill RL env, PPO/DAPG, BC, real-arm deploy
  sort_cubes_env.py, ppo.py, deploy_real.py, record_demos.py, ...

env/ sim/ scripts/        # earliest MuJoCo experiments (superseded)
```
*(checkpoints, datasets, videos, and run logs are git-ignored — they're large and derived.)*

---

## How to reproduce

### 0. Prereqs
- A server with [openpi](https://github.com/Physical-Intelligence/openpi) installed (`~/Projects/openpi`,
  uv-managed venv, JAX). ~24 GB+ GPU for LoRA.
- The SO-ARM101 leader + follower arms, an overhead camera (index 0) and a wrist camera (index 1),
  [LeRobot](https://github.com/huggingface/lerobot) on the arm-side machine.

### 1. Record demos (arm-side machine, 2 cameras)
```bash
lerobot-record --robot.type=so101_follower --robot.port=COM4 --robot.id=my_awesome_follower_arm \
  "--robot.cameras={ top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30} }" \
  --teleop.type=so101_leader --teleop.port=COM3 --teleop.id=my_awesome_leader_arm \
  --dataset.repo_id=local/so101_green_bowl "--dataset.single_task=Put the green cube in the bowl" \
  --dataset.num_episodes=10 --dataset.fps=30 --dataset.episode_time_s=30 --dataset.reset_time_s=10 \
  --dataset.push_to_hub=false
```
(`vla_sort/enum_cameras.py` helps identify which cv2 index is the top vs wrist camera.)

### 2. Train (server)
```bash
# copy so101_policy.py into openpi/src/openpi/policies/, then:
python  ~/Projects/so101_sort/setup_green_bowl_config.py   # register the TrainConfig
bash    ~/Projects/so101_sort/run_normstats_gb.sh          # compute norm stats
bash    ~/Projects/so101_sort/smoke_gb.sh                  # 5-step smoke test
bash    ~/Projects/so101_sort/train_gb.sh                  # full LoRA fine-tune
```

### 3. Validate (server, offline)
```bash
bash ~/Projects/so101_sort/run_diag_gb.sh 4000   # teacher-forcing + camera ablation for ckpt 4000
```

### 4. Serve + deploy (server serves, arm-side client connects)
```bash
# server:
bash ~/Projects/so101_sort/serve_gb.sh           # websocket policy server on :8000
# arm-side (via SSH tunnel `ssh -N -L 8000:localhost:8000 user@server`):
python vla_sort/so101_vla_deploy.py --server_host=localhost --robot_port=COM4 \
    --camera_index=0 --wrist_index=1
```

---

## Next: reinforcement learning

The VLA gets us from 0% to "occasionally grasps." To push to reliable, the plan is **RL on top of
the VLA** — the right lever once there's a non-zero success rate to bootstrap from:

- **Real-world online RL (RLT-style):** keep the VLA frozen, attach a small RL-token + actor-critic
  head, collect real rollouts with a sparse 0/1 reward (cube in bowl), and update the head. Sample-
  efficient (~minutes of real data) and avoids the sim2real gap that killed phase 1.
- **(Alternative) sim RL (SimpleVLA-RL / RIPT):** SFT prior + sparse 0/1 RL in sim. We have a
  ManiSkill env, but sim2real already bit us once — used only if a faithful sim is cheap.

Scaffolding lives under `vla_sort/` (see the RL design notes). RL is **not** run until phase 3b
deploy shows occasional grasps — without that, sparse-reward RL has nothing to climb.

---

## Acknowledgements
- [openpi](https://github.com/Physical-Intelligence/openpi) — π0/π0.5 and the LoRA fine-tuning stack (Physical Intelligence).
- [LeRobot](https://github.com/huggingface/lerobot) — SO-ARM101 drivers, teleop, dataset format (Hugging Face).
- [ManiSkill](https://github.com/haosulab/ManiSkill) — the simulation used in phase 1.

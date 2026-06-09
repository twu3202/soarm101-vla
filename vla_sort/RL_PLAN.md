# RL plan: from "occasionally grasps" to "reliable"

The π0.5 SFT (green-bowl, 2-cam) is meant to cross the **0% → occasionally-succeeds** threshold.
RL is the lever that takes **occasionally → reliable**. This doc is the concrete plan.

## Hard prerequisite (rung 0)

**Sparse-reward RL cannot start from 0% success.** It learns by contrasting successful vs failed
rollouts; if every rollout scores 0, the gradient is 0 and nothing moves. So:

> Before any RL: deploy the SFT policy and measure the real-arm success rate.
> - **≥ ~10–15%** → proceed to rung 1.
> - **~0%** → do **not** start RL. Fix the prior first: a few more/cleaner demos, better wrist
>   framing, or shorter `exec_horizon`. Then re-measure.

This is exactly why we simplified the task and added the wrist camera — to make a non-zero success
rate reachable from 10 demos.

## The bootstrap ladder

Climb only as far as needed; each rung is more powerful and more work than the last.

### Rung 1 — Iterative SFT / DAgger-lite  *(recommended first; simplest, reuses our stack)*
Loop:
1. Deploy current policy, collect ~20–40 on-policy rollouts (`collect_rollouts.py`).
2. Label each success/fail — auto via `bowl_success.py`, or manual keypress.
3. **Keep the successful rollouts** (optionally also operator-corrected partial ones = true DAgger).
4. Merge them into the training set and re-run `train_gb.sh`.
5. Repeat.

Why it works: the failures are caused by the policy drifting into states the 10 demos never
covered. On-policy successful rollouts are *exactly* the data that covers those states → each round
widens the known "tube". This is filtered behavior cloning = reward-weighted regression with a 0/1
reward, the core of most practical few-demo RL — with **zero new infra** (same openpi LoRA fine-tune).

Cost: human resets between episodes. Expect 2–4 rounds to go from ~15% → ~60–80%.

### Rung 2 — Advantage-weighted regression (AWR) / GRPO-style  *(if rung 1 plateaus)*
π0.5 is **flow-matching**, not autoregressive tokens, so token-level GRPO (SimpleVLA-RL / RIPT-VLA)
doesn't drop in directly. The clean equivalent for flow-matching is **AWR**: weight each rollout's
flow-matching SFT loss by `w = exp(A / β)`, where the advantage `A = reward − baseline`
(baseline = mean reward of rollouts from the same start; group-relative, like GRPO/RLOO).
- reward 0/1 (or shaped, see below). β controls greediness.
- Reduces to filtered-BC (rung 1) when `w ∈ {0,1}`; AWR just uses *soft* weights and down-weights
  (not discards) failures, so it extracts signal from near-misses too.
- Implementation = a one-line per-sample weight on the existing loss in openpi's train step + a
  rollout buffer with rewards. Much smaller than a full PPO/GRPO stack.

### Rung 3 — RLT (RL Token), research stretch  *(highest ceiling, most engineering)*
Freeze the VLA; attach an **RL-token readout + small actor-critic head** that *edits* the VLA's
predicted action; train the head online with real rollouts (sample-efficient, ~minutes of data,
per the RLT paper). Reserve for the last mile after rungs 1–2 plateau. Requires exposing VLA
features from the openpi server and an online update loop — non-trivial; only worth it if needed.

## Reward / success detection

Sparse reward = **"green cube ended inside the bowl."** Options:
- **Manual**: operator presses `y/n` at episode end. Reliable, fine for tens of rollouts.
- **Automatic** (`bowl_success.py`): top-camera HSV — green-cube centroid inside the bowl region for
  the last K frames. Validated offline against the demos (every demo ends in success). Removes the
  labeling bottleneck and is the reward fn for rungs 1–2.

Optional shaping (rung 2 only, keep mild to avoid the reward-farming traps we hit in sim — see
project memory): + small bonus for "cube lifted" and "cube above bowl" so near-misses get partial
credit. Verify any shaped reward with the γ-bounded farming check before use.

## Reset
Single cube + bowl → reset is cheap: operator returns the cube to a (varied) start pose between
episodes. Vary cube start across rollouts for coverage, same as demo collection.

## What's scaffolded in this repo
- `bowl_success.py` — HSV success detector + offline self-test on the demo dataset (the reward fn).
- `collect_rollouts.py` — deploy the served policy, record each episode (top+wrist+state+action),
  auto/manual success label, save per-episode `.npz` to a rollout folder (data engine for rung 1/2).
- Merge-and-retrain (rung 1): convert kept rollouts to a LeRobot dataset and re-run `train_gb.sh`
  on the union. (Conversion sketch in `collect_rollouts.py` docstring.)

## Recommendation
Start at **rung 1** the moment the SFT shows occasional grasps — it's the highest
return-on-effort and reuses everything we already built. Escalate to rung 2 (AWR) only if iterative
SFT stalls, and treat rung 3 (RLT) as research, not the default path.

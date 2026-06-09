"""Reward shape + cumulative diagnostic — script ideal/typical/bad trajectories and
print per-phase components and totals.

We DON'T run physics here — we directly invoke `env._compute_sort_reward` with
hand-constructed (tip_xyz, cube_pos) trajectories so the curves are easy to read
and reproducible.

What we want to confirm:
1. Reach component is monotone-increasing as tip approaches cube
2. Transport component dominates reach once the cube is grasped+lifted (so policy
   doesn't get stuck at "hover near cube")
3. Placement and done bonuses fire exactly once at the right moment
4. Total reward for an ideal trajectory >> for a bad-but-active trajectory
5. Step + smoothness penalties don't overwhelm gains

Outputs:
- printed per-phase summary
- matplotlib plot of component & cumulative reward over time → outputs/diagnostics/
"""
from __future__ import annotations

import sys
import pathlib
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from env.sort_env import (  # noqa: E402
    SortBlocksEnv, TARGET_POSITIONS, CUBE_REST_Z, CUBE_COLORS,
    GRASP_TIP_DIST_MAX, GRASP_LIFT_Z_MIN,
    PLACEMENT_TOL_IN, PLACEMENT_Z_MAX,
)

OUT_DIR = ROOT / "outputs" / "diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------- Trajectory builders ----------------

def _lerp(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """n linearly-spaced points from a to b inclusive."""
    t = np.linspace(0.0, 1.0, n, dtype=np.float64)[:, None]
    return (1 - t) * a + t * b


def ideal_trajectory_n1(cube_xy: tuple[float, float]) -> tuple[np.ndarray, np.ndarray, list[tuple[str, int]]]:
    """Hand-scripted "good policy" for num_active=1 (red only).

    Phases:
      A: REACH       — tip drops from home to just above cube  (30 steps)
      B: GRASP_TRANS — cube lifts to grasp_z while tip stays clamped  (5 steps)
      C: TRANSPORT   — both tip & cube glide to red-slot xy at grasp_z  (35 steps)
      D: PLACE       — cube descends to table at slot, tip stays high  (5 steps)
      E: RETREAT     — tip retracts up after release  (5 steps)

    Returns (tip_traj[T,3], cube_traj[T,3], phase_markers)
    """
    home_tip = np.array([0.30, 0.0, 0.18])
    cube_start = np.array([cube_xy[0], cube_xy[1], CUBE_REST_Z])
    slot = TARGET_POSITIONS[0]  # red slot
    grasp_z = 0.06   # lifted ~4 cm above table
    place_z = CUBE_REST_Z
    tip_lift_after = np.array([slot[0], slot[1], 0.15])

    # Phase A: reach — tip → just above cube
    tip_above_cube = np.array([cube_start[0], cube_start[1], grasp_z])
    a_tip = _lerp(home_tip, tip_above_cube, 30)
    a_cube = np.tile(cube_start, (30, 1))

    # Phase B: grasp transition — cube z rises from REST_Z to grasp_z (tip stays clamped)
    b_tip = np.tile(tip_above_cube, (5, 1))
    b_cube = np.zeros((5, 3))
    b_cube[:, 0] = cube_start[0]
    b_cube[:, 1] = cube_start[1]
    b_cube[:, 2] = np.linspace(CUBE_REST_Z, grasp_z, 5)

    # Phase C: transport — both glide laterally to slot at grasp_z
    cube_at_slot_grasp = np.array([slot[0], slot[1], grasp_z])
    c_tip = _lerp(tip_above_cube, cube_at_slot_grasp, 35)
    c_cube = _lerp(np.array([cube_start[0], cube_start[1], grasp_z]), cube_at_slot_grasp, 35)

    # Phase D: place — cube z drops to table, tip stays just above
    d_tip = np.tile(cube_at_slot_grasp, (5, 1))
    d_cube = np.zeros((5, 3))
    d_cube[:, 0] = slot[0]
    d_cube[:, 1] = slot[1]
    d_cube[:, 2] = np.linspace(grasp_z, place_z, 5)

    # Phase E: retreat — tip moves up & back, cube stays in slot
    e_tip = _lerp(cube_at_slot_grasp, tip_lift_after, 5)
    e_cube = np.tile(np.array([slot[0], slot[1], place_z]), (5, 1))

    tip = np.concatenate([a_tip, b_tip, c_tip, d_tip, e_tip], axis=0)
    cube = np.concatenate([a_cube, b_cube, c_cube, d_cube, e_cube], axis=0)
    markers = [("A reach", 0), ("B grasp", 30), ("C transport", 35),
               ("D place", 70), ("E retreat", 75)]
    return tip, cube, markers


def hover_no_grasp_traj(cube_xy: tuple[float, float], n: int = 80) -> tuple[np.ndarray, np.ndarray]:
    """Bad trajectory: tip hovers right above cube (in reach zone) but cube never lifts.

    This isolates "what if the policy gets the reach reward but never escalates" —
    cumulative reward should be bounded by W_REACH × T which is much less than placement.
    """
    cube_pos = np.array([cube_xy[0], cube_xy[1], CUBE_REST_Z])
    tip = np.tile(cube_pos + np.array([0, 0, 0.03]), (n, 1))  # 3 cm above cube
    cube = np.tile(cube_pos, (n, 1))
    return tip, cube


def random_traj(n: int = 80, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Bad trajectory: tip jitters randomly in workspace, cube stationary."""
    rng = np.random.default_rng(seed)
    tip_x = 0.15 + 0.1 * rng.uniform(-1, 1, n)
    tip_y = 0.0 + 0.1 * rng.uniform(-1, 1, n)
    tip_z = 0.12 + 0.05 * rng.uniform(-1, 1, n)
    tip = np.stack([tip_x, tip_y, tip_z], axis=1)
    cube = np.tile([0.17, 0.05, CUBE_REST_Z], (n, 1))
    return tip, cube


# ---------------- Runner ----------------

def run_trajectory(name: str, env: SortBlocksEnv, tip_traj: np.ndarray, cube_traj: np.ndarray,
                   phase_markers: list[tuple[str, int]] | None = None) -> dict:
    """Replay a (tip, cube) trajectory through env's reward fns. Returns per-step components."""
    T = tip_traj.shape[0]
    cube_pos_full = np.tile(env._cube_world_pos(), (1, 1))  # (5, 3) — copy current parked positions
    # Reset placed_mask for this run
    env._placed_mask = np.zeros(env.num_active_cubes if False else 5, dtype=bool)
    env._placed_mask[:] = False
    history: dict[str, list] = {
        "step": [], "phase": [],
        "reach": [], "transport": [], "grasp_bonus": [], "placement": [],
        "done": [], "smoothness": [], "step_penalty": [], "total": [],
        "is_grasping": [], "n_placed": [], "cumulative": [],
        "dist_tip_to_current": [], "dist_current_to_slot": [],
    }
    cum = 0.0
    last_action = np.zeros(6, dtype=np.float32)

    def phase_at(t: int) -> str:
        if not phase_markers:
            return ""
        cur = ""
        for label, start in phase_markers:
            if t >= start:
                cur = label
        return cur

    # Build cube_pos_full[0] = the cube we're trajectorying (red, idx 0). Others stay at their reset.
    base_cube_pos = env._cube_world_pos().copy()

    for t in range(T):
        cube_pos = base_cube_pos.copy()
        cube_pos[0] = cube_traj[t]
        new_placed = env._update_placed_mask(cube_pos)
        # Use zero action delta — we want to isolate task reward, not smoothness
        a = np.zeros(6, dtype=np.float32)
        r, comps = env._compute_sort_reward(
            cube_pos=cube_pos, action=a, last_action=last_action,
            prev_placed=env._placed_mask, new_placed=new_placed,
            tip_xyz=tip_traj[t],
        )
        env._placed_mask = new_placed
        cum += r

        history["step"].append(t)
        history["phase"].append(phase_at(t))
        for k in ("reach", "transport", "grasp_bonus", "placement",
                  "done", "smoothness", "step_penalty", "total",
                  "dist_tip_to_current", "dist_current_to_slot"):
            history[k].append(comps[k])
        history["is_grasping"].append(comps["is_grasping_current"])
        history["n_placed"].append(comps["n_placed"])
        history["cumulative"].append(cum)
        last_action = a

    print(f"\n=== {name} ===")
    print(f"  steps        : {T}")
    print(f"  cumulative   : {cum:+.3f}")
    print(f"  final placed : {history['n_placed'][-1]} / {env.num_active_cubes}")
    # Phase breakdown (if markers given)
    if phase_markers:
        starts = [m[1] for m in phase_markers] + [T]
        for i, (label, start) in enumerate(phase_markers):
            end = starts[i + 1]
            sub_total = sum(history["total"][start:end])
            sub_reach = sum(history["reach"][start:end])
            sub_transp = sum(history["transport"][start:end])
            sub_grasp = sum(history["grasp_bonus"][start:end])
            sub_place = sum(history["placement"][start:end])
            sub_done = sum(history["done"][start:end])
            print(f"  [{label:13s}] steps {start:3d}..{end-1:3d}  "
                  f"sum={sub_total:+7.3f}  reach={sub_reach:+6.2f}  "
                  f"transp={sub_transp:+6.2f}  grasp={sub_grasp:+5.2f}  "
                  f"place={sub_place:+6.2f}  done={sub_done:+6.2f}")
    return history


def plot_history(name: str, hist: dict, phase_markers: list[tuple[str, int]] | None = None) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not available — skipping plot)")
        return

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    s = np.array(hist["step"])

    ax = axes[0]
    ax.plot(s, hist["reach"], label="R_reach", color="C0")
    ax.plot(s, hist["transport"], label="R_transport", color="C1")
    ax.plot(s, hist["grasp_bonus"], label="R_grasp_bonus", color="C2")
    ax.plot(s, hist["placement"], label="R_placement", color="C3", linewidth=1.5)
    ax.plot(s, hist["done"], label="R_done", color="C4", linewidth=1.5)
    ax.set_ylabel("component reward")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title(f"{name} — reward decomposition")

    ax = axes[1]
    ax.plot(s, hist["dist_tip_to_current"], label="dist(tip → current cube)", color="C5")
    ax.plot(s, hist["dist_current_to_slot"], label="dist(current cube → slot)", color="C6")
    ax.plot(s, hist["is_grasping"], label="is_grasping (0/1)", color="k", linestyle="--", alpha=0.6)
    ax.set_ylabel("distance / flag")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(s, hist["total"], label="step total", color="C0", alpha=0.6)
    ax.plot(s, hist["cumulative"], label="cumulative", color="C3", linewidth=2)
    ax.set_xlabel("step")
    ax.set_ylabel("reward")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    if phase_markers:
        for label, start in phase_markers:
            for ax in axes:
                ax.axvline(start, color="gray", linestyle=":", alpha=0.6)
            axes[0].text(start, axes[0].get_ylim()[1] * 0.95, label,
                          rotation=90, va="top", fontsize=7, color="gray")

    fig.tight_layout()
    fname = OUT_DIR / f"reward_curve_{name.replace(' ', '_').replace('/', '_')}.png"
    fig.savefig(fname, dpi=110)
    plt.close(fig)
    print(f"  plot saved → {fname}")


def ideal_trajectory_n5() -> tuple[np.ndarray, np.ndarray, list[tuple[str, int]]]:
    """Chain 5 ideal single-cube trajectories — sort R→O→Y→G→B in order.

    Returns tip_traj (T,3), cube_traj (T, 5, 3) with all 5 cubes' positions per step.
    """
    # initial positions for the 5 cubes (xy spread across workspace)
    init_xys = np.array([
        [0.18,  0.08],   # red
        [0.13, -0.10],   # orange
        [0.20, -0.05],   # yellow
        [0.15,  0.05],   # green
        [0.21,  0.11],   # blue
    ])
    home_tip = np.array([0.30, 0.0, 0.18])
    grasp_z = 0.06
    place_z = CUBE_REST_Z

    tip_chunks: list[np.ndarray] = []
    cube_chunks: list[np.ndarray] = []
    markers: list[tuple[str, int]] = []
    t_so_far = 0
    current_tip = home_tip
    current_cubes = np.zeros((5, 3))
    current_cubes[:, :2] = init_xys
    current_cubes[:, 2] = CUBE_REST_Z

    color_letters = ["R", "O", "Y", "G", "B"]
    for ci in range(5):
        slot = TARGET_POSITIONS[ci]
        cube_start = current_cubes[ci].copy()
        tip_above_cube = np.array([cube_start[0], cube_start[1], grasp_z])
        cube_at_slot_grasp = np.array([slot[0], slot[1], grasp_z])
        cube_at_slot_place = np.array([slot[0], slot[1], place_z])

        # A reach (20 steps)
        n_a = 20
        a_tip = _lerp(current_tip, tip_above_cube, n_a)
        a_cubes = np.tile(current_cubes, (n_a, 1, 1))
        markers.append((f"reach {color_letters[ci]}", t_so_far))
        t_so_far += n_a

        # B grasp (4 steps)
        n_b = 4
        b_tip = np.tile(tip_above_cube, (n_b, 1))
        b_cubes = np.tile(current_cubes, (n_b, 1, 1))
        b_cubes[:, ci, 2] = np.linspace(CUBE_REST_Z, grasp_z, n_b)
        markers.append((f"grasp {color_letters[ci]}", t_so_far))
        t_so_far += n_b

        # C transport (24 steps)
        n_c = 24
        c_tip = _lerp(tip_above_cube, cube_at_slot_grasp, n_c)
        c_cubes = np.tile(current_cubes, (n_c, 1, 1))
        c_cubes[:, ci, 0] = np.linspace(cube_start[0], slot[0], n_c)
        c_cubes[:, ci, 1] = np.linspace(cube_start[1], slot[1], n_c)
        c_cubes[:, ci, 2] = grasp_z
        markers.append((f"transport {color_letters[ci]}", t_so_far))
        t_so_far += n_c

        # D place (4 steps)
        n_d = 4
        d_tip = np.tile(cube_at_slot_grasp, (n_d, 1))
        d_cubes = np.tile(current_cubes, (n_d, 1, 1))
        d_cubes[:, ci, 0] = slot[0]
        d_cubes[:, ci, 1] = slot[1]
        d_cubes[:, ci, 2] = np.linspace(grasp_z, place_z, n_d)
        markers.append((f"place {color_letters[ci]}", t_so_far))
        t_so_far += n_d

        tip_chunks.extend([a_tip, b_tip, c_tip, d_tip])
        cube_chunks.extend([a_cubes, b_cubes, c_cubes, d_cubes])

        # Update running positions for next iteration
        current_tip = cube_at_slot_grasp.copy()
        current_cubes[ci] = cube_at_slot_place

    tip = np.concatenate(tip_chunks, axis=0)
    cube = np.concatenate(cube_chunks, axis=0)  # (T, 5, 3)
    return tip, cube, markers


def run_full_trajectory(name: str, env: SortBlocksEnv,
                        tip_traj: np.ndarray, cube_traj_full: np.ndarray,
                        phase_markers: list[tuple[str, int]]) -> dict:
    """Like run_trajectory but takes per-step positions for ALL 5 cubes."""
    T = tip_traj.shape[0]
    env._placed_mask[:] = False
    history: dict[str, list] = {
        "step": [], "reach": [], "transport": [], "grasp_bonus": [], "placement": [],
        "done": [], "smoothness": [], "step_penalty": [], "total": [],
        "is_grasping": [], "n_placed": [], "cumulative": [],
        "dist_tip_to_current": [], "dist_current_to_slot": [],
        "current_color_idx": [],
    }
    cum = 0.0
    last_action = np.zeros(6, dtype=np.float32)

    for t in range(T):
        cube_pos = cube_traj_full[t].astype(np.float64)
        new_placed = env._update_placed_mask(cube_pos)
        a = np.zeros(6, dtype=np.float32)
        r, comps = env._compute_sort_reward(
            cube_pos=cube_pos, action=a, last_action=last_action,
            prev_placed=env._placed_mask, new_placed=new_placed,
            tip_xyz=tip_traj[t],
        )
        env._placed_mask = new_placed
        cum += r
        history["step"].append(t)
        for k in ("reach", "transport", "grasp_bonus", "placement", "done",
                  "smoothness", "step_penalty", "total",
                  "dist_tip_to_current", "dist_current_to_slot"):
            history[k].append(comps[k])
        history["is_grasping"].append(comps["is_grasping_current"])
        history["n_placed"].append(comps["n_placed"])
        history["current_color_idx"].append(comps["current_color_idx"])
        history["cumulative"].append(cum)
        last_action = a

    print(f"\n=== {name} ===")
    print(f"  steps        : {T}")
    print(f"  cumulative   : {cum:+.3f}")
    print(f"  final placed : {history['n_placed'][-1]} / {env.num_active_cubes}")

    # per-color breakdown — each color takes 52 steps (20+4+24+4)
    n_per_color = 52
    for ci in range(env.num_active_cubes):
        start = ci * n_per_color
        end = (ci + 1) * n_per_color
        sub = sum(history["total"][start:end])
        print(f"  [{CUBE_COLORS[ci]:7s}] steps {start:3d}..{end-1:3d}  cum_for_this_color={sub:+7.2f}")
    return history


def main() -> int:
    print("=== reward shape + cumulative diagnostic ===")
    env = SortBlocksEnv(num_active_cubes=1, render_images=False, seed=0)
    env.reset(seed=0)

    # Pick a representative starting cube position
    cube_xy = (0.18, 0.08)

    # 1) Ideal scripted trajectory
    tip_ideal, cube_ideal, markers = ideal_trajectory_n1(cube_xy)
    hist_ideal = run_trajectory("ideal n=1", env, tip_ideal, cube_ideal, markers)
    plot_history("ideal_n1", hist_ideal, markers)

    # 2) Hover-but-no-grasp baseline
    env.reset(seed=0)
    tip_hover, cube_hover = hover_no_grasp_traj(cube_xy, n=80)
    hist_hover = run_trajectory("hover no grasp", env, tip_hover, cube_hover)
    plot_history("hover_no_grasp", hist_hover)

    # 3) Random tip jitter baseline
    env.reset(seed=0)
    tip_rand, cube_rand = random_traj(n=80, seed=0)
    hist_rand = run_trajectory("random jitter", env, tip_rand, cube_rand)
    plot_history("random_jitter", hist_rand)

    # ----- Pathology checks -----
    print("\n=== sanity assertions ===")

    cum_ideal = hist_ideal["cumulative"][-1]
    cum_hover = hist_hover["cumulative"][-1]
    cum_rand = hist_rand["cumulative"][-1]

    ratio_ideal_to_hover = cum_ideal / max(cum_hover, 1e-3)
    ratio_ideal_to_rand = cum_ideal / max(cum_rand, 1e-3)

    print(f"  ideal cumulative  : {cum_ideal:+.2f}")
    print(f"  hover cumulative  : {cum_hover:+.2f}  (ratio ideal/hover = {ratio_ideal_to_hover:.1f}x)")
    print(f"  random cumulative : {cum_rand:+.2f}  (ratio ideal/random = {ratio_ideal_to_rand:.1f}x)")

    # Check 1: placement+done fire exactly once (n=1)
    assert sum(1 for x in hist_ideal["placement"] if x > 0) <= 1, \
        "placement bonus should fire at most once in n=1 ideal traj"
    assert any(x > 0 for x in hist_ideal["done"]), "done bonus must fire on ideal traj"
    print("  OK: placement + done fire exactly when expected")

    # Check 2: per-step total reward during transport > pre-grasp reach phase.
    # With linear -dist shaping both reach and transport are negative; the differentiator
    # is the +W_GRASP_BONUS that fires only while grasping. So the test is: does total/step
    # actually improve once the policy starts grasping?
    grasp_steps = [i for i, g in enumerate(hist_ideal["is_grasping"]) if g]
    if grasp_steps:
        gi = grasp_steps[0]
        pre_total_per_step = np.mean(hist_ideal["total"][:gi])
        post_total_per_step = np.mean(hist_ideal["total"][gi:gi + 30])
        print(f"  avg total/step (pre-grasp)   = {pre_total_per_step:+.3f}")
        print(f"  avg total/step (early grasp) = {post_total_per_step:+.3f}")
        assert post_total_per_step > pre_total_per_step + 0.5, (
            f"grasping should improve per-step reward by ≥0.5, got "
            f"{post_total_per_step - pre_total_per_step:+.3f}"
        )
        print("  OK: grasp+transport phase clearly beats reach phase per-step")

    # Check 3: ideal cumulative >> hover cumulative AND hover/random must be net-negative
    # (so a long episode of doing-nothing-near-cube is strictly worse than just terminating).
    assert cum_ideal > cum_hover + 50.0, \
        f"ideal should beat hover by >50 reward units, got ideal={cum_ideal:.2f}, hover={cum_hover:.2f}"
    assert cum_hover < 0, f"hover must be net-negative, got {cum_hover:+.2f}"
    assert cum_rand < 0, f"random jitter must be net-negative, got {cum_rand:+.2f}"
    print("  OK: ideal >> hover; both hover and random are net-negative (no farming exploit)")

    # Check 4: step+smooth penalties small vs task signal (gain from placement+done)
    pen_total = sum(hist_ideal["step_penalty"]) + sum(hist_ideal["smoothness"])
    print(f"  ideal total penalties (step + smooth) = {pen_total:+.3f} "
          f"(vs ideal cum +{cum_ideal:.1f})")
    assert abs(pen_total) < 0.2 * cum_ideal, \
        "step+smooth penalties should be < 20% of ideal task reward"
    print("  OK: penalties don't dominate signal")

    # 4) n=5 full ideal — chain 5 single-cube trajectories
    env5 = SortBlocksEnv(num_active_cubes=5, render_images=False, seed=0)
    env5.reset(seed=0)
    tip5, cube5, markers5 = ideal_trajectory_n5()
    hist5 = run_full_trajectory("ideal n=5 (full chain)", env5, tip5, cube5, markers5)
    plot_history("ideal_n5", hist5, markers5)
    cum5 = hist5["cumulative"][-1]
    print(f"\n  n=5 ideal cumulative = {cum5:+.2f} (expected ~+180 to +220)")
    assert cum5 > 150.0, f"n=5 ideal should hit cumulative > 150, got {cum5:.2f}"
    assert hist5["n_placed"][-1] == 5, f"all 5 should be placed at end, got {hist5['n_placed'][-1]}"
    # done bonus must fire exactly once
    done_firings = sum(1 for x in hist5["done"] if x > 0)
    assert done_firings == 1, f"done should fire exactly once, fired {done_firings} times"
    # placement bonus must fire exactly 5 times
    place_firings = sum(1 for x in hist5["placement"] if x > 0.5)  # threshold for v4 W_PLACE_BONUS=5
    assert place_firings == 5, f"placement bonus should fire 5 times, fired {place_firings}"
    print("  OK: n=5 chain reaches +150+ cumulative; done fires once, placement fires 5x")

    env.close()
    env5.close()
    print("\n=== PASS ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

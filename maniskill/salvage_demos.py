"""Salvage the recorded demos WITHOUT re-recording:
  1) clean cube_positions: any xy outside the reachable workspace box is a phantom (skin/edge)
     detection -> replace with last valid (forward-fill; leading invalids back-filled).
  2) re-label per-step task progress (cur) against the NEW SLOT_XY (user's own layout) + tol.
  3) rebuild the 42-dim state (cleaned cubes + correct cur/onehot + new slot_xy), keep actions.
Saves demos_clean.npz. Run inspect afterward to confirm cur now advances R->O->Y->G->B.

Usage: python salvage_demos.py demos.npz demos_clean.npz
"""
import sys
import numpy as np
import deploy_real as D  # SLOT_XY (new), PLACEMENT_XY_TOL, DETECT_WORKSPACE_BOX, compute_current_color_idx, build_state_vector

src = sys.argv[1] if len(sys.argv) > 1 else "demos.npz"
dst = sys.argv[2] if len(sys.argv) > 2 else "demos_clean.npz"
box = D.DETECT_WORKSPACE_BOX
cols = ["R", "O", "Y", "G", "B"]
print(f"salvage {src} -> {dst}")
print(f"workspace box {box} | new SLOT_XY:\n{D.SLOT_XY}\n tol={D.PLACEMENT_XY_TOL}\n")

d = np.load(src)
S, A, L = d["states"], d["actions"], d["ep_lens"]
newS = S.copy()
n_replaced = 0
i = 0
for ep, n in enumerate(L):
    s = S[i:i + n]
    cubes = s[:, 9:24].reshape(n, 5, 3).copy()
    qpos = s[:, 0:6]
    tcp = s[:, 39:42]
    # --- clean each cube's trajectory ---
    for k in range(5):
        inside = ((cubes[:, k, 0] >= box[0]) & (cubes[:, k, 0] <= box[1]) &
                  (cubes[:, k, 1] >= box[2]) & (cubes[:, k, 1] <= box[3]))
        if not inside.any():
            print(f"  ep{ep} cube {cols[k]}: NEVER inside box (left as-is)")
            continue
        first = int(np.argmax(inside))
        last_valid = cubes[first, k].copy()
        for t in range(n):
            if t < first or not inside[t]:
                cubes[t, k] = last_valid
                n_replaced += 1
            else:
                last_valid = cubes[t, k].copy()
    # --- re-label cur: MONOTONIC + debounce (sorting never un-places a cube) ---
    DEBOUNCE = 4  # consecutive in-slot frames required to advance to the next color (~0.27s @15Hz)
    cur = 0
    streak = 0
    seq = []
    for t in range(n):
        if cur < 5:
            dxy = np.linalg.norm(cubes[t, cur, :2] - D.SLOT_XY[cur])
            on_table = cubes[t, cur, 2] < D.PLACEMENT_Z_MAX
            if dxy < D.PLACEMENT_XY_TOL and on_table:
                streak += 1
                if streak >= DEBOUNCE:
                    cur += 1
                    streak = 0
            else:
                streak = 0
        done = cur >= 5
        cur_idx = 0 if done else cur            # mirror compute_current_color_idx's (0, True) on done
        newS[i + t] = D.build_state_vector(qpos[t], tcp[t], cubes[t], cur_idx, done)
        c = 5 if done else cur
        if not seq or seq[-1] != c:
            seq.append(c)
    seqstr = "->".join(cols[c] if c < 5 else "DONE" for c in seq)
    reached = max((c for c in seq if c < 5), default=-1)
    print(f"ep{ep:2d} n={n:3d}  cur: {seqstr:30s}  reached={cols[reached] if reached>=0 else '-'}  done={cur>=5}")
    i += n

np.savez(dst, states=newS, actions=A, ep_lens=L)
print(f"\nreplaced {n_replaced} glitched cube-position entries ({100*n_replaced/(len(S)*5):.1f}% of cube slots)")
print(f"saved {len(newS)} steps -> {dst}")

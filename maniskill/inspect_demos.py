"""Inspect recorded demos.npz: did the per-step task progress (cur/onehot) actually
advance, and where did cubes actually end up? Tells us if the demos are BC-usable."""
import sys
import numpy as np

f = sys.argv[1] if len(sys.argv) > 1 else "demos.npz"
d = np.load(f)
S, A, L = d["states"], d["actions"], d["ep_lens"]
print(f"{f}: {len(S)} steps, {len(L)} episodes\n")

SLOT_XY = np.array([[0.28,-0.10],[0.28,-0.05],[0.28,0.00],[0.28,0.05],[0.28,0.10]])
cols = ["R","O","Y","G","B"]
i = 0
for ep, n in enumerate(L):
    s = S[i:i+n]
    onehot = s[:, 24:29]                  # current color (0 if all_done)
    cur = onehot.argmax(1)                 # but argmax of all-zero -> 0, so check sum
    has = onehot.sum(1) > 0.5
    cur_eff = np.where(has, cur, 5)        # 5 == "done" (no onehot set)
    # progression: unique cur values in order
    seq = []
    for c in cur_eff:
        if not seq or seq[-1] != c:
            seq.append(int(c))
    seqstr = "->".join(cols[c] if c < 5 else "DONE" for c in seq)
    # where did each cube end up (final-step cube_positions) vs its slot
    final_cubes = s[-1, 9:24].reshape(5, 3)
    errs = [np.linalg.norm(final_cubes[k, :2] - SLOT_XY[k]) for k in range(5)]
    errstr = " ".join(f"{cols[k]}:{errs[k]*100:4.1f}cm" for k in range(5))
    reached_done = (cur_eff == 5).any()
    print(f"ep{ep:2d} n={n:3d}  cur: {seqstr:30s}  done={reached_done}")
    print(f"        final cube->slot dist:  {errstr}")
    i += n

# overall: how many steps are labeled which color
all_oh = S[:, 24:29]
allcur = np.where(all_oh.sum(1) > 0.5, all_oh.argmax(1), 5)
print("\nstep counts by current label:")
for c in range(6):
    name = cols[c] if c < 5 else "DONE"
    print(f"  {name}: {(allcur==c).sum()}")

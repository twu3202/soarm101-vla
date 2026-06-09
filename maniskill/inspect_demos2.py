"""Dump realized cube layout + position stability, to decide if demos are salvageable."""
import sys
import numpy as np
np.set_printoptions(precision=3, suppress=True)

f = sys.argv[1] if len(sys.argv) > 1 else "demos.npz"
d = np.load(f)
S, L = d["states"], d["ep_lens"]
SLOT_XY = np.array([[0.28,-0.10],[0.28,-0.05],[0.28,0.00],[0.28,0.05],[0.28,0.10]])
cols = ["R","O","Y","G","B"]

i = 0
final_all = []
for ep, n in enumerate(L):
    s = S[i:i+n]
    cubes = s[:, 9:24].reshape(n, 5, 3)     # (T,5,3)
    final = cubes[-1]                         # last step
    final_all.append(final[:, :2])
    # how much does each cube's xy jump step-to-step (max single-step jump, in cm)
    jumps = np.linalg.norm(np.diff(cubes[:, :, :2], axis=0), axis=2)  # (T-1,5)
    maxjump = jumps.max(0) * 100
    print(f"ep{ep:2d}  final XY (m):")
    for k in range(5):
        print(f"   {cols[k]}: ({final[k,0]:+.3f},{final[k,1]:+.3f}) z={final[k,2]:+.3f}  "
              f"max step-jump={maxjump[k]:5.1f}cm")
    i += n

final_all = np.array(final_all)             # (E,5,2)
print("\nMEAN realized final XY across episodes (vs defined slot):")
for k in range(5):
    m = final_all[:, k].mean(0)
    sd = final_all[:, k].std(0)
    print(f"  {cols[k]}: mean=({m[0]:+.3f},{m[1]:+.3f})  std=({sd[0]:.3f},{sd[1]:.3f})  "
          f"slot=({SLOT_XY[k,0]:+.2f},{SLOT_XY[k,1]:+.2f})")
print("\nrealized row spread: y from",
      f"{final_all[:,0,1].mean():+.3f} (R) to {final_all[:,4,1].mean():+.3f} (B);",
      f"x mean={final_all[:,:,0].mean():+.3f}")

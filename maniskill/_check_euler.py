import numpy as np
from scipy.spatial.transform import Rotation as R

def optical_axis(yaw, pitch, roll):
    Rm = R.from_euler("zyx", [yaw, pitch, roll], degrees=True).as_matrix()
    return Rm @ np.array([0, 0, 1.0])  # camera +Z (lens-forward) in robot frame

print("Target for true top-down: optical axis = [0, 0, -1] (straight DOWN)\n")
for (y, p, r) in [(0, -90, 0), (0, 90, 0), (0, 180, 0), (0, 0, 180),
                  (90, 180, 0), (-90, 180, 0), (180, 180, 0)]:
    ax = np.round(optical_axis(y, p, r), 3)
    down = "  <== points DOWN" if abs(ax[2] + 1) < 1e-3 else ""
    print(f"yaw={y:5} pitch={p:5} roll={r:5}  -> optical axis = {ax}{down}")

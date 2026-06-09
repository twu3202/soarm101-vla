"""On-policy rollout / evaluation harness for the green-bowl VLA (real arm + served policy).

Two jobs (see RL_PLAN.md):
  1. RUNG-0 GATE: run N episodes, AUTO-SCORE each with bowl_success.py, report the real-arm success
     rate. RL only makes sense once this is >~15%.
  2. Data engine seed: optionally dump each episode's (state, action arrays + final top frame +
     success) to an .npz for audit / failure analysis.

Full-trajectory image recording for rung-1 retraining is better done with `lerobot-record` targeting
the failure cases (it writes a clean LeRobot dataset that merges with the demos) — see RL_PLAN.md.

Run (server reachable via SSH tunnel on :8000):
  python collect_rollouts.py --robot_port=COM4 --camera_index=0 --wrist_index=1 --episodes=20

Per episode: reset cube/bowl, press ENTER to start, let the policy run; it auto-ends when the detector
sees success (cube in bowl) or after --max_cycles. Then confirm/override the auto-label.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from openpi_client import websocket_client_policy  # noqa: E402
from bowl_success import cube_in_bowl  # noqa: E402

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server_host", default="localhost")
    p.add_argument("--server_port", type=int, default=8000)
    p.add_argument("--robot_port", default="COM4")
    p.add_argument("--robot_id", default="my_awesome_follower_arm")
    p.add_argument("--camera_index", type=int, default=0)
    p.add_argument("--wrist_index", type=int, default=1)
    p.add_argument("--prompt", default="Put the green cube in the bowl")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--exec_horizon", type=int, default=5)
    p.add_argument("--hz", type=float, default=30.0)
    p.add_argument("--max_cycles", type=int, default=60, help="re-infer cycles per episode before giving up")
    p.add_argument("--success_hold", type=int, default=3, help="consecutive detected-success frames to auto-stop")
    p.add_argument("--out_dir", default="rollouts_green_bowl")
    p.add_argument("--save_frames", action="store_true", help="also save final top/wrist frame per episode")
    args = p.parse_args()

    import cv2
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out_dir)
    os.makedirs(out, exist_ok=True)

    from lerobot.robots.so101_follower.so101_follower import SO101Follower
    from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
    robot = SO101Follower(SO101FollowerConfig(port=args.robot_port, id=args.robot_id))
    robot.connect(calibrate=False)

    def open_cam(idx, name):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {name} cam {idx}")
        for _ in range(8):
            cap.read()
        return cap

    cap_top, cap_wrist = open_cam(args.camera_index, "TOP"), open_cam(args.wrist_index, "WRIST")
    grab = lambda c: cv2.cvtColor(c.read()[1], cv2.COLOR_BGR2RGB)
    policy = websocket_client_policy.WebsocketClientPolicy(host=args.server_host, port=args.server_port)
    period = 1.0 / args.hz
    results = []

    try:
        for ep in range(args.episodes):
            input(f"\n=== episode {ep+1}/{args.episodes}: reset cube+bowl, ENTER to start ===")
            states, actions = [], []
            hold, auto_succ, last_top, last_wrist = 0, False, None, None
            for cyc in range(args.max_cycles):
                obs = robot.get_observation()
                state = np.array([float(obs[f"{j}.pos"]) for j in JOINTS], np.float32)
                top, wrist = grab(cap_top), grab(cap_wrist)
                last_top, last_wrist = top, wrist
                act = np.asarray(policy.infer({"observation/image": top, "observation/wrist_image": wrist,
                                               "observation/state": state, "prompt": args.prompt})["actions"], np.float32)
                if act.ndim == 1:
                    act = act[None]
                states.append(state); actions.append(act[0])
                ok, _ = cube_in_bowl(top)
                hold = hold + 1 if ok else 0
                if hold >= args.success_hold:
                    auto_succ = True
                    break
                for a in act[:min(args.exec_horizon, len(act))]:
                    robot.send_action({f"{j}.pos": float(a[i]) for i, j in enumerate(JOINTS)})
                    time.sleep(period)
            # final check + operator override
            fin_ok, info = cube_in_bowl(last_top)
            auto = auto_succ or fin_ok
            ans = input(f"  auto-label success={auto} (detector dist={info.get('dist')}/{info.get('bowl_r')}). "
                        f"override? [y]=success [n]=fail [ENTER]=keep auto: ").strip().lower()
            succ = True if ans == "y" else False if ans == "n" else auto
            results.append(succ)
            rec = {"states": np.array(states, np.float32), "actions": np.array(actions, np.float32),
                   "success": bool(succ), "prompt": args.prompt}
            if args.save_frames:
                rec["final_top"], rec["final_wrist"] = last_top, last_wrist
            np.savez_compressed(os.path.join(out, f"ep_{ep:03d}.npz"), **rec)
            print(f"  saved ep_{ep:03d}.npz  success={succ}  running rate={sum(results)}/{len(results)}")
    except KeyboardInterrupt:
        print("\ninterrupted.")
    finally:
        cap_top.release(); cap_wrist.release(); robot.disconnect()

    n = len(results)
    rate = sum(results) / max(1, n)
    json.dump({"episodes": n, "successes": int(sum(results)), "rate": rate},
              open(os.path.join(out, "summary.json"), "w"), indent=2)
    print(f"\n==== SUCCESS RATE: {sum(results)}/{n} = {rate:.0%} ====")
    print("RUNG-0 GATE: >~15% => proceed to RL rung 1 (iterative SFT). ~0% => fix the prior first.")


if __name__ == "__main__":
    main()

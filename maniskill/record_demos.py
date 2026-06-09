"""Record teleoperated demos (leader -> follower) in the POLICY's obs/action space, for BC
fine-tuning. Reuses deploy_real's perception/FK/state so the recorded 42-dim state EXACTLY
matches what the policy sees at deploy. joint 4 (wrist_roll) is locked at REST_QPOS[4].

state  = build_state_vector(...) (42-dim, same as deploy)
action = sim-convention target-delta / ACTION_RANGE in [-1,1] (6-dim, same as policy output)

Usage:
  python record_demos.py --leader_port=COM3 --follower_port=COM4 --camera_index=0 \
      --calibration=calib.npz --num_active=1 --out=demos.npz
Per episode: ENTER to start teleop, grasp+place the cube, Ctrl-C to end, then y/n to keep.
Collect ~10-20 successful demos.
"""
import argparse
import time
import numpy as np
import deploy_real as D  # SO101Driver, SO100ForwardKinematics, CubeTracker, build_state_vector, ...


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--leader_port", default="COM3")
    p.add_argument("--follower_port", default="COM4")
    p.add_argument("--leader_id", default="my_awesome_leader_arm")
    p.add_argument("--follower_id", default="my_awesome_follower_arm")
    p.add_argument("--camera_index", type=int, default=0)
    p.add_argument("--calibration", default="calib.npz")
    p.add_argument("--num_active", type=int, default=1)
    p.add_argument("--hz", type=float, default=15.0)
    p.add_argument("--max_steps", type=int, default=4000)  # ~4.4 min @15Hz; episode ends on Ctrl-C anyway
    p.add_argument("--out", default="demos.npz")
    args = p.parse_args()

    from lerobot.teleoperators.so101_leader.so101_leader import SO101Leader
    from lerobot.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig
    leader = SO101Leader(SO101LeaderConfig(port=args.leader_port, id=args.leader_id))
    leader.connect(calibrate=False)
    print(f"Leader connected at {args.leader_port}")

    follower = D.SO101Driver(port=args.follower_port, robot_id=args.follower_id)
    fk = D.SO100ForwardKinematics()
    c = np.load(args.calibration)
    dist = c["dist"] if "dist" in c.files else None
    pos_off = c["pos_offset"] if "pos_offset" in c.files else None
    tracker = D.CubeTracker(camera_index=args.camera_index, K=c["K"], dist=dist,
                            T_robot_cam=c["T_robot_cam"], pos_offset=pos_off)

    period = 1.0 / args.hz
    lock_deg = float(np.rad2deg(D.REST_QPOS[4]))  # locked wrist_roll (deg)
    S, A, ep_lens = [], [], []
    ndemo = 0
    print(f"\nLeader {args.leader_port} -> Follower {args.follower_port} | wrist locked {lock_deg:.1f} deg "
          f"| num_active={args.num_active}\n")
    cols = ["R", "O", "Y", "G", "B"]
    try:
        while True:
            input(f"=== Demo {ndemo + 1}: place cube(s), ENTER to start teleop (Ctrl-C to end) ===")
            follower.reset_to_home()
            prev_tgt = follower.read_qpos_rad().copy()  # sim-convention rad
            es, ea = [], []
            try:
                for t in range(args.max_steps):
                    loop = time.time()
                    la = leader.get_action()                 # {"<joint>.pos": deg}
                    la["wrist_roll.pos"] = lock_deg          # LOCK joint 4
                    follower.robot.send_action(la)           # mirror leader -> follower
                    qpos = follower.read_qpos_rad()          # sim convention (elbow offset removed)
                    tcp = fk.tcp_pos(qpos)
                    cubes = tracker.get_cube_positions()
                    cur, done = D.compute_current_color_idx(cubes, num_active=args.num_active,
                                                            detected=tracker.last_detected)
                    state = D.build_state_vector(qpos, tcp, cubes, cur, done)
                    # action label = sim-convention target delta / ACTION_RANGE
                    tgt = np.array([la[f"{n}.pos"] for n in D.JOINT_NAMES], dtype=np.float32)
                    tgt[2] -= D.ELBOW_FLEX_OFFSET_DEG
                    tgt = np.deg2rad(tgt)
                    action = np.clip((tgt - prev_tgt) / D.ACTION_RANGE, -1.0, 1.0).astype(np.float32)
                    action[4] = 0.0
                    prev_tgt = tgt
                    es.append(state); ea.append(action)
                    if t % 15 == 0:
                        det = "SEE" if tracker.last_detected[cur] else "MISS"
                        print(f"\r step {t:3d} cur={cols[cur]}[{det}] tcp=({tcp[0]:+.2f},{tcp[1]:+.2f},"
                              f"{tcp[2]:+.2f}) cube=({cubes[cur][0]:+.2f},{cubes[cur][1]:+.2f}) done={done}  ",
                              end="", flush=True)
                    dt = time.time() - loop
                    if dt < period:
                        time.sleep(period - dt)
            except KeyboardInterrupt:
                pass
            print(f"\n  {len(es)} steps.")
            if input("  keep? success=y / discard=n: ").strip().lower() == "y" and len(es) > 5:
                S.extend(es); A.extend(ea); ep_lens.append(len(es)); ndemo += 1
                np.savez(args.out, states=np.array(S, np.float32), actions=np.array(A, np.float32),
                         ep_lens=np.array(ep_lens, np.int32))
                print(f"  SAVED. demos={ndemo}, steps={len(S)} -> {args.out}")
            if input("  record another? y/n: ").strip().lower() == "n":
                break
    finally:
        try: leader.disconnect()
        except Exception: pass
        follower.close(); tracker.close()
    print(f"\nDone: {ndemo} demos, {len(S)} steps -> {args.out}")


if __name__ == "__main__":
    main()

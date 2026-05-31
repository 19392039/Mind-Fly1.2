from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
if DEPLOY_DIR not in sys.path:
    sys.path.insert(0, DEPLOY_DIR)

from runtime_common import (  # noqa: E402
    OBS_RAYS,
    RuntimeState,
    add_model_args,
    build_observation,
    build_policy_cell,
    compute_goal_features,
    infer_action,
    load_limits,
    load_view_radius,
    safety_filter,
    sector_min,
    setup_mindspore,
)


class DryLidar:
    def __init__(self, max_distance_m: float = 10.0):
        self.max_distance_m = float(max_distance_m)
        self.i = 0

    def start(self) -> None:
        pass

    def get_ranges_m(self) -> np.ndarray:
        rays = np.full((OBS_RAYS,), self.max_distance_m, dtype=np.float32)
        if (self.i // 50) % 3 == 1:
            rays[: OBS_RAYS // 16] = 0.7
            rays[-OBS_RAYS // 16 :] = 0.7
        elif (self.i // 50) % 3 == 2:
            rays[: OBS_RAYS // 16] = 0.3
            rays[-OBS_RAYS // 16 :] = 0.3
        self.i += 1
        return rays

    def stop(self) -> None:
        pass


class DryPX4:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.last_t = time.time()

    def update_telemetry(self):
        return self.x, self.y, self.yaw

    def send_velocity_yawrate_cmd(self, vx: float, omega: float, vy: float = 0.0, vz: float = 0.0) -> None:
        now = time.time()
        dt = max(0.0, min(now - self.last_t, 0.2))
        self.last_t = now
        self.yaw += float(omega) * dt
        self.x += (float(vx) * math.cos(self.yaw) - float(vy) * math.sin(self.yaw)) * dt
        self.y += (float(vx) * math.sin(self.yaw) + float(vy) * math.cos(self.yaw)) * dt
        print(f"cmd vx={vx:+.3f} omega={omega:+.3f}")

    def stop_motion(self) -> None:
        self.send_velocity_yawrate_cmd(0.0, 0.0)


def parse_args():
    parser = argparse.ArgumentParser(description="MindSpore PPO real-flight loop for Orange Pi AIpro + PX4")
    add_model_args(parser)
    parser.add_argument("--dry-run", action="store_true", help="Run without lidar or PX4 hardware.")
    parser.add_argument("--lidar-port", default="/dev/ttyUSB0")
    parser.add_argument("--lidar-baud", type=int, default=115200)
    parser.add_argument("--lidar-angle-offset", type=float, default=180.0)
    parser.add_argument("--px4-port", default="/dev/ttyAMA1")
    parser.add_argument("--px4-baud", type=int, default=921600)
    parser.add_argument("--target-x", type=float, default=10.0)
    parser.add_argument("--target-y", type=float, default=0.0)
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--no-safety-filter", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_mindspore(args.device_target, args.graph_mode)
    limits = load_limits(args.env_config)
    view_radius_m = load_view_radius(args.env_config)
    net = build_policy_cell(args.env_config, args.ckpt)
    state = RuntimeState()

    if args.dry_run:
        lidar = DryLidar(view_radius_m)
        fc = DryPX4()
    else:
        from flight_control import PX4Controller
        from lidar_reader import LidarReader

        lidar = LidarReader(
            port=args.lidar_port,
            baudrate=args.lidar_baud,
            max_distance_m=view_radius_m,
            angle_offset_deg=args.lidar_angle_offset,
        )
        fc = PX4Controller(port=args.px4_port, baud=args.px4_baud)

    period = 1.0 / max(float(args.hz), 1e-6)
    lidar.start()
    print("[System] MindSpore PPO deployment loop started. Press Ctrl+C to stop.")

    try:
        while True:
            tick = time.time()
            x, y, yaw = fc.update_telemetry()
            ref_sin, ref_cos, dist_n, dist = compute_goal_features(
                x,
                y,
                yaw,
                args.target_x,
                args.target_y,
                view_radius_m,
            )
            lidar_m = lidar.get_ranges_m()
            obs = build_observation(
                lidar_m,
                state,
                limits,
                view_radius_m=view_radius_m,
                ref_sin=ref_sin,
                ref_cos=ref_cos,
                dist_n=dist_n,
            )
            vx, omega = infer_action(net, obs)
            if not args.no_safety_filter:
                vx, omega = safety_filter((vx, omega), lidar_m)

            fc.send_velocity_yawrate_cmd(vx, omega)
            state.update(vx, omega)

            front = sector_min(lidar_m, OBS_RAYS - OBS_RAYS // 16, OBS_RAYS // 16)
            left = sector_min(lidar_m, OBS_RAYS // 4 - OBS_RAYS // 16, OBS_RAYS // 4 + OBS_RAYS // 16)
            right = sector_min(lidar_m, 3 * OBS_RAYS // 4 - OBS_RAYS // 16, 3 * OBS_RAYS // 4 + OBS_RAYS // 16)
            print(
                f"\rpos=({x:+.2f},{y:+.2f}) yaw={math.degrees(yaw):+.0f} "
                f"dist={dist:.2f} lidar F/L/R={front:.2f}/{left:.2f}/{right:.2f} "
                f"cmd vx={vx:+.2f} omega={omega:+.2f}",
                end="",
            )

            elapsed = time.time() - tick
            if elapsed < period:
                time.sleep(period - elapsed)
    except KeyboardInterrupt:
        print("\n[System] stop requested")
    finally:
        try:
            fc.stop_motion()
        finally:
            lidar.stop()
        print("[System] stopped")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
if DEPLOY_DIR not in sys.path:
    sys.path.insert(0, DEPLOY_DIR)

from runtime_common import (  # noqa: E402
    RuntimeState,
    add_model_args,
    build_observation,
    build_policy_cell,
    compute_goal_features,
    infer_action,
    load_limits,
    load_view_radius,
    safety_filter,
    setup_mindspore,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark MindSpore PPO inference latency on the target device.")
    add_model_args(parser)
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations before measurement.")
    parser.add_argument("--iters", type=int, default=500, help="Measured iterations.")
    parser.add_argument(
        "--mode",
        choices=["infer", "dry-loop"],
        default="infer",
        help="infer measures model forward only; dry-loop includes obs build and safety filter.",
    )
    parser.add_argument("--report-every", type=int, default=0, help="Print live progress every N iterations.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def summarize(name: str, samples_ms) -> None:
    arr = np.asarray(samples_ms, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    p50 = float(np.percentile(arr, 50))
    p90 = float(np.percentile(arr, 90))
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))
    max_v = float(np.max(arr))
    min_v = float(np.min(arr))
    fps = 1000.0 / mean if mean > 0 else float("inf")

    print("")
    print(f"{name} benchmark")
    print("-" * 48)
    print(f"samples      : {arr.size}")
    print(f"mean latency : {mean:.3f} ms")
    print(f"std latency  : {std:.3f} ms")
    print(f"min latency  : {min_v:.3f} ms")
    print(f"p50 latency  : {p50:.3f} ms")
    print(f"p90 latency  : {p90:.3f} ms")
    print(f"p95 latency  : {p95:.3f} ms")
    print(f"p99 latency  : {p99:.3f} ms")
    print(f"max latency  : {max_v:.3f} ms")
    print(f"FPS / Hz     : {fps:.2f}")


def make_test_lidar(rng, view_radius_m: float, i: int) -> np.ndarray:
    rays = np.full((128,), view_radius_m, dtype=np.float32)
    noise = rng.uniform(-0.05, 0.05, size=128).astype(np.float32)
    rays = np.clip(rays + noise, 0.05, view_radius_m)
    if (i // 37) % 3 == 1:
        rays[:8] = 0.7
        rays[-8:] = 0.7
    elif (i // 37) % 3 == 2:
        rays[24:42] = 0.5
    return rays


def run_infer_benchmark(args, net, obs) -> None:
    for _ in range(args.warmup):
        infer_action(net, obs)

    samples = []
    for i in range(args.iters):
        t0 = time.perf_counter()
        infer_action(net, obs)
        samples.append((time.perf_counter() - t0) * 1000.0)
        if args.report_every > 0 and (i + 1) % args.report_every == 0:
            print(f"measured {i + 1}/{args.iters}")

    summarize("model-only inference", samples)


def run_dry_loop_benchmark(args, net, limits, view_radius_m: float, rng) -> None:
    state = RuntimeState()
    x, y, yaw = 0.0, 0.0, 0.0
    target_x, target_y = 10.0, 0.0

    for i in range(args.warmup):
        lidar_m = make_test_lidar(rng, view_radius_m, i)
        ref_sin, ref_cos, dist_n, _ = compute_goal_features(x, y, yaw, target_x, target_y, view_radius_m)
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
        vx, omega = safety_filter((vx, omega), lidar_m)
        state.update(vx, omega)

    samples = []
    for i in range(args.iters):
        t0 = time.perf_counter()
        lidar_m = make_test_lidar(rng, view_radius_m, i)
        ref_sin, ref_cos, dist_n, _ = compute_goal_features(x, y, yaw, target_x, target_y, view_radius_m)
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
        vx, omega = safety_filter((vx, omega), lidar_m)
        state.update(vx, omega)
        samples.append((time.perf_counter() - t0) * 1000.0)
        if args.report_every > 0 and (i + 1) % args.report_every == 0:
            print(f"measured {i + 1}/{args.iters}")

    summarize("dry control loop", samples)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    setup_mindspore(args.device_target, args.graph_mode)
    limits = load_limits(args.env_config)
    view_radius_m = load_view_radius(args.env_config)
    net = build_policy_cell(args.env_config, args.ckpt)

    state = RuntimeState()
    lidar_m = np.full((128,), view_radius_m, dtype=np.float32)
    obs = build_observation(lidar_m, state, limits, view_radius_m=view_radius_m)

    print(f"device-target : {args.device_target}")
    print(f"graph-mode    : {bool(args.graph_mode)}")
    print(f"mode          : {args.mode}")
    print(f"warmup/iters  : {args.warmup}/{args.iters}")

    if args.mode == "infer":
        run_infer_benchmark(args, net, obs)
    else:
        run_dry_loop_benchmark(args, net, limits, view_radius_m, rng)


if __name__ == "__main__":
    main()

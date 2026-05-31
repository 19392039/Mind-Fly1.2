import json
import os
import sys

import numpy as np
import mindspore as ms
from mindspore import Tensor, load_checkpoint, load_param_into_net, ops


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MIND_PPO_DIR = os.path.join(BASE_DIR, "mind_ppo")
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, MIND_PPO_DIR)

from mind_models import PPOPolicy


VEC_DIM = 135
ACTION_DIM = 2
HIDDEN = 64
D_MODEL = 128
NUM_QUERIES = 4
NUM_HEADS = 4
RAY_MAX_M = 10.0
CKPT_FILE = os.path.join(BASE_DIR, "runs", "ppo_exp1", "latest_policy.ckpt")


def _load_action_limits():
    cfg_path = os.path.join(BASE_DIR, "config", "env_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    limits = cfg.get("limits", {})
    return [
        float(limits.get("vx_max", 0.6)),
        float(limits.get("omega_max", 1.5)),
    ]


LIMITS = Tensor(_load_action_limits(), dtype=ms.float32)

policy = PPOPolicy(
    vec_dim=VEC_DIM,
    action_dim=ACTION_DIM,
    hidden=HIDDEN,
    d_model=D_MODEL,
    num_queries=NUM_QUERIES,
    num_heads=NUM_HEADS,
    learnable_queries=True,
)

param_dict = load_checkpoint(CKPT_FILE)
load_param_into_net(policy, param_dict)
policy.set_train(False)


def ppo_predict(
    laser_data,
    ref_sin=0.0,
    ref_cos=1.0,
    prev_vx_n=0.0,
    prev_omega_n=0.0,
    dvx_n=0.0,
    domega_n=0.0,
    dist_n=1.0,
):
    """Return deterministic PPO action from normalized lidar and task features.

    Observation layout matches SimRandomBatchEnv._build_obs:
    [128 lidar rays, ref_sin, ref_cos, prev_vx_n, prev_omega_n,
     dvx_n, domega_n, dist_n].
    """
    laser_data = np.asarray(laser_data, dtype=np.float32).reshape(-1)
    if laser_data.shape[0] != 128:
        raise ValueError(f"laser_data must have 128 values, got {laser_data.shape[0]}")

    pose_data = np.array(
        [ref_sin, ref_cos, prev_vx_n, prev_omega_n, dvx_n, domega_n, dist_n],
        dtype=np.float32,
    )
    obs = np.concatenate([laser_data, pose_data], axis=0).astype(np.float32)
    obs_tensor = Tensor(obs[None, :], dtype=ms.float32)

    mu, _, _ = policy._core(obs_tensor)
    action = (ops.tanh(mu) * LIMITS).asnumpy()[0]
    return float(action[0]), float(action[1])


def _sector_mask(n_rays, center_deg, width_deg):
    angles = np.arange(n_rays, dtype=np.float32) * 360.0 / n_rays
    delta = (angles - center_deg + 180.0) % 360.0 - 180.0
    return np.abs(delta) <= width_deg / 2.0


def make_lidar_case(case_name):
    """Build normalized 128-ray lidar inputs for repeatable behavior probes."""
    lidar = np.full(128, RAY_MAX_M, dtype=np.float32)

    if case_name == "open_space":
        pass
    elif case_name == "front_far_wall":
        lidar[_sector_mask(128, 0.0, 50.0)] = 1.5
    elif case_name == "front_wall":
        lidar[_sector_mask(128, 0.0, 50.0)] = 0.35
    elif case_name == "left_blocked_right_open":
        lidar[_sector_mask(128, 90.0, 90.0)] = 0.45
        lidar[_sector_mask(128, 0.0, 35.0)] = 0.9
    elif case_name == "right_blocked_left_open":
        lidar[_sector_mask(128, 270.0, 90.0)] = 0.45
        lidar[_sector_mask(128, 0.0, 35.0)] = 0.9
    elif case_name == "narrow_corridor":
        lidar[_sector_mask(128, 90.0, 70.0)] = 0.7
        lidar[_sector_mask(128, 270.0, 70.0)] = 0.7
    elif case_name == "boxed_in":
        lidar[:] = 0.45
    else:
        raise ValueError(f"Unknown lidar case: {case_name}")

    return np.clip(lidar / RAY_MAX_M, 0.0, 1.0).astype(np.float32)


def run_behavior_probe():
    cases = [
        ("open_space", 0.0, 1.0, "goal forward, expect vx positive and modest omega"),
        ("open_space", 1.0, 0.0, "goal left, expect omega sign differs from right-goal case"),
        ("open_space", -1.0, 0.0, "goal right, expect omega sign differs from left-goal case"),
        ("front_far_wall", 0.0, 1.0, "goal forward with far obstacle, expect some caution"),
        ("front_wall", 0.0, 1.0, "goal forward with near obstacle, expect lower vx or stronger turn"),
        ("left_blocked_right_open", 0.0, 1.0, "goal forward, left blocked/right open"),
        ("right_blocked_left_open", 0.0, 1.0, "goal forward, right blocked/left open"),
        ("narrow_corridor", 0.0, 1.0, "goal forward in corridor"),
        ("boxed_in", 0.0, 1.0, "goal forward but boxed in, expect low vx if policy learned safety"),
    ]

    limits_np = LIMITS.asnumpy()
    print("PPO behavior probe")
    print(f"checkpoint: {CKPT_FILE}")
    print(f"limits: vx={limits_np[0]:.3f}, omega={limits_np[1]:.3f}")
    print("-" * 98)
    print(f"{'case':<28} {'ref':>11} {'vx':>9} {'omega':>9}  expectation")
    print("-" * 98)

    for name, ref_sin, ref_cos, expectation in cases:
        vx, omega = ppo_predict(
            make_lidar_case(name),
            ref_sin=ref_sin,
            ref_cos=ref_cos,
            dist_n=1.0,
        )
        ref = f"[{ref_sin:+.0f},{ref_cos:+.0f}]"
        print(f"{name:<28} {ref:>11} {vx:>+9.3f} {omega:>+9.3f}  {expectation}")


if __name__ == "__main__":
    run_behavior_probe()

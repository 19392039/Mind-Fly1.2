from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Iterable, Tuple

warnings.filterwarnings(
    "ignore",
    message=r"The value of the smallest subnormal for.*type is zero\.",
    category=UserWarning,
)

import numpy as np
import mindspore as ms
from mindspore import Tensor, context, load_checkpoint, load_param_into_net


DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(DEPLOY_DIR)
MIND_PPO_DIR = os.path.join(PROJECT_ROOT, "mind_ppo")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if MIND_PPO_DIR not in sys.path:
    sys.path.insert(0, MIND_PPO_DIR)

from mind_models import PPOPolicy  # noqa: E402
from deploy_policy import DeployPolicy  # noqa: E402


OBS_RAYS = 128
POSE_DIM = 7
OBS_DIM = OBS_RAYS + POSE_DIM
ACTION_DIM = 2
DEFAULT_VIEW_RADIUS_M = 10.0
DEFAULT_CKPT = os.path.join(PROJECT_ROOT, "runs", "ppo_exp1", "latest_policy.ckpt")
DEFAULT_ENV_CONFIG = os.path.join(PROJECT_ROOT, "config", "env_config.json")


@dataclass
class RuntimeState:
    prev_vx: float = 0.0
    prev_omega: float = 0.0
    prev_prev_vx: float = 0.0
    prev_prev_omega: float = 0.0

    def update(self, vx: float, omega: float) -> None:
        self.prev_prev_vx = self.prev_vx
        self.prev_prev_omega = self.prev_omega
        self.prev_vx = float(vx)
        self.prev_omega = float(omega)


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--env-config", default=DEFAULT_ENV_CONFIG)
    parser.add_argument("--device-target", default="CPU", choices=["CPU", "Ascend"])
    parser.add_argument("--graph-mode", action="store_true")


def setup_mindspore(device_target: str = "CPU", graph_mode: bool = False) -> None:
    mode = context.GRAPH_MODE if graph_mode else context.PYNATIVE_MODE
    context.set_context(mode=mode, device_target=device_target)
    if device_target == "Ascend":
        context.set_context(enable_graph_kernel=False)


def load_env_config(path: str = DEFAULT_ENV_CONFIG) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_limits(env_config_path: str = DEFAULT_ENV_CONFIG) -> np.ndarray:
    cfg = load_env_config(env_config_path)
    limits = cfg.get("limits", {})
    return np.array(
        [
            float(limits.get("vx_max", 0.6)),
            float(limits.get("omega_max", 1.5)),
        ],
        dtype=np.float32,
    )


def load_view_radius(env_config_path: str = DEFAULT_ENV_CONFIG) -> float:
    cfg = load_env_config(env_config_path)
    return float(cfg.get("obs", {}).get("patch_meters", DEFAULT_VIEW_RADIUS_M))


def build_policy_cell(env_config_path: str, ckpt_path: str) -> DeployPolicy:
    policy = PPOPolicy(
        vec_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden=64,
        d_model=128,
        num_queries=4,
        num_heads=4,
        learnable_queries=True,
    )
    params = load_checkpoint(ckpt_path)
    load_result = load_param_into_net(policy, params)
    if isinstance(load_result, tuple) and len(load_result) == 2:
        missing_in_ckpt, unused_in_net = load_result
    else:
        missing_in_ckpt, unused_in_net = load_result, []
    if missing_in_ckpt:
        print(f"[WARN] model parameters missing in checkpoint: {missing_in_ckpt}")
    optimizer_unused = [name for name in unused_in_net if str(name).startswith("optimizer.")]
    other_unused = [name for name in unused_in_net if not str(name).startswith("optimizer.")]
    if optimizer_unused:
        print(f"[INFO] ignored {len(optimizer_unused)} optimizer checkpoint tensors for inference")
    if other_unused:
        print(f"[WARN] unused checkpoint tensors: {other_unused}")
    policy.set_train(False)

    net = DeployPolicy(policy, load_limits(env_config_path))
    net.set_train(False)
    return net


def normalize_lidar(lidar_m: Iterable[float], view_radius_m: float) -> np.ndarray:
    lidar = np.asarray(lidar_m, dtype=np.float32).reshape(-1)
    if lidar.shape[0] != OBS_RAYS:
        raise ValueError(f"lidar must have {OBS_RAYS} rays, got {lidar.shape[0]}")
    return np.clip(lidar / max(float(view_radius_m), 1e-6), 0.0, 1.0).astype(np.float32)


def build_observation(
    lidar_m_or_normalized: Iterable[float],
    state: RuntimeState,
    limits: np.ndarray,
    *,
    view_radius_m: float = DEFAULT_VIEW_RADIUS_M,
    lidar_is_normalized: bool = False,
    ref_sin: float = 0.0,
    ref_cos: float = 1.0,
    dist_n: float = 1.0,
) -> np.ndarray:
    if lidar_is_normalized:
        rays = np.asarray(lidar_m_or_normalized, dtype=np.float32).reshape(-1)
        if rays.shape[0] != OBS_RAYS:
            raise ValueError(f"lidar must have {OBS_RAYS} rays, got {rays.shape[0]}")
        rays = np.clip(rays, 0.0, 1.0)
    else:
        rays = normalize_lidar(lidar_m_or_normalized, view_radius_m)

    vx_lim = max(float(limits[0]), 1e-6)
    omega_lim = max(float(limits[1]), 1e-6)
    prev_vx_n = np.clip(state.prev_vx / vx_lim, -1.0, 1.0)
    prev_omega_n = np.clip(state.prev_omega / omega_lim, -1.0, 1.0)
    dvx_n = np.clip((state.prev_vx - state.prev_prev_vx) / (2.0 * vx_lim), -1.0, 1.0)
    domega_n = np.clip(
        (state.prev_omega - state.prev_prev_omega) / (2.0 * omega_lim),
        -1.0,
        1.0,
    )
    pose = np.array(
        [
            float(ref_sin),
            float(ref_cos),
            prev_vx_n,
            prev_omega_n,
            dvx_n,
            domega_n,
            np.clip(float(dist_n), 0.0, 1.0),
        ],
        dtype=np.float32,
    )
    obs = np.concatenate([rays, pose], axis=0).astype(np.float32)
    if obs.shape[0] != OBS_DIM:
        raise RuntimeError(f"observation dim mismatch: got {obs.shape[0]}, expected {OBS_DIM}")
    return obs


def compute_goal_features(
    current_x: float,
    current_y: float,
    yaw_rad: float,
    target_x: float,
    target_y: float,
    view_radius_m: float,
) -> Tuple[float, float, float, float]:
    dx = float(target_x) - float(current_x)
    dy = float(target_y) - float(current_y)
    dist = math.hypot(dx, dy)
    local_angle = math.atan2(dy, dx) - float(yaw_rad)
    ref_sin = math.sin(local_angle)
    ref_cos = math.cos(local_angle)
    dist_n = min(dist / max(float(view_radius_m), 1e-6), 1.0)
    return ref_sin, ref_cos, dist_n, dist


def infer_action(net: DeployPolicy, obs: np.ndarray) -> Tuple[float, float]:
    obs_t = Tensor(obs.reshape(1, OBS_DIM), ms.float32)
    action = net(obs_t).asnumpy()[0]
    return float(action[0]), float(action[1])


def sector_min(lidar_m: Iterable[float], start: int, end: int) -> float:
    rays = np.asarray(lidar_m, dtype=np.float32).reshape(-1)
    if start <= end:
        return float(np.min(rays[start:end]))
    return float(min(np.min(rays[start:]), np.min(rays[:end])))


def safety_filter(action: Tuple[float, float], lidar_m: Iterable[float]) -> Tuple[float, float]:
    vx, omega = float(action[0]), float(action[1])
    front_min = sector_min(lidar_m, OBS_RAYS - OBS_RAYS // 16, OBS_RAYS // 16)
    if front_min < 0.35:
        return min(vx, 0.0), omega
    if front_min < 0.80:
        return min(vx, 0.15), omega
    return vx, omega

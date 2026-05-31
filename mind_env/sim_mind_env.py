from __future__ import annotations

"""
Key points:
- No global map or goal; each step samples per-ray FOV distances according to the empty/obstacle ratio.
- Only yaw, world-frame linear velocity, and two-step command history are tracked for rewards; position is not tracked.
- The reference direction is randomly chosen from sectors satisfying the safety distance; if none exist, use the farthest sector and uniformly sample within its width.
- Episodes never terminate (done is always False) to simplify continuous control training.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List

import math
import numpy as np
from collections import OrderedDict
import mindspore as ms
from mindspore import ops, Tensor

from .ray import compute_ray_defaults

def _wrap_angle_pi(yaw: Tensor) -> Tensor:
    pi_tensor = ms.Tensor(math.pi, ms.float32)
    two_pi_tensor = ms.Tensor(2.0 * math.pi, ms.float32)
    return ((yaw + pi_tensor) % two_pi_tensor) - pi_tensor


@dataclass
class SimGPUEnvConfig:
    dt: float = 0.1
    n_envs: int = 256
    patch_meters: float = 10.0
    ray_step_m: float = 0.025
    n_rays: int = 0
    ray_max_gap: float = 0.25
    safe_distance_m: float = 0.75
    vx_max: float = 1.5
    omega_max: float = 2.0
    w_collision: float = 1.0
    w_progress: float = 0.01
    w_limits: float = 0.1
    orientation_verify: bool = False
    w_jerk: float = 0.0
    w_jerk_omega: float = 0.0
    w_danger: float = 0.0
    w_danger_speed: float = 0.0
    w_speed_match: float = 0.0
    danger_distance_m: float = 1.0
    stop_distance_m: float = 0.35
    reward_time: float = 0.0
    collision_done: bool = False
    blank_ratio_base: float = 40.0
    blank_ratio_randmax: float = 40.0
    blank_ratio_std_ratio: float = 0.33
    narrow_passage_gaussian: bool = False
    narrow_passage_std_ratio: float = 0.3
    front_obstacle_prob: float = 0.0
    front_obstacle_width_ratio: float = 0.18
    front_obstacle_min_m: float = 0.25
    front_obstacle_max_m: float = 1.0
    device: Optional[str] = None
    task_point_max_dist_m: float = 8.0
    task_point_success_radius_m: float = 0.25
    task_point_random_interval_max: int = 0
    w_success: float = 2.0
    w_intrinsic: float = 0.05
    max_reward_component: float = 2.0
    visit_count_grid_size: float = 1.0
    visit_count_max_entries: int = 50000
    intrinsic_decay_start: float = 0.6
    intrinsic_decay_range: float = 0.2
    enable_python_intrinsic: bool = False
    curriculum_enabled: bool = True


class VisitCounter:
    def __init__(self, grid_size: float = 1.0, max_entries: int = 50000):
        self.grid_size = grid_size
        self.max_entries = max_entries
        self._table: OrderedDict[int, int] = OrderedDict()

    def discretize(self, rays_n: np.ndarray, ref_feat: np.ndarray,
                   dist_n: float) -> int:
        n_rays = rays_n.shape[0]
        bin_size = max(n_rays // 8, 1)
        ray_bins = []
        for i in range(0, n_rays, bin_size):
            chunk = rays_n[i:i + bin_size]
            ray_bins.append(int(np.mean(chunk) * 10))
        ref_angle = math.atan2(ref_feat[0], ref_feat[1])
        dir_bin = int((ref_angle + math.pi) / (2 * math.pi) * 8) % 8
        dist_bin = int(dist_n * 10)
        return hash((tuple(ray_bins), dir_bin, dist_bin))

    def update_and_query(self, state_key: int) -> int:
        count = self._table.get(state_key, 0)
        if state_key in self._table:
            self._table[state_key] = count + 1
            self._table.move_to_end(state_key)
        else:
            if len(self._table) >= self.max_entries:
                self._table.popitem(last=False)
            self._table[state_key] = 1
        return count

    def reset(self):
        self._table.clear()


class ShapedRewardCalculator:
    def __init__(self, cfg: SimGPUEnvConfig, intrinsic_cfg: Optional[Dict] = None):
        self.w_progress = cfg.w_progress
        self.w_collision = cfg.w_collision
        self.w_success = cfg.w_success
        self.w_intrinsic = cfg.w_intrinsic
        self.w_jerk = cfg.w_jerk
        self.w_jerk_omega = cfg.w_jerk_omega
        self.w_danger = cfg.w_danger
        self.w_danger_speed = cfg.w_danger_speed
        self.w_speed_match = cfg.w_speed_match
        self.danger_distance_m = cfg.danger_distance_m
        self.stop_distance_m = cfg.stop_distance_m
        self.w_limits = cfg.w_limits
        self.reward_time = cfg.reward_time
        self.max_reward_component = cfg.max_reward_component
        self.task_point_max_dist_m = cfg.task_point_max_dist_m
        self.intrinsic_decay_start = cfg.intrinsic_decay_start
        self.intrinsic_decay_range = cfg.intrinsic_decay_range

        self._visit_counter = VisitCounter(
            grid_size=cfg.visit_count_grid_size,
            max_entries=cfg.visit_count_max_entries
        )
        self._train_progress = 0.0

    def _intrinsic_decay(self, progress: float) -> float:
        if progress < self.intrinsic_decay_start:
            return 1.0
        t = (progress - self.intrinsic_decay_start) / max(self.intrinsic_decay_range, 1e-6)
        return max(1.0 - t, 0.0)

    def set_train_progress(self, progress: float):
        self._train_progress = progress

    def compute(self, progress_val, collided, success, jerk_norm,
                jerk_omega_norm, limit_hit, v_ratio,
                task_dist, rays_n_np=None, ref_feat_np=None, dist_n_np=None,
                min_ray_dist_m=None, forward_speed_ratio=None, signed_speed_ratio=None,
                orientation_verify=True, cos_heading_val=None,
                delta_d=None) -> Tuple[Any, Dict[str, Any]]:
        mc = self.max_reward_component

        if min_ray_dist_m is None:
            danger = ops.zeros_like(progress_val)
        else:
            danger = ops.clamp(
                (self.danger_distance_m - min_ray_dist_m) / max(self.danger_distance_m, 1e-6),
                0.0,
                1.0,
            )

        # In dangerous frontal states, reduce but do not erase positive progress.
        pos_progress = ops.clamp(progress_val, 0.0, mc)
        neg_progress = ops.clamp(progress_val, -mc, 0.0)
        safe_progress = neg_progress + pos_progress * (0.3 + 0.7 * (1.0 - danger))
        rew_progress = ops.clamp(self.w_progress * safe_progress, -mc, mc)

        rew_collision = ops.clamp(
            -self.w_collision * collided.to(ms.float32),
            -mc, mc
        )

        success_bonus = ops.clamp(
            task_dist / max(self.task_point_max_dist_m, 1e-6),
            0.3, 1.0
        )
        rew_success = ops.clamp(
            self.w_success * success_bonus * success.to(ms.float32),
            -mc, mc
        )

        intrinsic_decay_factor = self._intrinsic_decay(self._train_progress)
        w_intrinsic_eff = self.w_intrinsic * intrinsic_decay_factor
        if w_intrinsic_eff > 0.0 and rays_n_np is not None and ref_feat_np is not None and dist_n_np is not None:
            rew_intrinsic_parts = []
            for i in range(rays_n_np.shape[0]):
                state_key = self._visit_counter.discretize(
                    rays_n_np[i], ref_feat_np[i], float(dist_n_np[i])
                )
                visit_count = self._visit_counter.update_and_query(state_key)
                rew_intrinsic_parts.append(
                    w_intrinsic_eff / math.sqrt(visit_count + 1)
                )
            rew_intrinsic = ops.clamp(
                Tensor(rew_intrinsic_parts, ms.float32),
                -mc, mc
            )
        else:
            rew_intrinsic = ops.zeros_like(progress_val)

        rew_jerk = ops.clamp(-self.w_jerk * jerk_norm, -mc, mc)
        rew_jerk_omega = ops.clamp(-self.w_jerk_omega * jerk_omega_norm, -mc, mc)
        if min_ray_dist_m is None:
            rew_danger = ops.zeros_like(progress_val)
            rew_danger_speed = ops.zeros_like(progress_val)
        else:
            rew_danger = ops.clamp(-self.w_danger * danger, -mc, mc)
            if forward_speed_ratio is None:
                forward_speed_ratio = ops.zeros_like(progress_val)
            rew_danger_speed = ops.clamp(
                -self.w_danger_speed * danger * ops.clamp(forward_speed_ratio, 0.0, 1.0),
                -mc,
                mc,
            )
        if min_ray_dist_m is None or signed_speed_ratio is None:
            rew_speed_match = ops.zeros_like(progress_val)
        else:
            slow_span = max(self.danger_distance_m - self.stop_distance_m, 1e-6)
            target_speed_ratio = ops.clamp(
                (min_ray_dist_m - self.stop_distance_m) / slow_span,
                0.0,
                1.0,
            )
            speed_err = signed_speed_ratio - target_speed_ratio
            rew_speed_match = ops.clamp(-self.w_speed_match * speed_err * speed_err, -mc, mc)
        rew_limits = ops.clamp(
            -self.w_limits * limit_hit.to(ms.float32),
            -mc, mc
        )
        rew_time = Tensor(-self.reward_time, ms.float32)

        total = (rew_progress + rew_collision + rew_success + rew_intrinsic
                 + rew_jerk + rew_jerk_omega + rew_danger + rew_danger_speed
                 + rew_speed_match + rew_limits + rew_time)

        components = {
            "rew_progress": rew_progress,
            "rew_collision": rew_collision,
            "rew_success": rew_success,
            "rew_intrinsic": rew_intrinsic,
            "rew_jerk": rew_jerk,
            "rew_jerk_omega": rew_jerk_omega,
            "rew_danger": rew_danger,
            "rew_danger_speed": rew_danger_speed,
            "rew_speed_match": rew_speed_match,
            "rew_limits": rew_limits,
            "rew_time": rew_time,
        }
        return total, components

    def get_component_names(self) -> List[str]:
        return ["rew_progress", "rew_collision", "rew_success",
                "rew_intrinsic", "rew_jerk", "rew_jerk_omega",
                "rew_danger", "rew_danger_speed", "rew_speed_match",
                "rew_limits", "rew_time"]

    def reset(self):
        self._visit_counter.reset()


class SimRandomBatchEnv:
    """Batch randomized ray environment with PPO-compatible observations and rewards.
    Adapted for MindSpore 2.x.
    """

    def __init__(self, cfg: SimGPUEnvConfig,
                 intrinsic_cfg: Optional[Dict] = None) -> None:
        self.cfg = cfg
        self.float_type = ms.float32
        self.int_type = ms.int32
        self.bool_type = ms.bool_

        n_rays = int(cfg.n_rays)
        if n_rays <= 0:
            n_rays, _, _ = compute_ray_defaults(
                {"ray_max_gap": float(cfg.ray_max_gap)},
                float(cfg.patch_meters),
            )
        self.n_rays = int(max(0, n_rays))
        self.view_radius_m = float(cfg.patch_meters)
        
        if self.view_radius_m <= 0.0:
            raise ValueError(f"patch_meters must be positive, got {self.view_radius_m}")
        if cfg.vx_max <= 0.0 or cfg.omega_max <= 0.0:
            raise ValueError("velocity limits must be positive")
            
        self.B = int(cfg.n_envs)
        
        self.t = ops.zeros((self.B,), self.int_type)
        self.yaw = ops.zeros((self.B,), self.float_type)
        self.vel_xy = ops.zeros((self.B, 2), self.float_type)
        self.pos_xy = ops.zeros((self.B, 2), self.float_type)
        self.prev_cmd = ops.zeros((self.B, 3), self.float_type)
        self.prev_prev_cmd = ops.zeros((self.B, 3), self.float_type)
        
        if self.n_rays > 0:
            self._ray_ang = ops.arange(self.n_rays, dtype=self.float_type) * (2.0 * math.pi / float(self.n_rays))
        else:
            self._ray_ang = ops.zeros((0,), dtype=self.float_type)
            
        self._rays_m = ops.zeros((self.B, self.n_rays), dtype=self.float_type)
        self._ref_vec = ops.zeros((self.B, 2), dtype=self.float_type)
        self._ref_feat = ops.zeros((self.B, 2), dtype=self.float_type)
        self._global_task_xy = ops.zeros((self.B, 2), dtype=self.float_type)
        self._local_task_xy = ops.zeros((self.B, 2), dtype=self.float_type)
        
        self.interval_max = int(getattr(self.cfg, "task_point_random_interval_max", 0))
        self._task_redraw_counter = ops.zeros((self.B,), dtype=self.int_type)
        
        if self.interval_max > 0:
            self._task_redraw_target = ops.randint(1, self.interval_max + 1, (self.B,), dtype=self.int_type)
        else:
            self._task_redraw_target = ops.zeros((self.B,), dtype=self.int_type)
            
        self._resample_fov_and_ref()
        self._sample_new_global_task_points(mask=ops.ones((self.B,), dtype=self.bool_type))

        self._reward_calc = ShapedRewardCalculator(cfg, intrinsic_cfg)
        self._last_reward_components: Dict[str, Any] = {}

    def get_limits(self) -> Tensor:
        return Tensor([self.cfg.vx_max, self.cfg.omega_max], dtype=self.float_type)

    def reset(self) -> Tensor:
        self.t = ops.zeros((self.B,), self.int_type)
        self.yaw = ops.zeros((self.B,), self.float_type)
        self.vel_xy = ops.zeros((self.B, 2), self.float_type)
        self.pos_xy = ops.zeros((self.B, 2), self.float_type)
        self.prev_cmd = ops.zeros((self.B, 3), self.float_type)
        self.prev_prev_cmd = ops.zeros((self.B, 3), self.float_type)
        
        self._resample_fov_and_ref()
        self._sample_new_global_task_points(mask=ops.ones((self.B,), dtype=self.bool_type))
        
        if self.interval_max > 0:
            self._task_redraw_counter = ops.zeros((self.B,), dtype=self.int_type)
            self._task_redraw_target = ops.randint(1, self.interval_max + 1, (self.B,), dtype=self.int_type)
        
        self._reward_calc.reset()
        return self.observe()

    def observe(self) -> Tensor:
        return self._build_obs(self._rays_m, self._ref_feat)

    def set_train_progress(self, progress: float):
        self._reward_calc.set_train_progress(progress)

    def get_reward_components(self) -> Dict[str, Any]:
        return self._last_reward_components

    def step(self, action: Tensor) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Any]]:
        vx_max = float(self.cfg.vx_max)
        om_max = float(self.cfg.omega_max)
        dt = float(self.cfg.dt)

        vx_cmd = ops.clamp(action[:, 0], -vx_max, vx_max)
        if action.shape[1] == 2:
            vy_cmd = ops.zeros_like(vx_cmd)
            om_cmd = ops.clamp(action[:, 1], -om_max, om_max)
        else:
            om_cmd = ops.clamp(action[:, 2], -om_max, om_max)
            vy_cmd = ops.zeros_like(vx_cmd)

        pos_prev = self.pos_xy.copy()
        dx_local = vx_cmd * dt
        dy_local = vy_cmd * dt
        
        yaw_end = _wrap_angle_pi(self.yaw + om_cmd * dt)
        c1 = ops.cos(yaw_end)
        s1 = ops.sin(yaw_end)
        
        vx_w_end = c1 * vx_cmd - s1 * vy_cmd
        vy_w_end = s1 * vx_cmd + c1 * vy_cmd
        
        self.vel_xy = ops.stack([vx_w_end, vy_w_end], axis=-1)
        self.yaw = yaw_end
        self.t = self.t + 1
        
        if self.interval_max > 0:
            self._task_redraw_counter = self._task_redraw_counter + 1
            
        pos_x_new = self.pos_xy[:, 0] + vx_w_end * dt
        pos_y_new = self.pos_xy[:, 1] + vy_w_end * dt
        self.pos_xy = ops.stack([pos_x_new, pos_y_new], axis=-1)
        
        travel = ops.sqrt(ops.square(dx_local) + ops.square(dy_local))
        ang = ops.atan2(dy_local, dx_local)
        ray_d_along = self._interp_ray_distance(self._rays_m, ang)
        
        pen = travel - ray_d_along
        collided = ops.logical_and((pen > 1e-6), (ray_d_along > 0.0))
        
        d_prev = ops.sqrt(ops.square(self._global_task_xy[:, 0] - pos_prev[:, 0]) + ops.square(self._global_task_xy[:, 1] - pos_prev[:, 1]))
        d_next = ops.sqrt(ops.square(self._global_task_xy[:, 0] - self.pos_xy[:, 0]) + ops.square(self._global_task_xy[:, 1] - self.pos_xy[:, 1]))
        delta_d = d_next - d_prev
        
        denom_progress = vx_max * dt
        jerk_x = (vx_cmd - 2.0 * self.prev_cmd[:, 0] + self.prev_prev_cmd[:, 0]) / vx_max
        jerk_omega = (om_cmd - 2.0 * self.prev_cmd[:, 2] + self.prev_prev_cmd[:, 2]) / om_max
        limit_hit = ops.logical_or((ops.abs(vx_cmd) >= vx_max - 1e-9), (ops.abs(om_cmd) >= om_max - 1e-9))

        min_ray_dist_m = self._rays_m.min(axis=-1)
        if self.n_rays > 0:
            front_width = max(1, int(round(self.n_rays / 8)))
            front_rays = self._rays_m[:, :front_width]
            front_ray_dist_m = front_rays.min(axis=-1)
        else:
            front_ray_dist_m = min_ray_dist_m
        task_resampled = ops.zeros((self.B,), dtype=self.bool_type)

        jerk_norm = ops.clamp((jerk_x * jerk_x) / 16.0, 0.0, 1.0)
        jerk_omega_norm = ops.clamp((jerk_omega * jerk_omega) / 16.0, 0.0, 1.0)
        v_lin = ops.sqrt(ops.square(vx_w_end) + ops.square(vy_w_end))
        v_ratio = ops.clamp(v_lin / vx_max, 0.0, 1.0)
        forward_speed_ratio = ops.clamp(vx_cmd / vx_max, 0.0, 1.0)
        signed_speed_ratio = ops.clamp(vx_cmd / vx_max, -1.0, 1.0)
        base_progress = (-delta_d) / denom_progress

        if bool(self.cfg.orientation_verify):
            dot_hv = c1 * vx_w_end + s1 * vy_w_end
            cos_heading_vel = ops.where(
                v_lin > 1e-9,
                ops.clamp(dot_hv / v_lin, -1.0, 1.0),
                ops.ones_like(v_lin)
            )
        else:
            cos_heading_vel = ops.ones_like(base_progress)

        progress_val = ops.where(delta_d > 0.0, -ops.abs(base_progress), base_progress)
        if bool(self.cfg.orientation_verify):
            allow_pos = ops.logical_and(delta_d < 0.0, cos_heading_vel > 0.0)
            progress_val = ops.where(allow_pos, progress_val, -ops.abs(progress_val))

        progress_val = ops.where(delta_d > 0.0, -ops.abs(progress_val), progress_val)

        self.prev_prev_cmd = self.prev_cmd
        zero_hist = ops.zeros_like(vx_cmd)
        self.prev_cmd = ops.stack([vx_cmd, zero_hist, om_cmd], axis=-1)

        # 碰撞终止与状态重置
        if bool(getattr(self.cfg, "collision_done", False)):
            term = collided.to(self.bool_type)
            self.t = ops.where(term, ops.zeros_like(self.t), self.t)
            self.yaw = ops.where(term, ops.zeros_like(self.yaw), self.yaw)
            term_2d = term.unsqueeze(-1).broadcast_to(self.vel_xy.shape)
            term_3d = term.unsqueeze(-1).broadcast_to(self.prev_cmd.shape)
            self.vel_xy = ops.where(term_2d, ops.zeros_like(self.vel_xy), self.vel_xy)
            self.pos_xy = ops.where(term_2d, ops.zeros_like(self.pos_xy), self.pos_xy)
            self.prev_cmd = ops.where(term_3d, ops.zeros_like(self.prev_cmd), self.prev_cmd)
            self.prev_prev_cmd = ops.where(term_3d, ops.zeros_like(self.prev_prev_cmd), self.prev_prev_cmd)
            self._sample_new_global_task_points(mask=term)
            task_resampled = ops.logical_or(task_resampled, term)
        else:
            term = ops.zeros((self.B,), dtype=self.bool_type)

        # 成功抵达判定
        u = self.pos_xy - pos_prev
        uu = u[:, 0] * u[:, 0] + u[:, 1] * u[:, 1]
        w0 = self._global_task_xy - pos_prev
        
        move_mask = uu > 0.0
        safe_uu = ops.where(move_mask, uu, ops.ones_like(uu))
        t_proj_raw = (w0[:, 0] * u[:, 0] + w0[:, 1] * u[:, 1]) / safe_uu
        t_proj = ops.where(move_mask, ops.clamp(t_proj_raw, 0.0, 1.0), ops.zeros_like(uu))
        
        nearest_x = pos_prev[:, 0] + t_proj * u[:, 0]
        nearest_y = pos_prev[:, 1] + t_proj * u[:, 1]
        dist2_near = ops.square(nearest_x - self._global_task_xy[:, 0]) + ops.square(nearest_y - self._global_task_xy[:, 1])
        
        r_s = float(self.cfg.task_point_success_radius_m)
        success = dist2_near <= (r_s * r_s)

        task_dist = d_prev
        
        self._sample_new_global_task_points(mask=success)
        task_resampled = ops.logical_or(task_resampled, success)

        self._resample_fov_and_ref()
        obs_next = self._build_obs(self._rays_m, self._ref_feat)

        # 奖励重塑计算
        rays_n_np = ref_feat_np = dist_n_np = None
        if float(getattr(self.cfg, "w_intrinsic", 0.0)) > 0.0 and bool(getattr(self.cfg, "enable_python_intrinsic", False)):
            rays_n_np = ops.clamp(self._rays_m / self.view_radius_m, 0.0, 1.0).asnumpy()
            ref_feat_np = self._ref_feat.asnumpy()
            local_dx = self._local_task_xy[:, 0] - self.pos_xy[:, 0]
            local_dy = self._local_task_xy[:, 1] - self.pos_xy[:, 1]
            local_dist = ops.sqrt(ops.square(local_dx) + ops.square(local_dy))
            dist_n_np = ops.clamp(local_dist / self.view_radius_m, 0.0, 1.0).asnumpy()

        rew, components = self._reward_calc.compute(
            progress_val=progress_val,
            collided=collided,
            success=success,
            jerk_norm=jerk_norm,
            jerk_omega_norm=jerk_omega_norm,
            limit_hit=limit_hit,
            v_ratio=v_ratio,
            min_ray_dist_m=front_ray_dist_m,
            forward_speed_ratio=forward_speed_ratio,
            signed_speed_ratio=signed_speed_ratio,
            task_dist=task_dist,
            rays_n_np=rays_n_np,
            ref_feat_np=ref_feat_np,
            dist_n_np=dist_n_np,
            orientation_verify=bool(self.cfg.orientation_verify),
            cos_heading_val=cos_heading_vel,
            delta_d=delta_d
        )
        self._last_reward_components = components

        info: Dict[str, Any] = {
            "limits": self.get_limits(),
            "success": success,
            "success_count": success.to(ms.int32).sum(),
            "timeout": ops.zeros((self.B,), dtype=self.bool_type),
            "min_ray_dist_m": min_ray_dist_m,
            "collided": collided,
            "reward_components": components,
        }
        return obs_next, rew, term, info

    def _resample_fov_and_ref(self) -> None:
        if self.n_rays <= 0:
            self._rays_m = ops.zeros_like(self._rays_m)
            self._update_local_task_points()
            self._update_ref_from_local()
            return

        base = float(self.cfg.blank_ratio_base)
        jitter = float(self.cfg.blank_ratio_randmax)
        std_ratio = float(getattr(self.cfg, "blank_ratio_std_ratio", 0.33))
        sigma = max(jitter * std_ratio, 1e-6)
        
        p_empty_raw = ops.standard_normal((self.B,)) * sigma + base
        p_empty = (ops.clamp(p_empty_raw, base, base + jitter) / 100.0).to(self.float_type)
        
        mask_empty = ops.uniform((self.B, self.n_rays), Tensor(0.0, ms.float32), Tensor(1.0, ms.float32)) < p_empty.view(-1, 1)
        
        use_gaussian = bool(getattr(self.cfg, "narrow_passage_gaussian", False))
        if use_gaussian:
            std_ratio = float(getattr(self.cfg, "narrow_passage_std_ratio", 0.3))
            sigma = max(self.view_radius_m * std_ratio, 1e-6)
            dist = ops.abs(ops.standard_normal((self.B, self.n_rays))) * sigma
            dist = ops.clamp(dist, 0.0, self.view_radius_m)
            rays_m = ops.where(mask_empty, ops.full_like(dist, self.view_radius_m), dist)
        else:
            prop = ops.uniform((self.B, self.n_rays), Tensor(0.0, ms.float32), Tensor(1.0, ms.float32))
            rays_m = prop * self.view_radius_m
            rays_m = ops.where(mask_empty, ops.full_like(rays_m, self.view_radius_m), rays_m)
            
        self._rays_m = rays_m.to(self.float_type)
        self._inject_front_obstacles()
        self._update_local_task_points()
        self._update_ref_from_local()

    def _inject_front_obstacles(self) -> None:
        prob = float(getattr(self.cfg, "front_obstacle_prob", 0.0))
        if self.n_rays <= 0 or prob <= 0.0:
            return

        width = max(1, int(round(self.n_rays * float(getattr(self.cfg, "front_obstacle_width_ratio", 0.18)))))
        width = min(width, self.n_rays)
        min_m = max(0.0, float(getattr(self.cfg, "front_obstacle_min_m", 0.25)))
        max_m = max(min_m, float(getattr(self.cfg, "front_obstacle_max_m", 1.0)))

        apply_mask = ops.uniform((self.B, 1), Tensor(0.0, ms.float32), Tensor(1.0, ms.float32)) < prob
        d = ops.uniform((self.B, 1), Tensor(min_m, ms.float32), Tensor(max_m, ms.float32))
        front = self._rays_m[:, :width]
        front = ops.where(apply_mask.broadcast_to(front.shape), ops.minimum(front, d.broadcast_to(front.shape)), front)
        self._rays_m = ops.concat([front, self._rays_m[:, width:]], axis=1)

    def _update_local_task_points(self, mask: Optional[Tensor] = None) -> None:
        dx = self._global_task_xy[:, 0] - self.pos_xy[:, 0]
        dy = self._global_task_xy[:, 1] - self.pos_xy[:, 1]
        dist_global = ops.sqrt(ops.square(dx) + ops.square(dy))

        nz = dist_global > 0.0
        safe_dist = ops.where(nz, dist_global, ops.ones_like(dist_global))
        dir_x = ops.where(nz, dx / safe_dist, ops.zeros_like(dx))
        dir_y = ops.where(nz, dy / safe_dist, ops.zeros_like(dy))

        ang_world = ops.atan2(dy, dx)
        ang_body = _wrap_angle_pi(ang_world - self.yaw)
        los_dist = ops.clamp(self._interp_ray_distance(self._rays_m, ang_body), min=0.0)
        travel = ops.minimum(dist_global, los_dist)

        new_x = self.pos_xy[:, 0] + travel * dir_x
        new_y = self.pos_xy[:, 1] + travel * dir_y

        if mask is None:
            self._local_task_xy = ops.stack([new_x, new_y], axis=-1)
        else:
            mask_2d = mask.unsqueeze(-1).broadcast_to(self._local_task_xy.shape)
            new_xy = ops.stack([new_x, new_y], axis=-1)
            self._local_task_xy = ops.where(mask_2d, new_xy, self._local_task_xy)

    def _update_ref_from_local(self, mask: Optional[Tensor] = None) -> None:
        dx = self._local_task_xy[:, 0] - self.pos_xy[:, 0]
        dy = self._local_task_xy[:, 1] - self.pos_xy[:, 1]
        n = ops.sqrt(ops.square(dx) + ops.square(dy))
        
        hx = ops.cos(self.yaw)
        hy = ops.sin(self.yaw)
        tx = ops.where(n > 1e-9, dx / ops.where(n > 1e-9, n, ops.ones_like(n)), hx)
        ty = ops.where(n > 1e-9, dy / ops.where(n > 1e-9, n, ops.ones_like(n)), hy)

        cos_th = ops.clamp(tx * hx + ty * hy, -1.0, 1.0)
        sin_th = ops.clamp(ty * hx - tx * hy, -1.0, 1.0)
        ref_feat = ops.stack([sin_th, cos_th], axis=-1)
        ref_vec = ops.stack([tx, ty], axis=-1)

        if mask is None:
            self._ref_vec = ref_vec
            self._ref_feat = ref_feat
        else:
            mask_2d = mask.unsqueeze(-1).broadcast_to(self._ref_vec.shape)
            self._ref_vec = ops.where(mask_2d, ref_vec, self._ref_vec)
            self._ref_feat = ops.where(mask_2d, ref_feat, self._ref_feat)

    def _sample_new_global_task_points(self, mask: Tensor) -> None:
        r_min = max(float(self.cfg.task_point_success_radius_m), 0.0)
        r_max = float(min(float(self.cfg.task_point_max_dist_m), float(self.view_radius_m)))
        r_max = max(r_max, r_min)
        
        u = ops.uniform((self.B,), Tensor(0.0, ms.float32), Tensor(1.0, ms.float32))
        dist = r_min + (r_max - r_min) * u
        safe = self._rays_m >= float(self.cfg.safe_distance_m)
        has_any = safe.any(axis=-1)
        
        argmax_idx = self._rays_m.argmax(axis=-1)
        rand_scores = ops.uniform((self.B, self.n_rays), Tensor(0.0, ms.float32), Tensor(1.0, ms.float32))
        scores = ops.where(safe, rand_scores, ops.full_like(rand_scores, float("-inf")))
        pick_any = ops.argmax(scores, dim=-1)
        
        idx = ops.where(has_any, pick_any, argmax_idx)
        if self.n_rays > 0:
            idx = ops.clamp(idx, 0, self.n_rays - 1)
            
        dth = (2.0 * math.pi) / float(max(self.n_rays, 1))
        jitter_local = (ops.uniform((self.B,), Tensor(0.0, ms.float32), Tensor(1.0, ms.float32)) - 0.5) * float(dth)
        
        ang = self._ray_ang[idx]
        th = _wrap_angle_pi(self.yaw + ang + jitter_local)
        
        new_x = self.pos_xy[:, 0] + dist * ops.cos(th)
        new_y = self.pos_xy[:, 1] + dist * ops.sin(th)
        new_xy = ops.stack([new_x, new_y], axis=-1)
        
        mask_2d = mask.unsqueeze(-1).broadcast_to(self._global_task_xy.shape)
        self._global_task_xy = ops.where(mask_2d, new_xy, self._global_task_xy)

        self._update_local_task_points(mask=mask)
        self._update_ref_from_local(mask=mask)

    def _build_obs(self, rays_m: Tensor, ref_feat: Tensor) -> Tensor:
        vx_lim = float(self.cfg.vx_max)
        om_lim = float(self.cfg.omega_max)
        prev_vx_n = self.prev_cmd[:, 0] / vx_lim
        prev_om_n = self.prev_cmd[:, 2] / om_lim
        prev_cmd_n = ops.stack([prev_vx_n, prev_om_n], axis=-1)
        
        dvx_n = (self.prev_cmd[:, 0] - self.prev_prev_cmd[:, 0]) / (2.0 * vx_lim)
        dom_n = (self.prev_cmd[:, 2] - self.prev_prev_cmd[:, 2]) / (2.0 * om_lim)
        dprev_all_n = ops.stack([dvx_n, dom_n], axis=-1)
        
        dist = ops.sqrt(ops.square(self._local_task_xy[:, 0] - self.pos_xy[:, 0]) + ops.square(self._local_task_xy[:, 1] - self.pos_xy[:, 1]))
        dist_n = ops.clamp(dist / self.view_radius_m, 0.0, 1.0).unsqueeze(-1)

        if self.n_rays <= 0:
            return ops.concat([ref_feat, prev_cmd_n, dprev_all_n, dist_n], axis=-1).to(self.float_type)

        rays_n = ops.clamp(rays_m / self.view_radius_m, 0.0, 1.0)
        parts = [rays_n, ref_feat, prev_cmd_n, dprev_all_n, dist_n]
        return ops.concat(parts, axis=-1).to(self.float_type)

    def _interp_ray_distance(self, rays_m: Tensor, angle: Tensor) -> Tensor:
        if self.n_rays <= 0:
            return ops.full((self.B,), float("inf"), dtype=self.float_type)
            
        R = float(self.n_rays)
        dth = (2.0 * math.pi) / R

        a = angle % (2.0 * math.pi)
        f = a / dth
        i0 = ops.floor(f).to(self.int_type)
        t = f - i0.to(self.float_type)
        
        if self.n_rays > 0:
            i0 = ops.clamp(i0, 0, self.n_rays - 1)
            i1 = (i0 + 1) % int(self.n_rays)
        else:
            i1 = i0
            
        ar = ops.arange(self.B, dtype=self.int_type)
        d0 = rays_m[ar, i0]
        d1 = rays_m[ar, i1]
        return d0 * (1.0 - t) + d1 * t

def infer_obs_dim(cfg) -> int:
    """独立于类之外的维度推断函数"""
    if int(cfg.n_rays) <= 0:
        return 7
    return int(cfg.n_rays) + 7
    
if __name__ == "__main__":
    cfg = SimGPUEnvConfig(n_envs=4)
    env = SimRandomBatchEnv(cfg)
    
    obs = env.reset()
    print(f"Observation shape: {obs.shape}")
    
    test_action = ms.ops.standard_normal((4, 3)) 
    obs_next, reward, terminated, info = env.step(test_action)
    
    print(f"Reward sample: {reward}")
    print(f"Collided mask: {info['collided']}")
    print(f"Success count: {info['success_count']}")
    print(f"Reward components: {list(info['reward_components'].keys())}")
    print("Successfully ran one step!")

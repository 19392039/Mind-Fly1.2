from __future__ import annotations

import os
import time
import argparse
import random
import json
import numpy as np
from typing import Any, Dict, Optional, List
from collections import deque, namedtuple
import sys

current_script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_script_dir)
sys.path.insert(0, root_dir)

import mindspore as ms
from mindspore import Tensor
from mindspore import nn, ops, context, save_checkpoint, load_checkpoint, load_param_into_net
from mindspore.common import dtype as mstype
from tensorboardX import SummaryWriter

from mind_models import PPOPolicy
from mind_buffer import RolloutBuffer
from mind_env import load_json_config
from mind_env.sim_mind_env import SimGPUEnvConfig, SimRandomBatchEnv, infer_obs_dim as _infer_obs_dim_sim


def load_train_config(path: Optional[str]) -> Dict[str, Any]:
    cfg = load_json_config(path) if path else {}
    cfg.setdefault("device", "Ascend")
    cfg.setdefault("env_config", "env_config.json")
    cfg.setdefault("mission_config", None)
    
    samp = cfg.setdefault("sampling", {})
    samp.setdefault("batch_env", 256)
    samp.setdefault("rollout_len", 128)
    samp.setdefault("reset_each_rollout", True)
    
    ppo = cfg.setdefault("ppo", {})
    ppo.setdefault("gamma", 0.99)
    ppo.setdefault("gae_lambda", 0.95)
    ppo.setdefault("clip_range", 0.2)
    ppo.setdefault("clip_range_final", 0.1)
    ppo.setdefault("lr", 3e-4)
    ppo.setdefault("value_lr", 3e-4)
    ppo.setdefault("lr_schedule", "linear")
    ppo.setdefault("lr_min_ratio", 0.01)
    ppo.setdefault("entropy_coef", 0.05)
    ppo.setdefault("entropy_coef_final", 0.02)
    ppo.setdefault("entropy_decay_progress", 0.9)
    ppo.setdefault("min_entropy", "auto")
    ppo.setdefault("value_coef", 0.5)
    ppo.setdefault("max_grad_norm", 0.5)
    ppo.setdefault("epochs", 4)
    ppo.setdefault("minibatch_size", 2048)
    ppo.setdefault("amp", False)
    ppo.setdefault("amp_bf16", True)
    ppo.setdefault("bootstrap", True)
    ppo.setdefault("log_std_min", -2.0)
    ppo.setdefault("log_std_max", 1.0)
    ppo.setdefault("collision_done", True)
    ppo.setdefault("value_clip", True)
    ppo.setdefault("grad_norm_adapt", True)
    ppo.setdefault("adam_eps", 1e-5)
    ppo.setdefault("log_std_weight_decay", 0.0)
    ppo.setdefault("nan_skip", True)
    ppo.setdefault("kl_target", 0.015)
    ppo.setdefault("kl_early_stop", True)
    ppo.setdefault("min_recovery_steps", 2000)
    ppo.setdefault("max_collapse_count", 5)
    ppo.setdefault("advantage_clip", 10.0)
    ppo.setdefault("gae_lambda_explore", 0.98)
    ppo.setdefault("entropy_safe_threshold", 0.3)
    
    diag = cfg.setdefault("diagnostics", {})
    diag.setdefault("stagnation_window", 10)
    diag.setdefault("stagnation_threshold", 1e-4)
    diag.setdefault("relative_stagnation_threshold", 0.05)
    diag.setdefault("kl_warning_threshold", 0.02)
    diag.setdefault("vf_drift_threshold", 1.0)
    
    model = cfg.setdefault("model", {})
    model.setdefault("num_queries", 4)
    model.setdefault("num_heads", 4)
    
    run = cfg.setdefault("run", {})
    run.setdefault("total_env_steps", 2_000_000)
    run.setdefault("ckpt_dir", "runs/ppo_exp1")
    run.setdefault("log_interval", 20000)
    run.setdefault("eval_every", 100000)

    cur = cfg.setdefault("curriculum", {})
    cur.setdefault("enabled", True)
    cur.setdefault("stage_hold_windows", 5)
    cur.setdefault("fallback_threshold", 0.1)
    cur.setdefault("skip_on_stall", False)
    cur.setdefault("skip_after_steps", 1_000_000)
    cur.setdefault("stages", [
        {"blank_ratio_base": 60.0, "narrow_passage_std_ratio": 0.3,
         "task_point_max_dist_m": 8.0, "success_threshold": 0.3},
        {"blank_ratio_base": 50.0, "narrow_passage_std_ratio": 0.2,
         "task_point_max_dist_m": 15.0, "success_threshold": 0.3},
        {"blank_ratio_base": 40.0, "narrow_passage_std_ratio": 0.15,
         "task_point_max_dist_m": 30.0, "success_threshold": 0.3},
    ])

    intr = cfg.setdefault("intrinsic_reward", {})
    intr.setdefault("w_intrinsic", 0.05)
    intr.setdefault("visit_count_grid_size", 1.0)
    intr.setdefault("visit_count_max_entries", 50000)
    intr.setdefault("intrinsic_decay_start", 0.6)
    intr.setdefault("intrinsic_decay_range", 0.2)

    return cfg

def _extract_seed_from_train_or_env(train_cfg: Dict[str, Any], env_cfg: Dict[str, Any]) -> Optional[int]:
    try:
        run = train_cfg.get("run", {}) or {}
        seed_v = run.get("seed", None)
        if seed_v in (None, "", "null"):
            seed_v = None
        if seed_v is not None:
            return int(seed_v)
        sim = env_cfg.get("sim", {}) or {}
        s2 = sim.get("seed", None)
        return None if s2 in (None, "", "null") else int(s2)
    except Exception:
        return None

class LRScheduler:
    def __init__(self, lr_init, value_lr_init, total_steps, schedule="linear", min_lr_ratio=0.01):
        self.lr_init = lr_init
        self.value_lr_init = value_lr_init
        self.total_steps = max(total_steps, 1)
        self.schedule = schedule
        self.min_lr_ratio = min_lr_ratio

    def step(self, current_step):
        progress = min(current_step / self.total_steps, 1.0)
        if self.schedule == "linear":
            factor = 1.0 - progress * (1.0 - self.min_lr_ratio)
        else:
            factor = 1.0
        factor = max(factor, self.min_lr_ratio)
        return self.lr_init * factor, self.value_lr_init * factor

    def state_dict(self):
        return {"lr_init": self.lr_init, "value_lr_init": self.value_lr_init,
                "total_steps": self.total_steps, "schedule": self.schedule,
                "min_lr_ratio": self.min_lr_ratio}


class ClipRangeScheduler:
    def __init__(self, clip_init, clip_final, total_steps, kl_target=0.015, kl_early_stop=True):
        self.clip_init = clip_init
        self.clip_final = clip_final
        self.total_steps = max(total_steps, 1)
        self.kl_target = kl_target
        self.kl_early_stop = kl_early_stop

    def step(self, current_step, approx_kl=0.0):
        progress = min(current_step / self.total_steps, 1.0)
        clip_range = self.clip_init + (self.clip_final - self.clip_init) * progress
        should_stop = False
        if self.kl_early_stop and approx_kl > self.kl_target * 1.5:
            clip_range *= 0.8
            if approx_kl > self.kl_target * 4.0:
                should_stop = True
        return max(clip_range, 0.01), should_stop


class EntropyControllerV2:
    def __init__(self, ent_coef_init, ent_coef_final, decay_progress, total_steps,
                 min_entropy=0.1, log_std_param=None, log_std_min=-2.0, log_std_max=1.0,
                 min_recovery_steps=2000, max_collapse_count=5,
                 action_dim=2):
        self.ent_coef_init = ent_coef_init
        self.ent_coef_final = max(ent_coef_final, 0.02)
        self.decay_progress = max(decay_progress, 0.9)
        self.total_steps = max(total_steps, 1)
        self.log_std_param = log_std_param
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.min_recovery_steps = min_recovery_steps
        self.max_collapse_count = max_collapse_count

        if min_entropy == "auto":
            self.min_entropy = max(
                0.5 * action_dim * (1.0 + np.log(2 * np.pi) + 2.0 * log_std_min),
                0.05
            )
        else:
            self.min_entropy = float(min_entropy)

        self.entropy_history = deque(maxlen=10)
        self.collapse_count = 0
        self._recovery_remaining = 0

    def step(self, current_step, current_entropy, is_collapsed=False):
        self.entropy_history.append(float(current_entropy))
        progress = min(current_step / self.total_steps, 1.0)

        if self._recovery_remaining > 0:
            self._recovery_remaining -= 1
            ent_coef = self.ent_coef_init
        else:
            if progress < self.decay_progress:
                t = progress / self.decay_progress
                ent_coef = self.ent_coef_init + (self.ent_coef_final - self.ent_coef_init) * t
            else:
                ent_coef = self.ent_coef_final

        if len(self.entropy_history) >= 5:
            avg_ent = np.mean(list(self.entropy_history)[-5:])
            if avg_ent < self.min_entropy or is_collapsed:
                self.collapse_count += 1
                ent_coef = self.ent_coef_init
                self._recovery_remaining = self.min_recovery_steps

                if self.collapse_count > self.max_collapse_count:
                    self._aggressive_recovery()
                    print(f"[CRITICAL] 连续坍缩第{self.collapse_count}次，执行激进恢复")
                else:
                    print(f"[WARN] 策略坍缩(entropy={avg_ent:.4f}<{self.min_entropy:.4f})，"
                          f"恢复entropy_coef={ent_coef:.6f}，保护窗口{self.min_recovery_steps}步")

        return max(ent_coef, self.ent_coef_final)

    def _aggressive_recovery(self):
        if self.log_std_param is not None:
            mid_val = (self.log_std_min + self.log_std_max) / 2.0
            reset_val = np.array([mid_val] * int(self.log_std_param.shape[0]), dtype=np.float32)
            current_log_std = float(self.log_std_param.asnumpy().mean())
            self.log_std_param.set_data(Tensor(reset_val, mstype.float32))
            print(f"[WARN] 激进恢复: 重置log_std从{current_log_std:.4f}到{mid_val:.4f}")

    def get_collapse_count(self):
        return self.collapse_count

    def state_dict(self):
        return {
            "collapse_count": self.collapse_count,
            "recovery_remaining": self._recovery_remaining,
            "min_entropy": self.min_entropy,
        }


class GradNormController:
    def __init__(self, max_grad_norm, adapt=True):
        self.max_grad_norm = max_grad_norm
        self.adapt = adapt
        self.clip_count_window = deque(maxlen=100)

    def check_and_adjust(self, was_clipped):
        if not self.adapt:
            return
        self.clip_count_window.append(1 if was_clipped else 0)
        if len(self.clip_count_window) >= 50:
            clip_freq = sum(self.clip_count_window) / len(self.clip_count_window)
            if clip_freq > 0.5:
                self.max_grad_norm *= 1.2
                print(f"[WARN] 梯度裁剪频率{clip_freq:.1%}>50%，增大max_grad_norm至{self.max_grad_norm:.4f}")
                self.clip_count_window.clear()


CollapseEvent = namedtuple("CollapseEvent", [
    "timestamp", "step", "entropy", "reward", "kl",
    "success_rate", "log_std_mean", "collapse_count", "recovery_action"
])

class DiagnosticsMonitorV2:
    def __init__(self, stagnation_window=10, stagnation_threshold=1e-4,
                 relative_stagnation_threshold=0.05,
                 kl_warning_threshold=0.02, vf_drift_threshold=1.0,
                 min_entropy=0.1):
        self.stagnation_window = stagnation_window
        self.stagnation_threshold = stagnation_threshold
        self.relative_stagnation_threshold = relative_stagnation_threshold
        self.kl_warning_threshold = kl_warning_threshold
        self.vf_drift_threshold = vf_drift_threshold
        self.min_entropy = min_entropy

        self.policy_loss_history = deque(maxlen=stagnation_window + 2)
        self.value_loss_history = deque(maxlen=stagnation_window + 2)
        self.kl_history = deque(maxlen=10)
        self.entropy_history = deque(maxlen=10)
        self.reward_moving_avg = deque(maxlen=100)
        self.success_rate_history = deque(maxlen=50)
        self._collapse_events: List[CollapseEvent] = []

    def _check_reward_stagnation(self) -> bool:
        if len(self.reward_moving_avg) < self.stagnation_window:
            return False
        recent = list(self.reward_moving_avg)[-self.stagnation_window:]
        mean_r = np.mean(recent)
        if abs(mean_r) < 1e-8:
            return False
        std_r = np.std(recent)
        return (std_r / abs(mean_r)) < self.relative_stagnation_threshold

    def update(self, policy_loss, value_loss, approx_kl, entropy,
               avg_reward=None, success_rate=None):
        warnings = []
        self.policy_loss_history.append(policy_loss)
        self.value_loss_history.append(value_loss)
        self.kl_history.append(approx_kl)
        self.entropy_history.append(entropy)
        if avg_reward is not None:
            self.reward_moving_avg.append(avg_reward)
        if success_rate is not None:
            self.success_rate_history.append(success_rate)

        is_collapsed = False

        if len(self.policy_loss_history) > self.stagnation_window:
            recent = list(self.policy_loss_history)[-self.stagnation_window:]
            change_rate = abs(recent[-1] - recent[0]) / (abs(recent[0]) + 1e-8)
            if change_rate < self.stagnation_threshold:
                warnings.append(f"policy_loss停滞(change_rate={change_rate:.2e}<{self.stagnation_threshold})")

        if len(self.kl_history) >= 3:
            if all(kl > self.kl_warning_threshold for kl in list(self.kl_history)[-3:]):
                warnings.append(f"KL散度持续异常(最近3次>{self.kl_warning_threshold})")

        entropy_low = len(self.entropy_history) >= 5 and np.mean(list(self.entropy_history)[-5:]) < self.min_entropy
        reward_stagnant = self._check_reward_stagnation()

        if entropy_low and reward_stagnant:
            is_collapsed = True
            warnings.append("策略坍缩联合检测: entropy低 AND reward停滞")
        elif entropy_low:
            warnings.append(f"策略熵偏低(近5轮平均{np.mean(list(self.entropy_history)[-5:]):.4f}<{self.min_entropy})")

        return warnings, is_collapsed

    def get_success_rate(self) -> Optional[float]:
        if not self.success_rate_history:
            return None
        return float(np.mean(list(self.success_rate_history)[-5:]))

    def get_collapse_events(self) -> List[CollapseEvent]:
        return list(self._collapse_events)

    def record_collapse_event(self, step, entropy, reward, kl,
                               success_rate, log_std_mean, collapse_count, recovery_action):
        evt = CollapseEvent(
            timestamp=time.time(), step=step, entropy=entropy,
            reward=reward, kl=kl, success_rate=success_rate,
            log_std_mean=log_std_mean, collapse_count=collapse_count,
            recovery_action=recovery_action
        )
        self._collapse_events.append(evt)

    def get_extended_metrics(self, grad_norm_before=0.0, grad_norm_after=0.0,
                             clip_ratio=0.0, value_error=0.0,
                             lr_pi=0.0, lr_vf=0.0, clip_range=0.0, entropy_coef=0.0):
        return {
            "grad/norm_before": grad_norm_before,
            "grad/norm_after": grad_norm_after,
            "metric/clip_ratio": clip_ratio,
            "metric/value_error": value_error,
            "lr/policy": lr_pi,
            "lr/value": lr_vf,
            "param/clip_range": clip_range,
            "param/entropy_coef": entropy_coef,
        }


class CurriculumScheduler:
    DEFAULT_STAGES = [
        {"blank_ratio_base": 60.0, "narrow_passage_std_ratio": 0.3,
         "task_point_max_dist_m": 8.0, "success_threshold": 0.3},
        {"blank_ratio_base": 50.0, "narrow_passage_std_ratio": 0.2,
         "task_point_max_dist_m": 15.0, "success_threshold": 0.3},
        {"blank_ratio_base": 40.0, "narrow_passage_std_ratio": 0.15,
         "task_point_max_dist_m": 30.0, "success_threshold": 0.3},
    ]

    def __init__(self, config: Dict, env_cfg: Dict):
        self.enabled = bool(config.get("enabled", True))
        self.stages = config.get("stages", self.DEFAULT_STAGES)
        self.hold_windows = int(config.get("stage_hold_windows", 5))
        self.fallback_threshold = float(config.get("fallback_threshold", 0.1))
        self.skip_on_stall = bool(config.get("skip_on_stall", False))
        self.skip_after_steps = int(config.get("skip_after_steps", 1_000_000))
        self.current_stage = 0
        self._promotion_counter = 0
        self._oscillation_counts = [0] * len(self.stages)
        self._locked = False
        self._locked_stage = 0

    def check_stage_transition(self, success_rate: float, current_step: int) -> Dict:
        if not self.enabled or self._locked:
            return {"stage": self.current_stage, "params": self.stages[self.current_stage],
                    "transitioned": False, "direction": None}

        result = {"stage": self.current_stage, "params": self.stages[self.current_stage],
                  "transitioned": False, "direction": None}

        stage_cfg = self.stages[self.current_stage]
        threshold = float(stage_cfg.get("success_threshold", 0.3))

        if self.current_stage < len(self.stages) - 1 and success_rate > threshold:
            self._promotion_counter += 1
            if self._promotion_counter >= self.hold_windows:
                old_stage = self.current_stage
                self.current_stage += 1
                self._promotion_counter = 0
                result = {"stage": self.current_stage, "params": self.stages[self.current_stage],
                          "transitioned": True, "direction": "promote"}
                print(f"[CURRICULUM] 晋升: 阶段{old_stage}→{self.current_stage}, success_rate={success_rate:.2%}")
        elif self.current_stage > 0 and success_rate < self.fallback_threshold:
            old_stage = self.current_stage
            self.current_stage -= 1
            self._promotion_counter = 0
            result = {"stage": self.current_stage, "params": self.stages[self.current_stage],
                      "transitioned": True, "direction": "fallback"}
            print(f"[CURRICULUM] 回退: 阶段{old_stage}→{self.current_stage}, success_rate={success_rate:.2%}")

            self._oscillation_counts[self.current_stage] += 1
            if self._oscillation_counts[self.current_stage] >= 3:
                self._locked = True
                self._locked_stage = self.current_stage
                self.stages[self.current_stage]["success_threshold"] = threshold * 1.5
                self.hold_windows = int(self.hold_windows * 1.5)
                print(f"[WARN] 课程阶段震荡，锁定阶段{self.current_stage}，提高晋升阈值")

        if self.skip_on_stall and self.current_stage == 0 and self._promotion_counter == 0 and current_step > self.skip_after_steps:
            if success_rate < self.fallback_threshold:
                print(f"[WARN] 课程学习停滞超过{self.skip_after_steps}步，跳过课程直接使用最终参数")
                self.current_stage = len(self.stages) - 1
                result = {"stage": self.current_stage, "params": self.stages[self.current_stage],
                          "transitioned": True, "direction": "skip"}

        return result

    def apply_to_env(self, env: SimRandomBatchEnv, params: Dict):
        env.cfg.blank_ratio_base = float(params.get("blank_ratio_base", env.cfg.blank_ratio_base))
        env.cfg.narrow_passage_std_ratio = float(params.get("narrow_passage_std_ratio", env.cfg.narrow_passage_std_ratio))
        env.cfg.task_point_max_dist_m = float(params.get("task_point_max_dist_m", env.cfg.task_point_max_dist_m))

    def get_current_stage(self):
        return self.current_stage

    def state_dict(self):
        return {
            "current_stage": self.current_stage,
            "promotion_counter": self._promotion_counter,
            "oscillation_counts": self._oscillation_counts,
            "locked": self._locked,
            "locked_stage": self._locked_stage,
        }


class AdvantagePostProcessor:
    def __init__(self, advantage_clip=10.0, gae_lambda_base=0.95,
                 gae_lambda_explore=0.98, entropy_safe_threshold=0.3):
        self.advantage_clip = advantage_clip
        self.gae_lambda_base = gae_lambda_base
        self.gae_lambda_explore = gae_lambda_explore
        self.entropy_safe_threshold = entropy_safe_threshold
        self._clip_ratio_history = deque(maxlen=10)

    def process(self, advantages: Tensor, current_entropy: float) -> Tuple[Tensor, float]:
        effective_lambda = self.gae_lambda_base
        if current_entropy < self.entropy_safe_threshold:
            effective_lambda = self.gae_lambda_explore

        clipped = ops.clamp(advantages, -self.advantage_clip, self.advantage_clip)
        total = int(advantages.shape[0])
        if total > 0:
            clipped_count = int((ops.abs(advantages) > self.advantage_clip).to(ms.int32).sum().asnumpy())
            clip_ratio = clipped_count / total
        else:
            clip_ratio = 0.0

        self._clip_ratio_history.append(clip_ratio)
        if len(self._clip_ratio_history) >= 10:
            avg_clip_ratio = np.mean(list(self._clip_ratio_history))
            if avg_clip_ratio > 0.5:
                self.advantage_clip *= 1.2
                print(f"[WARN] 优势裁剪比例>{avg_clip_ratio:.1%}，自动增大advantage_clip至{self.advantage_clip:.2f}")
                self._clip_ratio_history.clear()

        return clipped, effective_lambda

    def get_effective_lambda(self, current_entropy: float) -> float:
        if current_entropy < self.entropy_safe_threshold:
            return self.gae_lambda_explore
        return self.gae_lambda_base


class PPOTrainStep(nn.Cell):
    def __init__(self, policy, optimizer, clip_eps, ent_coef, vf_coef, max_grad_norm, use_amp, weights, value_clip=False):
        super().__init__()
        self.policy = policy
        self.optimizer = optimizer
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp
        self.weights = weights
        self.value_clip = Tensor(value_clip, mstype.bool_)
        self.grad_fn = ops.value_and_grad(self.forward_fn, None, self.weights, has_aux=True)

    def forward_fn(self, obs, actions, old_logp, advantages, returns, limits, old_values, clip_eps, ent_coef):
        new_logp, ent, v_pred = self.policy.evaluate_actions(obs, actions, limits)

        ratio = ops.exp(new_logp - old_logp)
        ratio = ops.clamp(ratio, 1e-6, 10.0)

        advantages_f32 = advantages.astype(mstype.float32)
        ratio_f32 = ratio.astype(mstype.float32)

        surr1 = ratio_f32 * advantages_f32
        surr2 = ops.clamp(ratio_f32, 1.0 - clip_eps, 1.0 + clip_eps) * advantages_f32

        pg_loss = -ops.reduce_mean(ops.minimum(surr1, surr2))

        v_clipped = old_values + ops.clamp(v_pred - old_values, -clip_eps, clip_eps)
        v_loss1 = ops.pow(returns - v_pred, 2)
        v_loss2 = ops.pow(returns - v_clipped, 2)
        v_loss_unclipped = 0.5 * ops.reduce_mean(v_loss1)
        v_loss_clipped = 0.5 * ops.reduce_mean(ops.maximum(v_loss1, v_loss2))
        v_loss = ops.select(self.value_clip, v_loss_clipped, v_loss_unclipped)

        ent_bonus = ops.reduce_mean(ent)
        total_loss = pg_loss + self.vf_coef * v_loss - ent_coef * ent_bonus
        approx_kl = ops.reduce_mean(old_logp - new_logp)

        return total_loss, (pg_loss, v_loss, ent_bonus, approx_kl)

    def construct(self, obs, actions, old_logp, advantages, returns, limits, old_values, clip_eps, ent_coef):
        (total_loss, aux_infos), grads = self.grad_fn(obs, actions, old_logp, advantages, returns, limits, old_values, clip_eps, ent_coef)
        pg_loss, v_loss, ent_bonus, approx_kl = aux_infos

        is_nan = ops.isnan(total_loss)
        norm_factor = ops.select(is_nan, Tensor(0.0, mstype.float32), Tensor(1.0, mstype.float32))
        safe_grads = tuple(g * norm_factor for g in grads)

        grads = ops.clip_by_global_norm(safe_grads, self.max_grad_norm)
        self.optimizer(grads)

        return pg_loss, v_loss, ent_bonus, approx_kl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config", type=str, default=os.path.join("config", "train_config.json"))
    args = parser.parse_args()

    cfg = load_train_config(args.train_config)
    cfg_dir = os.path.dirname(os.path.abspath(args.train_config))
    env_cfg_path = cfg.get("env_config", None)
    if env_cfg_path and not os.path.isabs(env_cfg_path):
        env_cfg_path = os.path.join(cfg_dir, env_cfg_path)

    device_target = cfg.get("device", "Ascend")
    ms.set_device(device_target)
    context.set_context(mode=context.GRAPH_MODE)
    if device_target == "Ascend":
        context.set_context(enable_graph_kernel=False)
    print(f"[PPO] device_target={device_target}")

    env_cfg = load_json_config(env_cfg_path) if env_cfg_path else {}
    obs_cfg = env_cfg.get('obs', {}) or {}
    sim_cfg = env_cfg.get('sim', {}) or {}
    lim_cfg = env_cfg.get('limits', {}) or {}
    rew_cfg = env_cfg.get('reward', {}) or {}
    ppo_cfg = cfg.get('ppo', {}) or {}

    B_env = int((cfg.get('sampling', {}) or {}).get('batch_env', 256))
    T_roll = int((cfg.get('sampling', {}) or {}).get('rollout_len', 128))
    reset_each_rollout = bool((cfg.get('sampling', {}) or {}).get('reset_each_rollout', True))
    run_seed = _extract_seed_from_train_or_env(cfg, env_cfg)
    if run_seed is not None:
        ms.set_seed(run_seed)
        np.random.seed(run_seed)
        random.seed(run_seed)

    safe_dist = float(sim_cfg.get('safe_distance', sim_cfg.get('warning_distance', 0.5)))
    sim = SimGPUEnvConfig(
        dt=float(sim_cfg.get('dt', 0.1)),
        n_envs=B_env,
        patch_meters=float(obs_cfg.get('patch_meters', 10.0)),
        ray_step_m=float(obs_cfg.get('ray_step_m', 0.025)),
        n_rays=int(obs_cfg.get('n_rays', 0)),
        ray_max_gap=float(obs_cfg.get('ray_max_gap', 0.25)),
        safe_distance_m=safe_dist,
        vx_max=float(lim_cfg.get('vx_max', 1.5)),
        omega_max=float(lim_cfg.get('omega_max', 2.0)),
        w_collision=float(rew_cfg.get('reward_collision', 1.0)),
        w_progress=float(rew_cfg.get('reward_progress', 1.0)),
        w_limits=float(rew_cfg.get('reward_limits', 0.1)),
        orientation_verify=bool(rew_cfg.get('orientation_verify', False)),
        w_jerk=float(rew_cfg.get('reward_jerk', 0.0)),
        w_jerk_omega=float(rew_cfg.get('reward_jerk_omega', 0.0)),
        w_danger=float(rew_cfg.get('reward_danger', 0.0)),
        w_danger_speed=float(rew_cfg.get('reward_danger_speed', 0.0)),
        w_speed_match=float(rew_cfg.get('reward_speed_match', 0.0)),
        danger_distance_m=float(rew_cfg.get('danger_distance', safe_dist)),
        stop_distance_m=float(rew_cfg.get('stop_distance', safe_dist * 0.5)),
        reward_time=float(rew_cfg.get('reward_time', 0.0)),
        blank_ratio_base=float((obs_cfg.get('blank_ratio_base', 40.0))),
        blank_ratio_randmax=float((obs_cfg.get('blank_ratio_randmax', 40.0))),
        blank_ratio_std_ratio=float(obs_cfg.get('blank_ratio_std_ratio', 0.33)),
        narrow_passage_gaussian=bool(obs_cfg.get('narrow_passage_gaussian', False)),
        narrow_passage_std_ratio=float(obs_cfg.get('narrow_passage_std_ratio', 0.3)),
        front_obstacle_prob=float(obs_cfg.get('front_obstacle_prob', 0.0)),
        front_obstacle_width_ratio=float(obs_cfg.get('front_obstacle_width_ratio', 0.18)),
        front_obstacle_min_m=float(obs_cfg.get('front_obstacle_min_m', 0.25)),
        front_obstacle_max_m=float(obs_cfg.get('front_obstacle_max_m', 1.0)),
        device=device_target,
        task_point_max_dist_m=float(sim_cfg.get('task_point_max_dist_m', 8.0)),
        task_point_success_radius_m=float(sim_cfg.get('task_point_success_radius_m', 0.25)),
        task_point_random_interval_max=int(sim_cfg.get('task_point_random_interval_max', 0)),
        collision_done=bool(ppo_cfg.get('collision_done', True)),
        w_success=float(rew_cfg.get('w_success', 2.0)),
        w_intrinsic=float(rew_cfg.get('w_intrinsic', 0.05)),
        max_reward_component=float(rew_cfg.get('max_reward_component', 2.0)),
        visit_count_grid_size=float((cfg.get('intrinsic_reward', {}) or {}).get('visit_count_grid_size', 1.0)),
        visit_count_max_entries=int((cfg.get('intrinsic_reward', {}) or {}).get('visit_count_max_entries', 50000)),
        intrinsic_decay_start=float((cfg.get('intrinsic_reward', {}) or {}).get('intrinsic_decay_start', 0.6)),
        intrinsic_decay_range=float((cfg.get('intrinsic_reward', {}) or {}).get('intrinsic_decay_range', 0.2)),
        enable_python_intrinsic=bool((cfg.get('intrinsic_reward', {}) or {}).get('enable_python_intrinsic', False)),
        curriculum_enabled=bool((cfg.get('curriculum', {}) or {}).get('enabled', True)),
    )

    intrinsic_cfg = cfg.get('intrinsic_reward', {}) or {}
    env = SimRandomBatchEnv(sim, intrinsic_cfg=intrinsic_cfg)
    obs = env.reset()
    vec_dim = int(obs.shape[1]) if len(obs.shape) == 2 else int(_infer_obs_dim_sim(sim))
    act_dim = 2

    model_cfg = cfg.get('model', {}) or {}
    policy = PPOPolicy(
        vec_dim=vec_dim,
        action_dim=act_dim,
        num_queries=int(model_cfg.get('num_queries', 4)),
        num_heads=int(model_cfg.get('num_heads', 4)),
        log_std_min=float((cfg.get('ppo', {}) or {}).get('log_std_min', -2.0)),
        log_std_max=float((cfg.get('ppo', {}) or {}).get('log_std_max', 1.0)),
    )

    ppo_cfg = cfg.get('ppo', {}) or {}
    lr_pi = float(ppo_cfg.get('lr', 3e-4))
    lr_vf = float(ppo_cfg.get('value_lr', lr_pi))
    adam_eps = float(ppo_cfg.get('adam_eps', 1e-5))
    log_std_wd = float(ppo_cfg.get('log_std_weight_decay', 0.0))

    encoder_params = list(policy.encoder.get_parameters())
    mu_params = list(policy.mu_head.get_parameters())
    log_std_param = [policy.log_std]
    vf_params = list(policy.value_head.get_parameters())
    pi_params = encoder_params + mu_params
    all_params = pi_params + log_std_param + vf_params

    optimizer = nn.Adam(
        params=[
            {"params": pi_params, "lr": lr_pi, "weight_decay": 1e-5},
            {"params": log_std_param, "lr": lr_pi, "weight_decay": log_std_wd},
            {"params": vf_params, "lr": lr_vf, "weight_decay": 1e-5}
        ],
        eps=adam_eps
    )

    use_amp = False
    ckpt_dir = cfg['run']['ckpt_dir']
    os.makedirs(ckpt_dir, exist_ok=True)
    global_step = 0
    last_log = 0
    
    print("[PPO] 将从随机初始化开始全新训练。")

    train_step = PPOTrainStep(
        policy=policy,
        optimizer=optimizer,
        clip_eps=float(ppo_cfg.get('clip_range', 0.2)),
        ent_coef=float(ppo_cfg.get('entropy_coef', 0.02)),
        vf_coef=float(ppo_cfg.get('value_coef', 0.5)),
        max_grad_norm=float(ppo_cfg.get('max_grad_norm', 0.5)),
        use_amp=use_amp,
        weights=all_params,
        value_clip=bool(ppo_cfg.get('value_clip', True))
    )
    train_step.set_train()

    total_env_steps = int(cfg['run']['total_env_steps'])
    if bool((cfg.get('run', {}) or {}).get('resume_as_additional', False)) and global_step > 0:
        total_env_steps = global_step + total_env_steps
        print(f"[PPO] 累计训练步数: {total_env_steps:,} (已加载 {global_step:,} + 新增 {cfg['run']['total_env_steps']:,})")

    lr_scheduler = LRScheduler(
        lr_init=lr_pi, value_lr_init=lr_vf,
        total_steps=total_env_steps,
        schedule=str(ppo_cfg.get('lr_schedule', 'linear')),
        min_lr_ratio=float(ppo_cfg.get('lr_min_ratio', 0.01))
    )
    clip_scheduler = ClipRangeScheduler(
        clip_init=float(ppo_cfg.get('clip_range', 0.2)),
        clip_final=float(ppo_cfg.get('clip_range_final', 0.1)),
        total_steps=total_env_steps,
        kl_target=float(ppo_cfg.get('kl_target', 0.015)),
        kl_early_stop=bool(ppo_cfg.get('kl_early_stop', True))
    )

    min_entropy_cfg = ppo_cfg.get('min_entropy', 'auto')
    entropy_controller = EntropyControllerV2(
        ent_coef_init=float(ppo_cfg.get('entropy_coef', 0.05)),
        ent_coef_final=float(ppo_cfg.get('entropy_coef_final', 0.02)),
        decay_progress=float(ppo_cfg.get('entropy_decay_progress', 0.9)),
        total_steps=total_env_steps,
        min_entropy=min_entropy_cfg,
        log_std_param=policy.log_std,
        log_std_min=float(ppo_cfg.get('log_std_min', -2.0)),
        log_std_max=float(ppo_cfg.get('log_std_max', 1.0)),
        min_recovery_steps=int(ppo_cfg.get('min_recovery_steps', 2000)),
        max_collapse_count=int(ppo_cfg.get('max_collapse_count', 5)),
        action_dim=act_dim,
    )
    grad_norm_controller = GradNormController(
        max_grad_norm=float(ppo_cfg.get('max_grad_norm', 0.5)),
        adapt=bool(ppo_cfg.get('grad_norm_adapt', True))
    )
    diag_cfg = cfg.get('diagnostics', {}) or {}
    diagnostics = DiagnosticsMonitorV2(
        stagnation_window=int(diag_cfg.get('stagnation_window', 10)),
        stagnation_threshold=float(diag_cfg.get('stagnation_threshold', 1e-4)),
        relative_stagnation_threshold=float(diag_cfg.get('relative_stagnation_threshold', 0.05)),
        kl_warning_threshold=float(diag_cfg.get('kl_warning_threshold', 0.02)),
        vf_drift_threshold=float(diag_cfg.get('vf_drift_threshold', 1.0)),
        min_entropy=entropy_controller.min_entropy,
    )

    curriculum = CurriculumScheduler(
        config=cfg.get('curriculum', {}),
        env_cfg=env_cfg
    )
    if curriculum.enabled and curriculum.current_stage == 0:
        curriculum.apply_to_env(env, curriculum.stages[0])
        print(f"[CURRICULUM] 初始阶段0: {curriculum.stages[0]}")

    advantage_processor = AdvantagePostProcessor(
        advantage_clip=float(ppo_cfg.get('advantage_clip', 10.0)),
        gae_lambda_base=float(ppo_cfg.get('gae_lambda', 0.95)),
        gae_lambda_explore=float(ppo_cfg.get('gae_lambda_explore', 0.98)),
        entropy_safe_threshold=float(ppo_cfg.get('entropy_safe_threshold', 0.3)),
    )

    gamma_val = float(ppo_cfg.get('gamma', 0.99))
    if abs(gamma_val - 0.99) > 1e-6:
        print(f"[WARN] gamma={gamma_val}被修改，强制使用0.99")
        gamma_val = 0.99

    t0 = time.time()
    log_interval = int(cfg['run']['log_interval'])
    ep_rew_acc = ms.Tensor(np.zeros(B_env), dtype=mstype.float32)
    ep_len_acc = ms.Tensor(np.zeros(B_env), dtype=mstype.int32)
    finished_rewards = []
    finished_lengths = []

    tb_log_dir = ckpt_dir
    os.makedirs(tb_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=ckpt_dir)

    vx_max = float(lim_cfg.get('vx_max', 1.5))
    omega_max = float(lim_cfg.get('omega_max', 2.0))
    base_limits_ms = ms.Tensor(np.array([vx_max, omega_max], dtype=np.float32), mstype.float32)
    current_limits = ops.broadcast_to(base_limits_ms.expand_dims(0), (B_env, 2))

    mb_size = int(ppo_cfg.get('minibatch_size', 2048))
    N_per_rollout = T_roll * B_env
    if mb_size > N_per_rollout // int(ppo_cfg.get('epochs', 4)):
        print(f"[WARN] minibatch_size({mb_size}) > N//epochs({N_per_rollout // int(ppo_cfg.get('epochs', 4))})，更新步数不足")
    if reset_each_rollout and T_roll < 128:
        print(f"[INFO] 频繁重置+短rollout(T={T_roll})可能影响GAE估计质量")

    while global_step < total_env_steps:
        if reset_each_rollout:
            obs = env.reset()

        rollout_success_count = 0
        rollout_total_steps = 0
        rollout_success_count_t = ms.Tensor(0, mstype.int32)
        rollout_reward_sum_t = ms.Tensor(0.0, mstype.float32)
        buf = RolloutBuffer(T_roll, B_env, obs_dim=vec_dim, act_dim=act_dim)

        policy_loss_vals = []
        value_loss_vals = []
        entropy_vals = []
        approx_kl_vals = []

        for _ in range(T_roll):
            out = policy.act(obs, current_limits)
            logp = out.logp
            act = out.action
            _, _, v = policy._core(obs)

            next_obs, reward_t, done_t, info = env.step(act)
            d = done_t.astype(mstype.float32)

            buf.add(
                obs=obs,
                act=act,
                logp=logp,
                rew=reward_t,
                done=d,
                val=v,
                limits=ops.broadcast_to(env.get_limits().expand_dims(0), (B_env, act_dim))
            )

            rt = reward_t.view(-1)
            done_mask = done_t.view(-1).astype(mstype.bool_)
            ep_rew_next = ep_rew_acc + rt
            ep_len_next = ep_len_acc + 1

            ep_rew_acc = ops.where(done_mask, ms.Tensor(0.0, mstype.float32), ep_rew_next)
            ep_len_acc = ops.where(done_mask, ms.Tensor(0, mstype.int32), ep_len_next)
            rollout_success_count_t = rollout_success_count_t + info.get("success_count", ms.Tensor(0, mstype.int32))
            rollout_reward_sum_t = rollout_reward_sum_t + reward_t.mean()
            rollout_total_steps += B_env

            global_step += B_env
            obs = next_obs

        train_progress = global_step / max(total_env_steps, 1)
        env.set_train_progress(train_progress)
        rollout_success_count = int(rollout_success_count_t.asnumpy())
        avg_rollout_reward = float((rollout_reward_sum_t / max(T_roll, 1)).asnumpy())

        if bool(ppo_cfg.get('bootstrap', True)):
            last_v = policy._core(obs)[2].view(B_env, 1)
            avg_ent_for_adv = float(np.mean(entropy_vals)) if entropy_vals else 0.5
            effective_lambda = advantage_processor.get_effective_lambda(avg_ent_for_adv)
            buf.compute_gae(last_v, gamma=gamma_val, lam=effective_lambda)

            adv_clipped, _ = advantage_processor.process(buf.advantages, avg_ent_for_adv)
            buf.advantages = adv_clipped
        else:
            buf.compute_mc_returns(gamma=gamma_val)

        current_lr_pi, current_lr_vf = lr_scheduler.step(global_step)
        if hasattr(optimizer, 'param_groups') and len(optimizer.param_groups) >= 3:
            for i, group in enumerate(optimizer.param_groups):
                if i < 2:
                    group['lr'] = current_lr_pi
                else:
                    group['lr'] = current_lr_vf

        avg_kl_for_clip = float(np.mean(approx_kl_vals)) if approx_kl_vals else 0.0
        current_clip_range, should_stop = clip_scheduler.step(global_step, avg_kl_for_clip)

        avg_ent_for_ctrl = float(np.mean(entropy_vals)) if entropy_vals else 0.5
        avg_rew_for_diag = avg_rollout_reward
        success_rate = rollout_success_count / max(rollout_total_steps, 1)

        diag_warnings, is_collapsed = diagnostics.update(
            float(np.mean(policy_loss_vals)) if policy_loss_vals else 0.0,
            float(np.mean(value_loss_vals)) if value_loss_vals else 0.0,
            avg_kl_for_clip,
            avg_ent_for_ctrl,
            avg_rew_for_diag,
            success_rate
        )

        current_ent_coef = entropy_controller.step(global_step, avg_ent_for_ctrl, is_collapsed)
        current_ent_coef_tensor = ms.Tensor(current_ent_coef, mstype.float32)

        current_clip_eps = ms.Tensor(current_clip_range, mstype.float32)

        progress = global_step / max(total_env_steps, 1)
        epochs = int(ppo_cfg.get('epochs', 4))
        if progress > 0.7:
            epochs = max(2, epochs - 1)

        log_std_min_val = float(ppo_cfg.get('log_std_min', -2.0))
        log_std_max_val = float(ppo_cfg.get('log_std_max', 1.0))
        for epoch_i in range(epochs):
            epoch_kl_vals = []
            for mb in buf.minibatches(mb_size):
                pg_loss, v_loss, ent_bonus, approx_kl = train_step(
                    mb.obs, mb.actions, mb.logp, mb.advantages, mb.returns, mb.limits,
                    mb.values, current_clip_eps, current_ent_coef_tensor
                )

                log_std_data = policy.log_std.asnumpy()
                if log_std_data.min() < log_std_min_val or log_std_data.max() > log_std_max_val:
                    clipped = np.clip(log_std_data, log_std_min_val, log_std_max_val)
                    policy.log_std.set_data(Tensor(clipped, mstype.float32))

                policy_loss_vals.append(float(pg_loss.asnumpy()))
                value_loss_vals.append(float(v_loss.asnumpy()))
                entropy_vals.append(float(ent_bonus.asnumpy()))
                kl_val = float(approx_kl.asnumpy())
                approx_kl_vals.append(kl_val)
                epoch_kl_vals.append(kl_val)

                if bool(ppo_cfg.get('kl_early_stop', True)):
                    if kl_val > float(ppo_cfg.get('kl_target', 0.015)) * 4.0:
                        print(f"[WARN] KL早停: approx_kl={kl_val:.4f}超过阈值")
                        break
            if bool(ppo_cfg.get('kl_early_stop', True)) and epoch_kl_vals and epoch_kl_vals[-1] > float(ppo_cfg.get('kl_target', 0.015)) * 4.0:
                break

        cur_result = curriculum.check_stage_transition(success_rate, global_step)
        if cur_result["transitioned"]:
            curriculum.apply_to_env(env, cur_result["params"])

        if global_step - last_log >= log_interval:
            elapsed = time.time() - t0
            fps = global_step / max(1e-3, elapsed)

            avg_policy_loss = float(np.mean(policy_loss_vals)) if policy_loss_vals else 0.0
            avg_value_loss = float(np.mean(value_loss_vals)) if value_loss_vals else 0.0
            avg_entropy = float(np.mean(entropy_vals)) if entropy_vals else 0.0
            avg_approx_kl = float(np.mean(approx_kl_vals)) if approx_kl_vals else 0.0

            print(f"[PPO] step={global_step:,} | fps={fps:.1f} | policy_loss={avg_policy_loss:.4f} | clip={current_clip_range:.4f} | ent_coef={current_ent_coef:.6f} | success_rate={success_rate:.2%} | curriculum_stage={curriculum.get_current_stage()}")

            writer.add_scalar("loss/policy", avg_policy_loss, global_step)
            writer.add_scalar("loss/value", avg_value_loss, global_step)
            writer.add_scalar("metric/entropy", avg_entropy, global_step)
            writer.add_scalar("metric/approx_kl", avg_approx_kl, global_step)
            writer.add_scalar("metric/success_rate", success_rate, global_step)
            writer.add_scalar("metric/curriculum_stage", curriculum.get_current_stage(), global_step)
            writer.add_scalar("speed/fps", fps, global_step)
            writer.add_scalar("lr/policy", current_lr_pi, global_step)
            writer.add_scalar("lr/value", current_lr_vf, global_step)
            writer.add_scalar("param/clip_range", current_clip_range, global_step)
            writer.add_scalar("param/entropy_coef", current_ent_coef, global_step)
            writer.add_scalar("param/gae_lambda", effective_lambda, global_step)
            writer.add_scalar("param/advantage_clip", advantage_processor.advantage_clip, global_step)

            if finished_rewards:
                avg_ep_rew = float(np.mean(finished_rewards))
                writer.add_scalar("episode/avg_reward", avg_ep_rew, global_step)
                print(f"[PPO] avg_ep_reward = {avg_ep_rew:.2f}")
            else:
                writer.add_scalar("rollout/avg_step_reward", avg_rollout_reward, global_step)
                print(f"[PPO] avg_step_reward = {avg_rollout_reward:.4f}")

            for w in diag_warnings:
                print(f"[DIAG] {w}")

            writer.flush()

            ms.save_checkpoint(policy, os.path.join(ckpt_dir, 'latest_policy.ckpt'))

            last_log = global_step
            finished_rewards.clear()
            finished_lengths.clear()
            policy_loss_vals.clear()
            value_loss_vals.clear()
            entropy_vals.clear()
            approx_kl_vals.clear()

    save_checkpoint(policy, os.path.join(ckpt_dir, 'final_policy.ckpt'))
    save_checkpoint(optimizer, os.path.join(ckpt_dir, 'final_optimizer.ckpt'))
    final_state = {
        "global_step": global_step,
        "entropy_controller": entropy_controller.state_dict(),
        "curriculum": curriculum.state_dict(),
    }
    with open(os.path.join(ckpt_dir, "final_state.json"), "w", encoding="utf-8") as f:
        json.dump(final_state, f, ensure_ascii=False, indent=2)

    writer.close()

if __name__ == "__main__":
    main()

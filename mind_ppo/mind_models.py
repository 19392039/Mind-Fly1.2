from __future__ import annotations

import math
import numpy as np
import mindspore as ms
from mindspore import ops, nn, Tensor, Parameter
from typing import NamedTuple
from mind_encoder import RayEncoder


# 推荐方案：使用 NamedTuple，兼顾可读性与 MindSpore 静态图性能
class PolicyOutput(NamedTuple):
    """
    MindSpore 策略输出结构
    """
    action: Tensor  # 形状 [B, A]，动作张量
    logp: Tensor    # 形状 [B, 1]，对数概率
    mu: Tensor      # 形状 [B, A]，网络输出的均值
    std: Tensor     # 形状 [B, A]，网络输出的标准差


def _tanh_log_det_jac(pre_tanh: Tensor) -> Tensor:
    use_asymptotic = ops.gt(ops.abs(pre_tanh), 20.0)
    softplus = ops.Softplus()
    exact = 2.0 * (math.log(2.0) - pre_tanh - softplus(-2.0 * pre_tanh))
    asymptotic = 2.0 * math.log(2.0) - 4.0 * ops.abs(pre_tanh)
    return ops.select(use_asymptotic, asymptotic, exact)

def _squash(mu: Tensor, log_std: Tensor, eps: Tensor, limits: Tensor):
    mu = ops.select(ops.isnan(mu), ops.zeros_like(mu), mu)
    log_std = ops.select(ops.isnan(log_std), ops.ones_like(log_std) * -1.0, log_std)

    log_std = ops.clip_by_value(log_std, -20.0, 2.0)
    std = ops.exp(log_std)
    std = std + 1e-6

    pre_tanh = mu + std * eps
    a = ops.tanh(pre_tanh)
    log_det = _tanh_log_det_jac(pre_tanh)

    mu = mu.astype(ms.float32)
    std = std.astype(ms.float32)
    pre_tanh = pre_tanh.astype(ms.float32)

    log_2pi = math.log(2 * math.pi)
    var = ops.pow(std, 2)
    logp_raw = -0.5 * (ops.pow(pre_tanh - mu, 2) / (var + 1e-8) + log_2pi + 2 * log_std)
    logp = (logp_raw - log_det).sum(-1, keepdims=True)

    a_scaled = a * limits
    return a_scaled, logp, std

def _inverse_squash(action_scaled: Tensor, limits: Tensor) -> Tensor:
    eps = 1e-12
    a = ops.clip_by_value(action_scaled / ops.maximum(limits, eps), -0.999999, 0.999999)
    return 0.5 * (ops.log1p(a) - ops.log1p(-a))

class PPOPolicy(nn.Cell):
    def __init__(self, 
                 vec_dim: int, 
                 action_dim: int = 3, 
                 hidden: int = 64, 
                 d_model: int = 128, 
                 num_queries: int = 4, 
                 num_heads: int = 4, 
                 learnable_queries: bool = True, 
                 log_std_min: float = -5.0, 
                 log_std_max: float = 2.0):
        super(PPOPolicy, self).__init__()

        # 1. 初始化 Encoder
        self.encoder = RayEncoder(
            vec_dim=vec_dim, 
            hidden=hidden, 
            d_model=d_model, 
            num_queries=num_queries, 
            num_heads=num_heads, 
            learnable_queries=learnable_queries
        )
        
        # 2. 策略头
        self.mu_head = nn.Dense(256, action_dim)
        self.log_std = Parameter(ops.zeros(action_dim, ms.float32), name="log_std")
        self._log_std_min = log_std_min
        self._log_std_max = log_std_max

        # 3. 价值头
        self.value_head = nn.SequentialCell([
            nn.Dense(256, 256),
            nn.ReLU(),
            nn.Dense(256, 1)
        ])
        self.softplus = ops.Softplus()
        self.exp = ops.Exp()

    def _core(self, obs_vec: Tensor):
        g, _, _ = self.encoder(obs_vec)
        mu = self.mu_head(g)
        log_std_raw = self.log_std.view(1, -1)
        log_std = ops.clip_by_value(log_std_raw, self._log_std_min, self._log_std_max)
        log_std = ops.broadcast_to(log_std, mu.shape)
        v = self.value_head(g)
        return mu, log_std, v

    def act(self, obs_vec: Tensor, limits: Tensor) -> PolicyOutput:
        mu, log_std, _ = self._core(obs_vec)
        eps = ops.standard_normal(mu.shape)
        mu_det = ops.stop_gradient(mu)
        log_std_det = ops.stop_gradient(log_std)
        a_scaled, logp, std = _squash(mu_det, log_std_det, eps, limits)
        
        return PolicyOutput(
            action=a_scaled, 
            logp=logp, 
            mu=mu_det, 
            std=std
        )

    def evaluate_actions(self, obs_vec: Tensor, actions_scaled: Tensor, limits: Tensor):
            # 1. 获取核心输出
            mu, log_std, v = self._core(obs_vec)
            # 确保类型统一为 float32，避免后续计算精度冲突
            mu = mu.astype(ms.float32)
            std = ops.exp(log_std).astype(ms.float32)+1e-6
            
            # 2. 逆向变换获取原始空间的值
            y = _inverse_squash(actions_scaled, limits)
            
            # 3. 手动计算 Log Probability (代替 dist.log_prob)
            # 公式: -0.5 * ((y - mu)/std)^2 - log(std) - 0.5 * log(2*pi)
            log_2pi = math.log(2 * math.pi)
            # 计算每个维度的 log_prob
            var = ops.pow(std, 2)
            logp_raw = -0.5 * (ops.pow(y - mu, 2) / (var + 1e-8) + log_2pi + 2 * log_std)
            
            # 4. 考虑 Tanh 变换的雅可比行列式修正
            log_det = _tanh_log_det_jac(y)
            # 在动作维度上求和
            logp = (logp_raw - log_det).sum(axis=-1, keepdims=True)
            
            # 5. 手动计算 Entropy (代替 dist.entropy)
            # 公式: 0.5 + 0.5 * log(2 * pi * std^2)
            ent_raw = 0.5 + 0.5 * log_2pi + log_std
            ent = ent_raw.sum(axis=-1, keepdims=True)
            
            return logp, ent, v

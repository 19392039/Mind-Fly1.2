from __future__ import annotations

from dataclasses import dataclass
import mindspore as ms
from mindspore import Tensor, ops, Parameter

@dataclass
class RolloutBatch:
    obs: Tensor         # [N, obs_dim]
    actions: Tensor     # [N, act_dim]
    logp: Tensor        # [N, 1]
    advantages: Tensor  # [N, 1]
    returns: Tensor     # [N, 1]
    values: Tensor      # [N, 1]
    limits: Tensor      # [N, act_dim]


class RolloutBuffer:
    def __init__(self, T: int, B: int, obs_dim: int, act_dim: int) -> None:
        N = int(T * B)
        self.T, self.B, self.N = T, B, N
        
        # MindSpore 中通常直接定义 Tensor，如果需要在网络内更新可使用 Parameter
        self.obs = Tensor(ops.zeros((N, obs_dim), ms.float32))
        self.actions = Tensor(ops.zeros((N, act_dim), ms.float32))
        self.logp = Tensor(ops.zeros((N, 1), ms.float32))
        self.rewards = Tensor(ops.zeros((N, 1), ms.float32))
        self.dones = Tensor(ops.zeros((N, 1), ms.float32))
        self.values = Tensor(ops.zeros((N, 1), ms.float32))
        self.limits = Tensor(ops.zeros((N, act_dim), ms.float32))
        self._ptr = 0

    def add(self, obs: Tensor, act: Tensor, logp: Tensor, rew: Tensor,
            done: Tensor, val: Tensor, limits: Tensor) -> None:
        n = obs.shape[0]
        i0, i1 = self._ptr, self._ptr + n
        
        # MindSpore 使用 slice_scatter 或简单的索引赋值
        self.obs[i0:i1] = obs
        self.actions[i0:i1] = act
        self.logp[i0:i1] = logp
        self.rewards[i0:i1] = rew.reshape(-1, 1)
        self.dones[i0:i1] = done.reshape(-1, 1).astype(ms.float32)
        self.values[i0:i1] = val.reshape(-1, 1)
        self.limits[i0:i1] = limits
        
        self._ptr = i1



    @ms.jit
    def compute_gae(self, last_value: Tensor, gamma: float, lam: float):
        T, B = self.T, self.B
        last_value = last_value.astype(ms.float32)

        last_value = ops.select(ops.isnan(last_value), ops.zeros_like(last_value), last_value)
        last_value = ops.select(ops.isinf(last_value), ops.zeros_like(last_value), last_value)

        r = self.rewards.reshape(T, B, 1).astype(ms.float32)
        d = self.dones.reshape(T, B, 1).astype(ms.float32)
        v = self.values.reshape(T, B, 1).astype(ms.float32)

        next_v = ops.concat([v[1:], last_value.reshape(1, B, 1)], axis=0)

        adv = ops.zeros((T, B, 1), ms.float32)
        gae = ops.zeros((B, 1), ms.float32)

        for t in reversed(range(T)):
            not_done = 1.0 - d[t]
            delta = r[t] + gamma * next_v[t] * not_done - v[t]
            gae = delta + gamma * lam * not_done * gae
            adv[t] = gae

        advantages_flat = adv.reshape(T * B, 1)
        returns_flat = advantages_flat + v.reshape(T * B, 1)

        adv_mean = advantages_flat.mean()
        adv_std = advantages_flat.std()
        adv_std = ops.maximum(adv_std, Tensor(1e-8, ms.float32))

        normalized_advantages = (advantages_flat - adv_mean) / adv_std

        self.advantages = normalized_advantages
        self.returns = returns_flat

        return normalized_advantages, returns_flat
    @ms.jit
    def compute_mc_returns(self, gamma: float):
        T, B = self.T, self.B
        
        # 维度重塑为 [T, B, 1]
        r = self.rewards.reshape(T, B, 1)
        d = self.dones.reshape(T, B, 1)
        v = self.values.reshape(T, B, 1)

        # 初始化存储结构
        ret = ms.numpy.zeros((T, B, 1))
        next_ret = ops.zeros((B, 1), ms.float32)

        # 逆序累加奖励
        for t in reversed(range(T)):
            not_done = 1.0 - d[t]
            # MC 核心公式：当前回报 = 立即奖励 + gamma * 下一步回报 (如果没结束)
            next_ret = r[t] + gamma * not_done * next_ret
            ret[t] = next_ret

        # 计算优势：实际回报 - 预测价值
        advantages = (ret - v).reshape(T * B, 1)
        returns = ret.reshape(T * B, 1)

        # 优势归一化（RL 训练稳定的关键）
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        adv_std = ops.maximum(adv_std, 1e-8) # 替代 torch.clamp_min
        
        advantages = (advantages - adv_mean) / adv_std
        
        self.advantages = advantages
        self.returns = returns
        
        return advantages, returns
    
    def minibatches(self, batch_size: int):
        # 1. 生成随机排列索引
        # MindSpore 对应的算子是 ops.randperm
        idx = ops.randperm(self.N)
        
        for i in range(0, self.N, batch_size):
            j = min(self.N, i + batch_size)
            
            # 2. 获取当前 batch 的索引切片
            sl = idx[i:j]
            
            # 3. 构造并产出 RolloutBatch 对象
            # 注意：MindSpore Tensor 支持直接用 Tensor 作为索引进行切片
            yield RolloutBatch(
                obs=self.obs[sl],
                actions=self.actions[sl],
                logp=self.logp[sl],
                advantages=self.advantages[sl],
                returns=self.returns[sl],
                values=self.values[sl],
                limits=self.limits[sl],
            )

from __future__ import annotations

import mindspore as ms
from mindspore import Tensor, nn, ops


class DeployPolicy(nn.Cell):
    """Deterministic PPO actor used for deployment.

    The training policy `act()` path samples noise. Real flight should use the
    actor mean, squash it with tanh, then scale it by the velocity limits.
    """

    def __init__(self, policy: nn.Cell, limits):
        super().__init__()
        self.policy = policy
        if isinstance(limits, Tensor):
            self.limits = limits.astype(ms.float32)
        else:
            self.limits = Tensor(limits, ms.float32)

    def construct(self, obs):
        mu, _, _ = self.policy._core(obs)
        return ops.tanh(mu.astype(ms.float32)) * self.limits

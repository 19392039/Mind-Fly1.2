from __future__ import annotations

import argparse
import os

import numpy as np
import mindspore as ms
from mindspore import Tensor, context, load, nn


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke-test exported MindIR policy.")
    parser.add_argument(
        "--mindir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts", "ppo_policy.mindir"),
    )
    parser.add_argument("--device-target", default="CPU", choices=["CPU", "Ascend"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    graph = nn.GraphCell(load(args.mindir))
    obs = Tensor(np.zeros((1, 135), dtype=np.float32), ms.float32)
    action = graph(obs).asnumpy()[0]
    print(f"vx={action[0]:+.6f} omega={action[1]:+.6f}")


if __name__ == "__main__":
    main()

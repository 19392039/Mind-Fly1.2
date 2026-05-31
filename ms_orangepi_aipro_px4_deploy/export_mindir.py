from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import mindspore as ms
from mindspore import Tensor

DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
if DEPLOY_DIR not in sys.path:
    sys.path.insert(0, DEPLOY_DIR)

from runtime_common import OBS_DIM, add_model_args, build_policy_cell, setup_mindspore  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Export the deterministic PPO policy to MindIR.")
    add_model_args(parser)
    parser.add_argument(
        "--output",
        default=os.path.join(DEPLOY_DIR, "artifacts", "ppo_policy"),
        help="Output path without extension.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_mindspore(args.device_target, graph_mode=True)
    net = build_policy_cell(args.env_config, args.ckpt)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    dummy = Tensor(np.zeros((1, OBS_DIM), dtype=np.float32), ms.float32)
    ms.export(net, dummy, file_name=args.output, file_format="MINDIR")
    print(f"exported: {args.output}.mindir")


if __name__ == "__main__":
    main()

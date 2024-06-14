import argparse
import os
import random
import time

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
import sys
import traceback

import wandb
from evil.irl.run_irl import run_irl
from evil.configs.irl_config import irl_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="JAXIRL",
    )
    parser.add_argument(
        "-e",
        "--env",
        type=str,
        choices=["ant", "hopper", "halfcheetah", "walker2d", "humanoid"],
    )
    args = parser.parse_args()
    irl_config["env"] = args.env
    run_irl(irl_config)

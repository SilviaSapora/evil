import argparse
import os
import random
import time

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
import sys
import traceback

import wandb
from evil.irl.run_irl import run_irl
from evil.configs.irl_plus_config import sweep_configuration


def pagent():
    try:
        run_irl()
    except Exception as e:
        print("Exception! Printing stack trace")
        # exit gracefully, so wandb logs the problem
        print(traceback.print_exc(), file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="JAXIRL",
    )
    parser.add_argument(
        "-r",
        "--restart",
        action="store_true",
    )
    parser.add_argument(
        "-s",
        "--sweep",
        type=str,
    )
    parser.add_argument(
        "-e",
        "--env",
        type=str,
        choices=["ant", "hopper", "halfcheetah", "walker2d", "humanoid"],
    )
    args = parser.parse_args()
    # seed = random.randint(0, 10)
    # sweep_configuration["parameters"]["seed"]["value"] = seed
    sweep_configuration["parameters"]["env"]["value"] = args.env
    if args.restart:
        print(f"RE-starting SWEEP ON ENV {args.env}")
        sweep = args.sweep
        wandb.agent(sweep, function=pagent, count=400)
    else:
        print(f"STARTING SWEEP ON ENV {args.env}")
        sweep_id = wandb.sweep(sweep=sweep_configuration, project="EvIL")
        print("Sweep ID: ", sweep_id)
        wandb.agent(sweep_id, function=pagent, count=400)

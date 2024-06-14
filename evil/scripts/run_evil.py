import argparse
import os
import time

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

from evil.irl.evil import main
from evil.configs.evil_config import run_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="EvIL",
    )
    parser.add_argument(
        "-e",
        "--env",
        type=str,
        choices=[
            "halfcheetah",
            "ant",
            "humanoid",
            "hopper",
            "walker2d",
        ],
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
    )
    parser.add_argument(
        "-u",
        "--updates",
        type=int,
        default=run_config["num_updates_two_step"],
    )
    parser.add_argument("--last", action="store_true")
    parser.add_argument("--real", action="store_true")
    parser.add_argument(
        "--reward_type", choices=["shaping", "reward"], default="shaping"
    )
    parser.add_argument("--reward_file", type=str)
    parser.add_argument(
        "--backend", choices=["positional", "generalized"], defaul="positional"
    )
    parser.add_argument("--across", action="store_true")
    parser.add_argument("-r", "--restart", type=str)
    parser.add_argument("-rg", "--restart_gen", type=int)
    parser.add_argument("-lr", type=float)
    args = parser.parse_args()
    run_config["seed"] = args.seed
    run_config["env"] = args.env
    run_config["num_updates_two_step"] = args.updates
    run_config["backend"] = args.backend
    run_config["restart_filename"] = args.restart
    run_config["restart_gen"] = args.restart_gen
    if args.across:
        run_config["reward_net_ensemble_params_type"] = "across_agents"
    else:
        run_config["reward_net_ensemble_params_type"] = "same_agent"
    if args.real:
        run_config["real_reward"] = "GROUND_TRUTH_REWARD"
        run_config["loss"] = "AUC"
    if args.reward_type == "shaping":
        run_config["reward_type"] = "SHAPING_STATE"
    if args.reward_file:
        run_config["reward_file"] = args.reward_file
    elif args.reward_type == "reward":
        run_config["reward_type"] = "REWARD_STATE"
    if run_config["env"] == "humanoid" or run_config["env"] == "ant":
        run_config["lrate_init"] = 0.001
    else:
        run_config["lrate_init"] = 0.002
    if args.lr:
        run_config["lrate_init"] = args.lr
    run_config["loss_last_only"] = args.last
    main(es_config=run_config)

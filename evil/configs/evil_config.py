import argparse
import os

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

from evil.irl.evil import main


run_config = {
    "irl_plus": True,
    "log_gen_every": 2,
    "reward_net_hsize": [64, 64],
    # "reward_net_hsize": [128, 128],
    "reward_net_sigmoid": False,
    "wandb_log": True,
    "plot": False,
    "train_rng": "DIFFERENT_IN_PAIRS",
    "save_to_file": True,
    "env": "hopper",
    "reward_normalize": True,
    "reward_all_normalize": False,
    "obs_normalize": True,
    "num_eval_steps": 1000,
    "num_expert_eval_envs": 50,
    "loss": "AUC_TWO_STEP",
    "seed": 1,
    "seeds": 5,
    "dual": False,
    "run_test": False,
    "num_eval_envs": 50,
    "inner_lr_linear": False,
    "inner_lr": 3e-4,
    "inner_steps": 10,
    # SHAPING
    "generations": 600,
    "real_reward": "IRL_STATE",
    "reward_type": "SHAPING_STATE",
    "lrate_init": 0.01,
    "popsize": 64,
    "reward_net_activation_fn": "tanh",
    "train_restart": "NONE",
    "restart_top_perc": 1.0,
    "max_buffer_size": 1,
    # IRL
    "discr_loss": "bce",
    "alpha": 0,
    "discr_l2_loss": 0.0,
    "discr_batch_size": 4096,
    "irl_lrate_init": 8e-3,
    "discr_schedule_type": "linear",
    "discr_trans_decay": 800,
    "discr_final_lr_diff": 3,
    "discr_updates_every": 1,
    "discr_updates": 14,
    "num_updates_inner_loop": 1,
    "num_updates_two_step": 1000,
    "retrain_from_scratch": True,
    "start_g": None,
    "outer_loop_restart": None,
    "auc_loss_type": "last_irl_return",
    "grad_penalty_coeff": 10,
    "discr_final_lr_diff": 2,
    "backend": "positional",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="JAXIRL",
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
        "--auc_loss_type", choices=["last_irl_return", "avg_episodic_irl_return"]
    )
    parser.add_argument("--reward_type", choices=["shaping", "reward"])
    parser.add_argument("--reward_file", type=str)
    parser.add_argument("--backend", choices=["positional", "generalized"])
    parser.add_argument("--across", action="store_true")
    parser.add_argument("-r", "--restart", type=str)
    parser.add_argument("-rg", "--restart_gen", type=int)
    parser.add_argument("-lr", type=float)
    args = parser.parse_args()
    run_config["seed"] = args.seed
    run_config["env"] = args.env
    run_config["num_updates_two_step"] = args.updates
    run_config["auc_loss_type"] = args.auc_loss_type
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

import argparse
import os
import random
import time

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
import sys
import traceback

import wandb

from evil.irl.run_irl import run_irl


irl_config = {
    "irl_plus": True,
    "log_gen_every": 2,
    "reward_net_hsize": [128, 128],
    "reward_net_sigmoid": True,
    "wandb_log": True,
    "plot": False,
    "train_rng": "DIFFERENT_IN_PAIRS",
    "save_to_file": True,
    "env": "ant",
    "reward_normalize": True,
    "reward_all_normalize": False,
    "obs_normalize": True,
    "num_eval_steps": 1000,
    "num_expert_eval_envs": 50,
    "loss": "IRL",
    "seed": 1,
    "seeds": 5,
    "num_discr": 5,
    "dual": False,
    "run_test": False,
    "num_eval_envs": 50,
    "inner_lr_linear": False,
    "inner_lr": 3e-4,
    "inner_steps": 10,
    # SHAPING
    "real_reward": "GROUND_TRUTH_REWARD",
    "reward_type": "REWARD_STATE",
    # IRL
    "discr_loss": "bce",
    "alpha": 0,
    "irl_generations": 2441,
    "discr_l2_loss": 0.0,
    "discr_batch_size": 4096,
    "irl_lrate_init": 4e-3,
    "discr_schedule_type": "linear",
    "discr_trans_decay": 400,
    "discr_final_lr_diff": 4,
    "discr_updates_every": 1,
    "discr_updates": 20,
    "num_updates_inner_loop": 1,
    "num_updates_two_step": 1000,
    "retrain_from_scratch": True,
    "grad_penalty_coeff": 10,
    "backend": "positional",
    "backend_test": "positional",
    "buffer_size_perc": 1,
    "reward_net_ensemble_type": "avg",
    "reward_net_ensemble_params_type": "same_agent",
    "expert_num_seeds": 1,
    "random_reinit": False,
    "random_reinit_prob": 0.03,
    "random_reinit_prob_final": 2,
    "random_reinit_decay": "linear",
}

import flax.linen as flax_nn

HOPPER_IRL_CONFIG = {
    "log_gen_every": 2,
    "reward_net_hsize": [128, 128],
    "reward_net_sigmoid": False,
    "wandb_log": True,
    "plot": False,
    "train_rng": "SAME",
    "env": "hopper",
    "save_to_file": False,
    "reward_type": "REWARD_STATE",
    "reward_normalize": True,
    "obs_normalize": False,
    "reward_net_activation_fn": "tanh",
    "real_reward_type": "IRL_STATE",
    "num_eval_steps": 1000,
    "loss": "IRL",
    "discr_batch_size": 4096,
    "seed": 1,
    "seeds": 1,
    "num_eval_envs": 10,
    "irl_lrate_init": 1e-2,
    "inner_lr_linear": True,
    "inner_lr": 3e-4,
    "inner_steps": 10,
    "discr_trans_decay": 700,
    "discr_updates_every": 1,
    "discr_updates": 20,
    "discr_l1_loss": 0.1,
    "dual": False,
    "discr_schedule_type": "linear",
    "discr_final_lr": 1e-6,
    "run_test": False,
    "discr_loss": "bce",
    "num_updates_inner_loop": 1,
    "buffer_size": 100000,
}

ANT_IRL_CONFIG = {
    "log_gen_every": 2,
    "reward_net_hsize": [128, 128],
    "reward_net_sigmoid": False,
    "wandb_log": True,
    "plot": False,
    "train_rng": "SAME",
    "save_to_file": False,
    "reward_type": "REWARD_STATE",
    "reward_normalize": True,
    "reward_net_activation_fn": "tanh",
    "real_reward_type": "IRL_STATE",
    "num_eval_steps": 1000,
    "loss": "IRL",
    "discr_batch_size": 4096,
    "seed": 1,
    "seeds": 5,
    "num_eval_envs": 100,
    "irl_lrate_init": 1e-3,
    "inner_lr_linear": True,
    "inner_lr": 4e-4,
    "inner_steps": 15,
    "discr_trans_decay": 200,
    "discr_updates_every": 1,
    "discr_updates": 5,
    "discr_l1_loss": 0.0,
    "dual": False,
    "discr_schedule_type": "linear",
    "discr_final_lr": 1e-6,
    "run_test": False,
    "discr_loss": "bce",
    "num_updates_inner_loop": 1,
    "buffer_size": 100000,
}

WALKER_IRL_CONFIG = {
    "log_gen_every": 2,
    "reward_net_hsize": [128, 128],
    "reward_net_sigmoid": False,
    "wandb_log": True,
    "plot": False,
    "train_rng": "SAME",
    "save_to_file": False,
    "reward_type": "REWARD_STATE",
    "reward_normalize": True,
    "reward_net_activation_fn": "tanh",
    "real_reward_type": "IRL_STATE",
    "num_eval_steps": 1000,
    "loss": "IRL",
    "discr_batch_size": 4096,
    "seed": 1,
    "seeds": 1,
    "num_eval_envs": 20,
    "inner_lr_linear": True,
    "inner_lr": 4e-4,
    "inner_steps": 10,
    "irl_lrate_init": 1e-2,
    "discr_trans_decay": 700,
    "discr_final_lr": 1e-6,
    "discr_updates_every": 1,
    "discr_updates": 15,
    "discr_l1_loss": 0.0,
    "dual": False,
    "discr_schedule_type": "linear",
    "run_test": False,
    "discr_loss": "bce",
    "num_updates_inner_loop": 1,
    "buffer_size": 100000,
}


HALFCHEETAH_IRL_CONFIG = {
    "log_gen_every": 2,
    "reward_net_hsize": [128, 128],
    "reward_net_sigmoid": False,
    "wandb_log": True,
    "plot": False,
    "train_rng": "SAME",
    "save_to_file": False,
    "reward_type": "REWARD_STATE",
    "reward_normalize": True,
    "real_reward_type": "IRL_STATE",
    "num_eval_steps": 1000,
    "loss": "IRL",
    "discr_batch_size": 4096,
    "seed": 1,
    "seeds": 5,
    "num_eval_envs": 100,
    "irl_lrate_init": 1e-4,
    "inner_lr_linear": True,
    "inner_lr": 4e-4,
    "inner_steps": 15,
    "discr_trans_decay": 200,
    "discr_updates_every": 1,
    "discr_updates": 5,
    "discr_l1_loss": 0.0,
    "dual": False,
    "discr_schedule_type": "constant",
    "discr_final_lr": 1e-4,
    "run_test": False,
    "discr_loss": "bce",
    "num_updates_inner_loop": 1,
    "buffer_size": 100000,
}

CARTPOLE_IRL_CONFIG = {
    "log_gen_every": 2,
    "reward_net_hsize": [128, 128],
    "reward_net_sigmoid": False,
    "wandb_log": True,
    "plot": False,
    "train_rng": "SAME",
    "env": "CartPole-v1",
    "save_to_file": False,
    "reward_type": "REWARD_STATE",
    "reward_normalize": True,
    "real_reward_type": "IRL_STATE",
    "num_eval_steps": 500,
    "loss": "IRL",
    "discr_batch_size": 4096,
    "seed": 1,
    "seeds": 1,
    "num_eval_envs": 100,
    "irl_lrate_init": 1e-3,
    "inner_lr_linear": True,
    "inner_lr": 2.5e-4,
    "inner_steps": 128,
    "discr_trans_decay": 200,
    "discr_updates_every": 1,
    "discr_updates": 5,
    "discr_l1_loss": 0.0,
    "dual": False,
    "discr_schedule_type": "linear",
    "discr_final_lr": 1e-5,
    "run_test": False,
    "discr_loss": "bce",
    "num_updates_inner_loop": 1,
    "buffer_size": 50000,
}

# CARTPOLE_IRL_CONFIG = {
#     "log_gen_every": 1000,
#     "reward_net_hsize": [128, 128],
#     "reward_net_sigmoid": False,
#     "wandb_log": True,
#     "plot": False,
#     "train_rng": "SAME",
#     "env": "CartPole-v1",
#     "save_to_file": False,
#     "reward_type": "REWARD_STATE",
#     "reward_normalize": True,
#     "real_reward_type": "IRL_STATE",
#     "num_eval_steps": 500,
#     "loss": "IRL",
#     "discr_batch_size": 4096,
#     "seed": 1,
#     "seeds": 1,
#     "num_eval_envs": 100,
#     "irl_lrate_init": 1e-3,
#     "inner_lr_linear": True,
#     "inner_lr": 2.5e-4,
#     "inner_steps": 128,
#     "discr_trans_decay": 200,
#     "discr_updates_every": 1,
#     "discr_updates": 5,
#     "discr_l1_loss": 0.0,
#     "dual": False,
#     "discr_schedule_type": "linear",
#     "discr_final_lr": 1e-5,
#     "run_test": False,
#     "discr_loss": "bce",
#     "num_updates_inner_loop": 1,
#     "buffer_size": 50000,
# }

# GRIDWORLD_IRL_CONFIG = {
#     "log_gen_every": 1000,
#     "reward_net_hsize": [128, 128],
#     "reward_net_sigmoid": False,
#     "wandb_log": True,
#     "plot": False,
#     "train_rng": "SAME",
#     "env": "CartPole-v1",
#     "save_to_file": False,
#     "reward_type": "REWARD_STATE",
#     "reward_normalize": True,
#     "real_reward_type": "IRL_STATE",
#     "num_eval_steps": 30,
#     "loss": "IRL",
#     "discr_batch_size": 4096,
#     "seed": 1,
#     "seeds": 1,
#     "num_eval_envs": 100,
#     "irl_lrate_init": 1e-3,
#     "inner_lr_linear": True,
#     "inner_lr": 1e-3,
#     "inner_steps": 30,
#     "discr_trans_decay": 200,
#     "discr_updates_every": 1,
#     "discr_updates": 5,
#     "discr_l1_loss": 0.0,
#     "dual": False,
#     "discr_schedule_type": "linear",
#     "discr_final_lr": 1e-5,
#     "run_test": False,
#     "discr_loss": "bce",
#     "num_updates_inner_loop": 1,
#     "buffer_size": 3000,
# }

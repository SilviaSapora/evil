import argparse
import pickle
from evil.plotting.create_plotting_data import get_expert_obsv_and_actions
import jax
import jax.numpy as jnp
import os
from evil.configs.evil_config import run_config as OUTER_CONFIG
from evil.utils.env_utils import get_env
from evil.utils.utils import get_irl_config
from evil.irl.irl_plus import IRLPlus
from evil.utils.randomize_env_func import randomize_env
from evil.training.ppo_v2_cont_irl import make_train as make_train_cont
from evil.training.ppo_v2_irl import make_train
import evosax
from evil.utils.utils import (
    RewardNetwork,
    get_irl_config,
    get_observation_size,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="EvIL",
    )
    parser.add_argument(
        "-e",
        "--env",
        type=str,
        choices=["ant", "hopper", "halfcheetah", "walker2d", "humanoid"],
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--reward_i",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--irl_reward_path",
        type=str,
    )
    parser.add_argument(
        "--shap_reward_path",
        type=str,
    )
    args = parser.parse_args()
    i = args.seed
    reward_i = args.reward_i
    es_config = OUTER_CONFIG
    es_config["env"] = args.env
    es_config["save_dir"] = None
    randomize_env(es_config["env"], use_default=True, seed=0)
    with open(
        f"{args.irl_reward_path}",
        "rb",
    ) as f:
        info = pickle.load(f)
        rew_params = info["rew_net_params"]
        rew_params = jax.tree_map(lambda x: x, rew_params)
        rew_params = jax.tree_map(
            lambda x: x.reshape(x.shape[0], -1, *x.shape[1:]), rew_params
        )
        print(jax.tree_map(lambda x: x.shape, rew_params))
        rew_norm = (info["data_stats"][0][0], info["data_stats"][1][0])

    env, env_params, original_training_config = get_env(env_name=es_config["env"])
    es_config, es_training_config = get_irl_config(es_config, original_training_config)
    print(es_training_config)
    if es_training_config["DISCRETE"]:
        action_num = env.action_space(env_params).n
        observation_shape = env.observation_space(env_params).shape[0]
        make_train = make_train
    else:
        action_num = env.action_space(env_params).shape[0]
        make_train = make_train_cont
        observation_shape = get_observation_size(env, env_params)[0]
    shaping_net = RewardNetwork(
        hsize=es_config["reward_net_hsize"],
        activation_fn=es_config["reward_net_activation_fn"],
        sigmoid=False,
    )
    shap_params = shaping_net.init(jax.random.PRNGKey(0), jnp.zeros(observation_shape))
    param_reshaper = evosax.ParameterReshaper(shap_params)
    with open(
        f"{args.shap_reward_path}",
        "rb",
    ) as f:
        info = pickle.load(f)
        shap_params = param_reshaper.reshape_single(info["state"].best_member)
        shap_params = jax.tree_map(lambda x: x.reshape(1, *x.shape), shap_params)
    irl_training_class = IRLPlus(
        env=env,
        env_params=env_params,
        training_config=es_training_config,
        es_config=es_config,
        logging_run=None,
        expert_data=(
            jnp.zeros((1, observation_shape)),
            jnp.zeros((1, env.action_size)),
        ),
        shaping_net=None,
    )
    irl_net = irl_training_class._reward_network

    # GET IRL++ RETRAINING TRANSFER
    irl_shaped_returns = get_expert_obsv_and_actions(
        env,
        env_params,
        es_config,
        make_train,
        original_training_config,
        1,
        i,
        rew_net=irl_net,
        rew_params=jax.tree_map(
            lambda x: jnp.expand_dims(x[reward_i], axis=0), rew_params
        ),
        rew_net_norm=rew_norm,
        is_transfer=True,
    )

    # GET IRL SHAPED TRANSFER
    irl_shaped_returns = get_expert_obsv_and_actions(
        env,
        env_params,
        es_config,
        make_train,
        original_training_config,
        1,
        i,
        shap_net=shaping_net,
        shap_params=jax.tree_map(
            lambda x: jnp.expand_dims(x[reward_i], axis=0), shap_params
        ),
        rew_net=irl_net,
        rew_params=jax.tree_map(
            lambda x: jnp.expand_dims(x[reward_i], axis=0), rew_params
        ),
        rew_net_norm=rew_norm,
        is_transfer=True,
    )

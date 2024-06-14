from functools import partial
from evil.utils.utils import CriticAsShaping
from evil.training.ppo_v2_cont_irl import get_network
import os

from evil.irl.irl_multi_discr import IRLMultiDiscr

# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".20"
import pickle
import evosax

import jax
import jax.numpy as jnp
from evil.training.wrappers import TremblingHandWrapper
from evil.utils.env_utils import get_env
from evil.utils.plot import setup_plot
from evil.utils.utils import (
    RewardNetwork,
    RewardNetworkPenalty,
    RewardWrapper,
    generate_config,
    get_irl_config,
    get_observation_size,
)
from evil.training.ppo_v2_cont_irl import (
    make_train as make_train_cont,
)
from evil.configs.hyperparam_search_new_shaping import run_config as OUTER_CONFIG
from evil.training.ppo_v2_irl import make_train
import matplotlib.pyplot as plt
import numpy as np
from evil.debug.randomize_env_func import randomize_env

env_info_runs = {}
# HOPPER
shaped = []
shape_original_gens = []
irl_reward = ""
irl_plus_reward = ""
irl_shaped_rewards = []
irl_shaped_gens = []
env_info_runs["hopper"] = {
    "shaped": shaped,
    "shape_original_gens": shape_original_gens,
    "irl_reward": irl_reward,
    "irl_plus_reward": irl_plus_reward,
    "irl_shaped_rewards": irl_shaped_rewards,
    "irl_shaped_gens": irl_shaped_gens,
}

# WALKER
shaped = []
shape_original_gens = []
irl_reward = ""
irl_plus_reward = ""
irl_shaped_rewards = []
irl_shaped_gens = []
env_info_runs["walker2d"] = {
    "shaped": shaped,
    "shape_original_gens": shape_original_gens,
    "irl_reward": irl_reward,
    "irl_plus_reward": irl_plus_reward,
    "irl_shaped_rewards": irl_shaped_rewards,
    "irl_shaped_gens": irl_shaped_gens,
}
# ANT
shaped = []
shape_original_gens = []

irl_reward = ""
irl_plus_reward = ""
irl_shaped_rewards = []
irl_shaped_gens = []
env_info_runs["ant"] = {
    "shaped": shaped,
    "shape_original_gens": shape_original_gens,
    "irl_reward": irl_reward,
    "irl_plus_reward": irl_plus_reward,
    "irl_shaped_rewards": irl_shaped_rewards,
    "irl_shaped_gens": irl_shaped_gens,
}
# HUMANOID
shaped = []
shape_original_gens = []
irl_reward = ""
irl_plus_reward = ""
irl_shaped_rewards = []
irl_shaped_gens = []
env_info_runs["humanoid"] = {
    "shaped": shaped,
    "shape_original_gens": shape_original_gens,
    "irl_reward": irl_reward,
    "irl_plus_reward": irl_plus_reward,
    "irl_shaped_rewards": irl_shaped_rewards,
    "irl_shaped_gens": irl_shaped_gens,
}

home_dir = "/home/silvias/docker/jaxirl/"


def get_expert_obsv_and_actions(
    env,
    env_params,
    es_config,
    make_train_fn,
    original_training_config,
    num_seeds,
    seed,
    rew_net=None,
    rew_params=None,
    rew_net_norm=None,
    shap_net=None,
    shap_params=None,
    shaping_critic=False,
    irl_plus=False,
    is_transfer=False,
    is_tremble=False,
):
    trained_standard_expert_path = f"{home_dir}/experts/{es_config['env']}.pkl"
    trained_new_standard_expert_path = (
        f"{home_dir}/experts/{es_config['env']}_positional_{num_seeds}.pkl"
    )
    if shap_net is None:
        trained_plotting_expert_path = f"{home_dir}/plotting_data/{es_config['env']}_{es_config['backend']}_{num_seeds}.pkl"
        data_mean = jnp.zeros(env.observation_size[0])
        data_var = jnp.ones(env.observation_size[0])
        agent_net = get_network(env, env_params, original_training_config)
        with open(trained_new_standard_expert_path, "rb") as f:
            expert_train_out = pickle.load(f)
            expert_params = expert_train_out["expert_params"]
            data_mean = expert_train_out["norm_mean"]
            data_var = expert_train_out["norm_var"]
        if shaping_critic:
            trained_new_standard_expert_path = (
                f"{home_dir}/experts/{es_config['env']}_positional_{1}.pkl"
            )
            print("getting critic")
            with open(trained_new_standard_expert_path, "rb") as f:
                expert_train_out = pickle.load(f)
                expert_params = jax.tree_map(
                    lambda x: jnp.stack([x[0]] * 5, axis=0),
                    expert_train_out["expert_params"],
                )
                data_mean = expert_train_out["norm_mean"]
                data_var = expert_train_out["norm_var"]
            trained_plotting_expert_path = f"{home_dir}/plotting_data/{es_config['env']}_{es_config['backend']}_{num_seeds}_value.pkl"
            shap_net = CriticAsShaping(agent_net)
            shap_params = expert_params
    else:
        trained_plotting_expert_path = f"{home_dir}/plotting_data/{es_config['env']}_{es_config['backend']}_{num_seeds}_shaped.pkl"
        with open(trained_standard_expert_path, "rb") as f:
            expert_train_out = pickle.load(f)
            print(expert_train_out.keys())
            try:
                data_mean = expert_train_out["norm_mean"]
                data_var = expert_train_out["norm_var"]
            except:
                data_mean = jnp.zeros(env.observation_size[0])
                data_var = jnp.ones(env.observation_size[0])

    if rew_net:
        if irl_plus:
            trained_plotting_expert_path = f"{home_dir}/plotting_data/{es_config['env']}_{es_config['backend']}_{num_seeds}_irl_plus.pkl"
        else:
            trained_plotting_expert_path = f"{home_dir}/plotting_data/{es_config['env']}_{es_config['backend']}_{num_seeds}_irl.pkl"
        data_mean, data_var = rew_net_norm[0], rew_net_norm[1]
        if shap_net:
            trained_plotting_expert_path = f"{home_dir}/plotting_data/{es_config['env']}_{es_config['backend']}_{num_seeds}_irl_shaped.pkl"
        if shaping_critic:
            print("IRL shaped by expert critic")
            trained_plotting_expert_path = f"{home_dir}/plotting_data/{es_config['env']}_{es_config['backend']}_{num_seeds}_irl_value.pkl"

    if is_transfer:
        trained_plotting_expert_path = trained_plotting_expert_path.replace(
            ".pkl", f"_transfer_{seed}.pkl"
        )
        print("TRANSFER", trained_plotting_expert_path)
    if is_tremble:
        trained_plotting_expert_path = trained_plotting_expert_path.replace(
            ".pkl", f"_tremble.pkl"
        )
        print("TREMBLE", trained_plotting_expert_path)

    rng = jax.random.PRNGKey(es_config["seed"])
    if not os.path.exists(trained_plotting_expert_path):

        def train_with_shap(rng, shap_params, rew_net_params, shap_net, rew_net):
            true_env = RewardWrapper(
                env,
                env_params,
                reward_network=rew_net,
                rew_network_params=rew_net_params,
                shaping_network=shap_net,
                shap_network_params=shap_params,
                training_config=original_training_config,
                invert_reward=True,
                data_stats=(data_mean, data_var),
            )
            if is_tremble:
                true_env = TremblingHandWrapper(true_env, 0.05)
            if shap_net is None:
                if rew_net is None:
                    print(
                        f"Baseline RL trained expert {es_config['backend']},  not found, retraining"
                    )
                else:
                    print("IRL retraining not found, retraining")
            else:
                if shap_params is None:
                    print("RL shaped expert GT not found, retraining")
                else:
                    if rew_net:
                        print("IRL shaped expert GT not found, retraining")
                    else:
                        print("Expert critic shaped not found, retraining")
            original_training_config["NUM_UPDATES"] = 2441
            train_fn = make_train_fn(
                config=original_training_config,
                env=true_env,
                env_params=env_params,
                runner_state_start=None,
                log_timestep_returns=True,
                return_obs_and_actions=False,
            )
            return train_fn(rng)

        rng = jax.random.PRNGKey(seed)
        multi_rng = jax.random.split(rng, num_seeds)
        ptrain_with_shap = partial(train_with_shap, shap_net=shap_net, rew_net=rew_net)
        expert_train_out = jax.jit(jax.vmap(ptrain_with_shap))(
            multi_rng, shap_params, rew_params
        )

        with open(trained_plotting_expert_path, "wb") as f:
            original_returns = expert_train_out["metrics"][
                "timestep_returned_episode_returns"
            ]
            pickle.dump(
                {
                    "returns": original_returns,
                    "expert_params": expert_train_out["runner_state"][0].params,
                },
                f,
            )
    else:
        if shap_net is None:
            if rew_net is None:
                print(f"FOUND - Baseline RL trained expert {es_config['backend']}")
            else:
                print("FOUND - IRL retrained expert")
        else:
            if shap_params is None:
                print("FOUND - Expert critic shaped")
            else:
                if rew_net:
                    print("FOUND - IRL shaped")
                else:
                    print("FOUND - RL shaped expert GT")

    with open(trained_plotting_expert_path, "rb") as f:
        expert_train_out = pickle.load(f)
        original_returns = expert_train_out["returns"]

    return original_returns


if __name__ == "__main__":
    for env_name in ["hopper", "walker2d", "ant", "humanoid"]:
        print("env name:", env_name)
        es_config = OUTER_CONFIG
        es_config["env"] = env_name
        es_config["save_dir"] = None
        randomize_env(es_config["env"], use_default=True, seed=0)
        env, env_params, original_training_config = get_env(env_name=es_config["env"])
        es_config, es_training_config = get_irl_config(
            es_config, original_training_config
        )
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
        shap_params = shaping_net.init(
            jax.random.PRNGKey(0), jnp.zeros(observation_shape)
        )
        param_reshaper = evosax.ParameterReshaper(shap_params)

        shaped_runs = env_info_runs[env_name]["shaped"]
        shaped_runs_gens = env_info_runs[env_name]["shape_original_gens"]
        irl_reward_run = env_info_runs[env_name]["irl_reward"]
        irl_plus_reward_run = env_info_runs[env_name]["irl_plus_reward"]
        irl_shaped_rewards_runs = env_info_runs[env_name]["irl_shaped_rewards"]
        irl_shaped_gens = env_info_runs[env_name]["irl_shaped_gens"]

        shap_params_init = shaping_net.init(
            jax.random.PRNGKey(0), jnp.zeros(observation_shape)
        )
        shap_params = jax.tree_map(lambda x: x.reshape(1, *x.shape), shap_params_init)
        for i, run_shaped in enumerate(shaped_runs):
            with open(
                f"{home_dir}/rewards/{es_config['env']}/{run_shaped}_{shaped_runs_gens[i]}.pkl",
                "rb",
            ) as f:
                print("adding ", run_shaped, "for shaping")
                info = pickle.load(f)
                new_shap_params = param_reshaper.reshape_single(info["state"].mean)
                new_shap_params = jax.tree_map(
                    lambda x: x.reshape(1, *x.shape), new_shap_params
                )
                shap_params = jax.tree_map(
                    lambda x, y: jnp.concatenate((x, y), axis=0),
                    shap_params,
                    new_shap_params,
                )
        shap_params = jax.tree_map(lambda x: x[1:], shap_params)

        # CREATE GROUND TRUTH SHAPED
        print("GT Shaped")
        rl_shaped_returns = get_expert_obsv_and_actions(
            env,
            env_params,
            es_config,
            make_train,
            original_training_config,
            5,
            0,
            shap_net=shaping_net,
            shap_params=shap_params,
        )
        # CREATE EXPERT CRITIC SHAPED
        print("GT Shaped Expert Critic")
        expert_baseline_rl_returns = get_expert_obsv_and_actions(
            env,
            env_params,
            es_config,
            make_train,
            original_training_config,
            5,
            0,
            shaping_critic=True,
        )
        irl_training_class = IRLMultiDiscr(
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
        print("IRL retraining")
        # GET IRL RETRAINING
        with open(
            f"{home_dir}/{irl_reward_run}",
            "rb",
        ) as f:
            info = pickle.load(f)
            rew_params = info["rew_net_params"]
            rew_params = jax.tree_map(lambda x: x[:5], rew_params)
            rew_params = jax.tree_map(
                lambda x: x.reshape(5, -1, *x.shape[1:]), rew_params
            )
            print(jax.tree_map(lambda x: x.shape, rew_params))
            rew_norm = (info["data_stats"][0][0], info["data_stats"][1][0])

        irl_returns = get_expert_obsv_and_actions(
            env,
            env_params,
            es_config,
            make_train,
            original_training_config,
            5,
            0,
            rew_net=irl_net,
            rew_params=rew_params,
            rew_net_norm=rew_norm,
        )

        # GET IRL++ RETRAINING
        print("IRL++ retraining")
        with open(
            f"{home_dir}/{irl_plus_reward_run}",
            "rb",
        ) as f:
            info = pickle.load(f)
            rew_params = info["rew_net_params"]
            rew_params = jax.tree_map(lambda x: jnp.swapaxes(x, 1, 0), rew_params)
            print("rew return")
            rew_norm = (info["data_stats"][0], info["data_stats"][1])

        print("rew plus shape", jax.tree_map(lambda x: x.shape, rew_params))
        irl_plus_returns = get_expert_obsv_and_actions(
            env,
            env_params,
            es_config,
            make_train,
            original_training_config,
            5,
            0,
            rew_net=irl_net,
            rew_params=rew_params,
            rew_net_norm=rew_norm,
            irl_plus=True,
        )

        # IRL++ SHAPED
        shap_params_init = shaping_net.init(
            jax.random.PRNGKey(0), jnp.zeros(observation_shape)
        )
        shap_params = jax.tree_map(lambda x: x.reshape(1, *x.shape), shap_params_init)
        print("IRL++ retraining Shaped")
        for i, irl_run_shaped in enumerate(irl_shaped_rewards_runs):
            with open(
                f"{home_dir}/rewards/{es_config['env']}/{irl_run_shaped}_{irl_shaped_gens[i]}.pkl",
                "rb",
            ) as f:
                print("adding ", irl_run_shaped, "for irl shaping")
                info = pickle.load(f)
                new_shap_params = param_reshaper.reshape_single(
                    info["state"].best_member
                )
                new_shap_params = jax.tree_map(
                    lambda x: x.reshape(1, *x.shape), new_shap_params
                )
                shap_params = jax.tree_map(
                    lambda x, y: jnp.concatenate((x, y), axis=0),
                    shap_params,
                    new_shap_params,
                )
        shap_params = jax.tree_map(lambda x: x[1:], shap_params)

        irl_shaped_returns = get_expert_obsv_and_actions(
            env,
            env_params,
            es_config,
            make_train,
            original_training_config,
            5,
            0,
            shap_net=shaping_net,
            shap_params=shap_params,
            rew_net=irl_net,
            rew_params=rew_params,
            rew_net_norm=rew_norm,
            irl_plus=True,
        )
        # IRL++ shaped by expert critic
        print("IRL++ shaped by expert critic")
        irl_shaped_returns = get_expert_obsv_and_actions(
            env,
            env_params,
            es_config,
            make_train,
            original_training_config,
            5,
            0,
            rew_net=irl_net,
            rew_params=rew_params,
            rew_net_norm=rew_norm,
            shaping_critic=True,
        )

        # TREMBLE
        print("TREMBLE")
        # GET EXPERT TREMBLE
        expert_transfer_returns = get_expert_obsv_and_actions(
            env,
            env_params,
            es_config,
            make_train,
            original_training_config,
            5,
            0,
            is_tremble=True,
        )
        # GET IRL++ RETRAINING TREMBLE
        irl_shaped_returns = get_expert_obsv_and_actions(
            env,
            env_params,
            es_config,
            make_train,
            original_training_config,
            5,
            0,
            rew_net=irl_net,
            rew_params=rew_params,
            rew_net_norm=rew_norm,
            is_tremble=True,
        )

        # GET IRL SHAPED TREMBLE
        irl_shaped_returns = get_expert_obsv_and_actions(
            env,
            env_params,
            es_config,
            make_train,
            original_training_config,
            5,
            0,
            shap_net=shaping_net,
            shap_params=shap_params,
            rew_net=irl_net,
            rew_params=rew_params,
            rew_net_norm=rew_norm,
            is_tremble=True,
        )

        # TRANSFER
        for i in range(5):
            randomize_env(es_config["env"], use_default=False, seed=i)
            env, env_params, original_training_config = get_env(
                env_name=es_config["env"]
            )
            # GET EXPERT TRANSFER
            expert_transfer_returns = get_expert_obsv_and_actions(
                env,
                env_params,
                es_config,
                make_train,
                original_training_config,
                1,
                i,
                is_transfer=True,
            )
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
                    lambda x: jnp.expand_dims(x[i], axis=0), rew_params
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
                    lambda x: jnp.expand_dims(x[i], axis=0), shap_params
                ),
                rew_net=irl_net,
                rew_params=jax.tree_map(
                    lambda x: jnp.expand_dims(x[i], axis=0), rew_params
                ),
                rew_net_norm=rew_norm,
                is_transfer=True,
            )

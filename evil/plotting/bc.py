from evil.irl.bc import BC
import jax
import importlib
import jaxirl.training.ppo_v2_cont_irl

importlib.reload(jaxirl.training.ppo_v2_cont_irl)
from evil.training.ppo_v2_cont_irl import eval as eval_cont
import os

from evil.irl.irl_multi_discr import IRLMultiDiscr
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
from evil.training.ppo_v2_cont_irl import make_train as make_train_cont, get_network
from evil.debug.randomize_env_func import randomize_env
from evil.configs.hyperparam_search_new_shaping import run_config as OUTER_CONFIG
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    for env_name in ["hopper", "walker2d", "ant", "humanoid"]:
        print("env name", env_name)
        es_config = OUTER_CONFIG
        es_config["env"] = env_name
        es_config["save_dir"] = None
        randomize_env(es_config["env"], use_default=True, seed=0)
        env, env_params, original_training_config = get_env(env_name=es_config["env"])
        es_config, es_training_config = get_irl_config(
            es_config, original_training_config
        )
        home_dir = "/home/silvias/docker/jaxirl/"
        trained_plotting_expert_path = (
            f"{home_dir}/plotting_data/{es_config['env']}_{es_config['backend']}_5.pkl"
        )
        original_env = RewardWrapper(
            env=env,
            env_params=env_params,
            reward_network=None,
            rew_network_params=None,
            shaping_network=None,
            shap_network_params=None,
            include_action=False,
            training_config=original_training_config,
            invert_reward=False,
            debug=False,
        )
        train_fn = make_train_cont(
            config=original_training_config,
            env=original_env,
            env_params=env_params,
            runner_state_start=None,
            log_timestep_returns=True,
            return_obs_and_actions=False,
        )
        seed = 1
        rng = jax.random.PRNGKey(seed)
        expert_train_out = jax.jit(train_fn)(rng)
        agent_net = get_network(env, env_params, original_training_config)
        obsv, actions, expert_reward, _ = eval_cont(
            100,
            1000,
            original_env,
            env_params,
            expert_train_out["runner_state"][0].params,
            rng,
            agent_net,
            True,
            False,
        )
        print(expert_reward.mean())

        print("BC not found, retraining")
        es_config["generations"] = 100000
        bc_training_class = BC(
            env=env,
            env_params=env_params,
            training_config=es_training_config,
            es_config=es_config,
            logging_run=None,
            expert_data=(obsv[:, :100], actions[:, :100]),
        )
        log_path = f"{home_dir}/plotting_data/{es_config['env']}_BC.pkl"
        if not os.path.exists(log_path):
            rng = jax.random.PRNGKey(2)
            bc_policy, _losses = jax.jit(bc_training_class.train)(rng=rng)

            rng = jax.random.split(rng, 5)
            _, _, cur_standard_reward, _ = jax.vmap(
                eval_cont,
                (None, None, None, None, None, 0, None, None, None),
            )(
                100,
                1000,
                original_env,
                env_params,
                bc_policy[0].params,
                rng,
                bc_training_class.agent_net,
                True,
                False,
            )
            print("reward on standard env shape", cur_standard_reward.shape)
            print("reward on standard env last return", cur_standard_reward[:, -1])
            tremble_basic_env = TremblingHandWrapper(original_env, p_tremble=0.05)
            _, _, cur_tremble_reward, _ = jax.vmap(
                eval_cont,
                (None, None, None, None, None, 0, None, None, None),
            )(
                100,
                1000,
                tremble_basic_env,
                env_params,
                bc_policy[0].params,
                rng,
                bc_training_class.agent_net,
                True,
                False,
            )
            standard_reward = cur_standard_reward.mean(-1)
            tremble_reward = cur_tremble_reward.mean(-1)

            print("standard BC reward", standard_reward)
            print("trmble BC reward", tremble_reward)
            with open(log_path, "wb") as ff:
                pickle.dump(
                    {
                        "BC_standard": standard_reward,
                        "BC_tremble": tremble_reward,
                        "BC_policy": bc_policy[0].params,
                        "obsv": obsv,
                        "actions": actions,
                    },
                    ff,
                )
        print("BC found, evaluating on TRANSFER")
        with open(log_path, "rb") as f:
            data = pickle.load(f)
            bc_policy = data["BC_policy"]
            standard_reward = data["BC_standard"]
            tremble_reward = data["BC_tremble"]

        transfer_rewards = jnp.zeros((5))
        for i in range(5):
            randomize_env(es_config["env"], use_default=False, seed=i)
            env, env_params, original_training_config = get_env(
                env_name=es_config["env"]
            )
            transfer_basic_env = RewardWrapper(
                env,
                env_params,
                None,
                None,
                None,
                None,
                include_action=False,
                training_config=es_training_config,
                invert_reward=False,
            )
            _, _, cur_transfer_reward, _ = eval_cont(
                100,
                1000,
                transfer_basic_env,
                env_params,
                bc_policy,
                jax.random.PRNGKey(i),
                bc_training_class.agent_net,
                True,
                False,
                # (norm_mean, norm_var),
            )
            print(cur_transfer_reward.mean().shape)
            transfer_rewards = transfer_rewards.at[i].set(cur_transfer_reward.mean())
        print(transfer_rewards)

        with open(log_path, "wb") as ff:
            pickle.dump(
                {
                    "BC_standard": standard_reward,
                    "BC_tremble": tremble_reward,
                    "BC_transfer": transfer_rewards,
                    "BC_policy": bc_policy,
                },
                ff,
            )
        randomize_env(es_config["env"], use_default=True, seed=i)

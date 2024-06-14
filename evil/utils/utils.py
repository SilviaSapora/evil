from enum import Enum
from functools import partial
import math
import pickle

import jax
import jax.numpy as jnp
import optax
import flax.linen as flax_nn
from typing import Any
import wandb
import os
from datetime import datetime

from evil.training.ppo_v2_irl import ActorCritic
from evil.training.ppo_v2_cont_irl import ActorCritic as ActorCriticCont
from evil.configs.outer_training_configs import (
    HALFCHEETAH_IRL_CONFIG,
    HOPPER_IRL_CONFIG,
    ANT_IRL_CONFIG,
    WALKER_IRL_CONFIG,
)
from evil.utils.env_utils import get_eval_config, get_test_params


class TrainRNG(Enum):
    SAME_ALWAYS = "SAME_ALWAYS"
    SAME_AT_STEP = "SAME_AT_STEP"
    DIFFERENT = "DIFFERENT"
    DIFFERENT_IN_PAIRS = "DIFFERENT_IN_PAIRS"


class TrainRestart(Enum):
    NONE = "NONE"
    RESTART_BEST = "RESTART_BEST"
    SAMPLE_INIT = "SAMPLE_INIT"
    SAMPLE_RECENT_INIT = "SAMPLE_RECENT_INIT"


class RewardType(Enum):
    REWARD_STATE = "REWARD_STATE"
    REWARD_STATE_ACTION = "REWARD_STATE_ACTION"
    SHAPING_STATE = "SHAPING_STATE"
    SHAPING_STATE_ACTION = "SHAPING_STATE_ACTION"
    NONE = "NONE"


class RealReward(Enum):
    IRL_STATE = "IRL_STATE"
    IRL_STATE_ACTION = "IRL_STATE_ACTION"
    GROUND_TRUTH_REWARD = "GROUND_TRUTH_REWARD"


class LossType(Enum):
    XE = "XE"
    IRL = "IRL"
    NONE = "NONE"
    BC = "BC"
    AUC = "AUC"
    AUC_TRANSFER = "AUC_TRANSFER"
    AUC_TWO_STEP = "AUC_TWO_STEP"


def get_plot_filename(es_config):
    if wandb.run is None:
        date_time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        return f"{os.getcwd()}/plots/{es_config['env']}_{date_time}.png"
    else:
        return f"{os.getcwd()}/plots/{es_config['env']}_{wandb.run.name}.png"


def _get_xentropy_match_score_expert(obsv, expert_actions, network_params, network):
    pi, _value = network.apply(network_params, obsv)
    # H(q, p)
    return optax.softmax_cross_entropy_with_integer_labels(
        pi.logits, expert_actions
    ).mean()


def get_xentropy_match_score_expert(
    params, expert_obsv, expert_actions, ppo_network, is_discrete
):
    if is_discrete:
        xe_function = _get_xentropy_match_score_expert
    else:
        xe_function = _get_nlog_l_expert
    partial_get_xentropy_match_score2 = partial(
        xe_function, network=ppo_network, network_params=params
    )
    xentropy = jax.vmap(partial_get_xentropy_match_score2, (0, 0), 0)(
        expert_obsv, expert_actions
    )
    return jnp.mean(xentropy)


def _get_nlog_l_expert(obsv, expert_actions, network_params, network):
    pi, _value = network.apply(network_params, obsv)
    # H(p, q)
    return jnp.mean(-pi.log_prob(expert_actions), axis=-1)


def get_action_size(env, env_params):
    if hasattr(env, "action_size"):
        return env.action_size
    elif hasattr(env.action_space(env_params), "n"):
        return env.action_space(env_params).n
    else:
        return env.action_space(env_params).shape[0]


def get_observation_size(env, env_params):
    if hasattr(env, "observation_size"):
        return env.observation_size
    else:
        return env.observation_space(env_params).shape


def get_network(action_size, training_config):
    if training_config["DISCRETE"]:
        return ActorCritic(
            action_size,
            activation=training_config["ACTIVATION"],
        )
    else:
        return ActorCriticCont(
            action_size,
            activation=training_config["ACTIVATION"],
        )


def maybe_concat_action(include_action, action_size, obs, action):
    if include_action:
        if len(action.shape) == 0:
            action = jax.nn.one_hot(action, action_size)
        return jnp.concatenate([obs, action], axis=-1)
    else:
        return obs


def get_feature_size(env, env_params, training_config, es_config):
    action_size = get_action_size(env, env_params)
    if training_config["DISCRETE"]:
        observation_shape = env.observation_space(env_params).shape[0]
    else:
        observation_shape = get_observation_size(env, env_params)[0]
    if is_state_only(es_config):
        in_features = observation_shape
    elif is_state_action(es_config):
        in_features = observation_shape + action_size
    elif RewardType[es_config["reward_type"]] == RewardType.NONE:
        raise NotImplementedError(
            f"reward type is None in IRL class, this shouldn't happen"
        )
    else:
        raise NotImplementedError(
            f"reward type not implemented {es_config['reward_type']}"
        )
    return observation_shape, action_size, in_features


class RewardNetwork(flax_nn.Module):
    hsize: Any
    activation_fn: Any
    sigmoid: bool = False

    @flax_nn.compact
    def __call__(self, x):
        for n in range(len(self.hsize)):
            x = flax_nn.Dense(features=self.hsize[n])(x)
            if self.activation_fn == "relu":
                x = flax_nn.relu(x)
            else:
                x = flax_nn.tanh(x)
        x = flax_nn.Dense(1, name="vals")(x)
        if self.sigmoid:
            return flax_nn.sigmoid(x)
        return x


def is_irl(es_config):
    return (
        RealReward[es_config["real_reward"]] == RealReward.IRL_STATE
        or RealReward[es_config["real_reward"]] == RealReward.IRL_STATE_ACTION
        or LossType[es_config["loss"]] == LossType.IRL
        or LossType[es_config["loss"]] == LossType.AUC_TWO_STEP
        or es_config["real_reward"] == RealReward.IRL_STATE
        or es_config["real_reward"] == RealReward.IRL_STATE_ACTION
        or es_config["loss"] == LossType.IRL
        or es_config["loss"] == LossType.AUC_TWO_STEP
    )


def is_reward(es_config):
    return (
        RewardType[es_config["reward_type"]] == RewardType.REWARD_STATE
        or RewardType[es_config["reward_type"]] == RewardType.REWARD_STATE_ACTION
        or es_config["reward_type"] == RewardType.REWARD_STATE
        or es_config["reward_type"] == RewardType.REWARD_STATE_ACTION
    )


def is_shaping(es_config):
    return (
        RewardType[es_config["reward_type"]] == RewardType.SHAPING_STATE
        or RewardType[es_config["reward_type"]] == RewardType.SHAPING_STATE_ACTION
        or es_config["reward_type"] == RewardType.SHAPING_STATE
        or es_config["reward_type"] == RewardType.SHAPING_STATE_ACTION
    )


def is_state_only(es_config):
    return (
        RewardType[es_config["reward_type"]] == RewardType.SHAPING_STATE
        or RewardType[es_config["reward_type"]] == RewardType.REWARD_STATE
        or es_config["reward_type"] == RewardType.REWARD_STATE
        or es_config["reward_type"] == RewardType.SHAPING_STATE
    )


def is_state_action(es_config):
    return (
        RewardType[es_config["reward_type"]] == RewardType.SHAPING_STATE_ACTION
        or RewardType[es_config["reward_type"]] == RewardType.REWARD_STATE_ACTION
        or es_config["reward_type"] == RewardType.REWARD_STATE_ACTION
        or es_config["reward_type"] == RewardType.SHAPING_STATE_ACTION
    )


def get_irl_config(es_config, original_training_config):
    if es_config is None:
        wandb.init(project="IRL")
        es_config = wandb.config

    print(es_config)
    es_config["discr_final_lr"] = es_config["irl_lrate_init"] * (
        0.1 ** es_config["discr_final_lr_diff"]
    )
    original_training_config["NORMALIZE_REWARD"] = es_config["reward_normalize"]
    original_training_config["NORMALIZE_OBS"] = es_config["obs_normalize"]
    es_training_config = original_training_config.copy()
    if "inner_lr_linear" in es_config:
        es_training_config["ANNEAL_LR"] = es_config["inner_lr_linear"]
    if "inner_lr" in es_config:
        es_training_config["LR"] = es_config["inner_lr"]
    if "inner_steps" in es_config:
        es_training_config["NUM_STEPS"] = es_config["inner_steps"]
    if "percentage_training" in es_config:
        es_training_config["NUM_UPDATES"] = int(
            original_training_config["NUM_UPDATES"] * es_config["percentage_training"]
        )
    elif "num_updates_inner_loop" in es_config:
        es_training_config["NUM_UPDATES"] = int(es_config["num_updates_inner_loop"])
        es_training_config["ORIG_NUM_UPDATES"] = int(
            original_training_config["ORIG_NUM_UPDATES"]
            * original_training_config["NUM_STEPS"]
            / (es_config["num_updates_inner_loop"] * es_training_config["NUM_STEPS"])
        )
    else:
        raise ValueError(
            "Either percentage_training or num_updates_inner_loop key must be present in the configuration"
        )
    # we set the total number of timesteps in IRL to be the same as standard RL
    if "irl_generations" not in es_config:
        if "percentage_training" in es_config:
            es_config["irl_generations"] = int(1 / es_config["percentage_training"])
        else:
            es_config["irl_generations"] = int(
                original_training_config["ORIG_NUM_UPDATES"]
                * original_training_config["NUM_STEPS"]
                / (
                    es_config["num_updates_inner_loop"]
                    * es_training_config["NUM_STEPS"]
                )
            )
    buffer_size = (
        es_config["num_eval_envs"]
        * es_config["inner_steps"]
        * es_config["irl_generations"]
    )
    if "irl_plus" in es_config:
        if not es_config["irl_plus"]:
            buffer_size = es_config["num_eval_envs"] * es_config["inner_steps"]
    if "buffer_size_perc" in es_config:
        es_config["buffer_size"] = int(es_config["buffer_size_perc"] * buffer_size)
    else:
        es_config["buffer_size"] = buffer_size
    print("Num IRL outer loop steps: ", es_config["irl_generations"])
    print("total timesteps RL", original_training_config["TOTAL_TIMESTEPS"])
    total_irl_timesteps = (
        es_config["irl_generations"]
        * es_training_config["NUM_UPDATES"]
        * es_training_config["NUM_STEPS"]
        * es_training_config["NUM_ENVS"]
    )
    print("Total timesteps IRL", total_irl_timesteps)
    print(
        "Total timesteps IRL inner loop",
        es_config["num_updates_inner_loop"]
        * es_training_config["NUM_STEPS"]
        * es_training_config["NUM_ENVS"],
    )
    return es_config, es_training_config


def generate_config(args, seed):
    if args.loss == "IRL" and args.env == "hopper":
        config = HOPPER_IRL_CONFIG.copy()
    elif args.loss == "IRL" and args.env == "ant":
        config = ANT_IRL_CONFIG.copy()
    elif args.loss == "IRL" and args.env == "halfcheetah":
        config = HALFCHEETAH_IRL_CONFIG.copy()
    elif args.loss == "IRL" and args.env == "walker2d":
        config = WALKER_IRL_CONFIG.copy()
    else:
        config = {}
    if args.generations is not None:
        config["generations"] = args.generations
    config["seed"] = seed
    config["wandb_log"] = args.log
    config["plot"] = args.plot
    config["save_to_file"] = args.save
    config["env"] = args.env
    config["loss"] = args.loss
    config = get_eval_config(config)

    return config


class RewardWrapper:
    def __init__(
        self,
        env,
        env_params,
        reward_network,
        rew_network_params,
        shaping_network,
        shap_network_params,
        include_action=False,
        training_config=None,
        invert_reward=False,
        data_stats=None,
        debug=False,
    ):
        self._env = env
        self.action_size = get_action_size(env, env_params)
        self.observation_size = get_observation_size(env, env_params)
        self.reward_network = reward_network
        self.rew_network_params = rew_network_params
        self.shaping_network = shaping_network
        self.shap_network_params = shap_network_params
        self.include_action = include_action
        self.gamma = training_config["GAMMA"]
        self.agent_net = get_network(self.action_size, training_config)
        self.invert_reward = invert_reward
        self.debug = debug
        if data_stats is None:
            self.data_avg = jnp.zeros(self.observation_size)
            self.data_var = jnp.ones(self.observation_size)
        else:
            self.data_avg = data_stats[0]
            self.data_var = data_stats[1]

    def normalize_obs(self, obs):
        norm_obs = (obs - self.data_avg) / jnp.sqrt(self.data_var + 1e-8)
        return norm_obs

    def reset(self, key, params=None):
        obsv, env_state = self._env.reset(key, params)
        return obsv, env_state

    def step(self, key, state, action, params=None, prev_done=False, agent_params=None):
        obs, next_state, real_reward, done, info = self._env.step(
            key, state, action, params
        )
        reward = real_reward
        if self.reward_network is not None:
            reward_input = maybe_concat_action(
                self.include_action,
                self.action_size,
                self.normalize_obs(self._get_obs(state, params)),
                # self._get_obs(state, params),
                action,
            )
            reward = self.reward_network.apply(self.rew_network_params, reward_input)
            if self.invert_reward:
                reward = reward * -1
            new_reward = jnp.squeeze(reward)
            reward = new_reward
            info["irl_reward"] = reward
        if self.shaping_network is not None:
            cur_state_shape_input = maybe_concat_action(
                self.include_action,
                self.action_size,
                self.normalize_obs(self._get_obs(state, params)),
                # self._get_obs(state, params),
                action,
            )
            next_state_obs = self.normalize_obs(self._get_obs(next_state, params))
            # next_state_obs = self._get_obs(next_state, params)
            pi, _ = self.agent_net.apply(agent_params, next_state_obs)
            key, action_key = jax.random.split(key)
            next_action = pi.sample(seed=action_key)
            next_state_shape_input = maybe_concat_action(
                self.include_action, self.action_size, next_state_obs, next_action
            )
            cur_state_shaping = jax.lax.select(
                prev_done,
                0.0,
                jnp.squeeze(
                    self.shaping_network.apply(
                        self.shap_network_params, cur_state_shape_input
                    )
                ),
            )
            next_state_shaping = jax.lax.select(
                done,
                0.0,
                jnp.squeeze(
                    self.shaping_network.apply(
                        self.shap_network_params, next_state_shape_input
                    )
                ),
            )
            reward = reward - cur_state_shaping * self.gamma + next_state_shaping
        info["real_reward"] = real_reward
        if self.debug:
            return obs, next_state, real_reward, done, info
        else:
            return obs, next_state, reward, done, info

    def _get_obs(self, state, params=None):
        if hasattr(state, "obs"):
            return state.obs
        elif hasattr(self._env, "get_obs"):
            try:
                return self._env.get_obs(state)
            except TypeError:
                return self._env.get_obs(state, params)

    def observation_space(self, params):
        return self._env.observation_space(params)

    def action_space(self, params):
        return self._env.action_space(params)


def get_expert_obsv_and_actions(
    true_env,
    env_params,
    es_config,
    make_train_fn,
    eval_fn,
    original_training_config,
    filename=None,
):
    if not filename:
        trained_expert_path = f"{os.getcwd()}/experts/{es_config['env']}_{es_config['backend']}_{es_config['expert_num_seeds']}.pkl"
    else:
        trained_expert_path = filename
    rng = jax.random.PRNGKey(es_config["seed"])
    if not os.path.exists(trained_expert_path):
        print(f"Trained expert {es_config['backend']} not found, retraining")
        # original_training_config["NUM_UPDATES"] = 2441
        # print("backend expert", true_env._env.backend)
        train_fn = make_train_fn(
            config=original_training_config,
            env=true_env,
            env_params=env_params,
            runner_state_start=None,
            log_timestep_returns=True,
            return_obs_and_actions=False,
        )
        rng = jax.random.PRNGKey(0)
        expert_train_out = jax.jit(train_fn)(rng)
        print("expert mean", expert_train_out["runner_state"][1].mean.shape)
        print("expert var", expert_train_out["runner_state"][1].var.shape)
        expert_obsv, expert_actions, expert_rewards, expert_dones = eval_fn(
            es_config["num_expert_eval_envs"],
            es_config["num_eval_steps"],
            true_env,
            env_params,
            expert_train_out["runner_state"][0].params,
            rng,
            true_env.agent_net,
            True,
            es_config["obs_normalize"],
            expert_train_out["runner_state"][1],
        )
        with open(trained_expert_path, "wb") as f:
            original_returns = expert_train_out["metrics"][
                "timestep_returned_episode_returns"
            ]
            expert_obsv = expert_obsv.transpose(1, 0, 2)
            expert_dones = expert_dones.transpose(1, 0)
            expert_rewards = expert_rewards.transpose(1, 0)
            expert_obsv = expert_obsv.reshape(
                expert_obsv.shape[0] * expert_obsv.shape[1], expert_obsv.shape[2], -1
            )
            expert_dones = expert_dones.reshape(
                expert_dones.shape[0] * expert_dones.shape[1], expert_dones.shape[2]
            )
            expert_rewards = expert_rewards.reshape(
                expert_rewards.shape[0] * expert_rewards.shape[1],
                expert_rewards.shape[2],
            )
            complete_idx = jnp.argmax(expert_dones[:, ::-1], axis=1)
            complete_idx = expert_dones.shape[1] - complete_idx - 1

            complete_ep_expert_states = expert_obsv[0, : complete_idx[0], :]
            for i in range(1, expert_obsv.shape[0]):  # envs
                complete_ep_expert_states = jnp.concatenate(
                    (complete_ep_expert_states, expert_obsv[i, : complete_idx[i], :]),
                    axis=0,
                )
            rng = jax.random.PRNGKey(0)
            complete_ep_expert_states = jax.random.shuffle(
                rng, complete_ep_expert_states, axis=0
            )
            print("complete expert states shape", complete_ep_expert_states.shape)
            pickle.dump(
                {
                    "returns": original_returns,
                    "final_return": original_returns[-1],
                    "expert_obsv": expert_obsv,
                    "complete_expert_states": complete_ep_expert_states,
                    "expert_actions": expert_actions,
                    "expert_rewards": expert_rewards,
                    "expert_dones": expert_dones,
                    "expert_params": expert_train_out["runner_state"][0].params,
                    "norm_mean": jnp.mean(
                        expert_train_out["runner_state"][1].mean, axis=0
                    ),
                    "norm_var": jnp.mean(
                        expert_train_out["runner_state"][1].var, axis=0
                    ),
                },
                f,
            )
    else:
        print(f"Trained TRAIN expert {es_config['backend']} FOUND")

    with open(trained_expert_path, "rb") as f:
        expert_train_out = pickle.load(f)
        original_returns = expert_train_out["returns"]
        last_return = expert_train_out["final_return"]
        expert_obsv_complete = expert_train_out["complete_expert_states"]
        expert_dones = expert_train_out["expert_dones"]
        expert_actions = expert_train_out["expert_actions"]
        # expert_rewards = expert_train_out["expert_rewards"]
        expert_obsv = expert_train_out["expert_obsv"]
        expert_norm_mean = expert_train_out["norm_mean"]
        expert_norm_var = expert_train_out["norm_var"]
    if es_config["wandb_log"]:
        wandb.log(
            step=0,
            data={
                "base_last_return": jnp.mean(last_return),
                # "original_train_plt": plt,
            },
        )
    print("Expert last return", last_return)
    return (
        expert_obsv_complete,
        expert_actions,
        (expert_obsv, expert_dones, None),
        original_returns,
        last_return,
        (expert_norm_mean, expert_norm_var),
    )


class RewardNetworkPenalty(flax_nn.Module):
    network: Any
    discount: Any = 0.0

    def apply(self, params, x):
        res = jax.vmap(self.network.apply, (0, None))(params, x)
        return jnp.mean(res, axis=0) + (jnp.std(res, axis=0) * self.discount)


class RewardNetworkPessimistic(flax_nn.Module):
    network: Any

    def apply(self, params, x):
        res = jax.vmap(self.network.apply, (0, None))(params, x)
        return jnp.max(res, axis=0)


class CriticAsShaping(flax_nn.Module):
    network: Any

    def apply(self, params, x):
        res = self.network.apply(params, x)
        return res[1]

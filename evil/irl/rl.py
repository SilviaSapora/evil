from abc import ABC
from functools import partial

import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import wandb
from evil.utils.utils import (
    RewardWrapper,
    get_action_size,
    get_network,
)

from evil.training.ppo_v2_irl import make_train, eval
from evil.training.ppo_v2_cont_irl import (
    make_train as make_train_cont,
    eval as eval_cont,
)


class RL(ABC):
    def __init__(
        self,
        env,
        training_config,
        outer_config,
        logging_run,
        env_params,
        reward_net=None,
        shaping_net=None,
    ) -> None:
        super().__init__()
        self._env = env
        self.action_size = get_action_size(env, env_params)
        self._reward_net = reward_net
        self._shaping_net = shaping_net
        print("Inner loop update steps", training_config["NUM_UPDATES"])
        if training_config["DISCRETE"]:
            self.make_train = make_train
            self.eval = eval
        else:
            self.make_train = make_train_cont
            self.eval = eval_cont
        self.agent_net = get_network(self.action_size, training_config)
        self._include_action = False

        self.env_params = env_params
        self._training_config = training_config
        self._outer_config = outer_config
        self._generations = outer_config["generations"]
        self._log_every = outer_config["log_gen_every"]
        self._training_config["LR"] = self._outer_config["inner_lr"]
        self._training_config["ANNEAL_LR"] = self._outer_config["inner_lr_linear"]
        self._training_config["NUM_UPDATES"] = 1
        if outer_config["save_to_file"]:
            self._save_dir = outer_config["save_dir"]
        self.num_gpus = len(jax.devices())

    def wandb_callback(self, gen, fitness, all_fitness, train_metrics, extra_info):
        if gen % self._log_every == 0:
            # last_return = train_metrics["last_return"].reshape(-1)
            # returns = train_metrics["returned_episode_returns"].reshape(-1)
            std_err = jnp.std(all_fitness, axis=0) / jnp.sqrt(all_fitness.shape[0])
            mean_fitness = jnp.mean(fitness)
            metrics = {
                "rl_mean_return": mean_fitness,
                "rl_std_err_up_return": mean_fitness + jnp.mean(std_err),
                "rl_std_err_down_return": mean_fitness - jnp.mean(std_err),
                # "avg_last_return": jnp.mean(last_return),
                # "avg_return": jnp.mean(returns),
            }
            wandb.log(
                step=int(gen),
                data=metrics | extra_info,
            )

    def train_agents(
        self,
        rng,
        runner_state=None,
        rew_net_params=None,
        shap_net_params=None,
        data_stats=None,
    ):
        invert_reward = False
        if rew_net_params:
            assert self._reward_net is not None
            invert_reward = True
        if shap_net_params:
            assert self._shaping_net is not None
        wrapped_env = RewardWrapper(
            self._env,
            self.env_params,
            self._reward_net,
            rew_net_params,
            self._shaping_net,
            shap_net_params,
            include_action=self._include_action,
            training_config=self._training_config,
            invert_reward=invert_reward,
            data_stats=data_stats,
        )
        training_config = self._training_config.copy()
        train_fn = self.make_train(
            config=training_config,
            env=wrapped_env,
            env_params=self.env_params,
            runner_state_start=runner_state,
            log_timestep_returns=True,
            return_obs_and_actions=False,
        )
        train_out = jax.jit(train_fn)(rng)
        return train_out["runner_state"], train_out["metrics"]

    def train_step(
        self, carry, unused, rew_net_params=None, shap_net_params=None, data_stats=None
    ):
        rng, runner_state, gen = carry
        rng, rng_train = jax.random.split(rng, 2)

        runner_state, metrics = self.train_agents(
            rng=rng_train,
            runner_state=runner_state,
            rew_net_params=rew_net_params,
            shap_net_params=shap_net_params,
            data_stats=data_stats,
        )
        fitness = jnp.mean(metrics["timestep_returned_episode_returns"])
        extra_info = {
            "rl_runner_state_mean": runner_state[1].mean.mean(),
            "rl_runner_state_var": runner_state[1].var.mean(),
        }
        # all_fitness = jax.lax.all_gather(fitness, axis_name="i")
        # jax.debug.print("{g} - return {f}", g=gen, f=fitness)
        # if self._outer_config["wandb_log"]:
        #     jax.debug.callback(
        #         self.wandb_callback, gen, fitness, all_fitness, metrics, extra_info
        #     )
        return (rng, runner_state, gen + 1), metrics

    def train(self, rng, rew_net_params=None, shap_net_params=None, data_stats=None):
        print("TRAIN RL ONLY")
        runner_state = None
        ptrain_step = partial(
            self.train_step,
            rew_net_params=rew_net_params,
            shap_net_params=shap_net_params,
            data_stats=data_stats,
        )
        carry_init, _ = jax.jit(ptrain_step)((rng, runner_state, 0), None)
        (_rng, last_runner_state, last_gen), metrics = jax.lax.scan(
            ptrain_step, carry_init, [jnp.zeros(self._generations)]
        )
        return last_runner_state, metrics

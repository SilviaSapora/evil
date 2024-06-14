from abc import ABC

import jax
import jax.numpy as jnp
import wandb
from evil.utils.utils import (
    RewardWrapper,
    get_action_size,
    get_network,
    get_observation_size,
)

from evil.training.supervised import SupervisedIL


class BC(ABC):
    def __init__(
        self,
        env,
        training_config,
        es_config,
        logging_run,
        env_params,
        expert_data=None,
    ) -> None:
        super().__init__()
        self._env = env
        self._reward_network = None
        self.action_size = get_action_size(env, env_params)
        self.observation_size = get_observation_size(env, env_params)[0]
        self.agent_net = get_network(self.action_size, training_config)
        self._include_action = False

        self.env_params = env_params
        self._training_config = training_config
        self._es_config = es_config
        self._generations = es_config["generations"]
        self._log_every = es_config["log_gen_every"]
        self._run = logging_run
        if es_config["save_to_file"]:
            self._save_dir = es_config["save_dir"]
        self.num_gpus = len(jax.devices())
        self.supervised_config = self.get_supervised_config()
        self.expert_obsv = expert_data[0]
        self.expert_actions = expert_data[1]
        self.il_agent_train = SupervisedIL(
            network_il=self.agent_net,
            expert_obsv=self.expert_obsv.reshape(-1, self.expert_obsv.shape[-1]),
            expert_actions=self.expert_actions.reshape(
                -1, self.expert_actions.shape[-1]
            ),
            config=self.supervised_config,
        )

    def get_supervised_config(self):
        # this is for IL
        return {
            "DISCRETE": self._training_config["DISCRETE"],
            "NUM_ACTIONS": self.action_size,
            "NUM_UPDATES": 1,
            "NUM_EPOCHS_PER_UPDATE": 8,
            "LR": 1e-4,
            "OBS_SIZE": self.observation_size,
        }

    def wandb_callback(self, gen, loss, reward):
        if gen % self._log_every == 0:
            metrics = {
                "mean_fitness": jnp.mean(loss),
                "avg_return": jnp.mean(reward),
            }
            wandb.log(
                step=int(gen),
                data=metrics,
            )

    def train_step(self, carry, unused):
        (rng, runner_state, gen) = carry
        train_rng, rng = jax.random.split(rng, 2)

        il_train = jax.jit(self.il_agent_train.train)
        il_train_state, loss_history = il_train(train_rng, runner_state)
        return (rng, il_train_state, gen + 1), loss_history.mean()

    def train(self, rng):
        runner_state = None
        carry_init, _ = jax.jit(self.train_step)((rng, runner_state, 0), None)
        (_rng, last_runner_state, last_gen), losses = jax.lax.scan(
            self.train_step, carry_init, xs=None, length=self._generations
        )
        return last_runner_state, losses

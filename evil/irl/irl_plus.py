from abc import ABC
from functools import partial

import jax
import jax.numpy as jnp
import optax
from evil.utils.buffer import ObsvActionBuffer
from flax.training.train_state import TrainState
from evil.utils.env_utils import get_test_params
from evil.utils.utils import (
    RewardNetworkPenalty,
    RewardNetworkPessimistic,
    RewardWrapper,
    RewardType,
    get_action_size,
    get_network,
    get_observation_size,
    get_xentropy_match_score_expert,
    is_state_action,
    is_state_only,
    maybe_concat_action,
)
from evil.irl.gail_discriminator import Discriminator

from evil.training.ppo_v2_irl import make_train, eval
from evil.training.ppo_v2_cont_irl import (
    make_train as make_train_cont,
    eval as eval_cont,
)


class IRLPlus(ABC):
    def __init__(
        self,
        env,
        training_config,
        es_config,
        logging_run,
        env_params,
        expert_data,
        shaping_net,
    ) -> None:
        super().__init__()
        self._env = env
        self._reward_network = None
        if shaping_net is not None:
            self._shaping_network = shaping_net
        else:
            self._shaping_network = None
        self.action_size = get_action_size(env, env_params)
        if training_config["DISCRETE"]:
            observation_shape = env.observation_space(env_params).shape[0]
            self.make_train = make_train
            self.eval = eval
        else:
            self.make_train = make_train_cont
            self.eval = eval_cont
            observation_shape = get_observation_size(env, env_params)[0]
        self.agent_net = get_network(self.action_size, training_config)
        self._include_action = False
        if is_state_only(es_config):
            self.in_features = observation_shape
        elif is_state_action(es_config):
            self.in_features = observation_shape + self.action_size
            self._include_action = True
        elif RewardType[es_config["reward_type"]] == RewardType.NONE:
            raise NotImplementedError(
                f"reward type is None in IRL class, this shouldn't happen"
            )
        else:
            raise NotImplementedError(
                f"reward type not implemented {es_config['reward_type']}"
            )

        self._training_config = training_config
        self._es_config = es_config
        self._generations = es_config["irl_generations"]
        # if self._es_config["loss"] == "IRL":
        self._log_every = es_config["log_gen_every"]
        # else:
        #     self._log_every = 200
        if "alpha" in es_config:
            self.alpha = es_config["alpha"]
        else:
            self.alpha = 0
        self._run = logging_run
        self.expert_obsv = expert_data[0].reshape(-1, expert_data[0].shape[-1])
        self.expert_obsv = self.expert_obsv[
            : self._es_config["num_expert_eval_envs"]
            * self._es_config["num_eval_steps"],
            :,
        ]
        self.expert_actions = expert_data[1].reshape(-1, expert_data[1].shape[-1])
        self.expert_actions = self.expert_actions[
            : self._es_config["num_expert_eval_envs"]
            * self._es_config["num_eval_steps"],
            :,
        ]
        if len(expert_data) == 3:
            self.expert_norm_mean = expert_data[2][0]
            self.expert_norm_var = expert_data[2][1]
        else:
            self.expert_norm_mean = jnp.zeros(self.in_features)
            self.expert_norm_var = jnp.ones(self.in_features)
            print("no expert mean and var passed, not normalising")
        self.inner_lr_linear = es_config["inner_lr_linear"]
        # TODO these two should probably be moved to the utils function at the beginning
        # of setting up the inner training config
        self._training_config["ANNEAL_LR"] = self.inner_lr_linear
        self._training_config["LR"] = es_config["inner_lr"]
        self.updates_every = es_config["discr_updates_every"]
        if es_config["save_to_file"]:
            self._save_dir = es_config["save_dir"]
        self.num_gpus = len(jax.devices())
        expert_data = maybe_concat_action(
            self._include_action,
            self.action_size,
            self.expert_obsv,
            self.expert_actions,
        )
        self.expert_data_flat = expert_data.reshape([-1, expert_data.shape[-1]])

        self.buffer = ObsvActionBuffer(
            obsv_shape=observation_shape,
            action_shape=self.action_size,
            include_action=self._include_action,
            ep_length=self._es_config["num_eval_steps"],
            envs=self._es_config["num_eval_envs"],
            max_size=es_config["buffer_size"],
        )
        self.discriminator_config = {
            "reward_net_hsize": es_config["reward_net_hsize"],
            "learning_rate": es_config["irl_lrate_init"],
            "discr_updates": es_config[
                "discr_updates"
            ],  # how many discriminator update steps should be done
            "n_features": self.in_features,
            "l2_loss": es_config["discr_l2_loss"],
            "transition_steps_decay": int(es_config["discr_trans_decay"]),
            "schedule_type": es_config["discr_schedule_type"],
            "discr_loss": es_config["discr_loss"],
            "discr_final_lr": es_config["discr_final_lr"],
            "buffer": self.buffer,
            "expert_data": self.expert_data_flat,
            "batch_size": es_config["discr_batch_size"],
        }
        self.discr = Discriminator(**self.discriminator_config)
        if self._es_config["reward_net_ensemble_type"] == "avg":
            self._reward_network = RewardNetworkPenalty(self.discr, 0.0)
        elif self._es_config["reward_net_ensemble_type"] == "min":
            self._reward_network = RewardNetworkPessimistic(self.discr)

    def reinit_train_state(self, rng, env_params, prev_runner_state):
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(self._env.observation_space(env_params).shape)
        init_network_params = self.agent_net.init(_rng, init_x)
        new_train_state = TrainState.create(
            apply_fn=prev_runner_state.apply_fn,
            params=init_network_params,
            tx=prev_runner_state.tx,
        )
        return new_train_state

    def wandb_callback(self, gen, episode_returns, train_metrics):
        (
            episode_returns_mean,
            episode_returns_min,
            episode_returns_max,
            episode_returns_all,
        ) = episode_returns
        if gen % self._log_every == 0:
            std_err = jnp.std(episode_returns_all) / jnp.sqrt(len(episode_returns_all))
            metrics = {
                "avg_return": jnp.mean(episode_returns_mean),
                "min_return": episode_returns_min,
                "max_return": episode_returns_max,
                "std_err_up": jnp.mean(episode_returns_mean) + std_err,
                "std_err_down": jnp.mean(episode_returns_mean) - std_err,
            }
            if self._run is not None:
                self._run.log(
                    step=int(gen),
                    data=metrics | train_metrics,
                )

    def xe_loss(self, runner_state):
        return get_xentropy_match_score_expert(
            params=runner_state[0].params,
            expert_obsv=self.expert_obsv,
            expert_actions=self.expert_actions,
            ppo_network=self.agent_net,
            is_discrete=self._training_config["DISCRETE"],
        )

    @partial(jax.jit, static_argnums=(0))
    def get_discriminator_loss(self, buffer_state, norm_info, discr_train_state, key):
        new_discr_train_state, discr_losses = self.discr.train_epoch(
            imit_data_buffer_state=buffer_state,
            norm_info=norm_info,
            train_state=discr_train_state,
            key=key,
        )
        return new_discr_train_state, discr_losses

    def train_agents(
        self,
        rew_network_params,
        rng,
        env_params,
        runner_state=None,
        shap_network_params=None,
        test=False,
        norm_info=None,
    ):
        current_config = self._training_config.copy()
        if test:
            current_config["NUM_UPDATES"] = current_config["ORIG_NUM_UPDATES"]
        if norm_info is None:
            norm_mean = jnp.zeros(self.in_features)
            norm_var = jnp.ones(self.in_features)
        else:
            norm_mean = norm_info[0]
            norm_var = norm_info[1]
        wrapped_env = RewardWrapper(
            self._env,
            env_params,
            self._reward_network,
            rew_network_params=rew_network_params.params,
            shaping_network=self._shaping_network,
            shap_network_params=shap_network_params,
            include_action=self._include_action,
            training_config=self._training_config,
            invert_reward=True,
            data_stats=(norm_mean, norm_var),
        )

        train_fn = self.make_train(
            config=current_config,
            env=wrapped_env,
            env_params=env_params,
            runner_state_start=runner_state,
            log_timestep_returns=False,
            return_obs_and_actions=True,
        )
        train_out = jax.jit(train_fn)(rng)
        return (
            train_out["runner_state"],
            train_out["metrics"],
            train_out["obs"],
            train_out["actions"],
        )

    def train_step(self, carry, unused, shap_net_params, env_params):
        rng, runner_state, discr_train_state, buffer_state, gen = carry

        rng, rng_loss, rng_train, rng_sample, rng_reinit = jax.random.split(rng, 5)
        rng_loss = jax.random.split(rng_loss, self._es_config["num_discr"])

        # if print_rng:
        #     jax.debug.print("rng {r}", r=rng_train)

        if self._es_config["dual"]:
            runner_state = None
        if self._es_config["random_reinit"] and runner_state is not None:
            new_train_state = self.reinit_train_state(
                rng_reinit, env_params, runner_state[0]
            )
            min_prob = self._es_config["random_reinit_prob"] * (
                0.1 ** self._es_config["random_reinit_prob_final"]
            )
            if self._es_config["random_reinit_decay"] == "harmonic":
                frac = 1.0 / (gen + 1)
                p = (self._es_config["random_reinit_prob"] - min_prob) * jnp.maximum(
                    frac, 0
                )
                p = jnp.maximum(self._es_config["random_reinit_prob"] * frac, min_prob)
            elif self._es_config["random_reinit_decay"] == "linear":
                frac = 1.0 - (gen / self._es_config["irl_generations"])
                p = (self._es_config["random_reinit_prob"] - min_prob) * jnp.maximum(
                    frac, 0
                )
                p = jnp.maximum(self._es_config["random_reinit_prob"] * frac, min_prob)

            reset = jax.random.bernoulli(rng_reinit, p=p)
            new_train_state = jax.tree_map(
                lambda x, y: jax.lax.select(reset, x, y),
                new_train_state,
                runner_state[0],
            )
            runner_state = (
                new_train_state,
                runner_state[1],
                runner_state[2],
                runner_state[3],
                runner_state[4],
            )
        runner_state, metrics, actor_obs, actor_actions = self.train_agents(
            rew_network_params=discr_train_state,
            rng=rng_train,
            env_params=env_params,
            runner_state=runner_state,
            shap_network_params=shap_net_params,
            norm_info=(self.expert_norm_mean, self.expert_norm_var),
            # norm_info=(buffer_state.norm_mean, buffer_state.norm_var),
        )
        buffer_state = self.buffer.add(
            actor_obs, actor_actions, rng_sample, buffer_state
        )
        new_discr_train_state, discr_loss = jax.vmap(
            self.get_discriminator_loss, (None, None, 0, 0), (0, 0)
        )(
            buffer_state,
            (self.expert_norm_mean, self.expert_norm_var),
            discr_train_state,
            rng_loss,
        )
        # xe_loss = self.xe_loss(runner_state)
        if self._es_config["dual"]:
            runner_state = None
        next_discr_train_state = jax.tree_map(
            lambda x, y: jax.lax.select(gen % self.updates_every == 0, x, y),
            new_discr_train_state,
            discr_train_state,
        )
        # test_runner_state, test_metrics = self.train_agents(discr_train_state.params, rng_train, None, None, test=True)
        # test_xe_loss = self.xe_test_loss(test_runner_state)
        episode_real_returns = metrics["returned_episode_real_returns"].mean()
        episode_irl_returns = metrics["returned_episode_irl_returns"].mean()
        train_metrics = {
            "returned_episode_real_returns": episode_real_returns,
            "returned_episode_irl_returns": episode_irl_returns,
            "discr_losses": discr_loss["loss"].mean(),
            "exp_loss": discr_loss["exp_loss"].mean(),
            "imit_loss": discr_loss["imit_loss"].mean(),
            "recent_imit_loss": discr_loss["recent_imit_loss"].mean(),
            "grad_penalty": discr_loss["grad_penalty"].mean(),
        }
        jax.debug.print(
            "{g} - return {r} - discr loss {l}",
            g=gen,
            r=episode_real_returns,
            l=discr_loss["loss"].mean(),
        )
        if self._es_config["wandb_log"] and self._es_config["loss"] == "IRL":
            jax.debug.callback(
                self.wandb_callback,
                gen,
                (
                    jax.lax.pmean(episode_real_returns, axis_name="i"),
                    jax.lax.pmin(episode_real_returns, axis_name="i"),
                    jax.lax.pmax(episode_real_returns, axis_name="i"),
                    jax.lax.all_gather(episode_real_returns, axis_name="i"),
                ),
                train_metrics,
            )
        return (
            (
                rng,
                runner_state,
                next_discr_train_state,
                buffer_state,
                gen + 1,
            ),
            train_metrics,
        )

    def train(self, rng, env_params, shaping_net_params=None, return_buffer=False):
        discr_rng, train_rng = jax.random.split(rng, 2)
        train_rng = jax.random.split(
            train_rng, self._es_config["seeds"] // jax.device_count()
        )
        discr_rng = jax.random.split(
            discr_rng, self._es_config["seeds"] // jax.device_count()
        )
        seeds_discr_rng = jax.vmap(jax.random.split, (0, None))(
            discr_rng, self._es_config["num_discr"]
        )
        discr_train_state, rng = jax.vmap(jax.vmap(self.discr._init_state))(
            seeds_discr_rng
        )
        buffer_state = jax.vmap(self.buffer.init_state)(discr_rng)
        ptrain_step = partial(
            self.train_step,
            shap_net_params=shaping_net_params,
            env_params=env_params,
        )

        runner_state = None
        vmap_train_step = jax.jit(
            jax.vmap(
                ptrain_step,
                in_axes=((0, 0, 0, 0, None), None),
                axis_name="i",
                out_axes=(
                    (0, 0, 0, 0, None),
                    (0),
                ),
            ),
        )
        carry_init, _ = vmap_train_step(
            (
                train_rng,
                runner_state,
                discr_train_state,
                buffer_state,
                0,
            ),
            None,
        )
        (
            _rng,
            last_runner_state,
            last_discr_state,
            buffer_state,
            last_gen,
        ), (
            irl_train_metrics
        ) = jax.lax.scan(vmap_train_step, carry_init, [jnp.zeros(self._generations)])

        if return_buffer:
            return (
                (
                    last_runner_state,
                    last_discr_state,
                    # (buffer_state.norm_mean, buffer_state.norm_var),
                    (self.expert_norm_mean, self.expert_norm_var),
                ),
                irl_train_metrics,
                buffer_state,
            )
        return (
            (
                last_runner_state,
                last_discr_state,
                # (buffer_state.norm_mean, buffer_state.norm_var),
                (self.expert_norm_mean, self.expert_norm_var),
            ),
            irl_train_metrics,
        )

import jax
import jax.numpy as jnp
import chex
import numpy as np
from flax import struct
from functools import partial
from typing import Optional, Tuple, Union, Any
from gymnax.environments import environment, spaces
from brax import envs
from brax.envs.wrappers.training import EpisodeWrapper, AutoResetWrapper


class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)


class FlattenObservationWrapper(GymnaxWrapper):
    """Flatten the observations of the environment."""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    def observation_space(self, params) -> spaces.Box:
        assert isinstance(
            self._env.observation_space(params), spaces.Box
        ), "Only Box spaces are supported for now."
        return spaces.Box(
            low=self._env.observation_space(params).low,
            high=self._env.observation_space(params).high,
            shape=(np.prod(self._env.observation_space(params).shape),),
            dtype=self._env.observation_space(params).dtype,
        )

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, state = self._env.reset(key, params)
        obs = jnp.reshape(obs, (-1,))
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        obs = jnp.reshape(obs, (-1,))
        return obs, state, reward, done, info


@struct.dataclass
class LogEnvStateRewardIRL:
    env_state: environment.EnvState
    episode_real_returns: float
    episode_irl_returns: float
    returned_episode_real_returns: float
    returned_episode_irl_returns: float
    timestep: int
    sum_of_real_returns: float
    sum_of_irl_returns: float
    dones: int


class LogWrapperRewardIRL(GymnaxWrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, env_state = self._env.reset(key, params)
        state = LogEnvStateRewardIRL(env_state, 0, 0, 0, 0, 0, 0, 0, 0)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
        prev_done: bool = False,
        agent_params: Any = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params, prev_done, agent_params
        )
        # with a reward wrapper, we might have 3 total rewards:
        # 1. the real reward
        # 2. irl_reward: the reward from the reward network (generally the IRL reward)
        # 3. the shaped reward (shaping can be either from the IRL reward or for the real reward)
        # we want to log the real reward and the sub reward
        real_reward = info.get("real_reward", reward)
        irl_reward = info.get("irl_reward", reward)
        state = LogEnvStateRewardIRL(
            env_state=env_state,
            episode_real_returns=(state.episode_real_returns + real_reward)
            * (1 - done),
            episode_irl_returns=(state.episode_irl_returns + irl_reward) * (1 - done),
            returned_episode_real_returns=state.returned_episode_real_returns
            * (1 - done)
            + (state.episode_real_returns + real_reward) * done,
            returned_episode_irl_returns=state.returned_episode_irl_returns * (1 - done)
            + (state.episode_irl_returns + irl_reward) * done,
            timestep=state.timestep + 1,
            sum_of_real_returns=state.sum_of_real_returns
            + (state.episode_real_returns + real_reward) * done,
            sum_of_irl_returns=state.sum_of_irl_returns
            + state.returned_episode_irl_returns * (1 - done)
            + (state.episode_irl_returns + irl_reward) * done,
            dones=state.dones + done,
        )
        info["timestep_returned_episode_returns"] = state.returned_episode_real_returns
        info["timestep_returned_episode_irl_returns"] = (
            state.returned_episode_irl_returns
        )
        return obs, state, reward, done, info


class BraxGymnaxWrapper:
    def __init__(self, env_name, backend="positional"):
        env = envs.get_environment(env_name=env_name, backend=backend)
        env = EpisodeWrapper(env, episode_length=1000, action_repeat=1)
        env = AutoResetWrapper(env)
        self._env = env
        self.backend = backend
        self.action_size = env.action_size
        self.observation_size = (env.observation_size,)

    def reset(self, key, params=None):
        state = self._env.reset(key)
        return state.obs, state

    def step(self, key, state, action, params=None):
        next_state = self._env.step(state, action)
        return next_state.obs, next_state, next_state.reward, next_state.done > 0.5, {}

    def observation_space(self, params):
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self._env.observation_size,),
        )

    def action_space(self, params):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._env.action_size,),
        )


class ClipActionRewardIRL(GymnaxWrapper):
    def __init__(self, env, low=-1.0, high=1.0):
        super().__init__(env)
        self.low = low
        self.high = high

    def step(self, key, state, action, params=None, prev_done=False, agent_params=None):
        action = jnp.clip(action, self.low, self.high)
        return self._env.step(key, state, action, params, prev_done, agent_params)


class TremblingHandWrapper(GymnaxWrapper):
    def __init__(self, env, p_tremble=0.1):
        super().__init__(env)
        self.env = env
        self.p_tremble = p_tremble

    def step(self, key, state, action, params=None, prev_done=False, agent_params=None):
        key_tremble, key_action, key = jax.random.split(key, 3)
        sampled_action = jax.random.uniform(
            key_action,
            shape=self.env.action_space(params).shape,
            minval=self.env.action_space(params).low,
            maxval=self.env.action_space(params).high,
        )
        action = jax.lax.select(
            jax.random.uniform(key_tremble) < self.p_tremble, sampled_action, action
        )
        return self.env.step(key, state, action, params, prev_done, agent_params)


class TransformObservation(GymnaxWrapper):
    def __init__(self, env, transform_obs):
        super().__init__(env)
        self.transform_obs = transform_obs

    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        return self.transform_obs(obs), state

    def step(self, key, state, action, params=None):
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        return self.transform_obs(obs), state, reward, done, info


class TransformReward(GymnaxWrapper):
    def __init__(self, env, transform_reward):
        super().__init__(env)
        self.transform_reward = transform_reward

    def step(self, key, state, action, params=None):
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        return obs, state, self.transform_reward(reward), done, info


class VecEnvRewardIRL(GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.reset = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step = jax.vmap(self._env.step, in_axes=(0, 0, 0, None, 0, None))


@struct.dataclass
class NormalizeVecObsEnvState:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    env_state: environment.EnvState


class NormalizeVecObservationIRL(GymnaxWrapper):
    def __init__(self, env, normalize_obs):
        super().__init__(env)
        self.normalize_obs = normalize_obs

    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        state = NormalizeVecObsEnvState(
            mean=jnp.zeros(obs.shape[-1]),
            var=jnp.ones(obs.shape[-1]),
            count=1e-4,
            env_state=state,
        )
        batch_mean = jnp.mean(obs, axis=0)
        batch_var = jnp.var(obs, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeVecObsEnvState(
            mean=jax.lax.select(self.normalize_obs, new_mean, state.mean),
            var=jax.lax.select(self.normalize_obs, new_var, state.var),
            count=jax.lax.select(self.normalize_obs, new_count, state.count),
            env_state=state.env_state,
        )

        return (
            jax.lax.select(
                self.normalize_obs, (obs - state.mean) / jnp.sqrt(state.var + 1e-8), obs
            ),
            obs,
            state,
        )

    def step(self, key, state, action, params=None, prev_done=False, agent_params=None):
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params, prev_done, agent_params
        )

        batch_mean = jnp.mean(obs, axis=0)
        batch_var = jnp.var(obs, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeVecObsEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            env_state=env_state,
        )
        return (
            jax.lax.select(
                self.normalize_obs,
                (obs - state.mean) / jnp.sqrt(state.var + 1e-8),
                obs,
            ),
            obs,
            state,
            reward,
            done,
            info,
        )


@struct.dataclass
class NormalizeVecRewIRLEnvState:
    real_reward_mean: jnp.ndarray
    real_reward_var: jnp.ndarray
    real_reward_return_val: float
    count: float
    env_state: environment.EnvState


class NormalizeVecRewardIRL(GymnaxWrapper):
    def __init__(self, env, gamma, normalize_reward, normalize_shaped_reward=False):
        super().__init__(env)
        self.gamma = gamma
        self.normalize_reward = normalize_reward
        self.normalize_shaped_reward = normalize_shaped_reward

    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        batch_count = obs.shape[0]

        state = NormalizeVecRewEnvState(
            mean=0.0,
            var=1.0,
            count=1e-4,
            return_val=jnp.zeros((batch_count,)),
            env_state=state,
        )
        return obs, state

    def step(self, key, state, action, params=None, prev_done=False, agent_params=None):
        obs, env_state, reward, done, info = self._env.step(
            key,
            state.env_state,
            action,
            params,
            prev_done,
            agent_params,
        )
        if self.normalize_shaped_reward:  # we normalize everything
            underlying_reward = reward
        else:  # we only normalize the underlying reward, not the shaping
            underlying_reward = info.get("irl_reward", info.get("real_reward", reward))
        return_val = state.return_val * self.gamma * (1 - done) + underlying_reward

        batch_mean = jnp.mean(return_val, axis=0)
        batch_var = jnp.var(return_val, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeVecRewEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            return_val=return_val,
            env_state=env_state,
        )
        if self.normalize_reward:
            norm_reward = underlying_reward / jnp.sqrt(state.var + 1e-8)
        else:
            norm_reward = underlying_reward
        # (norm_real_reward or norm_irl_reward) + shaped_reward
        # if the reward is not shaped, then we simply get
        # norm_reward + reward - reward -> norm_reward
        # if the reward is shaped, then we get
        # norm_reward + tot_reward - underlying_reward -> norm_reward + (underlying_reward + shaping - underlying_reward)
        # -> norm_reward + shaping
        final_reward = norm_reward + (reward - underlying_reward)
        return obs, state, final_reward, done, info

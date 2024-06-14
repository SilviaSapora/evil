from flax import struct
from typing import Any

import jax
import jax.numpy as jnp
from evil.utils.utils import TrainRestart


@struct.dataclass
class PrevParamsState:
    buffer: Any
    best_runner_state: Any
    buffer_size: int = 0
    # call_count: int = 0


class PrevParamsHandler:
    def __init__(
        self,
        train_restart,
        popsize=60,
        max_size=300,
        restart_percentage=0.5,
        reset_every=None,
    ):
        self.train_restart = TrainRestart[train_restart]
        self.popsize = popsize
        self.max_size = max_size
        self.restart_percentage = int(popsize * restart_percentage)
        self.reset_every = reset_every

    def init_state(self):
        return PrevParamsState(buffer=None, best_runner_state=None)

    def add_to_buffer(self, state, new_runner_state, fitness):
        if self.train_restart == TrainRestart.RESTART_BEST:
            best_fitness_idx = jnp.argmax(fitness)
            best_runner_state = jax.tree_map(
                lambda x: x[best_fitness_idx], new_runner_state
            )
            state = state.replace(best_runner_state=best_runner_state)
        elif self.train_restart == TrainRestart.SAMPLE_INIT:
            top_idx = jnp.argsort(fitness)
            top_idx = top_idx[-(self.restart_percentage) :]
            new_runner_state = jax.tree_map(lambda x: x[top_idx], new_runner_state)
            if state.buffer is None:
                state = state.replace(buffer=new_runner_state, buffer_size=self.popsize)
            else:
                new_buffer = jax.tree_map(
                    lambda x, y: jnp.concatenate((x, y), axis=0),
                    state.buffer,
                    new_runner_state,
                )
                state = state.replace(
                    buffer=new_buffer,
                    buffer_size=state.buffer_size + self.popsize,
                )
        elif self.train_restart == TrainRestart.SAMPLE_RECENT_INIT:
            top_idx = jnp.argsort(fitness)
            top_idx = top_idx[-(self.popsize // 4) :]
            new_runner_state = jax.tree_map(lambda x: x[top_idx], new_runner_state)
            if state.buffer is None:
                state = state.replace(buffer=new_runner_state, buffer_size=self.popsize)
            else:
                new_buffer = jax.tree_map(
                    lambda x, y: jnp.concatenate((x, y), axis=0)[-self.max_size :],
                    state.buffer,
                    new_runner_state,
                )
                state = state.replace(
                    buffer=new_buffer,
                    buffer_size=jnp.minimum(
                        state.buffer_size + self.popsize, self.max_size
                    ),
                )
        # return jax.lax.select(
        #     self.reset_every is not None and self.reset_every % state.call_count == 0,
        #     PrevParamsState(buffer=None, best_runner_state=None),
        #     state,
        # )
        return state

    def get_prev_params(self, state, key):
        if self.train_restart == TrainRestart.RESTART_BEST:
            if state.best_runner_state is None:
                return None
            res = jax.tree_map(
                lambda x: jnp.tile(x, reps=(self.popsize,)).reshape(
                    self.popsize, *x.shape
                ),
                state.best_runner_state,
            )
            return res
        elif (
            self.train_restart == TrainRestart.SAMPLE_INIT
            or self.train_restart == TrainRestart.SAMPLE_RECENT_INIT
        ):
            chosen_idx = jax.random.randint(
                key, shape=(self.popsize // 2,), minval=0, maxval=state.buffer_size
            )
            chosen_idx = jnp.concatenate((chosen_idx, chosen_idx), axis=0)
            return jax.tree_map(lambda x: x[chosen_idx], state.buffer)
        elif self.train_restart == TrainRestart.NONE:
            return None

from functools import partial
from typing import Tuple, Any, Callable
import jax
import jax.numpy as jnp
from flax import linen as flax_nn
import optax
import numpy as np
from flax.training.train_state import TrainState
from flax.core.frozen_dict import FrozenDict
from flax.linen.initializers import constant, orthogonal


class Discriminator(flax_nn.Module):
    reward_net_hsize: jnp.ndarray
    learning_rate: float
    discr_updates: int
    n_features: int  # usually OBS_SIZE + ACTION SIZE
    transition_steps_decay: int
    discr_final_lr: float
    buffer: Any
    expert_data: jnp.ndarray
    batch_size: int
    l2_loss: float = 0.0
    schedule_type: str = "linear"
    discr_loss: str = "bce"
    grad_penalty_coeff: float = 10.0

    def setup(self):
        self.g_layers = [
            flax_nn.Dense(
                layer_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
            )
            for layer_size in self.reward_net_hsize
        ] + [
            flax_nn.Dense(
                1, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
            )
        ]

    def __call__(self, x) -> jnp.ndarray:
        for i, layer in enumerate(self.g_layers):
            x = layer(x)
            if i != len(self.g_layers) - 1:
                x = flax_nn.relu(x)
            else:
                x = -flax_nn.sigmoid(x)
        return x

    def scheduler(self, step_number):
        lr = jax.lax.select(
            step_number == 0,
            self.learning_rate,
            self.learning_rate / jnp.maximum((step_number // self.discr_updates), 1),
        )
        return lr

    def linear_schedule(self, count):
        frac = 1.0 - (count // self.discr_updates) / self.transition_steps_decay
        lr = (self.learning_rate - self.discr_final_lr) * jnp.maximum(frac, 0)
        lr = jnp.maximum(lr, self.discr_final_lr)
        return lr

    def _init_state(
        self,
        rng: Any,
    ) -> Tuple[TrainState, Any]:
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(self.n_features)
        if self.schedule_type == "linear":
            schedule = optax.linear_schedule(
                init_value=self.learning_rate,
                end_value=self.discr_final_lr,
                transition_steps=self.transition_steps_decay * self.discr_updates,
            )
        elif self.schedule_type == "constant":
            schedule = self.learning_rate
        elif self.schedule_type == "harmonic":
            schedule = self.scheduler
        else:
            raise ValueError(f"Schedule type {self.schedule_type} not recognized")

        train_state = TrainState.create(
            apply_fn=self.apply,
            params=self.init(_rng, init_x),
            tx=optax.adamw(learning_rate=schedule, weight_decay=self.l2_loss, eps=1e-5),
        )
        return train_state, rng

    def batch_loss(
        self,
        params: FrozenDict,
        expert_batch: jnp.ndarray,
        imitation_batch: jnp.ndarray,
        key: Any,
    ) -> jnp.ndarray:
        def loss_exp_bce(
            expert_transition: jnp.ndarray,
        ) -> jnp.ndarray:
            exp_d = self.apply(params, expert_transition)
            # exp_d = optax.sigmoid_binary_cross_entropy(exp_d, 1.0)
            return exp_d

        def loss_imit_bce(imitation_transition: jnp.ndarray) -> jnp.ndarray:
            imit_d = self.apply(params, imitation_transition)
            # imit_d = optax.sigmoid_binary_cross_entropy(imit_d, 0.0)
            return imit_d

        def loss_exp_mse(
            expert_transition: jnp.ndarray,
        ) -> jnp.ndarray:
            exp_d = self.apply(params, expert_transition)
            return optax.l2_loss(exp_d, jnp.array([0.0]))

        def loss_imit_mse(imitation_transition: jnp.ndarray) -> jnp.ndarray:
            imit_d = self.apply(params, imitation_transition)
            return -optax.l2_loss(imit_d, jnp.array([1.0]))

        # @partial(jax.grad, argnums=1)
        def apply_scalar(
            params: FrozenDict,
            input: jnp.ndarray,
        ):
            return self.apply(params, input)[0]

        def interpolate(
            alpha: float,
            expert_batch: jnp.ndarray,
            imitation_batch: jnp.ndarray,
        ):
            return alpha * expert_batch + (1 - alpha) * imitation_batch

        if self.discr_loss == "mse":
            loss_expert = loss_exp_mse
            loss_imitation = loss_imit_mse
        elif self.discr_loss == "bce":
            loss_expert = loss_exp_bce
            loss_imitation = loss_imit_bce

        exp_loss = jnp.mean(jax.vmap(loss_expert)(expert_batch), axis=0)[0]
        imit_loss = jnp.mean(jax.vmap(loss_imitation)(imitation_batch), axis=0)[0]
        alpha = jax.random.uniform(key, (imitation_batch.shape[0],))

        interpolated = jax.vmap(interpolate)(alpha, expert_batch, imitation_batch)
        gradients = jax.vmap(jax.grad(fun=apply_scalar, argnums=1), (None, 0))(
            params, interpolated
        )
        gradients = gradients.reshape((expert_batch.shape[0], -1))
        gradients_norm = jnp.sqrt(jnp.sum(gradients**2, axis=1) + 1e-12)
        grad_penalty = ((gradients_norm - 0.4) ** 2).mean()

        # here we use 10 as a fixed parameter as a cost of the penalty.
        loss = exp_loss - imit_loss + self.grad_penalty_coeff * grad_penalty

        return loss, {
            "exp_loss": exp_loss,
            "imit_loss": imit_loss,
        }

    def update_step(
        self,
        loss_grad_fn: Callable[
            [FrozenDict, jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, FrozenDict]
        ],
        buffer_state: jnp.ndarray,
        norm_info: Any,
        carry: Tuple[TrainState, Any],
        unused: Any,
    ) -> Tuple[TrainState, jnp.ndarray]:
        train_state, key = carry
        key_exp_sel, key_imit_sel, train_key, key = jax.random.split(key, 4)
        expert_batch = jax.random.choice(
            key_exp_sel,
            self.expert_data,
            shape=(self.batch_size,),
            replace=False,
        )
        imitation_batch = self.buffer.sample(
            self.batch_size,
            key_imit_sel,
            buffer_state,
        )
        recent_imitation_batch = self.buffer.sample_recent(
            self.batch_size,
            key_imit_sel,
            buffer_state,
        )
        norm_imit_batch = (imitation_batch - norm_info[0]) / jnp.sqrt(
            norm_info[1] + 1e-8
        )
        norm_expert_batch = (expert_batch - norm_info[0]) / jnp.sqrt(
            norm_info[1] + 1e-8
        )
        norm_recent_imit_batch = (recent_imitation_batch - norm_info[0]) / jnp.sqrt(
            norm_info[1] + 1e-8
        )
        (loss_val, loss_info), grads = loss_grad_fn(
            train_state.params,
            jax.lax.stop_gradient(norm_expert_batch),
            jax.lax.stop_gradient(norm_imit_batch),
            train_key,
        )

        loss_info["loss"] = loss_val

        train_state = train_state.apply_gradients(grads=grads)

        recent_imit_pred = self.apply(train_state.params, norm_recent_imit_batch).mean()
        loss_info["recent_imit_loss"] = recent_imit_pred

        return (train_state, key), loss_info

    def train_epoch(
        self,
        imit_data_buffer_state: Any,
        norm_info: Any,
        train_state: TrainState,
        key: Any,
    ) -> Tuple[TrainState, jnp.ndarray]:
        loss_grad_fn = jax.value_and_grad(self.batch_loss, argnums=0, has_aux=True)
        _update_step = partial(
            self.update_step,
            loss_grad_fn,
            jax.lax.stop_gradient(imit_data_buffer_state),
            norm_info,
        )
        train_state, losses_info = jax.lax.scan(
            _update_step, (train_state, key), xs=None, length=self.discr_updates
        )
        return train_state[0], losses_info

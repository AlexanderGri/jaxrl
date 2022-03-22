"""Implementations of algorithms for continuous control."""

import functools
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxrl.agents.pg.actor import update as update_actor
from jaxrl.agents.pg.critic import update as update_critic
from jaxrl.datasets import PaddedTrajectoryData
from jaxrl.networks import critic_net, policies
from jaxrl.networks.common import InfoDict, Model, PRNGKey


@functools.partial(jax.jit,
                   static_argnames=('length',))
def _update_jit(
    rng: PRNGKey, actor: Model, critic: Model, data: PaddedTrajectoryData,
        discount: float, entropy_coef: float, length: int) -> Tuple[PRNGKey, Model, Model, InfoDict]:

    rng, key = jax.random.split(rng)
    new_critic, critic_info = update_critic(critic,
                                            data,
                                            discount,
                                            length)
    rng, key = jax.random.split(rng)
    new_actor, actor_info = update_actor(actor, new_critic, data, discount, entropy_coef)

    return rng, new_actor, new_critic, {
        **critic_info,
        **actor_info,
    }


class PGLearner(object):
    def __init__(self,
                 seed: int,
                 states: jnp.ndarray,
                 observations: jnp.ndarray,
                 available_actions: jnp.ndarray,
                 n_actions: int,
                 length: int,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 hidden_dims: Sequence[int] = (256, 256),
                 discount: float = 0.99,
                 entropy_coef: float = 1e-3):

        self.discount = discount
        self.entropy_coef = entropy_coef
        self.length = length

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key = jax.random.split(rng, 3)
        actor_def = policies.ConstrainedCategoricalPolicy(
            hidden_dims,
            n_actions,)
        actor = Model.create(actor_def,
                             inputs=[actor_key, observations, available_actions],
                             tx=optax.adam(learning_rate=actor_lr))

        critic_def = critic_net.StateValueCritic(hidden_dims)
        critic = Model.create(critic_def,
                              inputs=[critic_key, states],
                              tx=optax.adam(learning_rate=critic_lr))

        self.actor = actor
        self.critic = critic
        self.rng = rng

        self.step = 1

    def sample_actions(self,
                       observations: np.ndarray,
                       available_actions: np.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        rng, actions = policies.sample_constrained_actions(self.rng, self.actor.apply_fn,
                                                           self.actor.params, observations,
                                                           available_actions, temperature)
        self.rng = rng

        actions = np.asarray(actions)
        return actions

    def update(self, data: PaddedTrajectoryData) -> InfoDict:
        self.step += 1

        new_rng, new_actor, new_critic, info = _update_jit(
            self.rng, self.actor, self.critic, data, self.discount, self.entropy_coef, self.length,)

        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic

        return info

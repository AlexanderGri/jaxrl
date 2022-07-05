"""Implementations of algorithms for continuous control."""

import functools
from typing import Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxrl.agents.pg.actor import update as update_actor
from jaxrl.agents.pg.actor import update_recurrent as update_actor_recurrent
from jaxrl.agents.pg.critic import update as update_critic
from jaxrl.datasets import PaddedTrajectoryData
from jaxrl.networks import critic_net, policies
from jaxrl.networks.common import InfoDict, Model, PRNGKey, GRU


@functools.partial(jax.jit,
                   static_argnames=('length',))
def _update_jit_recurrent(
    rng: PRNGKey, actor: Model, critic: Model, data: PaddedTrajectoryData, init_carry: jnp.ndarray,
        discount: float, entropy_coef: float, length: int) -> Tuple[PRNGKey, Model, Model, InfoDict]:

    rng, key = jax.random.split(rng)
    new_critic, critic_info = update_critic(critic,
                                            data,
                                            discount,
                                            length)
    rng, key = jax.random.split(rng)
    new_actor, actor_info = update_actor_recurrent(actor, new_critic, data, init_carry, discount, entropy_coef)

    return rng, new_actor, new_critic, {
        **critic_info,
        **actor_info,
    }


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
                 critic_hidden_dims: Sequence[int] = (128, 128),
                 actor_hidden_dims: Sequence[int] = (64,),
                 actor_recurrent_hidden_dim: int = 64,
                 use_recurrent_policy: bool = True,
                 use_shared_policy: bool = True,
                 use_mc_return: bool = False,
                 discount: float = 0.99,
                 entropy_coef: float = 1e-3):

        self.discount = discount
        self.entropy_coef = entropy_coef
        self.length = length
        self.use_recurrent_policy = use_recurrent_policy
        self.actor_recurrent_hidden_dim = actor_recurrent_hidden_dim
        if use_mc_return:
            raise NotImplementedError

        _, _, n_agents, _ = observations.shape

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key = jax.random.split(rng, 3)
        if self.use_recurrent_policy:
            actor_def = policies.RecurrentConstrainedCategoricalPolicy(
                hidden_dims=actor_hidden_dims,
                recurrent_hidden_dim=actor_recurrent_hidden_dim,
                n_actions=n_actions,
                shared=use_shared_policy)
            carry = self.initialize_carry(1, n_agents)
            inputs = [actor_key, carry, observations, available_actions]
        else:
            actor_def = policies.ConstrainedCategoricalPolicy(
                hidden_dims=actor_hidden_dims,
                n_actions=n_actions,
                shared=use_shared_policy)
            inputs = [actor_key, observations, available_actions]

        actor = Model.create(actor_def,
                             inputs=inputs,
                             tx=optax.adam(learning_rate=actor_lr))

        critic_def = critic_net.StateValueCritic(critic_hidden_dims)
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
                       carry: Optional[jnp.ndarray] = None,
                       temperature: float = 1.0,
                       distribution: str = 'log_prob') -> \
            Union[Tuple[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        outputs = policies.sample_constrained_actions(self.rng, self.actor.apply_fn,
                                                      self.actor.params, observations,
                                                      available_actions, carry,
                                                      temperature, distribution)
        if carry is None:
            rng, actions, log_prob = outputs
            self.rng = rng
            actions = np.asarray(actions)
            return actions, log_prob
        else:
            rng, new_carry, actions, log_prob = outputs
            self.rng = rng
            actions = np.asarray(actions)
            return new_carry, actions, log_prob

    def update(self, data: PaddedTrajectoryData) -> InfoDict:
        self.step += 1

        if self.use_recurrent_policy:
            n_trajectories, _, n_agents = data.actions.shape
            init_carry = self.initialize_carry(n_trajectories, n_agents)
            new_rng, new_actor, new_critic, info = _update_jit_recurrent(
                self.rng, self.actor, self.critic, data, init_carry, self.discount, self.entropy_coef, self.length,)
        else:
            new_rng, new_actor, new_critic, info = _update_jit(
                self.rng, self.actor, self.critic, data, self.discount, self.entropy_coef, self.length, )

        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic

        return info

    def initialize_carry(self, n_trajectories, n_agents):
        return GRU.initialize_carry((n_trajectories, n_agents), self.actor_recurrent_hidden_dim)

    def save(self, path):
        self.actor.save(f'{path}_actor')
        self.critic.save(f'{path}_critic')

    def load(self, path):
        self.actor = self.actor.load(f'{path}_actor')
        self.critic = self.critic.load(f'{path}_critic')

"""Implementations of algorithms for continuous control."""

import functools
from typing import Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxrl.agents.meta_reward_pg.actor import update_intrinsic as update_intrinsic_actor
from jaxrl.agents.meta_reward_pg.critic import update_intrinsic as update_intrinsic_critics
from jaxrl.agents.meta_reward_pg.critic import update_extrinsic as update_extrinsic_critic
from jaxrl.agents.meta_reward_pg.reward import update as update_reward
from jaxrl.datasets import PaddedTrajectoryData
from jaxrl.networks import critic_net, policies
from jaxrl.networks.common import InfoDict, Model, PRNGKey, GRU


@functools.partial(jax.jit, static_argnames=('use_recurrent_policy',))
def _update_actor_jit(rng: PRNGKey, actor: Model, intrinsic_critics: Model,
                data: PaddedTrajectoryData, discount: float, entropy_coef: float,  mix_coef: float,
                use_recurrent_policy: bool, init_carry: Optional[jnp.ndarray] = None) \
        -> Tuple[PRNGKey, Model, InfoDict]:
    rng, key = jax.random.split(rng)
    new_actor, actor_info = update_intrinsic_actor(actor,
                                                   intrinsic_critics,
                                                   intrinsic_critics.params,
                                                   data, discount, entropy_coef,
                                                   mix_coef, use_recurrent_policy, init_carry)

    return rng, new_actor, actor_info


@functools.partial(jax.jit, static_argnames=('length', 'use_recurrent_policy'))
def _update_except_actor_jit(rng: PRNGKey, prev_actor: Model, intrinsic_critics: Model,
                             extrinsic_critic: Model, prev_data: PaddedTrajectoryData,
                             data: PaddedTrajectoryData, discount: float, entropy_coef: float,  mix_coef: float,
                             length: int, use_recurrent_policy: bool, init_carry: Optional[jnp.ndarray] = None) \
        -> Tuple[PRNGKey, Model, Model, InfoDict]:
    rng, key = jax.random.split(rng)
    new_extrinsic_critic, extrinsic_critic_info = update_extrinsic_critic(extrinsic_critic,
                                                                          data,
                                                                          discount,
                                                                          length)
    rng, key = jax.random.split(rng)
    new_intrinsic_critics, reward_info = update_reward(prev_actor,
                                                       intrinsic_critics,
                                                       new_extrinsic_critic,
                                                       prev_data,
                                                       data,
                                                       discount,
                                                       entropy_coef,
                                                       mix_coef,
                                                       use_recurrent_policy,
                                                       init_carry)
    rng, key = jax.random.split(rng)
    new_new_intrinsic_critics, intrinsic_critics_info = update_intrinsic_critics(new_intrinsic_critics,
                                                                                 data,
                                                                                 discount,
                                                                                 length,
                                                                                 mix_coef)

    info = {}
    for d, name in zip([extrinsic_critic_info, reward_info, intrinsic_critics_info],
                       ['extrinsic', 'outer', 'intrinsic']):
        info.update({f'{name}_{k}': v for k, v in d.items()})
    return rng, new_extrinsic_critic, new_new_intrinsic_critics, info


class MetaPGLearner(object):
    def __init__(self,
                 seed: int,
                 states: jnp.ndarray,
                 observations: jnp.ndarray,
                 available_actions: jnp.ndarray,
                 n_actions: int,
                 length: int,
                 n_agents: int,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 critic_hidden_dims: Sequence[int] = (128, 128),
                 actor_hidden_dims: Sequence[int] = (64,),
                 actor_recurrent_hidden_dim: int = 64,
                 use_recurrent_policy: bool = True,
                 use_shared_reward: bool = False,
                 discount: float = 0.99,
                 entropy_coef: float = 1e-3,
                 mix_coef: float = 0.01,):

        self.discount = discount
        self.entropy_coef = entropy_coef
        self.mix_coef = mix_coef
        self.length = length
        self.use_recurrent_policy = use_recurrent_policy
        self.actor_recurrent_hidden_dim = actor_recurrent_hidden_dim

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, extrinsic_critic_key, intrinsic_critics_key = jax.random.split(rng, 4)
        if self.use_recurrent_policy:
            actor_def = policies.RecurrentConstrainedCategoricalPolicy(
                hidden_dims=actor_hidden_dims,
                recurrent_hidden_dim=actor_recurrent_hidden_dim,
                n_actions=n_actions, )
            carry = self.initialize_carry(1, 1)
            inputs = [actor_key, carry, observations, available_actions]
        else:
            actor_def = policies.ConstrainedCategoricalPolicy(
                hidden_dims=actor_hidden_dims,
                n_actions=n_actions,)
            inputs = [actor_key, observations, available_actions]

        actor = Model.create(actor_def,
                             inputs=inputs,
                             tx=optax.adam(learning_rate=actor_lr, eps_root=1e-8))

        extrinsic_critic_def = critic_net.StateValueCritic(critic_hidden_dims)
        extrinsic_critic = Model.create(extrinsic_critic_def,
                                        inputs=[extrinsic_critic_key, states],
                                        tx=optax.adam(learning_rate=critic_lr))
        intrinsic_critic_def = critic_net.RewardAndCritics(critic_hidden_dims, n_agents, n_actions,
                                                           use_shared_reward)
        intrinsic_critics = Model.create(intrinsic_critic_def,
                                         inputs=[intrinsic_critics_key, states],
                                         tx=optax.adam(learning_rate=critic_lr))

        self.prev_data = None
        self.prev_actor = None
        self.actor = actor
        self.extrinsic_critic = extrinsic_critic
        self.intrinsic_critics = intrinsic_critics
        self.rng = rng

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

    def update_actor(self, data: PaddedTrajectoryData) -> InfoDict:
        n_trajectories, _, n_agents = data.actions.shape
        init_carry = self.initialize_carry(n_trajectories, n_agents)

        new_rng, new_actor, actor_info = _update_actor_jit(
            self.rng, self.actor, self.intrinsic_critics, data,
            self.discount, self.entropy_coef, self.mix_coef,
            self.use_recurrent_policy, init_carry)

        self.rng = new_rng
        self.actor = new_actor

        return {f'inner_{k}': v for k, v in actor_info.items()}

    def update_except_actor(self, prev_data: PaddedTrajectoryData,
                            data: PaddedTrajectoryData, prev_actor: Model) -> InfoDict:
        if prev_data is None or prev_actor is None:
            return {}
        n_trajectories, _, n_agents = data.actions.shape
        init_carry = self.initialize_carry(n_trajectories, n_agents)

        new_rng, new_extrinsic_critic, new_intrinsic_critics, info = _update_except_actor_jit(
            self.rng, prev_actor, self.intrinsic_critics, self.extrinsic_critic,
            prev_data, data, self.discount, self.entropy_coef, self.mix_coef, self.length,
            self.use_recurrent_policy, init_carry)

        self.rng = new_rng
        self.extrinsic_critic = new_extrinsic_critic
        self.intrinsic_critics = new_intrinsic_critics
        return info

    def initialize_carry(self, n_trajectories, n_agents):
        if self.use_recurrent_policy:
            return GRU.initialize_carry((n_trajectories, n_agents), self.actor_recurrent_hidden_dim)
        else:
            return None

    def save(self, path):
        self.actor.save(f'{path}_actor')
        self.intrinsic_critics.save(f'{path}_intrinsic_critics')
        self.extrinsic_critic.save(f'{path}_extrinsic_critic')

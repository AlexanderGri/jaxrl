import functools
from typing import Tuple, Optional

import jax.numpy as jnp
import jax

from jaxrl.datasets import PaddedTrajectoryData
from jaxrl.networks.common import InfoDict, Model, Params
from jaxrl.networks.critic_net import RewardAndCritics


@functools.partial(jax.jit, static_argnames=('use_recurrent_policy', 'use_importance_sampling',))
def get_actor_loss(actor: Model, actor_params, values: jnp.ndarray, next_values: jnp.ndarray,
                   data: PaddedTrajectoryData, rewards: jnp.ndarray, discount: float,
                   entropy_coef: float, use_recurrent_policy: bool, use_importance_sampling: bool = False,
                   init_carry: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, InfoDict]:
    advantages = rewards + discount * next_values - values

    if use_recurrent_policy:
        _, dist = actor.apply_fn({'params': actor_params},
                                 init_carry, data.observations, data.available_actions, )
    else:
        dist = actor.apply_fn({'params': actor_params},
                              data.observations, data.available_actions)
    log_probs = dist.log_prob(data.actions)
    if use_importance_sampling:
        old_log_probs = data.log_prob
        surrogate = advantages * jnp.exp(log_probs - old_log_probs)
    else:
        surrogate = advantages * log_probs
    agent_alive_normalized = data.agent_alive / data.agent_alive.sum()
    reward_loss = -(surrogate * agent_alive_normalized).sum()
    entropy = -(log_probs * agent_alive_normalized).sum()
    actor_loss = reward_loss - entropy_coef * entropy
    return actor_loss, {
        'reward_loss': reward_loss,
        'actor_loss': actor_loss,
        'entropy': entropy
    }


def update_extrinsic(actor: Model, extrinsic_critic: Model, data: PaddedTrajectoryData,
                     discount: float, entropy_coef: float,
                     use_recurrent_policy: bool, init_carry: Optional[jnp.ndarray] = None) -> Tuple[Model, InfoDict]:
    values = jnp.expand_dims(extrinsic_critic(data.states), axis=2)
    next_values = jnp.expand_dims(extrinsic_critic(data.next_states), axis=2)
    rewards = jnp.expand_dims(data.rewards, axis=2)
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        return get_actor_loss(actor, actor_params, values, next_values,
                              data, rewards, discount, entropy_coef,
                              use_recurrent_policy=use_recurrent_policy, init_carry=init_carry)

    new_actor, info = actor.apply_gradient(actor_loss_fn)
    return new_actor, info


def get_statistics(arr: jnp.array):
    info = {}
    for fun in [jnp.min, jnp.mean, jnp.max]:
        info[fun.__name__] = fun(arr)
    return info


def update_intrinsic(actor: Model, intrinsic_critics: Model, intrinsic_critics_params: Params,
                     data: PaddedTrajectoryData, discount: float, entropy_coef: float, mix_coef: float,
                     use_recurrent_policy: bool, init_carry: Optional[jnp.ndarray] = None) -> Tuple[Model, InfoDict]:
    all_meta_rewards = intrinsic_critics.apply_fn({'params': intrinsic_critics_params}, data.states,
                                                  method=RewardAndCritics.get_rewards)
    indices = (*jnp.indices(data.actions.shape), data.actions)
    meta_rewards = all_meta_rewards[indices]
    mixed_rewards = jnp.expand_dims(data.rewards, axis=2) + mix_coef * meta_rewards
    values = intrinsic_critics(data.states, method=RewardAndCritics.get_values)
    next_values = intrinsic_critics(data.next_states, method=RewardAndCritics.get_values)

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        return get_actor_loss(actor, actor_params, values, next_values,
                              data, mixed_rewards, discount, entropy_coef,
                              use_recurrent_policy=use_recurrent_policy, init_carry=init_carry)

    new_actor, info = actor.apply_gradient(actor_loss_fn)
    for arr, reward_type in zip([meta_rewards, data.rewards, mixed_rewards],
                         ['meta', 'env', 'mix']):
        stats = get_statistics(arr)
        info.update({f'{stat_name}_{reward_type}_reward': stat_value for stat_name, stat_value in stats.items()})
    return new_actor, info

import functools
from typing import Tuple, Optional

import jax.numpy as jnp
import jax

from jaxrl.datasets import PaddedTrajectoryData
from jaxrl.networks.common import InfoDict, Model, Params
from jaxrl.networks.critic_net import RewardAndCriticsModel
from jaxrl.agents.meta_reward_pg.critic import compute_returns, compute_returns_multiagent


@functools.partial(jax.jit, static_argnames=('use_recurrent_policy', 'use_importance_sampling',))
def get_actor_loss(actor: Model, advantages: jnp.ndarray, data: PaddedTrajectoryData,
                   entropy_coef: float, use_recurrent_policy: bool, use_importance_sampling: bool = False,
                   init_carry: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, InfoDict]:

    if use_recurrent_policy:
        _, dist = actor(init_carry, data.observations, data.available_actions,)
    else:
        dist = actor(data.observations, data.available_actions)
    log_probs = dist.log_prob(data.actions)
    if use_importance_sampling:
        old_log_probs = data.log_prob
        surrogate = advantages * jnp.exp(log_probs - old_log_probs)
    else:
        surrogate = advantages * log_probs
    agent_alive_normalized = data.agents_alive / data.agents_alive.sum()
    reward_loss = -(surrogate * agent_alive_normalized).sum()
    entropy = (dist.entropy() * agent_alive_normalized).sum()
    actor_loss = reward_loss - entropy_coef * entropy
    return actor_loss, {
        'reward_loss': reward_loss,
        'actor_loss': actor_loss,
        'entropy': entropy
    }


def compute_advantage(rewards: jnp.ndarray, dones: jnp.ndarray, values: jnp.ndarray, next_values: jnp.ndarray,
                      discount: float, time_limit: int, use_mc_return: bool = False) -> jnp.ndarray:
    if use_mc_return:
        last_values = next_values[:, -1]
        returns = compute_returns(rewards, dones, last_values, discount, time_limit)
        advantages = returns - values
    else:
        advantages = rewards + discount * next_values - values
    return advantages


def compute_advantage_multiagent(rewards_multiagent: jnp.ndarray, dones: jnp.ndarray, values_multiagent: jnp.ndarray,
                               next_values_multiagent: jnp.ndarray, discount: float, time_limit: int,
                               use_mc_return: bool = False) -> jnp.ndarray:
    if use_mc_return:
        last_values_multiagent = next_values_multiagent[:, -1]
        returns_multiagent = compute_returns_multiagent(rewards_multiagent, dones, last_values_multiagent, discount, time_limit)
        advantages = returns_multiagent - values_multiagent
    else:
        advantages = rewards_multiagent + discount * next_values_multiagent - values_multiagent
    return advantages


def get_statistics(arr: jnp.array):
    info = {}
    for fun in [jnp.min, jnp.mean, jnp.max]:
        info[fun.__name__] = fun(arr)
    return info


def update(actor: Model, intrinsic: RewardAndCriticsModel,
           data: PaddedTrajectoryData, discount: float, entropy_coef: float, mix_coef: float,
           time_limit: int, use_recurrent_policy: bool, init_carry: Optional[jnp.ndarray] = None,
           use_mc_return: bool = False) -> Tuple[Model, InfoDict]:
    all_meta_rewards = intrinsic.get_rewards(data.states)
    indices = (*jnp.indices(data.actions.shape), data.actions)
    meta_rewards = all_meta_rewards[indices]
    mixed_rewards = jnp.expand_dims(data.rewards, axis=2) + mix_coef * meta_rewards
    values = intrinsic.get_values(data.states)
    next_values = intrinsic.get_values(data.next_states)
    advantages = compute_advantage(mixed_rewards, data.is_ended, values, next_values, discount, time_limit, use_mc_return)

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        return get_actor_loss(actor.replace(params=actor_params), advantages,
                              data, entropy_coef=entropy_coef,
                              use_recurrent_policy=use_recurrent_policy, init_carry=init_carry)

    new_actor, info = actor.apply_gradient(actor_loss_fn)
    for arr, reward_type in zip([meta_rewards, data.rewards, mixed_rewards],
                         ['meta', 'env', 'mix']):
        stats = get_statistics(arr)
        info.update({f'{stat_name}_{reward_type}_reward': stat_value for stat_name, stat_value in stats.items()})
    return new_actor, info

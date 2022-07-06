import functools
from typing import Tuple, List

import jax.numpy as jnp
import jax

from jaxrl.datasets import PaddedTrajectoryData
from jaxrl.networks.common import InfoDict, Model, Params
from jaxrl.networks.critic_net import StateValueCritic, RewardAndCritics


def _compute_returns(rewards: jnp.ndarray,
                     done: bool,
                     last_value: float,
                     discount: float,
                     length: int) -> jnp.ndarray:
    # +1 because of last observation value
    coef = discount ** jnp.arange(length + 1)[::-1]
    # if done last_value is not needed
    rewards_with_last_value = jnp.append(rewards, last_value * ~done)
    discounted_returns = jnp.convolve(rewards_with_last_value, coef, mode='full')[length:-1]
    return discounted_returns


# |n_traj| x n_steps, |n_traj|, |n_traj|
compute_returns = jax.vmap(_compute_returns, in_axes=(0, 0, 0, None, None), out_axes=0)


# n_traj x n_steps x |n_agents|, n_traj, n_traj x |n_agents|
compute_returns_multiagent = jax.vmap(compute_returns, in_axes=(2, None, 1, None, None), out_axes=2)


def update_extrinsic(extrinsic_critic: Model, data: PaddedTrajectoryData,
                     discount: float, length: int) -> Tuple[Model, InfoDict]:
    last_values = extrinsic_critic(data.next_states[:, -1])
    target_values = compute_returns(data.rewards,
                                    data.dones,
                                    last_values,
                                    discount,
                                    length)

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        values = extrinsic_critic.apply_fn({'params': critic_params}, data.states)
        critic_loss = (values - target_values)**2
        all_agents_alive_normalized = data.all_agents_alive / data.all_agents_alive.sum()
        critic_loss = (critic_loss * all_agents_alive_normalized).sum()
        return critic_loss, {
            'critic_loss': critic_loss,
            'v': (values * all_agents_alive_normalized).sum(),
        }

    new_critic, info = extrinsic_critic.apply_gradient(critic_loss_fn)

    return new_critic, info


def get_grad_intrinsic(intrinsic_critics: Model, data: PaddedTrajectoryData,
                       discount: float, length: int, mix_coef: float) -> Tuple[Model, InfoDict]:
    last_values = intrinsic_critics(data.next_states[:, -1], method=RewardAndCritics.get_values)
    all_meta_rewards = intrinsic_critics(data.states, method=RewardAndCritics.get_rewards)
    indices = (*jnp.indices(data.actions.shape), data.actions)
    meta_rewards = all_meta_rewards[indices]
    mixed_rewards = jnp.expand_dims(data.rewards, axis=2) + mix_coef * meta_rewards

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        values_multiagent = intrinsic_critics.apply_fn({'params': critic_params}, data.states,
                                                       method=RewardAndCritics.get_values)
        returns_multiagent = compute_returns_multiagent(mixed_rewards,
                                                        data.dones,
                                                        last_values,
                                                        discount,
                                                        length)
        critic_loss = (values_multiagent - returns_multiagent)**2
        # here several ways of normalizing are possible
        agents_alive_normalized = data.agent_alive / data.agent_alive.sum()
        critic_loss = (critic_loss * agents_alive_normalized).sum()
        n_valid_states_per_agent = data.agent_alive.sum(0, keepdims=True).sum(1, keepdims=True)
        mean_values_per_agent = (values_multiagent * data.agent_alive / n_valid_states_per_agent).sum(1).sum(0)
        return critic_loss, {
            'critic_loss': critic_loss,
            **{f'v{i}': mean_v for i, mean_v in enumerate(mean_values_per_agent)}
        }

    (_, info), grad = jax.value_and_grad(critic_loss_fn, has_aux=True)(intrinsic_critics.params)

    return grad, info

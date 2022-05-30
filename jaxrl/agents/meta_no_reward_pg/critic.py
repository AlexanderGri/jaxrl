import functools
from typing import Tuple, List

import jax.numpy as jnp
import jax

from jaxrl.datasets import PaddedTrajectoryData
from jaxrl.networks.common import InfoDict, Model, Params


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


compute_returns = jax.vmap(_compute_returns, in_axes=(0, 0, 0, None, None))
compute_returns_agents = jax.vmap(compute_returns, in_axes=(2, None, 1, None, None), out_axes=2)


def update(critic: Model, data: PaddedTrajectoryData,
           discount: float, length: int) -> Tuple[Model, InfoDict]:
    last_values = critic(data.next_states[:, -1])
    target_values = compute_returns(data.rewards,
                                    data.dones,
                                    last_values,
                                    discount,
                                    length)

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        values = critic.apply_fn({'params': critic_params}, data.states)
        critic_loss = (values - target_values)**2
        all_agents_alive_normalized = data.all_agents_alive / data.all_agents_alive.sum()
        critic_loss = (critic_loss * all_agents_alive_normalized).sum()
        return critic_loss, {
            'critic_loss': critic_loss,
            'v': (values * all_agents_alive_normalized).sum(),
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info

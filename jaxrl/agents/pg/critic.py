import functools
from typing import Tuple

import jax.numpy as jnp
import jax

from jaxrl.datasets import PaddedTrajectoryData
from jaxrl.networks.common import InfoDict, Model, Params


# @jax.jit
# @functools.partial(jax.jit, static_argnums=(0,))
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


compute_returns = jax.jit(jax.vmap(_compute_returns, in_axes=(0, 0, 0, None, None)),
                          static_argnames=('length',))


def update(critic: Model, data: PaddedTrajectoryData, discount: float, length: int) -> Tuple[Model, InfoDict]:
    last_values = critic(data.next_states[:, -1])
    padded_target_v = compute_returns(data.rewards,
                                      data.dones,
                                      last_values,
                                      discount,
                                      length)
    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        padded_v = critic.apply_fn({'params': critic_params}, data.states)
        padded_critic_loss = (padded_v - padded_target_v)**2
        all_agents_alive_normalized = data.all_agents_alive / data.all_agents_alive.sum()
        critic_loss = (padded_critic_loss * all_agents_alive_normalized).sum()
        return critic_loss, {
            'critic_loss': critic_loss,
            'v': (padded_v * all_agents_alive_normalized).sum(),
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info

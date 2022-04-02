import functools
from typing import Tuple, Optional

import jax.numpy as jnp
import jax

from jaxrl.datasets import PaddedTrajectoryData
from jaxrl.networks.common import InfoDict, Model, Params
from jaxrl.networks.critic_net import RewardAndCritics, StateValueCritic
from jaxrl.agents.meta_reward_pg.actor import update_intrinsic as update_intrinsic_actor
from jaxrl.agents.meta_reward_pg.actor import get_actor_loss

# @functools.partial(jax.jit, static_argnames=('use_recurrent_policy',))
def update(prev_actor: Model, intrinsic_critics: Model, extrinsic_critic: Model,
           prev_data: PaddedTrajectoryData,
           data: PaddedTrajectoryData, discount: float, entropy_coef: float, mix_coef: float,
           use_recurrent_policy: bool, init_carry: Optional[jnp.ndarray] = None) -> Tuple[Model, InfoDict]:
    def reward_loss_fn(intrinsic_critics_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        actor, _ = update_intrinsic_actor(prev_actor, intrinsic_critics, intrinsic_critics_params,
                                          prev_data, discount, entropy_coef, mix_coef,
                                          use_recurrent_policy, init_carry)
        values = jnp.expand_dims(extrinsic_critic(data.states), axis=2)
        next_values = jnp.expand_dims(extrinsic_critic(data.next_states), axis=2)
        rewards = jnp.expand_dims(data.rewards, axis=2)
        return get_actor_loss(actor, actor.params, values, next_values,
                              data, rewards, discount, 0.,
                              use_recurrent_policy, init_carry)
    new_intrinsic_critics, info = intrinsic_critics.apply_gradient(reward_loss_fn)
    return new_intrinsic_critics, info

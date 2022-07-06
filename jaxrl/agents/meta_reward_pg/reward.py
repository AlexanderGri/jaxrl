import functools
from typing import Tuple, Optional

import jax.numpy as jnp
import jax

from jaxrl.datasets import PaddedTrajectoryData
from jaxrl.networks.common import InfoDict, Model, Params
from jaxrl.networks.critic_net import RewardAndCritics, StateValueCritic
from jaxrl.agents.meta_reward_pg.actor import update_intrinsic as update_intrinsic_actor
from jaxrl.agents.meta_reward_pg.actor import get_actor_loss, compute_advantage

# @functools.partial(jax.jit, static_argnames=('use_recurrent_policy',))
def get_grad(prev_actor: Model, intrinsic_critics: Model, extrinsic_critic: Model,
             prev_data: PaddedTrajectoryData,
             data: PaddedTrajectoryData, discount: float, entropy_coef: float, mix_coef: float,
             length, use_recurrent_policy: bool, sampling_scheme: str,
             init_carry: Optional[jnp.ndarray] = None, use_mc_return: bool = False) -> Tuple[Model, InfoDict]:
    if sampling_scheme == 'reuse':
        outer_update_data = data
        use_importance_sampling = False
    elif sampling_scheme == 'importance_sampling':
        outer_update_data = prev_data
        use_importance_sampling = True
    else:
        outer_update_data = None
        use_importance_sampling = None

    def reward_loss_fn(intrinsic_critics_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        actor, _ = update_intrinsic_actor(prev_actor, intrinsic_critics, intrinsic_critics_params,
                                          prev_data, discount, entropy_coef, mix_coef,
                                          length, use_recurrent_policy, init_carry)
        values = jnp.expand_dims(extrinsic_critic(data.states), axis=2)
        next_values = jnp.expand_dims(extrinsic_critic(data.next_states), axis=2)
        rewards = jnp.expand_dims(data.rewards, axis=2)
        advantages = compute_advantage(rewards, data.dones,  values, next_values, discount, use_mc_return)
        return get_actor_loss(actor, actor.params, advantages,
                              outer_update_data, entropy_coef=0.,
                              use_recurrent_policy=use_recurrent_policy,
                              use_importance_sampling=use_importance_sampling,
                              init_carry=init_carry,)

    (_, info), grad = jax.value_and_grad(reward_loss_fn, has_aux=True)(intrinsic_critics.params)
    return grad, info

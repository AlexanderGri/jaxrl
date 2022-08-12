from typing import Tuple, Optional

import jax.numpy as jnp

from jaxrl.data import PaddedTrajectoryData
from jaxrl.networks.common import InfoDict, Model, Params
from jaxrl.networks.critic_net import RewardAndCriticsModel
from jaxrl.algorithm.actor import update as update_actor
from jaxrl.algorithm.actor import get_actor_loss, compute_advantage


def update(prev_actor: Model, intrinsic: RewardAndCriticsModel, extrinsic_critic: Model,
           prev_data: PaddedTrajectoryData, data: PaddedTrajectoryData,
           discount: float, entropy_coef: float, mix_coef: float, time_limit: int,
           use_recurrent_policy: bool, sampling_scheme: str, init_carry: Optional[jnp.ndarray] = None,
           use_mc_return: bool = False) -> Tuple[RewardAndCriticsModel, InfoDict]:
    if sampling_scheme == 'reuse':
        outer_update_data = data
        use_importance_sampling = False
    elif sampling_scheme == 'importance_sampling':
        outer_update_data = prev_data
        use_importance_sampling = True
    else:
        outer_update_data = None
        use_importance_sampling = None

    def reward_loss_fn(params_reward: Params) -> Tuple[jnp.ndarray, InfoDict]:
        actor, _ = update_actor(actor=prev_actor, intrinsic=intrinsic.replace(params_reward=params_reward),
                                data=prev_data, discount=discount, entropy_coef=entropy_coef, mix_coef=mix_coef,
                                time_limit=time_limit, use_recurrent_policy=use_recurrent_policy, init_carry=init_carry,
                                use_mc_return=use_mc_return)
        values = jnp.expand_dims(extrinsic_critic(data.states), axis=2)
        next_values = jnp.expand_dims(extrinsic_critic(data.next_states), axis=2)
        rewards = jnp.expand_dims(data.rewards, axis=2)
        advantages = compute_advantage(rewards=rewards, dones=data.is_ended,  values=values, next_values=next_values,
                                       discount=discount, time_limit=time_limit, use_mc_return=use_mc_return)
        return get_actor_loss(actor=actor, advantages=advantages, data=outer_update_data,
                              entropy_coef=0., use_recurrent_policy=use_recurrent_policy,
                              use_importance_sampling=use_importance_sampling, init_carry=init_carry,)

    new_intrinsic, info = intrinsic.apply_gradient_reward(reward_loss_fn)
    return new_intrinsic, info

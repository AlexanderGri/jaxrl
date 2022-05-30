import functools
from typing import Tuple, Optional

import jax.numpy as jnp
import jax

from jaxrl.datasets import PaddedTrajectoryData
from jaxrl.networks.common import InfoDict, Model, Params
from jax.tree_util import tree_map


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
    entropy = (dist.entropy() * agent_alive_normalized).sum()
    actor_loss = reward_loss - entropy_coef * entropy
    return actor_loss, {
        'reward_loss': reward_loss,
        'actor_loss': actor_loss,
        'entropy': entropy
    }


def update_simple(actor: Model, actor_params: Params, critic: Model, data: PaddedTrajectoryData,
                  discount: float, entropy_coef: float, use_recurrent_policy: bool,
                  init_carry: Optional[jnp.ndarray] = None) -> Tuple[Model, InfoDict]:
    values = jnp.expand_dims(critic(data.states), axis=2)
    next_values = jnp.expand_dims(critic(data.next_states), axis=2)
    rewards = jnp.expand_dims(data.rewards, axis=2)

    def actor_loss_fn(inner_actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        loss, info = get_actor_loss(actor, inner_actor_params, values, next_values,
                                    data, rewards, discount, entropy_coef,
                                    use_recurrent_policy=use_recurrent_policy, init_carry=init_carry)
        return loss.mean(), info
    (_, info), grad = jax.value_and_grad(actor_loss_fn, has_aux=True)(actor_params)
    new_actor = actor.apply_gradient(grads=grad, has_aux=False)
    return new_actor, info


def update_complex(prev_actor: Model, actor: Model, prev_critic: Model, critic: Model,
                   prev_data: PaddedTrajectoryData, data: PaddedTrajectoryData,
                   discount: float, entropy_coef: float, use_recurrent_policy: bool,
                   init_carry: Optional[jnp.ndarray] = None) -> Tuple[Model, InfoDict]:
    values = jnp.expand_dims(critic(data.states), axis=2)
    next_values = jnp.expand_dims(critic(data.next_states), axis=2)
    rewards = jnp.expand_dims(data.rewards, axis=2)

    def prev_meta_loss_fn(inner_prev_actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        actor, _ = update_simple(prev_actor, inner_prev_actor_params, prev_critic, prev_data, discount,
                                 entropy_coef,
                                 use_recurrent_policy=use_recurrent_policy, init_carry=init_carry)
        loss, info = get_actor_loss(actor, actor.params, values, next_values,
                                    data, rewards, discount, entropy_coef,
                                    use_recurrent_policy=use_recurrent_policy, init_carry=init_carry)
        return loss.mean(), info
    (_, _), prev_meta_grad = jax.value_and_grad(prev_meta_loss_fn, has_aux=True)(prev_actor.params)
    def cur_loss_fn(inner_actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        loss, info = get_actor_loss(actor, inner_actor_params, values, next_values,
                                    data, rewards, discount, entropy_coef,
                                    use_recurrent_policy=use_recurrent_policy, init_carry=init_carry)
        return loss.mean(), info
    (_, info), cur_grad = jax.value_and_grad(cur_loss_fn, has_aux=True)(actor.params)

    full_grad = tree_map(lambda pt1, pt2: pt1 + pt2, prev_meta_grad, cur_grad)
    new_actor = actor.apply_gradient(grads=full_grad, has_aux=False)
    return new_actor, info

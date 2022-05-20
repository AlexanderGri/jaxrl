from typing import Tuple

import jax.numpy as jnp

from jaxrl.datasets import PaddedTrajectoryData
from jaxrl.networks.common import InfoDict, Model, Params


def update(actor: Model, critic: Model, data: PaddedTrajectoryData,
           discount: float, entropy_coef: float) -> Tuple[Model, InfoDict]:

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply_fn({'params': actor_params}, data.observations,
                              data.available_actions)
        padded_log_probs = dist.log_prob(data.actions)
        padded_next_values = critic(data.next_states)
        padded_values = critic(data.states)
        padded_advantages = data.rewards + discount * padded_next_values - padded_values
        padded_advantages_expanded = jnp.expand_dims(padded_advantages, axis=2)
        padded_surrogate = padded_advantages_expanded * padded_log_probs
        agent_alive_normalized = data.agent_alive / data.agent_alive.sum()
        reward_loss = -(padded_surrogate * agent_alive_normalized).sum()
        entropy = (dist.entropy() * agent_alive_normalized).sum()
        actor_loss = reward_loss - entropy_coef * entropy
        return actor_loss, {
            'reward_loss': reward_loss,
            'actor_loss': actor_loss,
            'entropy': entropy
        }

    new_actor, info = actor.apply_gradient(actor_loss_fn)

    return new_actor, info


def update_recurrent(actor: Model, critic: Model, data: PaddedTrajectoryData,
                     init_carry: jnp.ndarray, discount: float, entropy_coef: float) -> Tuple[Model, InfoDict]:

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        _, dist = actor.apply_fn({'params': actor_params}, init_carry,
                                 data.observations, data.available_actions)
        padded_log_probs = dist.log_prob(data.actions)
        padded_next_values = critic(data.next_states)
        padded_values = critic(data.states)
        padded_advantages = data.rewards + discount * padded_next_values - padded_values
        padded_advantages_expanded = jnp.expand_dims(padded_advantages, axis=2)
        padded_surrogate = padded_advantages_expanded * padded_log_probs
        agent_alive_normalized = data.agent_alive / data.agent_alive.sum()
        reward_loss = -(padded_surrogate * agent_alive_normalized).sum()
        entropy = (dist.entropy() * agent_alive_normalized).sum()
        actor_loss = reward_loss - entropy_coef * entropy
        return actor_loss, {
            'reward_loss': reward_loss,
            'actor_loss': actor_loss,
            'entropy': entropy
        }

    new_actor, info = actor.apply_gradient(actor_loss_fn)

    return new_actor, info

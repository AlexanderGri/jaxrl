from typing import Tuple

import jax.numpy as jnp

from jaxrl.datasets import PaddedTrajectoryData
from jaxrl.networks.common import InfoDict, Model, Params


def update(actor: Model, critic: Model, data: PaddedTrajectoryData,
           discount: float, entropy_coef: float) -> Tuple[Model, InfoDict]:

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply_fn({'params': actor_params}, data.observations,
                              data.available_actions)
        log_probs = dist.log_prob(data.actions)
        next_values = critic(data.next_states)
        values = critic(data.states)
        advantages = data.rewards + discount * next_values - values
        advantages_expanded = jnp.expand_dims(advantages, axis=2)
        surrogate = advantages_expanded * log_probs
        reward_loss = -(surrogate * data.agent_alive).mean()
        entropy = -(log_probs * data.agent_alive).mean()
        actor_loss = reward_loss - entropy_coef * entropy
        return actor_loss, {
            'reward_loss': reward_loss,
            'actor_loss': actor_loss,
            'entropy': entropy
        }

    new_actor, info = actor.apply_gradient(actor_loss_fn)

    return new_actor, info

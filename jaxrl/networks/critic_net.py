"""Implementations of algorithms for continuous control."""

from typing import Callable, Sequence, Tuple

import jax.numpy as jnp
from flax import linen as nn

from jaxrl.networks.common import MLP


class ValueCritic(nn.Module):
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        critic = MLP((*self.hidden_dims, 1))(observations)
        return jnp.squeeze(critic, -1)


class Critic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], -1)
        critic = MLP((*self.hidden_dims, 1),
                     activations=self.activations)(inputs)
        return jnp.squeeze(critic, -1)


class StateValueCritic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, states: jnp.ndarray) -> jnp.ndarray:
        critic = MLP((*self.hidden_dims, 1),
                     activations=self.activations)(states)
        return jnp.squeeze(critic, -1)


class RewardAndCritics(nn.Module):
    hidden_dims: Sequence[int]
    n_agents: int
    n_actions: int
    use_shared_reward: False
    use_shared_value: False
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    def setup(self):
        self.input_to_hidden = MLP((*self.hidden_dims[:-1], self.hidden_dims[-1]),
                                   activations=self.activations, activate_final=True)
        if self.use_shared_reward:
            self.hidden_to_rewards = nn.Dense(self.n_actions)
        else:
            self.hidden_to_rewards = nn.Dense(self.n_agents * self.n_actions)
        if self.use_shared_value:
            self.hidden_to_values = nn.Dense(1)
        else:
            self.hidden_to_values = nn.vmap(nn.Dense,
                                            variable_axes={'params': 0},
                                            split_rngs={'params': True},
                                            in_axes=None,
                                            out_axes=-1,
                                            axis_size=self.n_agents)(1)

    def __call__(self, states: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        hiddens = self.input_to_hidden(states)
        rewards = self.hidden_to_rewards(hiddens)
        if self.use_shared_reward:
            rewards = jnp.broadcast_to(rewards[..., None, :],
                                       (*rewards.shape[:-1], self.n_agents, self.n_actions))
        else:
            rewards = jnp.reshape(rewards, (*rewards.shape[:-1], self.n_agents, self.n_actions))
        rewards = jnp.tanh(rewards)
        if self.use_shared_value:
            values = self.hidden_to_values(hiddens)
            values = jnp.repeat(values, repeats=self.n_agents, axis=-1)
        else:
            values = self.hidden_to_values(hiddens)
            values = jnp.squeeze(values, -2)
        return rewards, values

    def get_rewards(self, states: jnp.ndarray) -> jnp.ndarray:
        hiddens = self.input_to_hidden(states)
        rewards = self.hidden_to_rewards(hiddens)
        if self.use_shared_reward:
            rewards = jnp.broadcast_to(rewards[..., None, :],
                                       (*rewards.shape[:-1], self.n_agents, self.n_actions))
        else:
            rewards = jnp.reshape(rewards, (*rewards.shape[:-1], self.n_agents, self.n_actions))
        rewards = jnp.tanh(rewards)
        return rewards

    def get_values(self, states: jnp.ndarray) -> jnp.ndarray:
        hiddens = self.input_to_hidden(states)
        if self.use_shared_value:
            values = self.hidden_to_values(hiddens)
            values = jnp.repeat(values, repeats=self.n_agents, axis=-1)
        else:
            values = self.hidden_to_values(hiddens)
            values = jnp.squeeze(values, -2)
        return values


class DoubleCritic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    num_qs: int = 2

    @nn.compact
    def __call__(self, states, actions):

        VmapCritic = nn.vmap(Critic,
                             variable_axes={'params': 0},
                             split_rngs={'params': True},
                             in_axes=None,
                             out_axes=0,
                             axis_size=self.num_qs)
        qs = VmapCritic(self.hidden_dims,
                        activations=self.activations)(states, actions)
        return qs

"""Implementations of algorithms for continuous control."""
import os
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from jaxrl.networks.common import MLP


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


PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
InfoDict = Dict[str, float]


def compute_update(params: Params,
                   tx: optax.GradientTransformation,
                   opt_state: optax.OptState,
                   loss_fn: Optional[Callable[[Params], Any]]):
    grads, aux = jax.grad(loss_fn, has_aux=True)(params)
    updates, new_opt_state = tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, aux


@flax.struct.dataclass
class RewardAndCriticsModel:
    step: int
    apply_fn: Callable[..., Any] = flax.struct.field(pytree_node=False)
    params_critic: Params
    params_reward: Params
    tx_critic: Optional[optax.GradientTransformation] = flax.struct.field(
        pytree_node=False)
    tx_reward: Optional[optax.GradientTransformation] = flax.struct.field(
        pytree_node=False)
    opt_state_critic: Optional[optax.OptState] = None
    opt_state_reward: Optional[optax.OptState] = None

    @classmethod
    def create(cls,
               model_def: nn.Module,
               inputs: Sequence[jnp.ndarray],
               tx_critic: Optional[optax.GradientTransformation] = None,
               tx_reward: Optional[optax.GradientTransformation] = None) -> 'RewardAndCriticsModel':
        variables = model_def.init(*inputs)

        params_reward_name = 'hidden_to_rewards'
        params_critic, params_reward_raw = variables['params'].pop(params_reward_name)
        params_reward = {params_reward_name: params_reward_raw}


        if tx_critic is not None:
            opt_state_critic = tx_critic.init(params_critic)
        else:
            opt_state_critic = None

        if tx_reward is not None:
            opt_state_reward = tx_reward.init(params_reward)
        else:
            opt_state_reward = None

        return cls(step=1,
                   apply_fn=model_def.apply,
                   params_critic=params_critic,
                   params_reward=params_reward,
                   tx_critic=tx_critic,
                   tx_reward=tx_reward,
                   opt_state_critic=opt_state_critic,
                   opt_state_reward=opt_state_reward)

    def __call__(self, params, *args, **kwargs):
        return self.apply_fn({'params': params}, *args, **kwargs)

    def get_values(self, *args, **kwargs):
        return self(self.params_critic,
                    *args, method=RewardAndCritics.get_values, **kwargs)

    def get_rewards(self, *args, **kwargs):
        return self(Params(**self.params_reward, **self.params_critic),
                    *args, method=RewardAndCritics.get_rewards, **kwargs)

    def apply_gradient_reward(
            self,
            loss_fn: Optional[Callable[[Params], Any]] = None,
            grads: Optional[Any] = None) -> Tuple['RewardAndCriticsModel', Any]:
        assert (loss_fn is not None or grads is not None,
                'Either a loss function or grads must be specified.')
        new_params, new_opt_state, aux = compute_update(self.params_reward, self.tx_reward, self.opt_state_reward,
                                                        loss_fn)
        new_model = self.replace(step=self.step + 1,
                                 params_reward=new_params,
                                 opt_state_reward=new_opt_state)
        return new_model, aux

    def apply_gradient_critic(
            self,
            loss_fn: Optional[Callable[[Params], Any]] = None,
            grads: Optional[Any] = None) -> Tuple['RewardAndCriticsModel', Any]:
        assert (loss_fn is not None or grads is not None,
                'Either a loss function or grads must be specified.')
        new_params, new_opt_state, aux = compute_update(self.params_critic, self.tx_critic, self.opt_state_critic,
                                                        loss_fn)
        new_model = self.replace(step=self.step + 1,
                                 params_critic=new_params,
                                 opt_state_critic=new_opt_state)
        return new_model, aux

    def get_attr_names(self):
        return ['params_critic', 'params_reward', 'opt_state_critic', 'opt_state_reward']

    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        attr_names = self.get_attr_names()
        for attr_name in attr_names:
            with open(save_path + '_' + attr_name, 'wb') as f:
                f.write(flax.serialization.to_bytes(getattr(self, attr_name)))

    def load(self, load_path: str) -> 'RewardAndCriticsModel':
        attr_names = self.get_attr_names()
        attr_values = []
        for attr_name in attr_names:
            with open(load_path + '_' + attr_name, 'rb') as f:
                attr_value = flax.serialization.from_bytes(getattr(self, attr_name), f.read())
                attr_values.append(attr_value)
        return self.replace(**dict(zip(attr_name, attr_values)))

"""Implementations of algorithms for continuous control."""

import functools
from typing import Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxrl.agents.meta_reward_pg.actor import update as update_actor
from jaxrl.agents.meta_reward_pg.critic import update_intrinsic as update_intrinsic_critic
from jaxrl.agents.meta_reward_pg.critic import update_extrinsic as update_extrinsic_critic
from jaxrl.agents.meta_reward_pg.reward import update as update_reward
from jaxrl.datasets import PaddedTrajectoryData
from jaxrl.networks import critic_net, policies
from jaxrl.networks.critic_net import RewardAndCriticsModel
from jaxrl.networks.common import InfoDict, Model, GRU


@functools.partial(jax.jit, static_argnames=('time_limit', 'use_recurrent_policy', 'use_mc_return'))
def _update_actor_jit(actor: Model, intrinsic: RewardAndCriticsModel,
                      data: PaddedTrajectoryData, discount: float, entropy_coef: float,  mix_coef: float, time_limit: int,
                      use_recurrent_policy: bool, use_mc_return: bool, init_carry: Optional[jnp.ndarray] = None) \
        -> Tuple[Model, InfoDict]:
    new_actor, actor_info = update_actor(actor,
                                         intrinsic,
                                         data, discount, entropy_coef,
                                         mix_coef, time_limit, use_recurrent_policy, init_carry,
                                         use_mc_return)
    return new_actor, actor_info


@functools.partial(jax.jit, static_argnames=('time_limit', 'use_recurrent_policy', 'sampling_scheme', 'use_mc_return'))
def _update_intrinsic_jit(prev_actor: Model, intrinsic: RewardAndCriticsModel,
                          extrinsic_critic: Model, prev_data: PaddedTrajectoryData,
                          data: PaddedTrajectoryData, discount: float, entropy_coef: float,  mix_coef: float,
                          time_limit: int, use_recurrent_policy: bool, sampling_scheme: str,
                          use_mc_return: bool, init_carry: Optional[jnp.ndarray] = None) \
        -> Tuple[RewardAndCriticsModel, InfoDict]:
    new_intrinsic, reward_info = update_reward(prev_actor,
                                               intrinsic,
                                               extrinsic_critic,
                                               prev_data,
                                               data,
                                               discount,
                                               entropy_coef,
                                               mix_coef,
                                               time_limit,
                                               use_recurrent_policy,
                                               sampling_scheme,
                                               init_carry,
                                               use_mc_return)

    new_new_intrinsic, critic_info = update_intrinsic_critic(new_intrinsic,
                                                             data,
                                                             discount,
                                                             time_limit,
                                                             mix_coef)
    info = {}
    for d, name in zip([reward_info, critic_info],
                       ['outer', 'intrinsic']):
        info.update({f'{name}_{k}': v for k, v in d.items()})
    return new_new_intrinsic, info


@functools.partial(jax.jit, static_argnames=('time_limit', 'use_recurrent_policy', 'sampling_scheme', 'use_mc_return'))
def _update_except_actor_jit(prev_actor: Model, intrinsic: RewardAndCriticsModel,
                             extrinsic_critic: Model, prev_data: PaddedTrajectoryData,
                             data: PaddedTrajectoryData, discount: float, entropy_coef: float,  mix_coef: float,
                             time_limit: int, use_recurrent_policy: bool, sampling_scheme: str,
                             use_mc_return: bool, init_carry: Optional[jnp.ndarray] = None) \
        -> Tuple[Model, RewardAndCriticsModel, InfoDict]:
    new_extrinsic_critic, extrinsic_critic_info = update_extrinsic_critic(extrinsic_critic,
                                                                          data,
                                                                          discount,
                                                                          time_limit)

    new_intrinsic, info = _update_intrinsic_jit(
        prev_actor, intrinsic, new_extrinsic_critic,
        prev_data, data, discount, entropy_coef, mix_coef, time_limit,
        use_recurrent_policy, sampling_scheme, use_mc_return, init_carry)

    for k, v in extrinsic_critic_info.items():
        info[f'extrinsic_{k}'] = v
    return new_extrinsic_critic, new_intrinsic, info


class MetaPGLearner(object):
    def __init__(self,
                 seed: int,
                 states: jnp.ndarray,
                 observations: jnp.ndarray,
                 available_actions: jnp.ndarray,
                 n_actions: int,
                 time_limit: int,
                 n_agents: int,
                 actor_lr: float = 3e-4,
                 reward_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 optimizer_name = 'adam',
                 critic_hidden_dims: Sequence[int] = (128, 128),
                 actor_hidden_dims: Sequence[int] = (64,),
                 actor_recurrent_hidden_dim: int = 64,
                 use_recurrent_policy: bool = True,
                 use_shared_reward: bool = False,
                 use_shared_value: bool = False,
                 use_shared_policy: bool = True,
                 discount: float = 0.99,
                 entropy_coef: float = 1e-3,
                 mix_coef: float = 0.01,
                 sampling_scheme: str = 'reuse',
                 use_mc_return: bool = False,):

        self.discount = discount
        self.entropy_coef = entropy_coef
        self.mix_coef = mix_coef
        self.sampling_scheme = sampling_scheme
        self.use_mc_return = use_mc_return
        self.actor_lr = actor_lr
        self.optimizer_name = optimizer_name
        self.time_limit = time_limit
        self.use_recurrent_policy = use_recurrent_policy
        self.actor_recurrent_hidden_dim = actor_recurrent_hidden_dim
        self.n_agents = n_agents

        assert use_shared_reward or not use_shared_value

        if self.optimizer_name == 'adam':
            Optimizer = functools.partial(optax.adam, eps_root=1e-8)
        elif self.optimizer_name == 'rmsprop':
            Optimizer = functools.partial(optax.rmsprop, eps=1e-8)
        elif self.optimizer_name == 'sgd':
            Optimizer = optax.sgd
        else:
            raise NotImplementedError()

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, extrinsic_critic_key, intrinsic_key = jax.random.split(rng, 4)
        if self.use_recurrent_policy:
            actor_def = policies.RecurrentConstrainedCategoricalPolicy(
                hidden_dims=actor_hidden_dims,
                recurrent_hidden_dim=actor_recurrent_hidden_dim,
                n_actions=n_actions,
                shared=use_shared_policy)
            carry = self.initialize_carry(1)
            inputs = [actor_key, carry, observations, available_actions]
        else:
            actor_def = policies.ConstrainedCategoricalPolicy(
                hidden_dims=actor_hidden_dims,
                n_actions=n_actions,
                shared=use_shared_policy)
            inputs = [actor_key, observations, available_actions]
        actor = Model.create(actor_def,
                             inputs=inputs,
                             tx=Optimizer(learning_rate=actor_lr))

        extrinsic_critic_def = critic_net.StateValueCritic(critic_hidden_dims)
        extrinsic_critic = Model.create(extrinsic_critic_def,
                                        inputs=[extrinsic_critic_key, states],
                                        tx=Optimizer(learning_rate=critic_lr))
        intrinsic_def = critic_net.RewardAndCritics(critic_hidden_dims, n_agents, n_actions,
                                                    use_shared_reward, use_shared_value)
        intrinsic = RewardAndCriticsModel.create(intrinsic_def,
                                                 inputs=[intrinsic_key, states],
                                                 tx_critic=Optimizer(learning_rate=critic_lr),
                                                 tx_reward=Optimizer(learning_rate=reward_lr))

        self.prev_data = None
        self.prev_actor = None
        self.actor = actor
        self.extrinsic_critic = extrinsic_critic
        self.intrinsic = intrinsic
        self.rng = rng

    def sample_actions(self,
                       observations: np.ndarray,
                       available_actions: np.ndarray,
                       carry: Optional[jnp.ndarray] = None,
                       temperature: float = 1.0,
                       distribution: str = 'log_prob') -> \
            Union[Tuple[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        outputs = policies.sample_constrained_actions(self.rng, self.actor.apply_fn,
                                                      self.actor.params, observations,
                                                      available_actions, carry,
                                                      temperature, distribution)
        if carry is None:
            rng, actions, log_prob = outputs
            self.rng = rng
            actions = np.asarray(actions)
            return actions, log_prob
        else:
            rng, new_carry, actions, log_prob = outputs
            self.rng = rng
            actions = np.asarray(actions)
            return new_carry, actions, log_prob

    def get_actor_update_kwargs(self, n_trajectories):
        return {
            'discount': self.discount,
            'entropy_coef': self.entropy_coef,
            'mix_coef': self.mix_coef,
            'time_limit': self.time_limit,
            'use_recurrent_policy': self.use_recurrent_policy,
            'use_mc_return': self.use_mc_return,
            'init_carry': self.initialize_carry(n_trajectories)
        }

    def update_actor(self, data: PaddedTrajectoryData) -> InfoDict:
        kwargs = self.get_actor_update_kwargs(n_trajectories=data.actions.shape[0])
        self.actor, actor_info = _update_actor_jit(self.actor, self.intrinsic, data, **kwargs)
        return {f'inner_{k}': v for k, v in actor_info.items()}

    def update_except_actor(self, prev_data: PaddedTrajectoryData,
                            data: PaddedTrajectoryData, prev_actor: Model,
                            update_only_intrinsic: bool = False) -> InfoDict:
        if prev_data is None or prev_actor is None:
            return {}

        args = (prev_actor, self.intrinsic, self.extrinsic_critic, prev_data, data)
        kwargs = self.get_actor_update_kwargs(n_trajectories=data.actions.shape[0])

        if update_only_intrinsic:
            self.intrinsic, info = _update_intrinsic_jit(*args, **kwargs, sampling_scheme=self.sampling_scheme)
        else:
            self.extrinsic_critic, self.intrinsic, info = _update_except_actor_jit(*args, **kwargs, sampling_scheme=self.sampling_scheme)
        return info

    def initialize_carry(self, n_trajectories):
        if self.use_recurrent_policy:
            return GRU.initialize_carry((n_trajectories, self.n_agents), self.actor_recurrent_hidden_dim)
        else:
            return None

    def save(self, path):
        self.actor.save(f'{path}_actor')
        self.intrinsic.save(f'{path}_intrinsic')
        self.extrinsic_critic.save(f'{path}_extrinsic_critic')

    def load(self, path):
        self.actor = self.actor.load(f'{path}_actor')
        self.intrinsic = self.intrinsic.load(f'{path}_intrinsic')
        self.extrinsic_critic = self.extrinsic_critic.load(f'{path}_extrinsic_critic')

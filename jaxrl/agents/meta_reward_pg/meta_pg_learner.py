"""Implementations of algorithms for continuous control."""

import functools
from typing import Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
import numpy as np
import optax

from jaxrl.agents.meta_reward_pg.actor import update_intrinsic as update_intrinsic_actor
from jaxrl.agents.meta_reward_pg.critic import get_grad_intrinsic as get_grad_instrinsic_critic
from jaxrl.agents.meta_reward_pg.critic import update_extrinsic as update_extrinsic_critic
from jaxrl.agents.meta_reward_pg.reward import get_grad as get_grad_reward
from jaxrl.datasets import PaddedTrajectoryData
from jaxrl.networks import critic_net, policies
from jaxrl.networks.common import InfoDict, Model, PRNGKey, GRU


@functools.partial(jax.jit, static_argnames=('length', 'use_recurrent_policy', 'use_mc_return'))
def _update_actor_jit(rng: PRNGKey, actor: Model, intrinsic_reward: Model, intrinsic_critic: Model,
                data: PaddedTrajectoryData, discount: float, entropy_coef: float,  mix_coef: float, length: int,
                use_recurrent_policy: bool, use_mc_return: bool, init_carry: Optional[jnp.ndarray] = None) \
        -> Tuple[PRNGKey, Model, InfoDict]:
    rng, key = jax.random.split(rng)
    new_actor, actor_info = update_intrinsic_actor(actor,
                                                   intrinsic_reward,
                                                   intrinsic_reward.params,
                                                   intrinsic_critic,
                                                   data, discount, entropy_coef,
                                                   mix_coef, length, use_recurrent_policy, init_carry,
                                                   use_mc_return)

    return rng, new_actor, actor_info


@functools.partial(jax.jit, static_argnames=('length', 'use_recurrent_policy', 'sampling_scheme', 'acc_intrinsic_grads', 'use_mc_return'))
def _update_intrinsic_jit(rng: PRNGKey, prev_actor: Model, intrinsic_reward: Model, intrinsic_critic: Model,
                          extrinsic_critic: Model, prev_data: PaddedTrajectoryData,
                          data: PaddedTrajectoryData, discount: float, entropy_coef: float,  mix_coef: float,
                          length: int, use_recurrent_policy: bool, sampling_scheme: str, acc_intrinsic_grads: bool,
                          use_mc_return: bool, init_carry: Optional[jnp.ndarray] = None) \
        -> Tuple[PRNGKey, Model, Model, InfoDict]:
    rng, key = jax.random.split(rng)
    grad_reward, reward_info = get_grad_reward(prev_actor,
                                               intrinsic_reward,
                                               intrinsic_critic,
                                               extrinsic_critic,
                                               prev_data,
                                               data,
                                               discount,
                                               entropy_coef,
                                               mix_coef,
                                               length,
                                               use_recurrent_policy,
                                               sampling_scheme,
                                               init_carry,
                                               use_mc_return)
    new_intrinsic_reward = intrinsic_reward.apply_gradient(grads=grad_reward, has_aux=False)

    rng, key = jax.random.split(rng)
    intrinsic_critic_grad, intrinsic_critics_info = get_grad_instrinsic_critic(new_intrinsic_reward,
                                                                               intrinsic_critic,
                                                                               data,
                                                                               discount,
                                                                               length,
                                                                               mix_coef)
    new_intrinsic_critic = intrinsic_critic.apply_gradient(grads=intrinsic_critic_grad, has_aux=False)

    info = {}
    for d, name in zip([reward_info, intrinsic_critics_info],
                       ['outer', 'intrinsic']):
        info.update({f'{name}_{k}': v for k, v in d.items()})
    return rng, new_intrinsic_reward, new_intrinsic_critic, info


@functools.partial(jax.jit, static_argnames=('length', 'use_recurrent_policy', 'sampling_scheme', 'acc_intrinsic_grads', 'use_mc_return'))
def _update_except_actor_jit(rng: PRNGKey, prev_actor: Model, intrinsic_reward: Model, intrinsic_critic: Model,
                             extrinsic_critic: Model, prev_data: PaddedTrajectoryData,
                             data: PaddedTrajectoryData, discount: float, entropy_coef: float,  mix_coef: float,
                             length: int, use_recurrent_policy: bool, sampling_scheme: str, acc_intrinsic_grads: bool,
                             use_mc_return: bool, init_carry: Optional[jnp.ndarray] = None) \
        -> Tuple[PRNGKey, Model, Model, Model, InfoDict]:
    rng, key = jax.random.split(rng)
    new_extrinsic_critic, extrinsic_critic_info = update_extrinsic_critic(extrinsic_critic,
                                                                          data,
                                                                          discount,
                                                                          length)

    rng, new_intrinsic_reward, new_intrinsic_critic, info = _update_intrinsic_jit(
        rng, prev_actor, intrinsic_reward, intrinsic_critic, new_extrinsic_critic,
        prev_data, data, discount, entropy_coef, mix_coef, length,
        use_recurrent_policy, sampling_scheme, acc_intrinsic_grads, use_mc_return, init_carry)

    for k, v in extrinsic_critic_info.items():
        info[f'extrinsic_{k}'] = v
    return rng, new_extrinsic_critic, new_intrinsic_reward, new_intrinsic_critic, info


class MetaPGLearner(object):
    def __init__(self,
                 seed: int,
                 states: jnp.ndarray,
                 observations: jnp.ndarray,
                 available_actions: jnp.ndarray,
                 n_actions: int,
                 length: int,
                 n_agents: int,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
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
                 acc_intrinsic_grads: bool = False,
                 use_mc_return: bool = False,
                 mimic_sgd: bool = False):

        self.discount = discount
        self.entropy_coef = entropy_coef
        self.mix_coef = mix_coef
        self.sampling_scheme = sampling_scheme
        self.acc_intrinsic_grads = acc_intrinsic_grads
        self.use_mc_return = use_mc_return
        self.actor_lr = actor_lr
        self.mimic_sgd = mimic_sgd
        self.length = length
        self.use_recurrent_policy = use_recurrent_policy
        self.actor_recurrent_hidden_dim = actor_recurrent_hidden_dim

        assert (not mimic_sgd) or (mimic_sgd and sampling_scheme == 'importance_sampling')
        assert use_shared_reward or not use_shared_value

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, extrinsic_critic_key, intrinsic_reward_key, intrinsic_critic_key = jax.random.split(rng, 5)
        if self.use_recurrent_policy:
            actor_def = policies.RecurrentConstrainedCategoricalPolicy(
                hidden_dims=actor_hidden_dims,
                recurrent_hidden_dim=actor_recurrent_hidden_dim,
                n_actions=n_actions,
                shared=use_shared_policy)
            carry = self.initialize_carry(1, n_agents)
            inputs = [actor_key, carry, observations, available_actions]
        else:
            actor_def = policies.ConstrainedCategoricalPolicy(
                hidden_dims=actor_hidden_dims,
                n_actions=n_actions,
                shared=use_shared_policy)
            inputs = [actor_key, observations, available_actions]
        actor = Model.create(actor_def,
                             inputs=inputs,
                             tx=optax.adam(learning_rate=actor_lr, eps_root=1e-8))

        extrinsic_critic_def = critic_net.StateValueCritic(critic_hidden_dims)
        extrinsic_critic = Model.create(extrinsic_critic_def,
                                        inputs=[extrinsic_critic_key, states],
                                        tx=optax.adam(learning_rate=critic_lr))
        intrinsic_reward_def = critic_net.IntrinsicReward(critic_hidden_dims, n_agents, n_actions,
                                                          use_shared_reward)
        intrinsic_reward = Model.create(intrinsic_reward_def,
                                        inputs=[intrinsic_reward_key, states],
                                        tx=optax.adam(learning_rate=critic_lr))
        intrinsic_critic_def = critic_net.IntrinsicCritic(critic_hidden_dims, n_agents, n_actions,
                                                          use_shared_value)
        intrinsic_critic = Model.create(intrinsic_critic_def,
                                        inputs=[intrinsic_critic_key, states],
                                        tx=optax.adam(learning_rate=critic_lr))

        self.prev_data = None
        self.prev_actor = None
        self.actor = actor
        self.extrinsic_critic = extrinsic_critic
        self.intrinsic_reward = intrinsic_reward
        self.intrinsic_critic = intrinsic_critic
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

    def update_actor(self, data: PaddedTrajectoryData) -> InfoDict:
        n_trajectories, _, n_agents = data.actions.shape
        init_carry = self.initialize_carry(n_trajectories, n_agents)

        new_rng, new_actor, actor_info = _update_actor_jit(
            self.rng, self.actor, self.intrinsic_reward, self.intrinsic_critic, data,
            self.discount, self.entropy_coef, self.mix_coef, self.length,
            self.use_recurrent_policy, self.use_mc_return, init_carry)

        self.rng = new_rng
        self.actor = new_actor

        return {f'inner_{k}': v for k, v in actor_info.items()}

    def update_except_actor(self, prev_data: PaddedTrajectoryData,
                            data: PaddedTrajectoryData, prev_actor: Model,
                            update_only_intrinsic: bool = False) -> InfoDict:
        if prev_data is None or prev_actor is None:
            return {}
        n_trajectories, _, n_agents = data.actions.shape
        init_carry = self.initialize_carry(n_trajectories, n_agents)

        if self.mimic_sgd:
            tx = optax.sgd(learning_rate=self.actor_lr)
            opt_state = tx.init(prev_actor.params)
            prev_actor = prev_actor.replace(tx=tx, opt_state=opt_state)

        args = (self.rng, prev_actor, self.intrinsic_reward, self.intrinsic_critic, self.extrinsic_critic,
                prev_data, data, self.discount, self.entropy_coef, self.mix_coef, self.length,
                self.use_recurrent_policy, self.sampling_scheme, self.acc_intrinsic_grads, self.use_mc_return, init_carry)
        if update_only_intrinsic:
            new_rng, new_intrinsic_reward, new_intrinsic_critic, info = _update_intrinsic_jit(*args)
            new_extrinsic_critic = self.extrinsic_critic
        else:
            new_rng, new_extrinsic_critic, new_intrinsic_reward, new_intrinsic_critic, info = _update_except_actor_jit(*args)


        self.rng = new_rng
        self.extrinsic_critic = new_extrinsic_critic
        self.intrinsic_reward = new_intrinsic_reward
        self.intrinsic_critic = new_intrinsic_critic
        return info

    def initialize_carry(self, n_trajectories, n_agents):
        if self.use_recurrent_policy:
            return GRU.initialize_carry((n_trajectories, n_agents), self.actor_recurrent_hidden_dim)
        else:
            return None

    def save(self, path):
        self.actor.save(f'{path}_actor')
        self.intrinsic_reward.save(f'{path}_intrinsic_reward')
        self.intrinsic_critic.save(f'{path}_intrinsic_critic')
        self.extrinsic_critic.save(f'{path}_extrinsic_critic')

    def load(self, path):
        self.actor = self.actor.load(f'{path}_actor')
        self.intrinsic_reward = self.intrinsic_reward.load(f'{path}_intrinsic_reward')
        self.intrinsic_critic = self.intrinsic_critic.load(f'{path}_intrinsic_critic')
        self.extrinsic_critic = self.extrinsic_critic.load(f'{path}_extrinsic_critic')

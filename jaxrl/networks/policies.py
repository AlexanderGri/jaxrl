import functools
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

from jaxrl.networks.common import MLP, Params, PRNGKey, default_init, Recurrent

LOG_STD_MIN = -10.0
LOG_STD_MAX = 2.0
IMPOSSIBLE_ACTION_LOGIT = -1e8


class RecurrentConstrainedCategoricalPolicy(nn.Module):
    hidden_dims: Sequence[int]
    recurrent_hidden_dim: int
    n_actions: int

    @nn.compact
    def __call__(self,
                 carry: jnp.ndarray,
                 observations: jnp.ndarray,
                 available_actions: jnp.ndarray,
                 temperature: float = 1.0,):
        backbone = Recurrent(self.hidden_dims,
                             self.recurrent_hidden_dim,
                             self.n_actions)
        # time dimension should be third
        # traj x time x agent x dim -> traj x agent x time x dim
        inputs = observations.transpose((0, 2, 1, 3))
        new_carry, outputs = backbone(carry, inputs)
        logits = outputs.transpose((0, 2, 1, 3))
        # set logits of unavailable actions to -inf
        masked_logits = jnp.where(available_actions, logits, IMPOSSIBLE_ACTION_LOGIT)
        base_dist = tfd.Categorical(logits=masked_logits)
        return new_carry, base_dist


class ConstrainedCategoricalPolicy(nn.Module):
    hidden_dims: Sequence[int]
    n_actions: int

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 available_actions: jnp.ndarray,
                 temperature: float = 1.0,
                 training: bool = False) -> tfd.Distribution:
        outputs = MLP(self.hidden_dims,
                      activate_final=True)(observations)
        logits = nn.Dense(self.n_actions)(outputs)
        # set logits of unavailable actions to -inf
        masked_logits = jnp.where(available_actions, logits, IMPOSSIBLE_ACTION_LOGIT)
        base_dist = tfd.Categorical(logits=masked_logits)
        return base_dist


class MSEPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 temperature: float = 1.0,
                 training: bool = False) -> jnp.ndarray:
        outputs = MLP(self.hidden_dims,
                      activate_final=True,
                      dropout_rate=self.dropout_rate)(observations,
                                                      training=training)

        actions = nn.Dense(self.action_dim,
                           kernel_init=default_init())(outputs)
        return nn.tanh(actions)


class NormalTanhPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    state_dependent_std: bool = True
    dropout_rate: Optional[float] = None
    final_fc_init_scale: float = 1.0
    log_std_min: Optional[float] = None
    log_std_max: Optional[float] = None
    tanh_squash_distribution: bool = True
    init_mean: Optional[jnp.ndarray] = None

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 temperature: float = 1.0,
                 training: bool = False) -> tfd.Distribution:
        outputs = MLP(self.hidden_dims,
                      activate_final=True,
                      dropout_rate=self.dropout_rate)(observations,
                                                      training=training)

        means = nn.Dense(self.action_dim,
                         kernel_init=default_init(
                             self.final_fc_init_scale))(outputs)
        if self.init_mean is not None:
            means += self.init_mean

        if self.state_dependent_std:
            log_stds = nn.Dense(self.action_dim,
                                kernel_init=default_init(
                                    self.final_fc_init_scale))(outputs)
        else:
            log_stds = self.param('log_stds', nn.initializers.zeros,
                                  (self.action_dim, ))

        log_std_min = self.log_std_min or LOG_STD_MIN
        log_std_max = self.log_std_max or LOG_STD_MAX
        log_stds = jnp.clip(log_stds, log_std_min, log_std_max)

        if not self.tanh_squash_distribution:
            means = nn.tanh(means)

        base_dist = tfd.MultivariateNormalDiag(loc=means,
                                               scale_diag=jnp.exp(log_stds) *
                                               temperature)
        if self.tanh_squash_distribution:
            return tfd.TransformedDistribution(distribution=base_dist,
                                               bijector=tfb.Tanh())
        else:
            return base_dist


class NormalTanhMixturePolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    num_components: int = 5
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 temperature: float = 1.0,
                 training: bool = False) -> tfd.Distribution:
        outputs = MLP(self.hidden_dims,
                      activate_final=True,
                      dropout_rate=self.dropout_rate)(observations,
                                                      training=training)

        logits = nn.Dense(self.action_dim * self.num_components,
                          kernel_init=default_init())(outputs)
        means = nn.Dense(self.action_dim * self.num_components,
                         kernel_init=default_init(),
                         bias_init=nn.initializers.normal(stddev=1.0))(outputs)
        log_stds = nn.Dense(self.action_dim * self.num_components,
                            kernel_init=default_init())(outputs)

        shape = list(observations.shape[:-1]) + [-1, self.num_components]
        logits = jnp.reshape(logits, shape)
        mu = jnp.reshape(means, shape)
        log_stds = jnp.reshape(log_stds, shape)

        log_stds = jnp.clip(log_stds, LOG_STD_MIN, LOG_STD_MAX)

        components_distribution = tfd.Normal(loc=mu,
                                             scale=jnp.exp(log_stds) *
                                             temperature)

        base_dist = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(logits=logits),
            components_distribution=components_distribution)

        dist = tfd.TransformedDistribution(distribution=base_dist,
                                           bijector=tfb.Tanh())

        return tfd.Independent(dist, 1)


@functools.partial(jax.jit, static_argnames=('actor_apply_fn', 'distribution'))
def _sample_actions(
        rng: PRNGKey,
        actor_apply_fn: Callable[..., Any],
        actor_params: Params,
        observations: np.ndarray,
        temperature: float = 1.0,
        distribution: str = 'log_prob') -> Tuple[PRNGKey, jnp.ndarray]:
    if distribution == 'det':
        return rng, actor_apply_fn({'params': actor_params}, observations,
                                   temperature)
    else:
        dist = actor_apply_fn({'params': actor_params}, observations,
                              temperature)
        rng, key = jax.random.split(rng)
        return rng, dist.sample(seed=key)


def sample_actions(
        rng: PRNGKey,
        actor_apply_fn: Callable[..., Any],
        actor_params: Params,
        observations: np.ndarray,
        temperature: float = 1.0,
        distribution: str = 'log_prob') -> Tuple[PRNGKey, jnp.ndarray]:
    return _sample_actions(rng, actor_apply_fn, actor_params, observations,
                           temperature, distribution)


@functools.partial(jax.jit, static_argnames=('actor_apply_fn', 'distribution'))
def _sample_constrained_actions(
        rng: PRNGKey,
        actor_apply_fn: Callable[..., Any],
        actor_params: Params,
        observations: np.ndarray,
        available_actions: np.ndarray,
        temperature: float = 1.0,
        distribution: str = 'log_prob') -> Tuple[PRNGKey, jnp.ndarray, jnp.ndarray]:
    dist = actor_apply_fn({'params': actor_params}, observations,
                          available_actions, temperature)
    if distribution == 'det':
        sample = dist.logits.argmax(axis=-1)
    else:
        rng, key = jax.random.split(rng)
        sample = dist.sample(seed=key)
    log_prob = dist.log_prob(sample)
    return rng, sample, log_prob


@functools.partial(jax.jit, static_argnames=('actor_apply_fn', 'distribution'))
def _sample_constrained_actions_recurrent(
        rng: PRNGKey,
        actor_apply_fn: Callable[..., Any],
        actor_params: Params,
        carry: np.ndarray,
        observations: np.ndarray,
        available_actions: np.ndarray,
        temperature: float = 1.0,
        distribution: str = 'log_prob') -> Tuple[PRNGKey, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    new_carry, dist = actor_apply_fn({'params': actor_params},
                                     carry, observations,
                                     available_actions, temperature)
    if distribution == 'det':
        sample = dist.logits.argmax(axis=-1)
    else:
        rng, key = jax.random.split(rng)
        sample = dist.sample(seed=key)
    log_prob = dist.log_prob(sample)
    return rng, new_carry, sample, log_prob


def sample_constrained_actions(
        rng: PRNGKey,
        actor_apply_fn: Callable[..., Any],
        actor_params: Params,
        observations: np.ndarray,
        available_actions: np.ndarray,
        carry: Optional[jnp.ndarray] = None,
        temperature: float = 1.0,
        distribution: str = 'log_prob') \
        -> Union[Tuple[PRNGKey, jnp.ndarray, jnp.ndarray], Tuple[PRNGKey, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    if carry is None:
        return _sample_constrained_actions(rng, actor_apply_fn, actor_params, observations,
                                           available_actions, temperature, distribution)
    else:
        return _sample_constrained_actions_recurrent(rng, actor_apply_fn, actor_params, carry,
                                                     observations, available_actions,
                                                     temperature, distribution)

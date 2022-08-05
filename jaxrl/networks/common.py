import functools
import os
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax


def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)


PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
PRNGKey = Any
Shape = Sequence[int]
Dtype = Any  # this could be a real type?
InfoDict = Dict[str, float]


class Recurrent(nn.Module):
    hidden_dims: Sequence[int]
    recurrent_hidden_dim: int
    n_actions: int

    @nn.compact
    def __call__(self,
                 carry: jnp.ndarray,
                 observations: jnp.ndarray,):
        inputs = MLP(self.hidden_dims, activate_final=True)(observations)
        new_carry, hiddens = GRU()(carry, inputs)
        outputs = nn.Dense(self.n_actions)(hiddens)
        return new_carry, outputs


class GRU(nn.Module):

    @functools.partial(
        nn.transforms.scan,
        variable_broadcast='params',
        in_axes=1, out_axes=1,
        split_rngs={'params': False})
    @nn.compact
    def __call__(self, carry: jnp.ndarray, x: jnp.ndarray):
        return nn.GRUCell()(carry, x)

    @staticmethod
    def initialize_carry(batch_dims: Shape,
                         hidden_size: int):
        assert len(batch_dims) == 2
        return nn.GRUCell.initialize_carry(jax.random.PRNGKey(0),
                                           batch_dims, hidden_size)


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init())(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.dropout_rate is not None:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training)
        return x


# TODO: Replace with TrainState when it's ready
# https://github.com/google/flax/blob/master/docs/flip/1009-optimizer-api.md#train-state
@flax.struct.dataclass
class Model:
    step: int
    apply_fn: Callable[..., Any] = flax.struct.field(pytree_node=False)
    params: Params
    tx: Optional[optax.GradientTransformation] = flax.struct.field(
        pytree_node=False)
    opt_state: Optional[optax.OptState] = None

    @classmethod
    def create(cls,
               model_def: nn.Module,
               inputs: Sequence[jnp.ndarray],
               tx: Optional[optax.GradientTransformation] = None) -> 'Model':
        variables = model_def.init(*inputs)

        _, params = variables.pop('params')

        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(step=1,
                   apply_fn=model_def.apply,
                   params=params,
                   tx=tx,
                   opt_state=opt_state)

    def __call__(self, *args, **kwargs):
        return self.apply_fn({'params': self.params}, *args, **kwargs)

    def apply_gradient(
            self,
            loss_fn: Optional[Callable[[Params], Any]] = None,
            grads: Optional[Any] = None,
            has_aux: bool = True) -> Union[Tuple['Model', Any], 'Model']:
        assert (loss_fn is not None or grads is not None,
                'Either a loss function or grads must be specified.')
        if grads is None:
            grad_fn = jax.grad(loss_fn, has_aux=has_aux)
            if has_aux:
                grads, aux = grad_fn(self.params)
            else:
                grads = grad_fn(self.params)
        else:
            assert (has_aux,
                    'When grads are provided, expects no aux outputs.')

        updates, new_opt_state = self.tx.update(grads, self.opt_state,
                                                self.params)
        new_params = optax.apply_updates(self.params, updates)

        new_model = self.replace(step=self.step + 1,
                                 params=new_params,
                                 opt_state=new_opt_state)
        if has_aux:
            return new_model, aux
        else:
            return new_model

    def get_attr_names(self):
        return ['params', 'opt_state']

    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        attr_names = self.get_attr_names()
        for attr_name in attr_names:
            with open(save_path + '_' + attr_name, 'wb') as f:
                f.write(flax.serialization.to_bytes(getattr(self, attr_name)))

    def load(self, load_path: str) -> 'Model':
        attr_names = self.get_attr_names()
        attr_values = []
        for attr_name in attr_names:
            with open(load_path + '_' + attr_name, 'rb') as f:
                attr_value = flax.serialization.from_bytes(getattr(self, attr_name), f.read())
                attr_values.append(attr_value)
        return self.replace(**dict(zip(attr_names, attr_values)))


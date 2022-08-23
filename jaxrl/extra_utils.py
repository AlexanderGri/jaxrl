from jax import dtypes, jit, lax, random
import jax.numpy as jnp
from jax._src.util import prod
from jax.ops import index, index_add, index_update
from jax.numpy.linalg import norm

import numpy as np


@jit
def factor_mgs(A):
    n, m = A.shape
    # n rows
    # m dimension
    if n > m:
        raise Exception("Number of rows is larger than dimension")
    Q = jnp.empty([n, m])
    R = jnp.zeros([n, n])
    for k in range(0, n):
        # fill the k-th diagonal entry in R
        atom = A[k]
        norm_a = norm(atom)
        R = index_update(R, index[k, k], norm_a)
        # Initialize the k-th vector in Q
        q = atom / norm_a
        Q = index_update(Q, index[k], q)
        # Compute the inner product of new q vector with each of the remaining rows in A
        products = A[k+1:n, :] @ q.T
        # Place in k-th column of R
        R = index_update(R, index[k+1:n, k], products)
        # Subtract the contribution of previous q vector from all remaining rows of A.
        rr = R[k+1:n, k:k+1]
        update =  -rr @ jnp.expand_dims(q, 0)
        A = index_add(A, index[k+1:n], update)
    return Q, R


def orthogonal(scale=1.0, column_axis=-1, dtype=jnp.float_):
  """
  Construct an initializer for uniformly distributed orthogonal matrices.

  If the shape is not square, the matrices will have orthonormal rows or columns
  depending on which side is smaller.
  """
  def init(key, shape, dtype=dtype):
    dtype = dtypes.canonicalize_dtype(dtype)
    if len(shape) < 2:
      raise ValueError("orthogonal initializer requires at least a 2D shape")
    n_rows, n_cols = prod(shape) // shape[column_axis], shape[column_axis]
    matrix_shape = (n_cols, n_rows) if n_rows < n_cols else (n_rows, n_cols)
    A = random.normal(key, matrix_shape, dtype)
    Q, R = factor_mgs(jnp.transpose(A))
    Q = -jnp.transpose(Q)
    R = -jnp.transpose(R)
    diag_sign = lax.broadcast_to_rank(jnp.sign(jnp.diag(R)), rank=Q.ndim)
    Q *= diag_sign # needed for a uniform distribution
    if n_rows < n_cols: Q = Q.T
    Q = jnp.reshape(Q, tuple(np.delete(shape, column_axis)) + (shape[column_axis],))
    Q = jnp.moveaxis(Q, -1, column_axis)
    return scale * Q
  return init

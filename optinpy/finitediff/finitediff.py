"""Automatic and finite-difference derivatives implemented with JAX.

The historical ``jacobian`` name denotes the gradient of a scalar objective.
Inputs are never mutated. Functions must accept JAX arrays.
"""
import jax
import jax.numpy as jnp


def as_vector(x):
    x = jnp.asarray(x)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("x must be a nonempty one-dimensional array")
    if jnp.issubdtype(x.dtype, jnp.complexfloating):
        raise TypeError("x must be real-valued")
    return x.astype(jnp.result_type(x.dtype, float))


def _step(x, epsilon, order):
    if epsilon is not None:
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        return jnp.asarray(epsilon, dtype=x.dtype)
    return jnp.asarray(jnp.finfo(x.dtype).eps ** (1 / order), dtype=x.dtype)


def jacobian(fun, x0, epsilon=None, algorithm="autodiff"):
    """Gradient via autodiff (default), central, forward, or backward differences.

    ``algorithm`` and ``epsilon`` are static options when compiling with JAX.
    """
    x = as_vector(x0)
    if algorithm == "autodiff":
        return jax.grad(fun)(x)
    if algorithm not in ("central", "forward", "backward"):
        raise ValueError(f"Unknown differentiation algorithm: {algorithm}")
    h = _step(x, epsilon, 3 if algorithm == "central" else 2)
    offsets = h * jnp.eye(x.size, dtype=x.dtype)
    if algorithm == "central":
        return jax.vmap(lambda e: (fun(x + e) - fun(x - e)) / (2 * h))(offsets)
    sign = 1 if algorithm == "forward" else -1
    return jax.vmap(lambda e: (fun(x + sign * e) - fun(x)) / (sign * h))(offsets)


def hessian(fun, x0, epsilon=None, algorithm="autodiff", initial=None):
    """Hessian; central differences retain the original five-point diagonal.

    ``initial`` is accepted for legacy parameter dictionaries.
    """
    x = as_vector(x0)
    if algorithm == "autodiff":
        return jax.hessian(fun)(x)
    if algorithm not in ("central", "forward", "backward"):
        raise ValueError(f"Unknown differentiation algorithm: {algorithm}")
    h = _step(x, epsilon, 4 if algorithm == "central" else 3)
    offsets = h * jnp.eye(x.size, dtype=x.dtype)
    if algorithm == "central":
        # Evaluate only one triangle, as in the original implementation, and
        # share f(x) across the fourth-order diagonal probes.
        # The probe layout depends only on the static vector size. Construct it
        # while tracing instead of running triangular-index generation per call.
        pairs = [(i, j) for i in range(x.size) for j in range(i + 1, x.size)]
        i, j = jnp.asarray(pairs, dtype=jnp.int32).reshape(-1, 2).T
        def entry(e, v):
            return (fun(x + e + v) - fun(x + e - v)
                    - fun(x - e + v) + fun(x - e - v)) / (4 * h * h)
        mixed = jax.vmap(entry)(offsets[i], offsets[j])
        center = fun(x)
        diagonal = jax.vmap(lambda e: (-fun(x + 2*e) + 16*fun(x + e)
                                       - 30*center + 16*fun(x - e) - fun(x - 2*e))
                            / (12*h*h))(offsets)
        return jnp.diag(diagonal).at[i, j].set(mixed).at[j, i].set(mixed)
    else:
        sign = 1 if algorithm == "forward" else -1

        def entry(e, v):
            return (fun(x + sign * (e + v)) - fun(x + sign * e)
                    - fun(x + sign * v) + fun(x)) / (h * h)
    return jax.vmap(lambda e: jax.vmap(lambda v: entry(e, v))(offsets))(offsets)

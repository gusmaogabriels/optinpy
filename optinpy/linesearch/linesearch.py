"""Native, bounded JAX line searches. No external optimizer calls."""
import jax
import jax.numpy as jnp
from jax import lax

from ..finitediff.finitediff import as_vector, jacobian


def xstep(x0, d, alpha):
    return as_vector(x0) + alpha * jnp.asarray(d)


def _prepare(fun, x0, d, gradient, kwargs, value=None):
    x = as_vector(x0)
    g = jacobian(fun, x, **kwargs) if gradient is None else jnp.asarray(gradient)
    d = -g if d is None else jnp.asarray(d, dtype=x.dtype)
    if d.shape != x.shape or g.shape != x.shape:
        raise ValueError("x, direction, and gradient must have the same shape")
    f = jnp.asarray(fun(x) if value is None else value, dtype=x.dtype)
    if f.ndim != 0:
        raise ValueError("The objective must return a scalar")
    return x, d, f, jnp.vdot(g, d).real, g


def _armijo_search(fun, x0, d, alpha, rho, c, max_iter, gradient, interpolate,
                   alpha_min, kwargs, value):
    if not 0 < rho < 1 or not 0 < c < 1 or int(max_iter) < 0:
        raise ValueError("Require 0 < rho, c < 1 and max_iter >= 0")
    if not 0 < alpha_min < rho:
        raise ValueError("Require 0 < alpha_min < rho")
    x, d, f0, slope, _ = _prepare(fun, x0, d, gradient, kwargs, value)
    alpha = jnp.asarray(alpha, dtype=x.dtype)
    phi = lambda a: jnp.asarray(fun(x + a * d), dtype=x.dtype)
    valid = jnp.isfinite(f0) & jnp.isfinite(slope) & (slope <= 0) & (alpha > 0)

    def accepted(a, f):
        return jnp.isfinite(f) & (f <= f0 + c * a * slope) & ((f < f0) | (slope == 0))

    def cond(state):
        a, f, _, _, i = state
        return valid & ~accepted(a, f) & (i < int(max_iter))

    def body(state):
        a, f, prev_a, prev_f, i = state
        proposal = rho * a
        if interpolate:
            quadratic = -slope * a * a / (2 * (f - f0 - a * slope))
            r = (f - f0 - a * slope) / (a * a)
            prev_r = (prev_f - f0 - prev_a * slope) / (prev_a * prev_a)
            cubic_a = (r - prev_r) / (a - prev_a)
            cubic_b = r - cubic_a * a
            discriminant = cubic_b * cubic_b - 3 * cubic_a * slope
            cubic = (-cubic_b + jnp.sqrt(jnp.maximum(discriminant, 0))) / (3 * cubic_a)
            cubic = jnp.where(jnp.abs(cubic_a) > jnp.finfo(x.dtype).tiny,
                              cubic, -slope / (2 * cubic_b))
            candidate = jnp.where(i == 0, quadratic, cubic)
            proposal = jnp.where(jnp.isfinite(candidate) & (candidate > 0),
                                 jnp.clip(candidate, alpha_min * a, rho * a), proposal)
        return proposal, phi(proposal), a, f, i + 1

    first_value = phi(alpha)
    initial = (alpha, first_value, alpha, first_value, jnp.asarray(0))
    a, f, _, _, iters = lax.while_loop(cond, body, initial)
    success = valid & accepted(a, f)
    a = jnp.where(success, a, 0)
    result = {"x": x + a * d, "f": jnp.where(success, f, f0), "alpha": a,
              "iterations": iters, "success": success}
    if interpolate:
        result["inner_iterations"] = {"first_order": jnp.asarray(1),
                                      "second_order": jnp.minimum(iters, 1),
                                      "third_order": jnp.maximum(iters - 1, 0)}
    return result


def backtracking(fun, x0, d=None, alpha=1., rho=0.6, c=1e-4, max_iter=50,
                 *, gradient=None, value=None, **kwargs):
    """Armijo backtracking. Failure returns alpha=0 and success=False.

    ``gradient`` and ``value`` reuse known derivatives and fun(x0). Other keywords
    select the differentiation algorithm used when the gradient is omitted.
    """
    return _armijo_search(fun, x0, d, alpha, rho, c, max_iter, gradient,
                          False, min(0.1, rho / 2), kwargs, value)


def interp23(fun, x0, d=None, alpha=1., c=1e-4, alpha_min=0.1, rho=0.5,
             max_iter=50, *, gradient=None, value=None, **kwargs):
    """Safeguarded quadratic/cubic interpolation satisfying Armijo decrease."""
    return _armijo_search(fun, x0, d, alpha, rho, c, max_iter, gradient,
                          True, alpha_min, kwargs, value)


def _bounded(fun, x0, d, b, threshold, max_iter, gradient, key, kwargs, value):
    if threshold <= 0 or int(max_iter) < 0:
        raise ValueError("Require threshold > 0 and max_iter >= 0")
    x, d, f0, _, _ = _prepare(fun, x0, d, gradient, kwargs, value)
    b = jnp.asarray(b, dtype=x.dtype)
    phi = lambda a: jnp.asarray(fun(x + a * d), dtype=x.dtype)
    randomized = key is not None

    def cond(state):
        lo, hi, i, _ = state
        return (hi - lo > threshold) & (i < int(max_iter))

    def body(state):
        lo, hi, i, rng = state
        rng, subkey = jax.random.split(rng)
        fractions = jnp.sort(jax.random.uniform(subkey, (2,), dtype=x.dtype))
        a1, a2 = lo + fractions * (hi - lo)
        f1, f2 = phi(a1), phi(a2)
        right = jnp.where(jnp.isfinite(f1), f1, jnp.inf) > jnp.where(jnp.isfinite(f2), f2, jnp.inf)
        return jnp.where(right, a1, lo), jnp.where(right, hi, a2), i + 1, rng

    if randomized:
        lo, hi, iters, _ = lax.while_loop(cond, body, (jnp.zeros_like(b), b, jnp.asarray(0), key))
    else:
        # A golden-section reduction retains one probe. Reuse its function
        # value so each subsequent reduction evaluates only one new point.
        fraction = jnp.asarray(0.6180339887498949, dtype=x.dtype)
        a1, a2 = (1-fraction)*b, fraction*b

        def golden_cond(s):
            lo, hi, _, _, _, _, i = s
            return (hi-lo > threshold) & (i < int(max_iter))

        def golden_step(s):
            lo, hi, a1, a2, f1, f2, i = s
            right = jnp.where(jnp.isfinite(f1), f1, jnp.inf) > jnp.where(jnp.isfinite(f2), f2, jnp.inf)
            lo, hi = jnp.where(right, a1, lo), jnp.where(right, hi, a2)
            probe = jnp.where(right, lo + fraction*(hi-lo), lo + (1-fraction)*(hi-lo))
            value = phi(probe)
            return (lo, hi, jnp.where(right, a2, probe), jnp.where(right, probe, a1),
                    jnp.where(right, f2, value), jnp.where(right, value, f1), i+1)

        lo, hi, _, _, _, _, iters = lax.while_loop(
            golden_cond, golden_step, (jnp.zeros_like(b), b, a1, a2, phi(a1), phi(a2), jnp.asarray(0)))
    candidates = jnp.stack([jnp.zeros_like(b), (lo + hi) / 2, b])
    values = jnp.stack([f0, phi(candidates[1]), phi(b)])
    index = jnp.argmin(jnp.where(jnp.isfinite(values), values, jnp.inf))
    a, f = candidates[index], values[index]
    success = (b > 0) & (hi - lo <= threshold) & jnp.isfinite(f) & (f <= f0)
    a = jnp.where(success, a, 0)
    return {"x": x + a * d, "f": jnp.where(success, f, f0), "alpha": a,
            "iterations": iters, "success": success}


def golden_section(fun, x0, d=None, b=1., threshold=1e-5, max_iter=100,
                   *, gradient=None, value=None, **kwargs):
    """Golden-section minimization over [0, b], including both endpoints."""
    return _bounded(fun, x0, d, b, threshold, max_iter, gradient, None, kwargs, value)


def unimodality(fun, x0, d=None, b=1., threshold=1e-5, max_iter=200,
                *, gradient=None, key=None, value=None, **kwargs):
    """Random interval reduction; pass a JAX key for explicit reproducibility."""
    key = jax.random.key(0) if key is None else key
    return _bounded(fun, x0, d, b, threshold, max_iter, gradient, key, kwargs, value)


def strong_wolfe(fun, x0, d=None, alpha=1., c1=1e-4, c2=0.9,
                 max_iter=60, alpha_max=1e6, *, gradient=None, gradient_fn=None, value=None, **kwargs):
    """Strong Wolfe search with bracketing and safeguarded bisection zoom.

    Tests sufficient decrease and absolute directional-derivative reduction.
    Failed searches return the original point with alpha=0.
    The result includes the gradient at the returned x for reuse by a solver.
    """
    if not 0 < c1 < c2 < 1 or max_iter < 0 or alpha_max <= 0:
        raise ValueError("Require 0<c1<c2<1, max_iter>=0, alpha_max>0")
    if gradient is None and gradient_fn is not None:
        gradient = gradient_fn(as_vector(x0))
    x, d, f0, slope, g0 = _prepare(fun, x0, d, gradient, kwargs, value)
    alpha = jnp.minimum(jnp.asarray(alpha, dtype=x.dtype), alpha_max)
    if gradient_fn is None and not kwargs:
        value_grad = jax.value_and_grad(fun)
    else:
        derivative_fn = gradient_fn if gradient_fn is not None else lambda y: jacobian(fun, y, **kwargs)
        value_grad = lambda y: (fun(y), derivative_fn(y))

    def evaluate(a):
        f, g = value_grad(x + a * d)
        g = jnp.asarray(g, dtype=x.dtype)
        return jnp.asarray(f, dtype=x.dtype), jnp.asarray(jnp.vdot(g, d), dtype=x.dtype), g

    valid = jnp.isfinite(f0) & jnp.isfinite(slope) & (slope < 0) & (alpha > 0)

    def accepted(a, f, derivative):
        return (jnp.isfinite(f) & jnp.isfinite(derivative)
                & (f <= f0 + c1 * a * slope) & (jnp.abs(derivative) <= -c2 * slope))

    def cond(s):
        a, f, derivative, _, _, _, i, _ = s
        return valid & ~accepted(a, f, derivative) & (i < int(max_iter))

    def body(s):
        a, f, derivative, lo, hi, flo, i, _ = s
        upper = (~jnp.isfinite(f) | ~jnp.isfinite(derivative)
                 | (f > f0 + c1 * a * slope) | ((lo > 0) & (f >= flo)) | (derivative >= 0))
        hi = jnp.where(upper, a, hi)
        lo, flo = jnp.where(upper, lo, a), jnp.where(upper, flo, f)
        a = jnp.where(jnp.isfinite(hi), (lo + hi) / 2, jnp.minimum(2 * a, alpha_max))
        f, derivative, g = evaluate(a)
        return a, f, derivative, lo, hi, flo, i + 1, g

    f, derivative, g = evaluate(alpha)
    a, f, derivative, _, _, _, i, g = lax.while_loop(
        cond, body, (alpha, f, derivative, jnp.zeros_like(alpha),
                     jnp.full_like(alpha, jnp.inf), f0, jnp.asarray(0), g))
    success = valid & accepted(a, f, derivative)
    a = jnp.where(success, a, 0)
    return {"x": x + a * d, "f": jnp.where(success, f, f0), "alpha": a,
            "gradient": jnp.where(success, g, g0), "iterations": i, "success": success}

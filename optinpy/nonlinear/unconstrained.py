"""Optinpy's own unconstrained optimization algorithms, implemented in JAX."""
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from ..finitediff.finitediff import as_vector, jacobian, hessian
from ..linesearch import backtracking, interp23, golden_section, unimodality, strong_wolfe


_METHODS = {"gradient", "newton", "modified-newton", "conjugate-gradient",
            "hessian-conjugate-gradient", "fletcher-reeves", "bfgs", "dfp", "lbfgs", "adam", "sgd"}
_SEARCHES = {"backtracking": backtracking, "interp23": interp23,
             "golden-section": golden_section, "unimodality": unimodality, "strong-wolfe": strong_wolfe}


class _State(NamedTuple):
    x: jax.Array
    f: jax.Array
    g: jax.Array
    old_g: jax.Array
    d: jax.Array
    inverse: jax.Array
    steps: jax.Array
    changes: jax.Array
    count: jax.Array
    first_moment: jax.Array
    second_moment: jax.Array
    iterations: jax.Array
    ls_iterations: jax.Array
    status: jax.Array


def gradient_direction(g):
    """Steepest-descent direction from a precomputed gradient."""
    return -as_vector(g)


sgd_direction = gradient_direction


def newton_direction(g, Q):
    """Solve Q d = -g for a precomputed Hessian Q (no explicit inverse).

    A singular Hessian may produce a nonfinite direction. ``minimize`` applies
    its descent fallback after this primitive; a manual driver must do likewise.
    """
    g = as_vector(g)
    Q = jnp.asarray(Q, dtype=g.dtype)
    if Q.shape != (g.size, g.size):
        raise ValueError("Hessian must have shape (len(g), len(g))")
    return jnp.linalg.solve(Q, -g)


def modified_newton_direction(g, Q, *, sigma=1e-4):
    """Newton direction with symmetric Hessian eigenvalues floored at sigma.

    This retains the JAX port's positive floor, not the original absolute-value
    transformation. ``sigma`` is static when this function is compiled.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    g = as_vector(g)
    Q = jnp.asarray(Q, dtype=g.dtype)
    if Q.shape != (g.size, g.size):
        raise ValueError("Hessian must have shape (len(g), len(g))")
    eigenvalues, vectors = jnp.linalg.eigh((Q + Q.T) / 2)
    Q = (vectors * jnp.maximum(eigenvalues, sigma)) @ vectors.T
    return newton_direction(g, Q)


def _matching_vectors(g, *vectors):
    g = as_vector(g)
    vectors = tuple(jnp.asarray(v, dtype=g.dtype) for v in vectors)
    if any(v.shape != g.shape for v in vectors):
        raise ValueError("Direction and gradient vectors must have matching shapes")
    return (g, *vectors)


def conjugate_gradient_direction(g, old_g, old_d, *, restart=False):
    """Polak–Ribiere+ direction; restart=True gives steepest descent.

    This is the modern PR+ method. The original optinpy recurrence is exposed
    separately as ``hessian_conjugate_gradient_direction``.
    """
    g, old_g, old_d = _matching_vectors(g, old_g, old_d)
    denominator = jnp.maximum(jnp.vdot(old_g, old_g), jnp.finfo(g.dtype).tiny)
    beta = jnp.maximum(jnp.vdot(g, g - old_g) / denominator, 0)
    return -g + jnp.where(restart, 0, beta) * old_d


def fletcher_reeves_direction(g, old_g, old_d, *, restart=False):
    """Fletcher–Reeves direction from current/previous gradients and direction."""
    g, old_g, old_d = _matching_vectors(g, old_g, old_d)
    denominator = jnp.maximum(jnp.vdot(old_g, old_g), jnp.finfo(g.dtype).tiny)
    beta = jnp.maximum(jnp.vdot(g, g) / denominator, 0)
    return -g + jnp.where(restart, 0, beta) * old_d


def hessian_conjugate_gradient_direction(g, old_d, Qd):
    """Original Q-conjugate direction using a precomputed Hessian product.

    Qd is Q @ old_d, obtained from a Hessian or jvp(grad(fun)). Nonpositive or
    nonfinite directional curvature gives steepest descent. The driver controls
    periodic restarts and need not evaluate Qd when restarting.
    """
    g, old_d, Qd = _matching_vectors(g, old_d, Qd)
    denominator = jnp.vdot(old_d, Qd)
    usable = jnp.isfinite(denominator) & (denominator > 0)
    beta = jnp.vdot(g, Qd) / jnp.where(usable, denominator, 1.)
    return -g + jnp.where(usable, beta, 0.) * old_d


def quasi_newton_direction(g, inverse):
    """Direction -H g from an inverse-Hessian approximation H."""
    g = as_vector(g)
    inverse = jnp.asarray(inverse, dtype=g.dtype)
    if inverse.shape != (g.size, g.size):
        raise ValueError("Inverse Hessian must have shape (len(g), len(g))")
    return -inverse @ g


def _curvature_ok(p, y):
    return jnp.vdot(p, y) > jnp.finfo(p.dtype).eps * jnp.linalg.norm(p) * jnp.linalg.norm(y)


def _inverse_update(inverse, p, y, formula):
    p, y = _matching_vectors(p, y)
    inverse = jnp.asarray(inverse, dtype=p.dtype)
    if inverse.shape != (p.size, p.size):
        raise ValueError("Inverse Hessian must have shape (len(p), len(p))")
    sy = jnp.vdot(p, y)

    def update(H):
        candidate = formula(H, p, y, sy)
        return jnp.where(jnp.all(jnp.isfinite(candidate)), (candidate + candidate.T) / 2, H)

    return lax.cond(_curvature_ok(p, y), update, lambda H: H, inverse)


def bfgs_update(inverse, p, y):
    """BFGS inverse-Hessian update, with p=x_new-x and y=g_new-g.

    Uses the original expanded O(n^2) rank-two formula. Insufficient curvature
    or a nonfinite candidate retains the previous inverse; inputs are immutable.
    """
    def formula(H, p, y, sy):
        Hy = H @ y
        rho = 1 / sy
        return (H + rho * (1 + rho * jnp.vdot(y, Hy)) * jnp.outer(p, p)
                - rho * (jnp.outer(p, Hy) + jnp.outer(Hy, p)))
    return _inverse_update(inverse, p, y, formula)


def dfp_update(inverse, p, y):
    """DFP inverse-Hessian update using the same safeguards as BFGS."""
    def formula(H, p, y, sy):
        Hy = H @ y
        return H + jnp.outer(p, p) / sy - jnp.outer(Hy, Hy) / jnp.vdot(y, Hy)
    return _inverse_update(inverse, p, y, formula)


def lbfgs_direction(g, steps, changes, count):
    """Limited-memory inverse-Hessian product using the two-loop recursion."""
    g = as_vector(g)
    steps, changes = jnp.asarray(steps, dtype=g.dtype), jnp.asarray(changes, dtype=g.dtype)
    if steps.ndim != 2 or steps.shape[0] < 1 or steps.shape[1] != g.size or changes.shape != steps.shape:
        raise ValueError("L-BFGS histories must have matching shape (memory_size, len(g))")
    size = steps.shape[0]
    active = jnp.arange(size) < count
    sy = jnp.sum(steps * changes, axis=1)
    rho = jnp.where(active, 1 / jnp.where(active, sy, 1), 0)

    def first(i, state):
        q, alphas = state
        alpha = rho[i] * jnp.vdot(steps[i], q)
        return q - alpha * changes[i], alphas.at[i].set(alpha)

    q, alphas = lax.fori_loop(0, size, first, (g, jnp.zeros(size, dtype=g.dtype)))
    yy = jnp.vdot(changes[0], changes[0])
    scale = jnp.where(count > 0, sy[0] / jnp.maximum(yy, jnp.finfo(g.dtype).tiny), 1)

    def second(i, r):
        k = size - 1 - i
        beta = rho[k] * jnp.vdot(changes[k], r)
        return r + steps[k] * (alphas[k] - beta)

    return -lax.fori_loop(0, size, second, scale * q)


def lbfgs_update(steps, changes, count, p, y):
    """Remember a valid curvature pair, newest first; return both histories/count.

    Allocate zero histories of shape (memory_size, len(x)) and count=0 initially.
    Histories passed to ``lbfgs_direction`` must obey this update's curvature rule.
    """
    p, y = _matching_vectors(p, y)
    steps, changes = jnp.asarray(steps, dtype=p.dtype), jnp.asarray(changes, dtype=p.dtype)
    if steps.ndim != 2 or steps.shape[0] < 1 or steps.shape[1] != p.size or changes.shape != steps.shape:
        raise ValueError("L-BFGS histories must have matching shape (memory_size, len(p))")

    def remember(mem):
        ps, ys, count = mem
        return (jnp.roll(ps, 1, axis=0).at[0].set(p),
                jnp.roll(ys, 1, axis=0).at[0].set(y), jnp.minimum(count + 1, steps.shape[0]))

    return lax.cond(_curvature_ok(p, y), remember, lambda mem: mem, (steps, changes, jnp.asarray(count)))


def adam_direction(g, first_moment, second_moment, iteration, *, beta1=.9, beta2=.999, epsilon=1e-8):
    """Return (direction, first_moment, second_moment) for one Adam update.

    iteration counts completed updates (zero initially). Multiply the returned
    direction by the learning rate. Hyperparameters are static under jit.
    """
    if not 0 <= beta1 < 1 or not 0 <= beta2 < 1 or epsilon <= 0:
        raise ValueError("Require beta1/beta2 in [0, 1) and epsilon > 0")
    g, m, v = _matching_vectors(g, first_moment, second_moment)
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g ** 2
    t = iteration + 1
    d = -(m / (1 - beta1 ** t)) / (jnp.sqrt(v / (1 - beta2 ** t)) + epsilon)
    return d, m, v


def _direction_for(method, objective, hessian_fn, args, *, sigma, beta1, beta2, adam_epsilon):
    """Resolve composition during setup/tracing, outside the numerical loop."""
    def unchanged_moments(direction):
        return lambda s: (direction(s), s.first_moment, s.second_moment)

    if callable(method):
        def custom(s):
            d = jnp.asarray(method(s.x, s.g, *args), dtype=s.x.dtype)
            if d.shape != s.x.shape:
                raise ValueError("Custom direction must match x.shape")
            return d
        return unchanged_moments(custom)
    if method in ("gradient", "sgd"):
        return unchanged_moments(lambda s: gradient_direction(s.g))
    if method in ("bfgs", "dfp"):
        return unchanged_moments(lambda s: quasi_newton_direction(s.g, s.inverse))
    if method == "lbfgs":
        return unchanged_moments(lambda s: lbfgs_direction(s.g, s.steps, s.changes, s.count))
    if method == "newton":
        return unchanged_moments(lambda s: newton_direction(s.g, hessian_fn(s.x)))
    if method == "modified-newton":
        return unchanged_moments(lambda s: modified_newton_direction(s.g, hessian_fn(s.x), sigma=sigma))
    if method in ("conjugate-gradient", "fletcher-reeves"):
        direction = conjugate_gradient_direction if method == "conjugate-gradient" else fletcher_reeves_direction
        return unchanged_moments(lambda s: direction(s.g, s.old_g, s.d, restart=s.iterations % s.x.size == 0))
    if method == "hessian-conjugate-gradient":
        def direction(s):
            def conjugate(_):
                Qd = (jax.jvp(jax.grad(objective), (s.x,), (s.d,))[1]
                      if hessian_fn is None else hessian_fn(s.x) @ s.d)
                return hessian_conjugate_gradient_direction(s.g, s.d, Qd)
            return lax.cond(s.iterations % s.x.size == 0, lambda _: gradient_direction(s.g), conjugate, None)
        return unchanged_moments(direction)
    if method == "adam":
        return lambda s: adam_direction(s.g, s.first_moment, s.second_moment, s.iterations,
                                        beta1=beta1, beta2=beta2, epsilon=adam_epsilon)
    raise ValueError(f"Unknown direction method: {method}")


def _update_for(method):
    def retain(s, p, y):
        return s.inverse, s.steps, s.changes, s.count
    if callable(method):
        return retain
    if method in ("bfgs", "dfp"):
        update = bfgs_update if method == "bfgs" else dfp_update
        return lambda s, p, y: (update(s.inverse, p, y), s.steps, s.changes, s.count)
    if method == "lbfgs":
        return lambda s, p, y: (s.inverse, *lbfgs_update(s.steps, s.changes, s.count, p, y))
    return retain


def minimize(fun, x0, *, args=(), method="bfgs", tol=1e-6, max_iter=1000,
             linesearch="strong-wolfe", linesearch_options=None,
             learning_rate=0.01, beta1=0.9, beta2=0.999, adam_epsilon=1e-8,
             memory_size=10, sigma=1e-4, jac=None, hess=None,
             initial_hessian=None, history=False):
    """Minimize a real scalar function of a nonempty real vector.

    Methods: gradient (Armijo), newton, modified-newton, conjugate-gradient
    (Polak-Ribiere+), hessian-conjugate-gradient (original Q-conjugacy),
    fletcher-reeves, bfgs, dfp, lbfgs, adam, and sgd.
    Derivatives default to JAX autodiff; optional callables take ``(x, *args)``.
    ``initial_hessian`` is an initial *inverse* Hessian for BFGS/DFP.

    ``method`` may also be a stateless direction callable ``(x, gradient, *args)
    -> direction``. ``linesearch`` accepts a name or a callable with signature
    ``(fun, x, d, *, gradient, value, **options)`` returning x, success, iterations.
    Custom searches get the same descent/finiteness checks; their final gradient
    is recomputed using the configured derivative. For stateful custom algorithms,
    compose the public direction/update primitives in your own driver.

    Results contain x, f, gradient, gradient_norm, iterations, ls_iterations,
    success, and status: 0 converged, 1 iteration limit, 2 line-search failure
    or stagnation, 3 nonfinite objective/gradient. Convergence requires
    ||gradient||_2 <= tol. This is a local first-order convergence criterion.

    With history=False this is jit/vmap compatible. Method names, tolerances,
    iteration limits and solver options must be static when jitting. History
    uses a Python driver and adds x_history/f_history. Dynamic solve loops do
    not support reverse-mode differentiation through the solver itself.
    """
    if isinstance(method, str):
        method = method.lower().replace("l-bfgs", "lbfgs")
    if not callable(method) and method not in _METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {sorted(_METHODS)}")
    if not callable(linesearch) and linesearch not in _SEARCHES:
        raise ValueError(f"Unknown line search: {linesearch}")
    if tol <= 0 or int(max_iter) != max_iter or max_iter < 0:
        raise ValueError("Require tol > 0 and a nonnegative integer max_iter")
    if memory_size < 1 or int(memory_size) != memory_size:
        raise ValueError("memory_size must be a positive integer")
    if learning_rate <= 0 or sigma <= 0 or adam_epsilon <= 0:
        raise ValueError("learning_rate, sigma, and adam_epsilon must be positive")
    if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
        raise ValueError("Adam beta1 and beta2 must be in [0, 1)")
    x = as_vector(x0)
    objective = lambda y: jnp.asarray(fun(y, *args), dtype=x.dtype)
    if jac is None:
        value_grad = jax.value_and_grad(objective)
    else:
        value_grad = lambda y: (objective(y), jnp.asarray(jac(y, *args), dtype=x.dtype))
    hessian_fn = jax.hessian(objective) if hess is None else lambda y: jnp.asarray(hess(y, *args), dtype=x.dtype)
    f, g = value_grad(x)
    if f.ndim != 0 or g.shape != x.shape:
        raise ValueError("Objective must be scalar and gradient must match x")
    dense = not callable(method) and method in ("bfgs", "dfp")
    inverse = jnp.eye(x.size, dtype=x.dtype) if dense else jnp.empty((0, 0), dtype=x.dtype)
    if initial_hessian is not None:
        if not dense:
            raise ValueError("initial_hessian is supported only for BFGS and DFP")
        inverse = jnp.asarray(initial_hessian, dtype=x.dtype)
        if inverse.shape != (x.size, x.size):
            raise ValueError("initial_hessian must have shape (len(x), len(x))")
    memory = int(memory_size) if method == "lbfgs" else 0
    zeros = jnp.zeros_like(x)
    finite = jnp.isfinite(f) & jnp.all(jnp.isfinite(g)) & jnp.all(jnp.isfinite(x))
    state = _State(x, f, g, zeros, zeros, inverse,
                   jnp.zeros((memory, x.size), dtype=x.dtype),
                   jnp.zeros((memory, x.size), dtype=x.dtype), jnp.asarray(0),
                   zeros, zeros, jnp.asarray(0), jnp.asarray(0),
                   jnp.where(finite, 0, 3))
    search = linesearch if callable(linesearch) else _SEARCHES[linesearch]
    wolfe = search is strong_wolfe
    use_search = callable(method) or method not in ("adam", "sgd")
    direction = _direction_for(method, objective,
                               None if method == "hessian-conjugate-gradient" and hess is None else hessian_fn,
                               args, sigma=sigma, beta1=beta1, beta2=beta2, adam_epsilon=adam_epsilon)
    update = _update_for(method)
    search_options = dict(linesearch_options or {})
    if wolfe and jac is not None:
        search_options["gradient_fn"] = lambda y: jnp.asarray(jac(y, *args), dtype=x.dtype)
    # A line-search-only derivative override must not replace the solver's
    # configured objective gradient or alter its convergence test.
    reuse_search_gradient = jac is not None or not any(
        name in search_options for name in ('gradient_fn', 'algorithm', 'epsilon'))

    def cond(s):
        return (s.status == 0) & (s.iterations < int(max_iter)) & (jnp.linalg.norm(s.g) > tol)

    def step(s):
        d, m, v = direction(s)
        if use_search:
            # Indefinite/singular Newton systems and roundoff may lose descent.
            descent = jnp.all(jnp.isfinite(d)) & (jnp.vdot(d, s.g) < 0)
            d = jnp.where(descent, d, -s.g)
            ls = search(objective, s.x, d, gradient=s.g, value=s.f, **search_options)
            xn = jnp.asarray(ls["x"], dtype=x.dtype)
            accepted = jnp.asarray(ls["success"], dtype=bool)
            lsiters = jnp.asarray(ls["iterations"], dtype=s.ls_iterations.dtype)
            if xn.shape != x.shape or accepted.ndim != 0 or lsiters.ndim != 0:
                raise ValueError("Line search must return x matching x.shape and scalar success/iterations")
        else:
            xn, accepted, lsiters = s.x + learning_rate * d, jnp.asarray(True), jnp.asarray(0)
        if use_search and wolfe and reuse_search_gradient:
            fn, gn = ls["f"], ls["gradient"]
        else:
            fn, gn = value_grad(xn)
        finite = jnp.isfinite(fn) & jnp.all(jnp.isfinite(gn)) & jnp.all(jnp.isfinite(xn))
        stalled = jnp.all(xn == s.x) & (jnp.linalg.norm(gn) > tol)
        status = jnp.where(~finite, 3, jnp.where(~accepted | stalled, 2, 0))
        p, y = xn - s.x, gn - s.g
        inverse, steps, changes, count = update(s, p, y)
        return _State(xn, fn, gn, s.g, d, inverse, steps, changes, count, m, v,
                      s.iterations + 1, s.ls_iterations + lsiters, status)

    if history:
        xs, fs = [state.x], [state.f]
        compiled_step = jax.jit(step)
        while bool(cond(state)):
            state = compiled_step(state)
            xs.append(state.x)
            fs.append(state.f)
    else:
        state = lax.while_loop(cond, step, state)
    norm = jnp.linalg.norm(state.g)
    success = (state.status == 0) & (norm <= tol)
    status = jnp.where(success, 0, jnp.where(state.status != 0, state.status, 1))
    result = {"x": state.x, "f": state.f, "gradient": state.g, "gradient_norm": norm,
              "iterations": state.iterations, "ls_iterations": state.ls_iterations,
              "success": success, "status": status}
    if history:
        result.update(x_history=jnp.stack(xs), f_history=jnp.stack(fs))
    return result


def compile_minimizer(fun, *, donate_x0=False, **options):
    """Prepare a reusable compiled solve(x0, *args) with fixed solver options.

    Keep the returned callable and pass changing problem data through args to
    reuse compilation for matching shapes/dtypes. It is vmap-compatible.
    History uses a Python driver and is unavailable here. donate_x0=True is
    opt-in: the caller must not use the original x0 buffer after a call.
    """
    if options.get('history', False):
        raise ValueError("Compiled minimizers require history=False")
    if 'args' in options:
        raise ValueError("Pass objective args to the returned callable: solve(x0, *args)")
    options = dict(options)
    if options.get('linesearch_options') is not None:
        options['linesearch_options'] = dict(options['linesearch_options'])

    def solve(x0, *args):
        return minimize(fun, x0, args=args, **options)

    return jax.jit(solve, donate_argnums=(0,) if donate_x0 else ())


class unconstrained:
    """Compatibility interface using the original shared parameter dictionary."""

    def __init__(self, parameters):
        self.params = parameters

    def fminunc(self, fun, x0, threshold=1e-6, vectorized=False, **kwargs):
        """Legacy API: threshold bounds the *squared* gradient norm.

        vectorized=True returns the iterate history, as in the original API.
        For batches of independent solves, use jax.vmap with minimize instead.
        conjugate-gradient retains the original Hessian-conjugate formula.
        """
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        selected = self.params["fminunc"]["method"]
        options = dict(self.params["fminunc"]["params"].get(selected, {}))
        method = selected
        if method == "conjugate-gradient":
            method = "hessian-conjugate-gradient"
        if method == "quasi-newton":
            update = options.pop("hessian_update", "bfgs").lower()
            aliases = {"davidon-fletcher-powell": "dfp", "broyden-fletcher-goldfarb-shanno": "bfgs"}
            method = aliases.get(update, update)
        options.update(kwargs)
        jac_options = dict(self.params["jacobian"])
        hess_options = dict(self.params["hessian"])
        initial = hess_options.pop("initial", None)
        if initial is not None:
            options.setdefault("initial_hessian", initial)
        result = minimize(fun, x0, method=method, tol=threshold ** 0.5,
                          linesearch=self.params["linesearch"]["method"],
                          linesearch_options=self.params["linesearch"]["params"][self.params["linesearch"]["method"]],
                          jac=lambda x: jacobian(fun, x, **jac_options),
                          hess=(None if hess_options.get("algorithm", "autodiff") == "autodiff"
                                else lambda x: hessian(fun, x, **hess_options)),
                          history=vectorized, **options)
        if vectorized:
            result["x"] = result.pop("x_history")
            result["f"] = result.pop("f_history")
        return result

"""Native constrained solvers with JAX derivatives and linear algebra.

Active sets and continuation use Python orchestration. These two legacy APIs
are eager solvers, not whole-solve jit/vmap or differentiable layers.
"""
import math

import jax
import jax.numpy as jnp
from jax import lax

from ..finitediff.finitediff import as_vector, jacobian
from ..linesearch import backtracking
from .unconstrained import compile_minimizer


def _constraints(A, b, x, name):
    A = jnp.empty((0, x.size), dtype=x.dtype) if A is None else jnp.asarray(A, dtype=x.dtype)
    b = jnp.empty((0,), dtype=x.dtype) if b is None else jnp.asarray(b, dtype=x.dtype)
    if A.size == 0 and b.size == 0:
        return jnp.empty((0, x.size), dtype=x.dtype), jnp.empty((0,), dtype=x.dtype)
    if A.ndim != 2 or A.shape[1] != x.size or b.shape != (A.shape[0],):
        raise ValueError(f"{name} must have shape (m, len(x)) and its RHS shape (m,)")
    if not bool(jnp.all(jnp.isfinite(A)) & jnp.all(jnp.isfinite(b))):
        raise ValueError("Linear constraint coefficients must be finite")
    return A, b


@jax.jit
def _violation(x, A, b, Aeq, beq):
    return jnp.maximum(jnp.max(jnp.maximum(A @ x - b, 0), initial=0),
                       jnp.max(jnp.abs(Aeq @ x - beq), initial=0))


@jax.jit
def _feasible(x, A, b, Aeq, beq, tol, max_steps):
    """Dykstra projections onto halfspaces/hyperplanes to find a feasible point."""
    rows, rhs = jnp.concatenate((A, Aeq)), jnp.concatenate((b, beq))
    corrections = jnp.zeros_like(rows)
    if rows.shape[0] == 0:
        return x

    def cond(s):
        y, _, i = s
        return (_violation(y, A, b, Aeq, beq) > tol) & (i < max_steps)

    def cycle(s):
        def project(i, carry):
            y, corrections = carry
            z = y + corrections[i]
            residual = jnp.vdot(rows[i], z) - rhs[i]
            residual = jnp.where(i < A.shape[0], jnp.maximum(residual, 0), residual)
            denom = jnp.vdot(rows[i], rows[i])
            y = z - residual / jnp.where(denom > 0, denom, 1) * rows[i]
            return y, corrections.at[i].set(z - y)
        y, corrections = lax.fori_loop(0, rows.shape[0], project, s[:2])
        return y, corrections, s[2] + 1

    return lax.while_loop(cond, cycle, (x, corrections, jnp.asarray(0)))[0]


@jax.jit
def _working_set(Ak):
    """One SVD supplies both the nullspace projection and KKT multipliers."""
    u, singular, vh = jnp.linalg.svd(Ak, full_matrices=True)
    cutoff = jnp.finfo(Ak.dtype).eps * max(Ak.shape) * singular[0]
    rank = jnp.sum(singular > cutoff)
    tangent = vh.T * (jnp.arange(Ak.shape[1]) >= rank)
    # Match JAX pinv's default relative cutoff for the multiplier solve.
    keep = singular > 10 * cutoff
    reciprocal = jnp.where(keep, 1 / jnp.where(keep, singular, 1.), 0.)
    multiplier_map = -(u[:, :singular.size] * reciprocal) @ vh[:singular.size]
    return tangent, multiplier_map


@jax.jit
def _project_gradient(g, tangent, multiplier_map):
    d = -tangent @ (tangent.T @ g)
    return d, multiplier_map @ g, jnp.linalg.norm(d)


@jax.jit
def _active_mask(A, b, x, tol):
    return jnp.abs(A @ x - b) <= tol


@jax.jit
def _step_limit(A, b, x, d, active):
    denom = A @ d
    eligible = (denom > jnp.finfo(x.dtype).eps) & ~active
    ratios = jnp.where(eligible, (b - A @ x) / jnp.where(eligible, denom, 1.), jnp.inf)
    return jnp.minimum(1., jnp.min(ratios, initial=jnp.inf))


class constrained:
    def __init__(self, parameters, unconstrained):
        self.params = parameters
        self.unconstrained = unconstrained

    def fmincon(self, fun, x0, A=None, b=None, Aeq=None, beq=None,
                threshold=1e-6, vectorized=False, *, max_iter=None,
                constraint_tol=1e-7, feasibility_max_iter=2000):
        """Projected gradient with a working set for Ax<=b and Aeq x=beq.

        Finds a feasible start by bounded projections. Status 4 means phase I
        failed to find feasibility within its budget, not a proof of infeasibility.
        No implicit nonnegativity bound is imposed on x.
        """
        if self.params["fmincon"]["method"] != "projected-gradient":
            raise ValueError("fmincon supports method='projected-gradient'")
        if threshold <= 0 or constraint_tol <= 0 or feasibility_max_iter < 1:
            raise ValueError("Tolerances and feasibility_max_iter must be positive")
        max_iter = self.params["fmincon"]["params"]["projected-gradient"]["max_iter"] if max_iter is None else max_iter
        if max_iter < 0 or int(max_iter) != max_iter:
            raise ValueError("max_iter must be a nonnegative integer")
        x = as_vector(x0)
        if not bool(jnp.all(jnp.isfinite(x))):
            raise ValueError("x0 must be finite")
        A, b = _constraints(A, b, x, "A")
        Aeq, beq = _constraints(Aeq, beq, x, "Aeq")
        tol = math.sqrt(threshold)
        x = _feasible(x, A, b, Aeq, beq, constraint_tol, int(feasibility_max_iter))
        objective = jax.jit(fun)
        jac_options = dict(self.params["jacobian"])
        derivative = (jax.value_and_grad(fun) if jac_options.get('algorithm', 'autodiff') == 'autodiff'
                      else lambda y: (fun(y), jacobian(fun, y, **jac_options)))
        value_gradient = jax.jit(derivative)
        line_step = jax.jit(lambda y, d, cap, g, f: backtracking(
            fun, y, d, alpha=cap, gradient=g, value=f))
        active = {i for i, flag in enumerate(_active_mask(A, b, x, constraint_tol).tolist()) if flag}
        xs, fs = ([x], [objective(x)]) if vectorized else ([], [])
        previous_set, factors = None, None
        status, iters, lsiters = 1, 0, 0
        projected_norm = jnp.asarray(jnp.inf, dtype=x.dtype)
        if float(_violation(x, A, b, Aeq, beq)) > constraint_tol:
            status = 4
        else:
            for _ in range(int(max_iter) + 1):
                fx, g = value_gradient(x)
                if not bool(jnp.isfinite(fx) & jnp.all(jnp.isfinite(g))):
                    status = 3
                    break
                # Remove inequalities with negative KKT multipliers when stuck.
                for _ in range(len(active) + 1):
                    indices = sorted(active)
                    key = tuple(indices)
                    if key != previous_set:
                        Ak = jnp.concatenate((A[jnp.asarray(indices, dtype=jnp.int32)], Aeq), axis=0)
                        factors = _working_set(Ak) if Ak.shape[0] else None
                        active_rows = jnp.zeros(A.shape[0], dtype=bool).at[jnp.asarray(indices, dtype=jnp.int32)].set(True)
                        previous_set = key
                    if factors is not None:
                        d, multipliers, projected_norm = _project_gradient(g, *factors)
                    else:
                        multipliers, d = jnp.empty((0,), dtype=x.dtype), -g
                        projected_norm = jnp.linalg.norm(d)
                    if float(projected_norm) <= tol and indices and float(jnp.min(multipliers[:len(indices)])) < -tol:
                        active.remove(indices[int(jnp.argmin(multipliers[:len(indices)]))])
                    else:
                        break
                if float(projected_norm) <= tol:
                    status = 0
                    break
                if iters >= max_iter:
                    break
                cap = _step_limit(A, b, x, d, active_rows)
                ls = line_step(x, d, cap, g, fx)
                lsiters += int(ls["iterations"])
                if not bool(ls["success"]) or bool(jnp.all(ls["x"] == x)):
                    status = 2
                    break
                x = ls["x"]
                active.update(i for i, flag in enumerate(_active_mask(A, b, x, constraint_tol).tolist()) if flag)
                iters += 1
                if vectorized:
                    xs.append(x)
                    fs.append(ls['f'])
        violation = _violation(x, A, b, Aeq, beq)
        success = status == 0 and float(violation) <= constraint_tol
        return {"x": jnp.stack(xs) if vectorized else x,
                "f": jnp.stack(fs) if vectorized else objective(x),
                "iterations": iters, "ls_iterations": lsiters, "success": success,
                "status": status if not (status == 0 and not success) else 4,
                "constraint_violation": violation, "projected_gradient_norm": projected_norm}

    def fminnlcon(self, fun, x0, g, c=1., beta=10., threshold=1e-6,
                  vectorized=False, *, max_iter=None, inner_max_iter=1000, inner_options=None):
        """Quadratic penalty or reciprocal/log barrier for g_i(x)<=0.

        Barrier methods require a strictly feasible starting point. Outer and
        inner iteration counts are bounded. Nonlinear solves are local and
        experimental. Success checks feasibility, stationarity, and complementarity.
        ``inner_options`` overrides the inner ``minimize`` method, line search,
        derivatives and other options. Defaults retain modified Newton. Inner
        derivative/direction callbacks receive ``(x, weight)`` / ``(x, g, weight)``.
        """
        method = self.params["fminnlcon"]["method"]
        if method not in ("penalty", "barrier", "log-barrier"):
            raise ValueError(f"Unknown nonlinear constraint method: {method}")
        if c <= 0 or beta <= 1 or threshold <= 0:
            raise ValueError("Require c>0, beta>1, and threshold>0")
        max_iter = self.params["fminnlcon"]["params"][method]["max_iter"] if max_iter is None else max_iter
        if max_iter < 0 or int(max_iter) != max_iter or inner_max_iter < 0:
            raise ValueError("Iteration budgets must be nonnegative integers")
        constraints = tuple(g)
        x = as_vector(x0)
        values = lambda y: jnp.stack([jnp.asarray(fn(y)) for fn in constraints]) if constraints else jnp.empty((0,), dtype=y.dtype)
        gx = values(x)
        if gx.ndim != 1:
            raise ValueError("Each inequality constraint must return a scalar")
        if not bool(jnp.all(jnp.isfinite(gx))):
            raise ValueError("Initial constraint values must be finite")
        if method != "penalty" and not bool(jnp.all(gx < 0)):
            raise ValueError("Barrier methods require a strictly feasible x0")
        tol = math.sqrt(threshold)
        xs, fs, cs, errors = ([x], [fun(x)], [c], []) if vectorized else ([], [], [], [])
        status, total, lsiters, outer = 1, 0, 0, 0
        error = jnp.asarray(jnp.inf, dtype=x.dtype)

        def weighted(y, weight):
            residual = values(y)
            if method == "penalty":
                return fun(y) + weight / 2 * jnp.sum(jnp.maximum(residual, 0) ** 2)
            safe = jnp.minimum(residual, -jnp.finfo(y.dtype).tiny)
            term = -jnp.log(-safe) if method == "log-barrier" else -1 / safe
            return jnp.where(jnp.all(residual < 0), fun(y) + jnp.sum(term) / weight, jnp.inf)

        inner_settings = dict(method="modified-newton", tol=tol * 0.1, max_iter=int(inner_max_iter))
        inner_settings.update(inner_options or {})
        inner_solve = compile_minimizer(weighted, **inner_settings)

        @jax.jit
        def kkt_error(y, weight):
            gx, pullback = jax.vjp(values, y)
            if method == "penalty":
                multipliers = weight * jnp.maximum(gx, 0)
            elif method == "log-barrier":
                multipliers = -1 / (weight * gx)
            else:
                multipliers = 1 / (weight * gx ** 2)
            # KKT needs J.T @ multipliers, not the full constraint Jacobian.
            stationarity = jax.grad(fun)(y) + pullback(multipliers)[0]
            violation = jnp.max(jnp.maximum(gx, 0), initial=0)
            complementarity = jnp.max(jnp.abs(multipliers * gx), initial=0)
            return jnp.maximum(violation, jnp.maximum(jnp.linalg.norm(stationarity), complementarity))

        for outer in range(1, int(max_iter) + 1):
            weight = jnp.asarray(c, dtype=x.dtype)
            sol = inner_solve(x, weight)
            x = sol["x"]
            total += int(sol["iterations"])
            lsiters += int(sol["ls_iterations"])
            error = kkt_error(x, weight)
            if vectorized:
                xs.append(x)
                fs.append(fun(x))
                cs.append(c)
                errors.append(error)
            if bool(jnp.isfinite(error)) and float(error) <= tol:
                status = 0
                break
            if not bool(jnp.isfinite(error)) or int(sol["status"]) == 3:
                status = 3
                break
            if outer < max_iter:
                c *= beta
        return {"x": jnp.stack(xs) if vectorized else x,
                "f": jnp.asarray(fs) if vectorized else fun(x),
                "c": jnp.asarray(cs) if vectorized else c,
                "err": jnp.asarray(errors) if vectorized else error,
                "iterations": outer, "inner_iterations": total, "ls_iterations": lsiters,
                "constraint_violation": jnp.max(jnp.maximum(values(x), 0), initial=0),
                "success": status == 0, "status": status}

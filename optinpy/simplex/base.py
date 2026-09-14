"""Optinpy's two-phase tableau simplex with JAX array arithmetic.

Pivot selection and iteration run in compiled JAX phases. Python manages
bound transformations and phase-I cleanup. No external LP solver is used.
"""
import math
from functools import partial

import jax
import jax.numpy as jnp


@jax.jit
def _finite_problem(A, b, c):
    return jnp.all(jnp.isfinite(A)) & jnp.all(jnp.isfinite(b)) & jnp.all(jnp.isfinite(c))


@jax.jit
def _invalid_bounds(lb, ub):
    return jnp.any(jnp.isnan(lb) | jnp.isnan(ub) | (lb > ub) | (lb == jnp.inf) | (ub == -jnp.inf))


@jax.jit
def _pivot_tableau(tableau, i, j):
    """One compiled Gauss-Jordan pivot; keep the basis column exact."""
    row = tableau[i] / tableau[i, j]
    pivot_row = jnp.arange(tableau.shape[0])[:, None] == i
    pivot_column = jnp.arange(tableau.shape[1])[None, :] == j
    # Broadcast selections fuse into the elimination pass instead of updating
    # the full tableau twice with separate row/column scatters.
    updated = jnp.where(pivot_row, row, tableau - tableau[:, j, None] * row)
    return jnp.where(pivot_column, pivot_row.astype(tableau.dtype), updated)


@jax.jit
def _objective_tableau(tableau, basic, cost):
    """Recompute reduced costs and their cancellation scales from the basis."""
    basic_cost = cost[basic]
    row = jnp.concatenate((cost, jnp.zeros(1, dtype=cost.dtype)))
    row = (row - basic_cost @ tableau[:-1]).at[basic].set(0.)
    magnitude = jnp.abs(cost) + jnp.abs(basic_cost) @ jnp.abs(tableau[:-1, :-1])
    return tableau.at[-1].set(row), magnitude


@jax.jit
def _reduced_tolerance(tableau, basic, cost, magnitude, tol):
    """Account for roundoff left in theoretically zero pivoted coefficients.

    Relative cancellation scaling alone treats a residual column coefficient
    as an exact small improvement. Estimate its rounding scale using the basic
    costs and the full column, including rows whose basic cost is zero. Allow
    for accumulation across the basis, rather than one arithmetic operation.
    With zero basic costs this floor is zero, preserving small genuine costs.
    """
    column_scale = jnp.max(jnp.abs(tableau[:-1, :-1]), axis=0, initial=0.)
    eps = jnp.finfo(tableau.dtype).eps
    accumulation = 2 * max(1, basic.size) * eps
    roundoff = accumulation * jnp.sum(jnp.abs(cost[basic])) * column_scale
    return jnp.maximum(tol * magnitude, roundoff)


@partial(jax.jit, static_argnames=('pricing',))
def _entering_column(tableau, improving, force_bland, *, pricing):
    bland = jnp.argmax(improving)
    if pricing == 'bland':
        return bland
    score = tableau[-1, :-1]
    if pricing == 'steepest-edge':
        score = score / jnp.sqrt(1. + jnp.sum(tableau[:-1, :-1] ** 2, axis=0))
    selected = jnp.argmin(jnp.where(improving, score, jnp.inf))
    return jnp.where(force_bland, bland, selected)


@partial(jax.jit, static_argnames=('pricing',))
def _primal_phase(tableau, basic, cost, tol, budget, stalled=0, force_bland=False, *, pricing='dantzig'):
    """Run a fixed-shape phase without a Python/device round trip per pivot.

    Internal status -1 continues, 0 means phase optimum, 1 exhausts budget,
    2 finds an unbounded ray, and 4 reports numerical failure. Budget and
    all numerical data are dynamic inputs, so changing them reuses the JIT.
    """
    def step(state):
        tableau, basic, pivots, _, stalled, force_bland = state
        invalid = (~jnp.all(jnp.isfinite(tableau)) |
                   jnp.any(tableau[:-1, -1] < -tol))
        tableau, magnitude = _objective_tableau(tableau, basic, cost)
        invalid = invalid | ~jnp.all(jnp.isfinite(tableau))
        nonbasic = jnp.ones(cost.shape, dtype=bool).at[basic].set(False)
        threshold = _reduced_tolerance(tableau, basic, cost, magnitude, tol)
        improving = nonbasic & (tableau[-1, :-1] < -threshold)
        # Empty decision/basis arrays occur for unconstrained and all-fixed LPs.
        if not cost.size or not basic.size:
            status = jnp.where(invalid, 4, jnp.where(jnp.any(improving), 2, 0))
            return tableau, basic, pivots, status, stalled, force_bland
        entering = _entering_column(tableau, improving, force_bland, pricing=pricing).astype(basic.dtype)
        column, rhs = tableau[:-1, entering], tableau[:-1, -1]
        eligible = column > tol
        ratios = jnp.where(eligible, jnp.maximum(rhs, 0.) / jnp.where(eligible, column, 1.), jnp.inf)
        best = jnp.min(ratios)
        # True minimum ratios only; Bland ordering breaks exact ties.
        leaving = jnp.argmin(jnp.where(eligible & (ratios == best), basic, cost.size))
        status = jnp.where(invalid, 4,
                           jnp.where(~jnp.any(improving), 0,
                                     jnp.where(~jnp.any(eligible), 2,
                                               jnp.where(pivots >= budget, 1, -1))))

        def pivot(operands):
            tableau, basic, pivots, stalled, force_bland = operands
            stalled = jnp.where(best <= tol, stalled + 1, 0)
            force_bland = force_bland | (stalled >= 8)
            return (_pivot_tableau(tableau, leaving, entering),
                    basic.at[leaving].set(entering), pivots + 1, stalled, force_bland)

        tableau, basic, pivots, stalled, force_bland = jax.lax.cond(
            status == -1, pivot, lambda operands: operands, (tableau, basic, pivots, stalled, force_bland))
        return tableau, basic, pivots, status, stalled, force_bland

    state = (tableau, basic, jnp.asarray(0), jnp.asarray(-1), jnp.asarray(stalled), jnp.asarray(force_bland))
    return jax.lax.while_loop(lambda state: state[3] == -1, step, state)


@jax.jit
def _prepare_problem(A, b, c, column_rows, column_signs, shift, upper_indices, upper_bounds):
    """Build and equilibrate the bounded standard form in one dispatch."""
    nv = column_rows.size
    transform = jnp.zeros((c.size, nv), dtype=c.dtype).at[column_rows, jnp.arange(nv)].set(column_signs)
    matrix, rhs = A[:, column_rows] * column_signs, b - A @ shift
    upper = jnp.eye(nv, dtype=c.dtype)[upper_indices]
    matrix, rhs = jnp.concatenate((matrix, upper)), jnp.concatenate((rhs, upper_bounds))
    # Nonnegative rows imply safe upper ranges for nonnegative variables.
    eligible = (jnp.all(matrix >= 0, axis=1) & (rhs >= 0))[:, None] & (matrix > 0)
    ranges = jnp.min(jnp.where(eligible, rhs[:, None] / jnp.where(eligible, matrix, 1.),
                               jnp.inf), axis=0, initial=jnp.inf)
    range_scale = jnp.where(jnp.isfinite(ranges) & (ranges > 0), ranges, 1.)
    matrix, transform = matrix * range_scale, transform * range_scale
    row_norm = jnp.max(jnp.abs(matrix), axis=1, initial=0.)
    impossible = jnp.any((row_norm == 0) & (rhs < 0))
    row_scale = jnp.where(row_norm > 0, row_norm, 1.)
    matrix, rhs = matrix / row_scale[:, None], rhs / row_scale
    column_norm = jnp.max(jnp.abs(matrix), axis=0, initial=0.)
    raw_cost = jnp.abs(transform.T @ c)
    unbounded_scale = jnp.where(raw_cost > 0, raw_cost, 1.)
    column_scale = 1. / jnp.where(column_norm > 0, column_norm, unbounded_scale)
    matrix, transform = matrix * column_scale, transform * column_scale
    cost_norm = jnp.max(jnp.abs(transform.T @ c), initial=0.)
    cost_scale = jnp.where(cost_norm > 0, cost_norm, 1.)
    decision_cost = transform.T @ (c / cost_scale)
    valid = (jnp.all(jnp.isfinite(matrix)) & jnp.all(jnp.isfinite(rhs)) &
             jnp.all(jnp.isfinite(transform)) & jnp.isfinite(cost_scale) &
             jnp.all(jnp.isfinite(decision_cost)))
    return matrix, rhs, transform, row_scale, cost_scale, decision_cost, impossible, valid


@jax.jit
def _initial_tableau(matrix, rhs, negative, decision_cost):
    nr, nv = matrix.shape
    sign = jnp.where(rhs < 0, -1., 1.)
    rows = jnp.concatenate((sign[:, None] * matrix, jnp.diag(sign),
                            jnp.eye(nr, dtype=matrix.dtype)[:, negative]), axis=1)
    basic = jnp.arange(nv, nv + nr, dtype=jnp.int32)
    artificial = jnp.arange(nv + nr, rows.shape[1], dtype=jnp.int32)
    basic = basic.at[negative].set(artificial)
    tableau = jnp.concatenate((jnp.concatenate((rows, jnp.abs(rhs)[:, None]), axis=1),
                               jnp.zeros((1, rows.shape[1] + 1), dtype=matrix.dtype)))
    cost = jnp.zeros(rows.shape[1], dtype=matrix.dtype)
    if negative.size:
        cost = cost.at[artificial].set(1.)
    else:
        cost = cost.at[:nv].set(decision_cost)
    tableau, _ = _objective_tableau(tableau, basic, cost)
    return tableau, cost


@jax.jit
def _solution_metrics(tableau, basic, transform, shift, A, b, c, lb, ub):
    values = jnp.zeros(tableau.shape[1] - 1, dtype=c.dtype).at[basic].set(tableau[:-1, -1])
    x = shift + transform @ values[:transform.shape[1]]
    f = c @ x
    row_norm = jnp.max(jnp.abs(A), axis=1, initial=0.)
    residual = jnp.maximum(A @ x - b, 0.)
    bound_residual = jnp.maximum(jnp.maximum(lb - x, x - ub), 0.)
    absolute = jnp.maximum(jnp.max(residual, initial=0.), jnp.max(bound_residual, initial=0.))
    scaled = jnp.maximum(jnp.max(residual / jnp.where(row_norm > 0, row_norm, 1.), initial=0.),
                         jnp.max(bound_residual, initial=0.))
    valid = jnp.all(jnp.isfinite(x)) & jnp.isfinite(f) & jnp.isfinite(scaled)
    return x, f, absolute, scaled, valid


@partial(jax.jit, static_argnames=('columns',))
def _drop_artificials(tableau, basic, decision_cost, *, columns):
    """Remove the appended artificial columns and rebuild phase-II costs."""
    tableau = jnp.concatenate((tableau[:, :columns], tableau[:, -1:]), axis=1)
    cost = jnp.pad(decision_cost, (0, columns - decision_cost.size))
    tableau, _ = _objective_tableau(tableau, basic, cost)
    return tableau, cost


class simplex:
    """Minimize (or maximize) c @ x subject to A @ x <= b and lb <= x <= ub.

    Bounds default to 0 <= x < infinity. Infinite bounds and fixed variables
    are supported. ``solve`` runs both phases. ``primal`` makes one primal
    step in the current phase; ``dual`` makes one dual-feasible pivot.
    Rows and objectives are normalized before pivoting. Results use status
    0 optimal, 1 budget exhausted, 2 unbounded, 3 infeasible, 4 numerical failure.
    ``tol`` applies to the normalized problem; 64-bit JAX is recommended.
    ``pricing`` selects 'dantzig' (default), 'steepest-edge', or 'bland'.
    Eight consecutive near-zero primal steps permanently activate Bland
    pricing for this instance, including subsequent calls to ``solve``.
    """
    def __init__(self, A, b, c, mode='min', *, lb=None, ub=None, tol=None, pricing='dantzig'):
        if pricing not in ('bland', 'dantzig', 'steepest-edge'):
            raise ValueError("pricing must be 'bland', 'dantzig', or 'steepest-edge'")
        self.pricing = pricing
        self._stalled, self._force_bland = 0, False
        self.c = jnp.asarray(c, dtype=float)
        if self.c.ndim != 1 or not self.c.size:
            raise ValueError("c must be a nonempty vector")
        self.m = self.c.size
        self.A = jnp.asarray(A, dtype=self.c.dtype)
        if self.A.shape == (0,):
            self.A = jnp.empty((0, self.m), dtype=self.c.dtype)
        self.b = jnp.asarray(b, dtype=self.c.dtype)
        if self.A.ndim != 2 or self.A.shape[1] != self.m or self.b.shape != (self.A.shape[0],):
            raise ValueError("A must have shape (n, len(c)) and b shape (n,)")
        if not bool(_finite_problem(self.A, self.b, self.c)):
            raise ValueError("A, b, and c must be finite")
        if mode not in ('min', 'max'):
            raise ValueError("mode must be 'min' or 'max'")
        self.mode, self.z, self.n = mode, 1 if mode == 'min' else -1, self.A.shape[0]
        self.tol = max(1e-9, 10 * float(jnp.finfo(self.c.dtype).eps)) if tol is None else tol
        if not math.isfinite(self.tol) or self.tol <= 0:
            raise ValueError("tol must be finite and positive")
        self.lb = jnp.broadcast_to(jnp.asarray(0. if lb is None else lb, dtype=self.c.dtype), (self.m,))
        self.ub = jnp.broadcast_to(jnp.asarray(jnp.inf if ub is None else ub, dtype=self.c.dtype), (self.m,))
        if bool(_invalid_bounds(self.lb, self.ub)):
            raise ValueError("Bounds must satisfy lb <= ub with valid infinity signs")
        columns, signs, shift, upper_rows = [], [], [], []
        for i, (low, high) in enumerate(zip(self.lb.tolist(), self.ub.tolist())):
            if low == high:
                # Fixed coordinates must not pollute objective scaling with
                # coefficients that cannot affect the solution.
                shift.append(low)
            elif math.isfinite(low):
                shift.append(low)
                columns.append(i)
                signs.append(1.)
                if math.isfinite(high):
                    upper_rows.append((len(columns) - 1, high - low))
            elif math.isfinite(high):
                shift.append(high)
                columns.append(i)
                signs.append(-1.)
            else:
                shift.append(0.)
                columns.extend((i, i))
                signs.extend((1., -1.))
        self._shift = jnp.asarray(shift, dtype=self.c.dtype)
        self._nv = len(columns)
        matrix, rhs, self._transform, self._row_scale, self._cost_scale, decision_cost, impossible, valid = _prepare_problem(
            self.A, self.b, self.c, jnp.asarray(columns, dtype=jnp.int32),
            jnp.asarray(signs, dtype=self.c.dtype), self._shift,
            jnp.asarray([i for i, _ in upper_rows], dtype=jnp.int32),
            jnp.asarray([bound for _, bound in upper_rows], dtype=self.c.dtype))
        self._decision_cost = self.z * decision_cost
        nr = len(rhs)
        negative = [i for i, value in enumerate(rhs.tolist()) if value < 0]
        self.basic = list(range(self._nv, self._nv + nr))
        self._artificial = list(range(self._nv + nr, self._nv + nr + len(negative)))
        for i, column in zip(negative, self._artificial):
            self.basic[i] = column
        self.tableau, self._cost = _initial_tableau(matrix, rhs, jnp.asarray(negative, dtype=jnp.int32), self._decision_cost)
        self.phase = 1 if negative else 2
        self.status, self.iterations = (3 if bool(impossible) else None if bool(valid) else 4), 0
        self._pivot_blocked = False

    @property
    def non_basic(self):
        return [i for i in range(self.tableau.shape[1] - 1) if i not in self.basic]

    @property
    def pivot_matrix(self):
        return self.tableau

    @property
    def delta_x(self):
        values = jnp.zeros(self.tableau.shape[1] - 1, dtype=self.c.dtype)
        if self.basic:
            values = values.at[jnp.asarray(self.basic)].set(self.tableau[:-1, -1])
        return values

    @property
    def x(self):
        return self._shift + self._transform @ self.delta_x[:self._nv]

    def _objective(self, cost):
        self._cost = cost
        self.tableau, magnitude = _objective_tableau(
            self.tableau, jnp.asarray(self.basic, dtype=jnp.int32), cost)
        return magnitude

    def pivot(self, i, j):
        if not 0 <= i < len(self.basic) or not 0 <= j < self.tableau.shape[1] - 1:
            raise IndexError("Invalid pivot index")
        value = float(self.tableau[i, j])
        if not math.isfinite(value) or abs(value) <= self.tol:
            raise ValueError("Pivot element is nonfinite or numerically zero")
        if j in self.basic and self.basic[i] != j:
            raise ValueError("An entering variable must be nonbasic")
        self.tableau = _pivot_tableau(self.tableau, i, j)
        self.basic[i] = j
        self.iterations += 1
        return self.tableau

    def _artificial_mass(self):
        # Read artificial basic values directly instead of relying on an
        # accumulated objective row, which can suffer cancellation.
        indices = [i for i, j in enumerate(self.basic) if j in self._artificial]
        if not indices:
            return 0.
        rhs = self.tableau[:-1, -1].tolist()
        return sum(max(rhs[i], 0.) for i in indices)

    def _phase_two(self, allow_pivot=True):
        if self._artificial_mass() > self.tol:
            self.status = 3
            return
        for i in reversed(range(len(self.basic))):
            if self.basic[i] in self._artificial:
                row = self.tableau[i, :-1].tolist()
                choices = [j for j in self.non_basic if j not in self._artificial and abs(row[j]) > self.tol]
                if choices:
                    if allow_pivot:
                        self.pivot(i, choices[0])
                    else:
                        self._pivot_blocked = True
                else:
                    self.tableau = jnp.delete(self.tableau, i, axis=0)
                    self.basic.pop(i)
                # One cleanup operation per call keeps phase-I pivots inside
                # the same iteration budget as ordinary primal pivots.
                return
        # Artificial columns are appended at construction and never reordered.
        # All remaining basic indices therefore retain their original numbers.
        self.tableau, self._cost = _drop_artificials(
            self.tableau, jnp.asarray(self.basic, dtype=jnp.int32),
            self._decision_cost, columns=self._artificial[0])
        self._artificial, self.phase = [], 2

    def primal(self, *, _allow_pivot=True):
        """One primal pivot under the selected pricing rule, or a phase operation."""
        if self.status is not None:
            return self.tableau
        if bool(jnp.any(self.tableau[:-1, -1] < -self.tol)):
            raise ValueError("Primal pivot requires primal feasibility")
        # Recompute reduced costs from the current basis. A single absolute
        # threshold would ignore small but uncancelled improving coefficients
        # beside a large positive cost. Scale each tolerance by the terms in
        # its own reduced-cost calculation to account for cancellation.
        cost_magnitude = self._objective(self._cost)
        reduced_tolerance = _reduced_tolerance(
            self.tableau, jnp.asarray(self.basic, dtype=jnp.int32),
            self._cost, cost_magnitude, self.tol).tolist()
        reduced = self.tableau[-1, :-1].tolist()
        choices = [j for j in self.non_basic if reduced[j] < -reduced_tolerance[j]]
        if not choices:
            if self.phase == 1:
                self._phase_two(_allow_pivot)
            else:
                self.status = 0
            return self.tableau
        improving = jnp.zeros(len(reduced), dtype=bool).at[jnp.asarray(choices)].set(True)
        j = int(_entering_column(self.tableau, improving, self._force_bland, pricing=self.pricing))
        column, rhs = self.tableau[:-1, j].tolist(), self.tableau[:-1, -1].tolist()
        candidates = [i for i in range(len(self.basic)) if column[i] > self.tol]
        if not candidates:
            if self.phase == 1 and self._artificial_mass() <= self.tol:
                # Zero auxiliary objective certifies phase-I optimality even
                # when roundoff leaves an apparently improving reduced cost.
                self._phase_two(_allow_pivot)
            else:
                self.status = 4 if self.phase == 1 else 2
            return self.tableau
        ratios = {i: max(rhs[i], 0.) / column[i] for i in candidates}
        best = min(ratios.values())
        # An additive tolerance on the ratio may select a step *past* the
        # tightest bound and produce arbitrarily large infeasibility.
        i = min((i for i in candidates if ratios[i] == best), key=lambda i: self.basic[i])
        if not _allow_pivot:
            self._pivot_blocked = True
            return self.tableau
        self._stalled = self._stalled + 1 if best <= self.tol else 0
        self._force_bland = self._force_bland or self._stalled >= 8
        return self.pivot(i, j)

    def dual(self):
        """One dual-simplex pivot; requires nonnegative reduced costs."""
        if bool(jnp.any(self.tableau[-1, :-1] < -self.tol)):
            raise ValueError("Dual pivot requires dual feasibility")
        rows = [i for i in range(len(self.basic)) if float(self.tableau[i, -1]) < -self.tol]
        if not rows:
            return self.tableau
        i = min(rows, key=lambda i: self.basic[i])
        choices = [j for j in self.non_basic if float(self.tableau[i, j]) < -self.tol]
        if not choices:
            self.status = 3
            return self.tableau
        j = min(choices, key=lambda j: (float(self.tableau[-1, j] / -self.tableau[i, j]), j))
        self.status = None
        return self.pivot(i, j)

    def solve(self, max_iter=1000):
        """Solve with at most max_iter additional pivots; interrupted solves resume.

        Optimality checks and phase transitions without pivots do not consume
        budget. A solution reached on the last allowed pivot is successful.
        ``primal_violation`` reports absolute residuals in original units;
        ``scaled_primal_violation`` normalizes constraint rows and leaves
        variable-bound residuals in the original variable units.
        """
        if not math.isfinite(max_iter) or max_iter < 0 or int(max_iter) != max_iter:
            raise ValueError("max_iter must be a nonnegative integer")
        start = self.iterations
        limit = start + int(max_iter)
        while self.status is None:
            self.tableau, basic, pivots, phase_status, stalled, force_bland = _primal_phase(
                self.tableau, jnp.asarray(self.basic, dtype=jnp.int32),
                self._cost, self.tol, limit - self.iterations, self._stalled, self._force_bland, pricing=self.pricing)
            basic, pivots, phase_status, stalled, force_bland = jax.device_get((basic, pivots, phase_status, stalled, force_bland))
            self.basic = basic.tolist()
            self._stalled, self._force_bland = int(stalled), bool(force_bland)
            self.iterations += int(pivots)
            if phase_status == 1:
                break
            if self.phase == 1 and phase_status == 2 and self._artificial_mass() <= self.tol:
                # The auxiliary objective has lower bound zero. Once its
                # artificial mass is zero, a roundoff-sized reduced cost with
                # no eligible pivot cannot invalidate that feasibility proof.
                phase_status = 0
            if phase_status != 0:
                # A phase-I objective is bounded below by zero. Its apparent
                # unbounded ray is a numerical failure, not a primal certificate.
                self.status = 4 if self.phase == 1 and phase_status == 2 else int(phase_status)
                break
            if self.phase == 2:
                self.status = 0
                break
            self._pivot_blocked = False
            self._phase_two(allow_pivot=self.iterations < limit)
            if self._pivot_blocked:
                break
        x, f, absolute, scaled, valid = _solution_metrics(
            self.tableau, jnp.asarray(self.basic, dtype=jnp.int32), self._transform,
            self._shift, self.A, self.b, self.c, self.lb, self.ub)
        if self.status == 0 and (not bool(valid) or float(scaled) > self.tol):
            self.status = 4
        status = 1 if self.status is None else self.status
        messages = {0: "Optimal solution found", 1: "Pivot budget exhausted",
                    2: "Problem is unbounded", 3: "Problem is infeasible",
                    4: "Numerical failure; inspect scaling and use 64-bit arithmetic"}
        return {"x": x, "f": f, "iterations": self.iterations,
                "iterations_this_solve": self.iterations - start,
                "pricing": self.pricing, "bland_fallback": self._force_bland,
                "success": status == 0, "status": status, "message": messages[status],
                "primal_violation": absolute, "scaled_primal_violation": scaled}

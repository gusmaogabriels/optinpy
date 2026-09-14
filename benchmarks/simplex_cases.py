"""Deterministic LP fixtures shared by regression tests and the comparison CLI."""
from dataclasses import dataclass

import numpy as np


@dataclass
class LP:
    name: str
    A: np.ndarray
    b: np.ndarray
    c: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    mode: str = 'min'
    expected_status: str = 'optimal'
    expected_objective: float | None = None
    category: str = 'edge'


def case(name, A, b, c, *, lb=0., ub=np.inf, mode='min',
         status='optimal', objective=None, category='edge'):
    c = np.asarray(c, dtype=float)
    return LP(name, np.asarray(A, dtype=float).reshape(-1, c.size),
              np.asarray(b, dtype=float), c,
              np.broadcast_to(lb, c.shape).astype(float).copy(),
              np.broadcast_to(ub, c.shape).astype(float).copy(),
              mode, status, objective, category)


def legacy_flow_case():
    """Original README network, including Big-M artificial root arcs."""
    arcs = [(1, 0, 20), (1, 2, 1), (1, 3, 8), (1, 4, 1),
            (2, 0, 20), (2, 3, 2), (3, 0, 20), (3, 4, 1), (3, 5, 4),
            (0, 4, 20), (0, 5, 20), (4, 5, 12), (5, 2, 7)]
    incidence = np.array([[float(u == node)-float(v == node) for u, v, _ in arcs]
                          for node in range(1, 6)])
    balances = np.array([10., 4., 0., -6., -8.])
    return case('legacy_minimum_cost_flow', np.vstack((incidence, -incidence)),
                np.concatenate((balances, -balances)), [cost for _, _, cost in arcs], objective=58.)


def curated_cases():
    return [
        case('legacy_bound_example', [[-3,2,-4,-5],[4,-2,5,3],[2,4,1,2],[3,2,-2,4]],
             [-10,10,10,15], [-2,-2,-3,-3], lb=-10., ub=5.),
        case('negative_lower_bound', [], [], [1.], lb=-5., ub=3., objective=-5.),
        case('upper_only_bound', [], [], [-1.], lb=-np.inf, ub=3., objective=-3.),
        case('fixed_variable', [], [], [2.], lb=2., ub=2., objective=4.),
        case('free_variable', [[1.],[-1.]], [-1.,2.], [1.], lb=-np.inf, objective=-2.),
        case('phase_one', [[-1.],[1.]], [-1.,3.], [1.], objective=1.),
        case('redundant_equalities', [[1.,1.],[-1.,-1.],[2.,2.]], [1.,-1.,2.], [1.,2.], objective=1.),
        case('multiple_optima', [[1.,1.]], [1.], [-1.,-1.], objective=-1.),
        case('maximization', [[1.,1.],[1.,0.]], [4.,2.], [3.,2.], mode='max', objective=10.),
        case('zero_objective', [[1.,1.]], [1.], [0.,0.], objective=0.),
        case('zero_row', [[0.],[1.]], [0.,1.], [-1.], objective=-1.),
        case('contradictory_zero_row', [[0.]], [-1.], [0.], status='infeasible'),
        case('infeasible', [[1.],[-1.]], [1.,-2.], [1.], status='infeasible'),
        case('infeasible_fixed', [[-1.]], [-1.], [1.], lb=0., ub=0., status='infeasible'),
        case('infeasible_before_unbounded', [[0.,1.],[0.,-1.]], [1.,-2.], [-1.,0.], status='infeasible'),
        case('unbounded', [], [], [-1.], status='unbounded'),
        case('unbounded_free', [[1.,0.]], [1.], [0.,-1.], lb=[0.,-np.inf], status='unbounded'),
        case('near_ratio_tie', [[1.],[1e6]], [1.+5e-10,1e6], [-1.], objective=-1.),
        case('small_row_units', [[1e-12]], [1e-12], [-1.], objective=-1.),
        case('small_objective_units', [[1.]], [1.], [-1e-12], objective=-1e-12),
        case('large_row_units', [[1e12]], [1e12], [-1.], objective=-1.),
        case('cycling_beale', [[.5,-5.5,-2.5,9.],[.5,-1.5,-.5,1.],[1.,0.,0.,0.]],
             [0.,0.,1.], [10.,-57.,-9.,-24.], mode='max', objective=1.),
        case('klee_minty_3', [[1.,0.,0.],[20.,1.,0.],[200.,20.,1.]],
             [1.,100.,10000.], [100.,10.,1.], mode='max', objective=10000.),
        legacy_flow_case(),
    ]


def random_cases(count=80, seed=20260913):
    """Box-bounded dense LPs with mixed explicit bounds and a known feasible point."""
    rng = np.random.default_rng(seed)
    result = []
    for k in range(count):
        n = (4, 8, 16, 24)[k % 4]
        low = rng.uniform(-4., 0., n)
        high = low + rng.uniform(.2, 6., n)
        witness = rng.uniform(low, high)
        A = rng.normal(size=(2*n, n))
        b = np.einsum('ij,j->i', A, witness) + rng.uniform(0., 3., 2*n)
        # Encoding the same box in A guarantees boundedness even when explicit
        # bounds are free, one-sided, or fixed. It also exercises redundant rows.
        A = np.vstack((A, np.eye(n), -np.eye(n)))
        b = np.concatenate((b, high, -low))
        lb, ub = low.copy(), high.copy()
        lb[np.arange(n) % 4 == 0] = -np.inf
        ub[np.arange(n) % 4 == 1] = np.inf
        lb[np.arange(n) % 4 == 2] = -np.inf
        ub[np.arange(n) % 4 == 2] = np.inf
        if k % 7 == 0:
            lb[-1] = ub[-1] = witness[-1]
        c = rng.normal(size=n)
        mode = 'max' if k % 2 else 'min'
        result.append(case(f'random_{k:03d}', A, b, c, lb=lb, ub=ub, mode=mode, category=f'dense_{n}'))
    return result


def normalize(lp):
    """Common row/objective units for all comparison solvers, before timing.

    Scaling avoids comparing different solvers' unit-sensitive stopping tests.
    Separate analytic tests exercise optinpy directly on the unscaled data.
    """
    row_scale = np.max(np.abs(lp.A), axis=1) if lp.A.shape[0] else np.empty(0)
    row_scale = np.where(row_scale > 0, row_scale, 1.)
    cost_scale = np.max(np.abs(lp.c)) or 1.
    return case(lp.name, lp.A / row_scale[:, None], lp.b / row_scale, lp.c / cost_scale,
                lb=lp.lb, ub=lp.ub, mode=lp.mode, status=lp.expected_status,
                objective=None if lp.expected_objective is None else lp.expected_objective/cost_scale,
                category=lp.category)


def violation(lp, x):
    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)):
        return float('inf')
    return float(max(np.max(np.einsum('ij,j->i', lp.A, x) - lp.b, initial=0.),
                     np.max(lp.lb - x, initial=0.), np.max(x - lp.ub, initial=0.)))

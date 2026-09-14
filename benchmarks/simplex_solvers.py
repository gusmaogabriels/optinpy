"""Reference adapters. External solver calls are confined to benchmarks/tests."""
import numpy as np

from .simplex_cases import violation


STATUS = {0: 'optimal', 1: 'limit', 2: 'unbounded', 3: 'infeasible', 4: 'numerical'}


def optinpy_solve(lp, *, dtype='float64', tol=None):
    import jax
    from optinpy import simplex
    # A local context exercises float32 independently of the rest of the suite.
    precision_context = getattr(jax, 'enable_x64', None)
    if precision_context is None:  # JAX 0.4.x compatibility.
        precision_context = jax.experimental.enable_x64
    with precision_context(dtype == 'float64'):
        solver = simplex(lp.A, lp.b, lp.c, lb=lp.lb, ub=lp.ub, mode=lp.mode, tol=tol)
        result = solver.solve(max_iter=10000)
        x = np.asarray(jax.device_get(result['x']), dtype=float)
        objective = float(lp.c @ x)
    return {'status': STATUS[result['status']], 'objective': objective,
            'violation': violation(lp, x), 'iterations': result['iterations'],
            'x': x.tolist()}


def highs_solve(lp, *, method='highs-ds'):
    from scipy.optimize import linprog
    result = linprog(lp.c if lp.mode == 'min' else -lp.c,
                     A_ub=lp.A if len(lp.b) else None, b_ub=lp.b if len(lp.b) else None,
                     bounds=list(zip(lp.lb, lp.ub)), method=method,
                     options={'primal_feasibility_tolerance': 1e-9,
                              'dual_feasibility_tolerance': 1e-9,
                              'ipm_optimality_tolerance': 1e-10})
    return {'status': {0:'optimal', 1:'limit', 2:'infeasible', 3:'unbounded', 4:'numerical'}[result.status],
            'objective': float(lp.c @ result.x) if result.success else None,
            'violation': violation(lp, result.x) if result.success else None,
            'iterations': result.nit, 'x': result.x.tolist() if result.success else None}


def clarabel_solve(lp):
    import clarabel
    from scipy import sparse
    n = len(lp.c)
    rows, rhs = [lp.A], [lp.b]
    identity = np.eye(n)
    lower, upper = np.isfinite(lp.lb), np.isfinite(lp.ub)
    rows.extend((-identity[lower], identity[upper]))
    rhs.extend((-lp.lb[lower], lp.ub[upper]))
    A, b = sparse.csc_matrix(np.vstack(rows)), np.concatenate(rhs)
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_iter = 300
    settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = 1e-9
    solver = clarabel.DefaultSolver(sparse.csc_matrix((n, n)),
                                    lp.c if lp.mode == 'min' else -lp.c,
                                    A, b, [clarabel.NonnegativeConeT(len(b))], settings)
    result = solver.solve()
    status = str(result.status)
    classification = None
    if status == 'DualInfeasible':
        # A dual-infeasibility certificate alone does not distinguish an
        # unbounded primal from simultaneous primal/dual infeasibility.
        feasibility = clarabel.DefaultSolver(sparse.csc_matrix((n, n)), np.zeros(n),
                                             A, b, [clarabel.NonnegativeConeT(len(b))], settings).solve()
        classification = str(feasibility.status)
        if classification == 'PrimalInfeasible':
            status = 'PrimalInfeasible'
        elif classification not in ('Solved', 'AlmostSolved'):
            status = 'UnclassifiedDualInfeasible'
    mapping = {'Solved':'optimal', 'AlmostSolved':'approximate',
               'PrimalInfeasible':'infeasible', 'DualInfeasible':'unbounded',
               'AlmostPrimalInfeasible':'approximate_infeasible',
               'AlmostDualInfeasible':'approximate_unbounded',
               'MaxIterations':'limit', 'MaxTime':'limit'}
    solved = status in ('Solved', 'AlmostSolved')
    return {'status': mapping.get(status, 'numerical'), 'raw_status': str(result.status),
            'feasibility_classification': classification,
            'objective': float(lp.c @ result.x) if solved else None,
            'violation': violation(lp, result.x) if solved else None,
            'iterations': result.iterations, 'x': list(result.x) if solved else None}

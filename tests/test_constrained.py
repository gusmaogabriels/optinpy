from copy import deepcopy

import jax.numpy as jnp
import numpy as np
import pytest

import optinpy as op
from optinpy.nonlinear.constrained import constrained


@pytest.fixture
def solver():
    return constrained(deepcopy(op.params), op.unconstrained)


def test_linear_constraints_and_feasible_history(solver):
    r = solver.fmincon(lambda x: jnp.sum((x-jnp.array([2., 1.]))**2),
                       [4., 4.], A=[[1., 1.]], b=[1.], vectorized=True, threshold=1e-12)
    assert r['success'], r
    np.testing.assert_allclose(r['x'][-1], [1., 0.], atol=1e-6)
    assert jnp.all(jnp.sum(r['x'], axis=1) <= 1+1e-7)


def test_equality_and_negative_coordinates(solver):
    r = solver.fmincon(lambda x: jnp.sum((x-jnp.array([-2., 1.]))**2),
                       [0., 0.], Aeq=[[1., 1.]], beq=[-1.], threshold=1e-12)
    assert r['success']
    np.testing.assert_allclose(r['x'], [-2., 1.], atol=1e-6)


def test_releases_inactive_bound_and_handles_redundant_equalities(solver):
    r = solver.fmincon(lambda x: jnp.sum((x-1)**2), [0., 0.],
                       A=[[-1., 0.]], b=[0.], Aeq=[[1., -1.], [2., -2.]],
                       beq=[0., 0.], threshold=1e-12)
    assert r['success'], r
    np.testing.assert_allclose(r['x'], [1., 1.], atol=1e-6)


def test_infeasible_phase_one_is_bounded(solver):
    r = solver.fmincon(lambda x: jnp.sum(x*x), [0.], A=[[1.], [-1.]],
                       b=[0., -1.], feasibility_max_iter=10)
    assert not r['success'] and r['status'] == 4


@pytest.mark.parametrize('method', ['penalty', 'barrier', 'log-barrier'])
def test_nonlinear_constraint_boundary(solver, method):
    solver.params['fminnlcon']['method'] = method
    r = solver.fminnlcon(lambda x: (x[0]-2)**2, [0.], [lambda x: x[0]-1], threshold=1e-8)
    assert r['success'], r
    np.testing.assert_allclose(r['x'], [1.], atol=1e-4)
    assert r['constraint_violation'] <= 1e-4


def test_feasible_start_still_minimizes_objective(solver):
    r = solver.fminnlcon(lambda x: (x[0]-0.5)**2, [0.], [lambda x: x[0]-1], threshold=1e-10)
    assert r['success']
    np.testing.assert_allclose(r['x'], [0.5], atol=1e-5)


def test_barrier_rejects_infeasible_start(solver):
    solver.params['fminnlcon']['method'] = 'log-barrier'
    with pytest.raises(ValueError, match='strictly feasible'):
        solver.fminnlcon(lambda x: x[0]**2, [2.], [lambda x: x[0]-1])


def test_outer_budget_exhaustion(solver):
    r = solver.fminnlcon(lambda x: (x[0]-2)**2, [0.], [lambda x: x[0]-1], max_iter=0)
    assert not r['success'] and r['iterations'] == 0


@pytest.mark.parametrize('method', ['penalty', 'barrier', 'log-barrier'])
def test_many_nonlinear_constraints_match_analytic_kkt_residual(solver, method):
    solver.params['fminnlcon']['method'] = method
    A = np.random.default_rng(12).normal(size=(8, 2))
    constraints = [lambda x, a=jnp.asarray(a): (a @ x)**2-1 for a in A]
    result = solver.fminnlcon(lambda x: jnp.sum((x-2)**2), [0., 0.], constraints,
                              max_iter=1, c=2., threshold=1e-10)
    x = np.asarray(result['x'])
    residual = (A @ x)**2-1
    if method == 'penalty':
        multipliers = 2*np.maximum(residual, 0)
    elif method == 'log-barrier':
        multipliers = -1/(2*residual)
    else:
        multipliers = 1/(2*residual**2)
    J = 2*(A @ x)[:, None]*A
    expected = max(np.linalg.norm(2*(x-2)+J.T @ multipliers),
                   np.maximum(residual, 0).max(), np.abs(multipliers*residual).max())
    np.testing.assert_allclose(result['err'], expected, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize('method', ['penalty', 'barrier', 'log-barrier'])
def test_nonlinear_history_matches_scalar_result(solver, method):
    solver.params['fminnlcon']['method'] = method
    fun = lambda x: (x[0]-2)**2
    constraints = [lambda x: x[0]-1]
    scalar = solver.fminnlcon(fun, [0.], constraints, threshold=1e-8)
    history = solver.fminnlcon(fun, [0.], constraints, threshold=1e-8, vectorized=True)
    assert scalar['success'] and history['success']
    np.testing.assert_allclose(history['x'][-1], scalar['x'])
    np.testing.assert_allclose(history['f'][-1], scalar['f'])
    np.testing.assert_allclose(history['err'][-1], scalar['err'])
    assert history['x'].shape[0] == history['iterations'] + 1

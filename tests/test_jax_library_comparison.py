"""Numerical checks for optional benchmark references and shared acceptance."""
import json
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import optinpy
from benchmarks.jax_library_cases import analytic, assess, cases
from benchmarks.jax_library_solvers import objective, optax_solve, solvers


@pytest.mark.parametrize('group', ['linear', 'nonlinear', 'first-order'])
def test_benchmark_problems_have_known_solutions(group):
    for case in cases(group):
        optimum = {'x': case.expected, 'success': True, 'iterations': 0, 'status': 0}
        assert assess(case, optimum)['passed']
        if case.kind != 'linear':
            value, gradient = analytic(case, case.x0)
            actual, derivative = jax.value_and_grad(objective(case.kind))(
                jnp.asarray(case.x0), *map(jnp.asarray, case.args))
            np.testing.assert_allclose(actual, value, atol=1e-10, rtol=1e-10)
            np.testing.assert_allclose(derivative, gradient, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize('group,package', [('linear', 'lineax'), ('nonlinear', 'optimistix'),
                                         ('first-order', 'optax')])
def test_optional_adapters_solve_same_quick_problem(group, package):
    pytest.importorskip(package)
    case = cases(group, quick=True)[0]
    for name, (solve, _) in solvers(case, group).items():
        result = jax.jit(solve)(*map(jnp.asarray, case.inputs))
        check = assess(case, result)
        assert check['passed'], (name, check)


def test_lineax_general_lu_accepts_indefinite_system():
    pytest.importorskip('lineax')
    case = cases('linear')[-1]
    assert not case.positive_definite
    for name, (solve, _) in solvers(case, 'linear').items():
        check = assess(case, jax.jit(solve)(*map(jnp.asarray, case.inputs)))
        assert check['passed'], (name, check)


@pytest.mark.parametrize('method', ['adam', 'sgd'])
@pytest.mark.parametrize('dtype', [jnp.float32, jnp.float64])
def test_optax_updates_match_native_trajectory(method, dtype):
    optax = pytest.importorskip('optax')
    case = cases('first-order', quick=True)[0]
    fun = objective(case.kind)
    x, *args = [jnp.asarray(v, dtype=dtype) for v in case.inputs]
    transform = optax.adam(.01) if method == 'adam' else optax.sgd(.01)
    reference = jax.jit(lambda x, *args: optax_solve(
        fun, x, args, transform, tol=1e-12, max_iter=10))(x, *args)
    native = jax.jit(lambda x, *args: optinpy.minimize(
        fun, x, args=args, method=method, learning_rate=.01, tol=1e-12, max_iter=10))(x, *args)
    np.testing.assert_allclose(native['x'], reference['x'], atol=2e-6 if dtype == jnp.float32 else 1e-12,
                               rtol=2e-6 if dtype == jnp.float32 else 1e-12)
    assert int(native['iterations']) == int(reference['iterations']) == 10
    assert not native['success'] and not reference['success']


@pytest.mark.parametrize('group', ['linear', 'nonlinear', 'first-order'])
def test_acceptance_rejects_wrong_or_unsuccessful_results(group):
    case = cases(group, quick=True)[0]
    result = {'x': case.expected, 'success': False, 'iterations': 100, 'status': 1}
    assert not assess(case, result)['passed']
    result.update(success=True, x=case.expected+1.)
    assert not assess(case, result)['passed']
    result['x'] = np.full_like(case.expected, np.nan)
    check = assess(case, result)
    assert not check['passed'] and not check['finite']
    result['x'] = np.full_like(case.expected, np.finfo(float).max)
    check = assess(case, result)
    assert not check['passed']
    json.dumps(check, allow_nan=False)

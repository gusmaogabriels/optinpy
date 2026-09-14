from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import optinpy as op
from optinpy.nonlinear.unconstrained import unconstrained

METHODS = ['gradient', 'newton', 'modified-newton', 'conjugate-gradient',
           'hessian-conjugate-gradient', 'fletcher-reeves', 'bfgs', 'dfp', 'lbfgs', 'adam', 'sgd']


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('dtype', [jnp.float32, jnp.float64])
def test_quadratic_all_methods(method, dtype):
    target = jnp.array([1., -2.], dtype=dtype)
    fn = lambda x: jnp.sum(jnp.array([1., 3.], dtype=dtype)*(x-target)**2)
    x = jnp.array([3., 2.], dtype=dtype)
    result = op.minimize(fn, x, method=method, tol=2e-4, max_iter=3000, learning_rate=0.03)
    assert result['success'], result
    assert result['status'] == 0
    assert result['x'].dtype == dtype
    np.testing.assert_allclose(result['x'], target, atol=2e-4)


@pytest.mark.parametrize('method', ['bfgs', 'dfp', 'lbfgs', 'newton', 'modified-newton'])
def test_rosenbrock(method):
    fn = lambda x: (1-x[0])**2 + 100*(x[1]-x[0]**2)**2
    r = op.minimize(fn, [-1.2, 1.], method=method, tol=1e-6)
    assert r['success'], r
    np.testing.assert_allclose(r['x'], [1., 1.], atol=2e-5)
    assert r['gradient_norm'] <= 1e-6


def test_jit_vmap_and_args():
    fn = lambda x, target: jnp.sum((x-target)**2)
    solve = jax.jit(jax.vmap(lambda target: op.minimize(fn, jnp.zeros(2), args=(target,))))
    targets = jnp.array([[1., 2.], [-2., 1.]])
    result = solve(targets)
    np.testing.assert_allclose(result['x'], targets, atol=1e-6)
    assert jnp.all(result['success'])


@pytest.mark.parametrize('method', METHODS)
def test_compiled_minimizer_reuses_dynamic_data_and_batches(method):
    traces = []
    def fn(x, target):
        traces.append(None)
        return jnp.sum((x-target)**2)
    solve = op.compile_minimizer(fn, method=method, tol=2e-4, max_iter=3000, learning_rate=0.03)
    x0 = jnp.zeros(2)
    first = solve(x0, jnp.array([1., -2.]))
    assert first['success']
    traced = len(traces)
    second = solve(x0, jnp.array([-2., 1.]))
    assert second['success'] and len(traces) == traced
    np.testing.assert_allclose(second['x'], [-2., 1.], atol=2e-4)
    batch = jax.jit(jax.vmap(solve, in_axes=(None, 0)))(x0, jnp.array([[1., 2.], [-2., 1.]]))
    assert jnp.all(batch['success'])
    np.testing.assert_allclose(batch['x'], [[1., 2.], [-2., 1.]], atol=2e-4)
    np.testing.assert_array_equal(x0, [0., 0.])


def test_compiled_minimizer_rejects_history_and_bound_dynamic_args():
    fn = lambda x: jnp.sum(x*x)
    with pytest.raises(ValueError, match='history'):
        op.compile_minimizer(fn, history=True)
    with pytest.raises(ValueError, match='args'):
        op.compile_minimizer(fn, args=())


def test_compiled_minimizer_opt_in_donation():
    solve = op.compile_minimizer(lambda x: jnp.sum((x-1)**2), donate_x0=True)
    result = jax.block_until_ready(solve(jnp.zeros(3)))
    assert result['success']
    np.testing.assert_allclose(result['x'], [1., 1., 1.], atol=1e-6)


def test_line_search_derivative_override_does_not_replace_objective_gradient():
    fn = lambda x: jnp.sum(jnp.array([1., 3.])*(x-jnp.array([1., -2.]))**2)
    approximate = lambda x: 0.5*jax.grad(fn)(x)
    result = op.compile_minimizer(fn, max_iter=1,
                                  linesearch_options={'gradient_fn': approximate})(jnp.zeros(2))
    assert result['iterations'] == 1
    np.testing.assert_allclose(result['gradient'], jax.grad(fn)(result['x']))


def test_initial_optimum_and_budget_exhaustion():
    fn = lambda x: jnp.sum(x*x)
    r = op.minimize(fn, [0., 0.], max_iter=0)
    assert r['success'] and r['iterations'] == 0
    r = op.minimize(fn, [1., 1.], max_iter=0)
    assert not r['success'] and r['status'] == 1


def test_nonfinite_and_failed_line_search():
    r = op.minimize(lambda x: jnp.log(x[0]), [-1.])
    assert not r['success'] and r['status'] == 3
    r = op.minimize(lambda x: 100*jnp.sum(x*x), [1.], method='gradient',
                    linesearch='backtracking', linesearch_options={'max_iter': 0})
    assert not r['success'] and r['status'] == 2
    np.testing.assert_array_equal(r['x'], [1.])


def test_singular_newton_falls_back_to_descent():
    fn = lambda x: (x[0]-1)**2
    r = op.minimize(fn, [0., 3.], method='newton')
    assert r['success']
    np.testing.assert_allclose(r['x'], [1., 3.])


def test_adam_first_step_matches_equations():
    r = op.minimize(lambda x: jnp.sum(x*x), [1., -2.], method='adam', learning_rate=0.1, max_iter=1)
    np.testing.assert_allclose(r['x'], [0.9, -1.9], atol=1e-8)
    assert r['status'] == 1


def test_history_and_legacy_api():
    params = deepcopy(op.params)
    params['fminunc']['method'] = 'quasi-newton'
    solver = unconstrained(params)
    fn = lambda x: jnp.sum((x-2)**2)
    r = solver.fminunc(fn, [0., 0.], vectorized=True, threshold=1e-12)
    assert r['success']
    assert r['x'].shape[0] == int(r['iterations']) + 1
    np.testing.assert_allclose(r['x'][-1], [2., 2.], atol=1e-6)
    np.testing.assert_allclose(r['f'], jax.vmap(fn)(r['x']))


@pytest.mark.parametrize('derivative', ['autodiff', 'central'])
def test_original_conjugate_gradient_preserves_hessian_orthogonality(derivative):
    params = deepcopy(op.params)
    params['fminunc']['method'] = 'conjugate-gradient'
    params['hessian'].update(algorithm=derivative, epsilon=1e-2)
    weights = jnp.array([1., 3., 7.])
    fn = lambda x: 0.5*jnp.sum(weights*x*x)
    r = unconstrained(params).fminunc(fn, [1., 2., 3.], max_iter=2, vectorized=True)
    assert r['iterations'] == 2
    first, second = np.diff(r['x'], axis=0)
    # This property distinguishes the original Hessian-conjugate recurrence
    # from PR+ under the inexact Armijo line search used by the original API.
    np.testing.assert_allclose(first @ np.diag(weights) @ second, 0., atol=1e-8)
    assert r['f'][-1] < r['f'][0]


@pytest.mark.parametrize('kwargs', [{'method':'bad'}, {'tol':0}, {'max_iter':-1},
                                   {'max_iter':1.5}, {'memory_size':0}, {'beta1':1},
                                   {'learning_rate':0}, {'linesearch':'bad'}])
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        op.minimize(lambda x: jnp.sum(x*x), [1.], **kwargs)

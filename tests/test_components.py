"""Standalone algorithm identities and interchangeable solver components."""
from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import optinpy as op
from optinpy import nonlinear as nl
from optinpy.nonlinear.unconstrained import unconstrained
from optinpy.nonlinear.constrained import constrained


@pytest.mark.parametrize('dtype', [jnp.float32, jnp.float64])
def test_standalone_newton_and_conjugate_directions(dtype):
    Q = jnp.array([[4., 1.], [1., 2.]], dtype=dtype)
    g, old_g, old_d = [jnp.array(v, dtype=dtype) for v in ([2., 3.], [4., 1.], [-1., -2.])]
    newton = jax.jit(nl.newton_direction)(g, Q)
    np.testing.assert_allclose(Q @ newton, -g, atol=2e-6)
    hcg = jax.jit(nl.hessian_conjugate_gradient_direction)(g, old_d, Q @ old_d)
    np.testing.assert_allclose(old_d @ Q @ hcg, 0., atol=5e-6)
    pr_beta = max(float(g @ (g-old_g) / (old_g @ old_g)), 0.)
    fr_beta = float((g @ g) / (old_g @ old_g))
    for fn, beta in ((nl.conjugate_gradient_direction, pr_beta), (nl.fletcher_reeves_direction, fr_beta)):
        np.testing.assert_allclose(jax.jit(fn)(g, old_g, old_d), -g+beta*old_d, atol=2e-6)
        np.testing.assert_array_equal(jax.jit(fn)(g, old_g, old_d, restart=True), -g)
    np.testing.assert_array_equal(nl.hessian_conjugate_gradient_direction(g, old_d, -old_d), -g)
    np.testing.assert_array_equal(nl.sgd_direction(g), -g)
    modified = jax.jit(lambda g: nl.modified_newton_direction(g, jnp.diag(jnp.array([-2., 3.], dtype=dtype)), sigma=.5))(g)
    np.testing.assert_allclose(modified, -g/jnp.array([.5, 3.], dtype=dtype), atol=2e-6)
    assert newton.dtype == hcg.dtype == modified.dtype == dtype


@pytest.mark.parametrize('update', [nl.bfgs_update, nl.dfp_update])
@pytest.mark.parametrize('dtype', [jnp.float32, jnp.float64])
def test_inverse_updates_preserve_secant_symmetry_and_positive_definiteness(update, dtype):
    H = jnp.array([[2., .2], [.2, 1.]], dtype=dtype)
    p, y = jnp.array([1., 2.], dtype=dtype), jnp.array([3., 4.], dtype=dtype)
    result = jax.jit(update)(H, p, y)
    np.testing.assert_allclose(result @ y, p, atol=2e-6)
    np.testing.assert_allclose(result, result.T, atol=1e-7)
    assert np.linalg.eigvalsh(np.asarray(result)).min() > 0
    assert result.dtype == dtype
    for bad_y in (-p, jnp.zeros_like(p), jnp.full_like(p, jnp.nan)):
        np.testing.assert_array_equal(jax.jit(update)(H, p, bad_y), H)
    batched = jax.jit(jax.vmap(update, in_axes=(None, 0, 0)))(H, jnp.stack([p, 2*p]), jnp.stack([y, 2*y]))
    np.testing.assert_allclose(batched, jnp.stack([result, result]), atol=2e-6)


@pytest.mark.parametrize('dtype', [jnp.float32, jnp.float64])
def test_lbfgs_history_matches_explicit_inverse_product(dtype):
    steps = jnp.zeros((3, 2), dtype=dtype)
    changes, count = jnp.zeros_like(steps), jnp.asarray(0)
    pairs = []
    for p in ([1., 2.], [2., -1.], [.3, .7], [-.2, .6]):
        p = jnp.array(p, dtype=dtype)
        y = jnp.array([[3., .5], [.5, 2.]], dtype=dtype) @ p
        pairs.append((np.asarray(p), np.asarray(y)))
        steps, changes, count = jax.jit(nl.lbfgs_update)(steps, changes, count, p, y)
    assert int(count) == 3
    p, y = pairs[-1]
    H = np.eye(2) * (np.dot(p, y)/np.dot(y, y))
    for p, y in pairs[-3:]:
        rho = 1/np.dot(p, y)
        V = np.eye(2)-rho*np.outer(p, y)
        H = V @ H @ V.T + rho*np.outer(p, p)
    g = jnp.array([1., -3.], dtype=dtype)
    d = jax.jit(nl.lbfgs_direction)(g, steps, changes, count)
    np.testing.assert_allclose(d, -H @ np.asarray(g), atol=2e-6, rtol=2e-6)
    assert d.dtype == dtype
    rejected = jax.jit(nl.lbfgs_update)(steps, changes, count, g, -g)
    for before, after in zip((steps, changes, count), rejected):
        np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize('dtype', [jnp.float32, jnp.float64])
def test_adam_primitive_uses_bias_corrected_moments(dtype):
    g = jnp.array([2., -4.], dtype=dtype)
    d, m, v = jax.jit(nl.adam_direction)(g, jnp.zeros_like(g), jnp.zeros_like(g), jnp.asarray(0))
    np.testing.assert_allclose(d, -g/(jnp.abs(g)+1e-8), atol=8e-6)
    np.testing.assert_allclose(m, .1*g, atol=1e-7)
    np.testing.assert_allclose(v, .001*g*g, atol=1e-7)
    assert d.dtype == m.dtype == v.dtype == dtype


@pytest.mark.parametrize('method', ['gradient', 'newton', 'bfgs', 'dfp', 'hessian-conjugate-gradient'])
@pytest.mark.parametrize('search', ['backtracking', 'interp23', 'strong-wolfe', 'golden-section', 'unimodality'])
@pytest.mark.parametrize('derivative', ['autodiff', 'central'])
def test_legacy_mix_and_match_methods_searches_and_derivatives(method, search, derivative):
    params = deepcopy(op.params)
    params['fminunc']['method'] = 'conjugate-gradient' if method == 'hessian-conjugate-gradient' else method
    params['linesearch']['method'] = search
    params['jacobian'].update(algorithm=derivative, epsilon=1e-3)
    params['hessian'].update(algorithm=derivative, epsilon=1e-2)
    fun = lambda x: jnp.sum(jnp.array([1., 3.])*(x-jnp.array([.25, -.5]))**2)
    solve = jax.jit(lambda x: unconstrained(params).fminunc(fun, x, threshold=1e-8, max_iter=500))
    r = solve(jnp.array([1., 2.]))
    assert r['success'], r
    np.testing.assert_allclose(r['x'], [.25, -.5], atol=1e-4)
    assert np.linalg.norm(jax.grad(fun)(r['x'])) <= 1.1e-4


def test_custom_direction_and_search_with_dynamic_data_and_batching():
    # A supplied derivative remains authoritative, including at search results.
    fun = lambda x, target, Q: jax.lax.stop_gradient(.5*(x-target) @ Q @ (x-target))
    jac = lambda x, target, Q: Q @ (x-target)
    traces = []
    def direction(x, g, target, Q):
        traces.append(None)
        return nl.newton_direction(g, Q)
    def search(fun, x, d, *, gradient, value, **options):
        return op.linesearch.interp23(fun, x, d, gradient=gradient, value=value, **options)
    solve = op.compile_minimizer(fun, jac=jac, method=direction, linesearch=search)
    Q = jnp.array([[3., .5], [.5, 2.]])
    x = jnp.zeros(2)
    first = solve(x, jnp.array([1., 2.]), Q)
    assert first['success']
    compiled_traces = len(traces)
    second = solve(x, jnp.array([-2., 1.]), Q)
    assert second['success'] and len(traces) == compiled_traces
    targets = jnp.array([[1., 2.], [-2., 1.]])
    batch = jax.jit(jax.vmap(solve, in_axes=(None, 0, None)))(x, targets, Q)
    np.testing.assert_allclose(batch['x'], targets, atol=1e-6)
    assert jnp.all(batch['success'])


@pytest.mark.parametrize('name', ['backtracking', 'interp23', 'strong_wolfe', 'golden_section', 'unimodality'])
def test_standalone_search_functions_plug_into_minimize(name):
    fun = lambda x: jnp.sum((x-2.)**2)
    r = op.compile_minimizer(fun, linesearch=getattr(op.linesearch, name))(jnp.zeros(2))
    assert r['success']
    np.testing.assert_allclose(r['x'], [2., 2.], atol=1e-6)


def test_custom_search_has_failure_and_shape_checks():
    def failed(fun, x, d, **kwargs):
        return {'x': x, 'success': False, 'iterations': 0}
    fun = lambda x: jnp.sum((x-1.)**2)
    result = op.compile_minimizer(fun, linesearch=failed)(jnp.zeros(2))
    assert not result['success'] and result['status'] == 2
    np.testing.assert_array_equal(result['x'], jnp.zeros(2))
    def fixed(fun, x, d, **kwargs):
        return {'x': x+.5*d, 'success': True, 'iterations': 0}
    assert op.compile_minimizer(fun, method='gradient', linesearch=fixed)(jnp.zeros(2))['success']
    with pytest.raises(ValueError, match='Custom direction'):
        op.minimize(fun, [0., 0.], method=lambda x, g: g[0])
    with pytest.raises(ValueError, match='Line search must return'):
        op.minimize(fun, [0., 0.], linesearch=lambda fun, x, d, **kwargs: dict(x=0., success=True, iterations=0))


@pytest.mark.parametrize('outer', ['penalty', 'barrier', 'log-barrier'])
@pytest.mark.parametrize('inner', ['bfgs', 'newton'])
def test_constraint_outer_method_accepts_selected_inner_components(outer, inner):
    params = deepcopy(op.params)
    params['fminnlcon']['method'] = outer
    solver = constrained(params, unconstrained(params))
    visited = []
    def search(fun, x, d, **kwargs):
        visited.append(True)
        return op.linesearch.backtracking(fun, x, d, **kwargs)
    options = dict(method=inner, linesearch=search)
    r = solver.fminnlcon(lambda x: (x[0]-2.)**2, [0.], [lambda x: x[0]-1.],
                        threshold=1e-8, inner_options=options)
    assert visited and r['success'], r
    assert options['method'] == inner and options['linesearch'] is search
    np.testing.assert_allclose(r['x'], [1.], atol=2e-4)

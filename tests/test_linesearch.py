import jax
import jax.numpy as jnp
import numpy as np
import pytest

from optinpy import linesearch as ls


@pytest.mark.parametrize('name', ['backtracking', 'interp23', 'strong_wolfe', 'golden_section', 'unimodality'])
def test_descent_and_default_direction(name):
    fun = lambda x: jnp.sum(x*x)
    result = jax.jit(lambda x: getattr(ls, name)(fun, x))(jnp.array([1., 2.]))
    assert result['success']
    assert result['f'] < fun(jnp.array([1., 2.]))
    assert result['alpha'] > 0


@pytest.mark.parametrize('name', ['backtracking', 'interp23'])
def test_armijo_and_failure(name):
    fun = lambda x: 100*jnp.sum(x*x)
    x = jnp.array([1., 2.])
    g = jax.grad(fun)(x)
    result = getattr(ls, name)(fun, x, -g)
    assert result['success']
    assert result['f'] <= fun(x) - 1e-4*result['alpha']*jnp.vdot(g, g)
    failed = getattr(ls, name)(fun, x, -g, max_iter=0)
    assert not failed['success']
    assert failed['alpha'] == 0
    np.testing.assert_array_equal(failed['x'], x)
    assert not getattr(ls, name)(fun, x, g)['success']


def test_strong_wolfe_conditions():
    fun = lambda x: jnp.exp(x[0]) + x[0]**2
    x, d = jnp.array([2.]), jnp.array([-1.])
    result = ls.strong_wolfe(fun, x, d)
    initial_slope = jnp.vdot(jax.grad(fun)(x), d)
    final_slope = jnp.vdot(jax.grad(fun)(result['x']), d)
    assert result['success']
    assert result['f'] <= fun(x) + 1e-4*result['alpha']*initial_slope
    assert abs(final_slope) <= 0.9*abs(initial_slope)


def test_golden_section_known_minimum_and_boundary():
    f = lambda x: (x[0]-0.3)**2
    r = ls.golden_section(f, [0.], [1.], threshold=1e-7)
    np.testing.assert_allclose(r['x'], [0.3], atol=1e-6)
    r = ls.golden_section(lambda x: -x[0], [0.], [1.], b=2.)
    assert r['x'][0] == 2


def test_random_search_reproducibility():
    fn = lambda x: jnp.sum(x*x)
    key = jax.random.key(19)
    a = ls.unimodality(fn, [1., 2.], key=key)
    b = ls.unimodality(fn, [1., 2.], key=key)
    np.testing.assert_array_equal(a['x'], b['x'])


def test_nonfinite_trial_is_rejected():
    fn = lambda x: jnp.where(x[0] > 0, -jnp.log(x[0]), jnp.inf)
    r = ls.backtracking(fn, [1.], [-10.])
    assert not r['success']  # ascent direction
    assert r['alpha'] == 0


def test_strong_wolfe_uses_supplied_derivative_at_trial_points():
    fn = lambda x: jax.lax.stop_gradient(jnp.sum(x*x))
    gradient = lambda x: 2*x
    r = ls.strong_wolfe(fn, [1.], [-1.], alpha=0.1, c2=0.1, gradient_fn=gradient)
    assert r['success']
    assert abs(gradient(r['x'])[0]) <= 0.2


def test_strong_wolfe_returns_gradient_at_accepted_and_failed_points():
    fn = lambda x: jnp.exp(x[0]) + x[0]**2
    x = jnp.array([2.])
    g = jax.grad(fn)(x)
    accepted = jax.jit(lambda y: ls.strong_wolfe(fn, y, [-1.], gradient=g, value=fn(y)))(x)
    assert accepted['success']
    np.testing.assert_allclose(accepted['gradient'], jax.grad(fn)(accepted['x']))
    failed = ls.strong_wolfe(fn, x, [1.], gradient=g, value=fn(x))
    assert not failed['success']
    np.testing.assert_array_equal(failed['gradient'], g)

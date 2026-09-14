import jax
import jax.numpy as jnp
import numpy as np
import pytest

from optinpy.finitediff import jacobian, hessian


def objective(x):
    return 3*x[0]**2 + 2*x[0]*x[1] + 4*x[1]**2 + jnp.sin(x[0])


@pytest.mark.parametrize('algorithm', ['autodiff', 'central', 'forward', 'backward'])
def test_derivatives_match_analytic_values_without_mutating_input(algorithm):
    x = np.array([0.3, -0.2])
    original = x.copy()
    eps = None if algorithm == 'autodiff' else 1e-3
    grad = jax.jit(lambda y: jacobian(objective, y, epsilon=eps, algorithm=algorithm))(x)
    H = jax.jit(lambda y: hessian(objective, y, epsilon=eps, algorithm=algorithm))(x)
    np.testing.assert_allclose(grad, [6*x[0]+2*x[1]+np.cos(x[0]), 2*x[0]+8*x[1]], atol=5e-3)
    np.testing.assert_allclose(H, [[6-np.sin(x[0]), 2], [2, 8]], atol=5e-3)
    np.testing.assert_array_equal(x, original)
    assert isinstance(grad, jax.Array)


def test_autodiff_is_composable_and_batchable():
    x = jnp.array([0.3, -0.2])
    np.testing.assert_allclose(jax.jacfwd(lambda y: jacobian(objective, y))(x), hessian(objective, x))
    result = jax.vmap(lambda y: jacobian(objective, y))(jnp.stack([x, x+1]))
    assert result.shape == (2, 2)


@pytest.mark.parametrize('dtype', [jnp.float32, jnp.float64])
@pytest.mark.parametrize('size', [1, 3])
def test_original_five_point_hessian_is_exact_for_quartic_polynomials(dtype, size):
    fn = lambda x: jnp.sum(x**4) + jnp.sum(x)**2
    points = jnp.ones((2, size), dtype=dtype)*jnp.array([0.5, 1.], dtype=dtype)[:, None]
    evaluate = jax.jit(jax.vmap(lambda x: hessian(fn, x, epsilon=0.1, algorithm='central')))
    actual = evaluate(points)
    expected = jax.vmap(lambda x: jnp.diag(12*x*x) + 2*jnp.ones((size, size)))(points)
    # A three-point diagonal has an O(h^2) error for this polynomial.
    np.testing.assert_allclose(actual, expected, atol=2e-3, rtol=0)
    assert actual.dtype == dtype


@pytest.mark.parametrize('x', [[], [[1., 2.]], [1+2j]])
def test_invalid_inputs(x):
    with pytest.raises((ValueError, TypeError)):
        jacobian(objective, x)


@pytest.mark.parametrize('fn', [jacobian, hessian])
def test_invalid_options(fn):
    with pytest.raises(ValueError):
        fn(objective, [1., 2.], algorithm='unknown')
    with pytest.raises(ValueError):
        fn(objective, [1., 2.], algorithm='central', epsilon=0)

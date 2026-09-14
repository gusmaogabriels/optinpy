import jax.numpy as jnp
import numpy as np
import pytest

from optinpy import simplex


@pytest.mark.parametrize('kwargs, expected', [({}, [2., 2.]), ({'mode':'max'}, [0., 0.])])
def test_known_linear_program(kwargs, expected):
    s = simplex([[1., 1.], [1., 0.], [0., 1.]], [4., 2., 3.], [-3., -2.], **kwargs)
    r = s.solve()
    assert r['success'], r
    np.testing.assert_allclose(r['x'], expected)


@pytest.mark.parametrize('lb, ub, c, expected', [([-5.], [3.], [1.], [-5.]),
                                                ([-np.inf], [3.], [-1.], [3.]),
                                                ([2.], [2.], [1.], [2.])])
def test_variable_bounds(lb, ub, c, expected):
    r = simplex([], [], c, lb=lb, ub=ub).solve()
    assert r['success'], r
    np.testing.assert_allclose(r['x'], expected)


def test_free_variable_and_phase_one():
    r = simplex([[1.], [-1.]], [-1., 2.], [1.], lb=[-np.inf]).solve()
    assert r['success']
    np.testing.assert_allclose(r['x'], [-2.])


def test_infeasible_and_unbounded_status():
    r = simplex([[1.], [-1.]], [1., -2.], [1.]).solve()
    assert not r['success'] and r['status'] == 3
    r = simplex([], [], [-1.]).solve()
    assert not r['success'] and r['status'] == 2


def test_phase_one_redundant_constraints():
    r = simplex([[-1.], [-2.], [1.]], [-1., -2., 3.], [1.]).solve()
    assert r['success']
    np.testing.assert_allclose(r['x'], [1.])


def test_dual_pivot_restores_feasibility():
    s = simplex([[-1.]], [1.], [1.])
    s.tableau = s.tableau.at[0, -1].set(-1.)
    s.dual()
    assert float(s.tableau[0, -1]) >= 0
    np.testing.assert_allclose(s.x, [1.])


def test_iteration_limit_is_reported():
    r = simplex([[1.]], [1.], [-1.]).solve(max_iter=0)
    assert not r['success'] and r['status'] == 1


def test_optimum_on_last_allowed_pivot_is_successful():
    r = simplex([[1.]], [1.], [-1.]).solve(max_iter=1)
    assert r['success'] and r['iterations'] == 1


def test_initial_optimum_needs_no_pivot_budget():
    r = simplex([[1.]], [1.], [1.]).solve(max_iter=0)
    assert r['success'] and r['iterations'] == 0


def test_solve_can_resume_with_a_fresh_pivot_budget():
    s = simplex([[1., 0.], [0., 1.]], [1., 1.], [-1., -1.])
    first = s.solve(max_iter=1)
    assert first['status'] == 1 and first['iterations_this_solve'] == 1
    second = s.solve(max_iter=1)
    assert second['success'] and second['iterations'] == 2
    assert second['iterations_this_solve'] == 1


@pytest.mark.parametrize('row_scale', [1e-12, 1., 1e12])
@pytest.mark.parametrize('cost_scale', [1e-12, 1., 1e12])
def test_equivalent_units_do_not_change_the_optimum(row_scale, cost_scale):
    r = simplex([[row_scale]], [row_scale], [-cost_scale]).solve()
    assert r['success'], r
    np.testing.assert_allclose(r['x'], [1.], atol=1e-12)
    np.testing.assert_allclose(r['f']/cost_scale, -1., atol=1e-12)


def test_small_decision_column_is_not_treated_as_an_unbounded_ray():
    r = simplex([[1., 1e-12]], [1.], [0., -1.]).solve()
    assert r['success'], r
    np.testing.assert_allclose(r['x'], [0., 1e12], rtol=1e-12)


@pytest.mark.parametrize('first_upper', [0., 1.])
def test_small_cost_over_large_variable_range_is_not_ignored(first_upper):
    r = simplex([], [], [1., -1e-12], ub=[first_upper, 1e12]).solve()
    assert r['success'], r
    np.testing.assert_allclose(r['x'], [0., 1e12], rtol=1e-12)
    np.testing.assert_allclose(r['f'], -1., atol=1e-12)


def test_implied_variable_range_is_scaled():
    r = simplex([[1., 1.]], [1e12], [1., -1e-12]).solve()
    assert r['success'], r
    np.testing.assert_allclose(r['f'], -1., atol=1e-12)


def test_unconstrained_small_cost_has_an_unbounded_ray():
    r = simplex([], [], [1., 1e-12], lb=[0., -np.inf], ub=[1., 1e12]).solve()
    assert not r['success'] and r['status'] == 2


@pytest.mark.parametrize('pricing', ['bland', 'dantzig', 'steepest-edge'])
@pytest.mark.parametrize('residual', [1e-17, 4e-16])
def test_phase_one_ignores_roundoff_in_zero_cost_column(pricing, residual):
    from optinpy.simplex.base import _primal_phase
    # x0's first coefficient is mathematically zero after elimination, but
    # the compiler may leave a rounding residual. Its apparent improvement
    # has no eligible pivot. x1 removes the remaining artificial variable.
    table = jnp.array([[residual, 1., 1., 0., 1.],
                       [-1., 0., 0., 1., 0.],
                       [0., 0., 0., 0., 0.]])
    basic = jnp.array([2, 3], dtype=jnp.int32)
    cost = jnp.array([0., 0., 1., 0.])
    result, basis, pivots, status, _, _ = _primal_phase(
        table, basic, cost, 1e-9, 10, pricing=pricing)
    assert int(status) == 0 and int(pivots) == 1
    np.testing.assert_array_equal(basis, [1, 3])
    np.testing.assert_allclose(result[:-1, -1], [1., 0.])


def test_zero_basic_costs_preserve_tiny_real_improvements():
    from optinpy.simplex.base import _primal_phase
    table = jnp.array([[1., 1., 1.], [0., 0., 0.]])
    result, basis, pivots, status, _, _ = _primal_phase(
        table, jnp.array([1], dtype=jnp.int32), jnp.array([-1e-20, 0.]),
        1e-9, 10, pricing='bland')
    assert int(status) == 0 and int(pivots) == 1
    np.testing.assert_array_equal(basis, [0])
    assert float(result[0, -1]) == 1.


def test_ratio_near_tie_does_not_cross_the_tightest_constraint():
    r = simplex([[1.], [1e6]], [1.+5e-10, 1e6], [-1.]).solve()
    assert r['success'], r
    np.testing.assert_allclose(r['x'], [1.], atol=1e-12)
    assert r['primal_violation'] == 0


@pytest.mark.parametrize('pricing', ['bland', 'dantzig', 'steepest-edge'])
def test_pricing_rules_handle_classical_cycling_example(pricing):
    # Beale's cycling LP; Bland's rule should reach objective 1 without cycling.
    r = simplex([[.5, -5.5, -2.5, 9.], [.5, -1.5, -.5, 1.], [1., 0., 0., 0.]],
                [0., 0., 1.], [10., -57., -9., -24.], mode='max', pricing=pricing).solve(max_iter=100)
    assert r['success'], r
    np.testing.assert_allclose(r['f'], 1., atol=1e-10)


def test_cycling_guard_persists_across_one_pivot_budgets():
    solver = simplex([[.5, -5.5, -2.5, 9.], [.5, -1.5, -.5, 1.], [1., 0., 0., 0.]],
                     [0., 0., 1.], [10., -57., -9., -24.], mode='max', pricing='dantzig')
    # Equilibration can break Beale's cycle by changing the pricing. Exercise
    # its canonical slack basis directly, retaining the same original LP.
    import jax.numpy as jnp
    solver.tableau = jnp.array([[.5, -5.5, -2.5, 9., 1., 0., 0., 0.],
                               [.5, -1.5, -.5, 1., 0., 1., 0., 0.],
                               [1., 0., 0., 0., 0., 0., 1., 1.],
                               [-10., 57., 9., 24., 0., 0., 0., 0.]])
    solver._cost = jnp.array([-10., 57., 9., 24., 0., 0., 0.])
    solver._transform = jnp.eye(4)
    for _ in range(100):
        result = solver.solve(max_iter=1)
        assert result['iterations_this_solve'] <= 1
        if result['success']:
            break
    assert result['success'] and result['bland_fallback']
    np.testing.assert_allclose(result['f'], 1., atol=1e-10)


@pytest.mark.parametrize('pricing, expected', [('bland', [1., 0.]), ('dantzig', [0., 1.]), ('steepest-edge', [0., 1.])])
def test_selected_pricing_rule_controls_the_first_pivot(pricing, expected):
    result = simplex([[1., 0.], [0., 1.]], [1., 1.], [-1., -3.], pricing=pricing).solve(max_iter=1)
    np.testing.assert_allclose(result['x'], expected)


def test_invalid_pricing():
    with pytest.raises(ValueError, match='pricing'):
        simplex([[1.]], [1.], [1.], pricing='unknown')


def test_phase_one_cleanup_respects_budget():
    s = simplex([[-1.], [1.]], [-1., 1.], [1.])
    r = s.solve(max_iter=1)
    assert r['iterations_this_solve'] <= 1
    if not r['success']:
        r = s.solve(max_iter=10)
    assert r['success']
    np.testing.assert_allclose(r['x'], [1.])


def test_exactly_inconsistent_zero_row_is_infeasible():
    r = simplex([[0.]], [-1e-20], [0.]).solve(max_iter=0)
    assert r['status'] == 3


def test_nonfinite_tableau_never_reports_optimal():
    s = simplex([[1.]], [1.], [-1.])
    s.tableau = s.tableau.at[0, 0].set(np.nan)
    r = s.solve()
    assert not r['success'] and r['status'] == 4


def test_overflow_during_scaling_is_a_numerical_failure():
    r = simplex([], [], [-1e308], ub=[1e308]).solve()
    assert not r['success'] and r['status'] == 4


@pytest.mark.parametrize('tol', [0., -1., np.nan, np.inf])
def test_invalid_tolerance(tol):
    with pytest.raises(ValueError, match='tol'):
        simplex([[1.]], [1.], [1.], tol=tol)

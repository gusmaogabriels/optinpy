"""Independent numerical checks; comparison libraries are optional test extras."""
import pytest

pytest.importorskip('scipy')
from benchmarks.simplex_cases import curated_cases, random_cases, normalize
from benchmarks.simplex_solvers import optinpy_solve, highs_solve, clarabel_solve


@pytest.fixture(scope='module', params=curated_cases()+random_cases(8), ids=lambda lp: lp.name)
def problem(request):
    lp = normalize(request.param)
    return lp, optinpy_solve(lp)


@pytest.mark.parametrize('reference', ['highs-ds', 'highs-ipm', 'clarabel'])
def test_matches_independent_solver(problem, reference):
    if reference == 'clarabel':
        pytest.importorskip('clarabel')
    lp, own = problem
    ref = clarabel_solve(lp) if reference == 'clarabel' else highs_solve(lp, method=reference)
    assert own['status'] == lp.expected_status, (lp.name, own)
    if lp.expected_status == 'optimal':
        assert ref['status'] in ('optimal', 'approximate'), (lp.name, ref)
        assert own['violation'] <= 1e-7, (lp.name, own)
        assert ref['violation'] <= 1e-7, (lp.name, ref)
        assert abs(own['objective']-ref['objective']) <= 1e-7*(1+abs(ref['objective'])), (lp.name, own, ref)
        if lp.expected_objective is not None:
            assert abs(own['objective']-lp.expected_objective) <= 1e-7*(1+abs(lp.expected_objective))
    else:
        assert ref['status'] == lp.expected_status, (lp.name, ref)


def test_individual_pivots_match_compiled_solve(problem):
    """The exposed one-step API retains the same LP behavior as the fast driver."""
    from optinpy import simplex
    lp, compiled = problem
    solver = simplex(lp.A, lp.b, lp.c, lb=lp.lb, ub=lp.ub, mode=lp.mode)
    for _ in range(10000):
        if solver.status is not None:
            break
        solver.primal()
    assert solver.status is not None, lp.name
    stepped = solver.solve(max_iter=0)
    from benchmarks.simplex_solvers import STATUS
    assert STATUS[stepped['status']] == compiled['status'], lp.name
    if stepped['success']:
        assert abs(float(stepped['f'])-compiled['objective']) <= 1e-7*(1+abs(compiled['objective']))
        assert float(stepped['scaled_primal_violation']) <= 1e-7


@pytest.mark.parametrize('pricing', ['bland', 'steepest-edge'])
def test_alternative_pricing_matches_reference(problem, pricing):
    from optinpy import simplex
    from benchmarks.simplex_solvers import STATUS
    lp, default = problem
    result = simplex(lp.A, lp.b, lp.c, lb=lp.lb, ub=lp.ub, mode=lp.mode, pricing=pricing).solve()
    assert STATUS[result['status']] == default['status'], (lp.name, pricing, result)
    if result['success']:
        reference = highs_solve(lp)
        assert abs(float(result['f'])-reference['objective']) <= 1e-7*(1+abs(reference['objective']))
        assert float(result['scaled_primal_violation']) <= 1e-7

"""Deterministic problems and independent analytic checks for JAX libraries."""
from dataclasses import dataclass

import numpy as np


def _matvec(A, x):
    # Explicit contractions keep reference checks independent of JAX and avoid
    # spurious NumPy 2.1/Accelerate BLAS floating-point flags on macOS.
    return np.einsum('ij,j->i', A, x)


@dataclass
class Case:
    name: str
    kind: str
    args: tuple
    expected: np.ndarray
    x0: np.ndarray | None = None
    positive_definite: bool = False

    @property
    def inputs(self):
        return self.args if self.x0 is None else (self.x0, *self.args)


def _matrix(n, condition=30., indefinite=False):
    q, _ = np.linalg.qr(np.random.default_rng(20260914+n).normal(size=(n, n)))
    eigenvalues = np.geomspace(1., condition, n)
    if indefinite:
        eigenvalues[::2] *= -1
    A = np.einsum('ik,jk->ij', q*eigenvalues, q)
    return (A+A.T)/2


def _quadratic(n):
    target = np.linspace(-.8, .8, n)
    return Case(f'quadratic_{n}', 'quadratic', (_matrix(n), target), target, np.zeros(n))


def _softplus(n):
    A = np.random.default_rng(19+n).normal(size=(2*n, n))/np.sqrt(n)
    target = np.linspace(-.5, .5, n)
    b = _matvec(A.T, 1/(1+np.exp(-_matvec(A, target)))) + .3*target
    return Case(f'softplus_{n}', 'softplus', (A, b), target, np.zeros(n))


def cases(group, quick=False):
    if group == 'linear':
        result = []
        for n, indefinite in ([(16, False)] if quick else
                              [(16, False), (64, False), (256, False), (64, True)]):
            A = _matrix(n, indefinite=indefinite)
            expected = np.linspace(-.8, .8, n)
            result.append(Case(f'{"indefinite" if indefinite else "spd"}_{n}', 'linear',
                               (A, _matvec(A, expected)), expected, positive_definite=not indefinite))
        return result
    if quick:
        return [_quadratic(16)]
    if group == 'first-order':
        return [_quadratic(16), _quadratic(64), _softplus(16)]
    if group == 'nonlinear':
        result = [_quadratic(16), _quadratic(64), _softplus(16), _softplus(64)]
        for n in (2, 8):
            result.append(Case(f'rosenbrock_{n}', 'rosenbrock', (), np.ones(n),
                               np.tile([-1.2, 1.], n//2)))
        return result
    raise ValueError(f'Unknown benchmark group: {group}')


def analytic(case, x):
    """Objective and gradient computed with NumPy, independent of solver autodiff."""
    if case.kind == 'quadratic':
        A, target = case.args
        y = x-target
        gradient = _matvec(A, y)
        return float(.5*np.sum(y*gradient)), gradient
    if case.kind == 'softplus':
        A, b = case.args
        z = _matvec(A, x)
        probability = np.exp(-np.logaddexp(0., -z))
        return float(np.logaddexp(0., z).sum()+.15*np.sum(x*x)-np.sum(b*x)), _matvec(A.T, probability)+.3*x-b
    if case.kind == 'rosenbrock':
        difference = x[1:]-x[:-1]**2
        g = np.zeros_like(x)
        g[:-1] = -400*x[:-1]*difference+2*(x[:-1]-1)
        g[1:] += 200*difference
        return float(np.sum(100*difference**2+(1-x[:-1])**2)), g
    raise ValueError(case.kind)


def assess(case, result):
    """Common acceptance rules, independently of each library's stopping rule."""
    x = np.asarray(result['x'], dtype=float)
    finite = bool(np.all(np.isfinite(x)))
    metrics = {'backend_success': bool(result['success']), 'status': str(result['status']),
               'iterations': int(result['iterations']), 'finite': finite, 'passed': False}
    if not finite:
        return metrics
    with np.errstate(over='ignore', invalid='ignore'):
        solution_error = float(np.max(np.abs(x-case.expected))/(1+np.max(np.abs(case.expected))))
    metrics['relative_solution_error'] = solution_error if np.isfinite(solution_error) else None
    if case.kind == 'linear':
        A, b = case.args
        with np.errstate(over='ignore', invalid='ignore'):
            residual = float(np.max(np.abs(_matvec(A, x)-b))/(1+np.max(np.abs(b))))
        metrics['relative_residual'] = residual if np.isfinite(residual) else None
        accurate = residual <= 1e-8 and solution_error <= 1e-7
    else:
        with np.errstate(over='ignore', invalid='ignore'):
            value, gradient = analytic(case, x)
            expected, _ = analytic(case, case.expected)
        error = abs(value-expected)/(1+abs(expected))
        with np.errstate(over='ignore', invalid='ignore'):
            norm = float(np.linalg.norm(gradient))
        # Keep JSON valid even when a finite iterate overflows the objective.
        metrics.update(objective=value if np.isfinite(value) else None,
                       relative_objective_error=error if np.isfinite(error) else None,
                       gradient_norm=norm if np.isfinite(norm) else None)
        accurate = error <= 1e-7 and norm <= 1e-5 and solution_error <= 1e-4
    metrics['passed'] = bool(metrics['backend_success'] and accurate)
    return metrics

"""Optional library adapters; no reference optimizer is used by runtime code."""
import jax
import jax.numpy as jnp

import optinpy


def objective(kind):
    if kind == 'quadratic':
        return lambda x, A, target: .5*(x-target) @ A @ (x-target)
    if kind == 'softplus':
        return lambda x, A, b: jnp.sum(jax.nn.softplus(A @ x))+.15*jnp.sum(x*x)-b @ x
    if kind == 'rosenbrock':
        return lambda x: jnp.sum(100*(x[1:]-x[:-1]**2)**2+(1-x[:-1])**2)
    raise ValueError(kind)


def optax_solve(fun, x0, args, transform, *, tol=1e-6, max_iter=10000):
    """Full-objective loop around Optax updates, matching optinpy's stop rule."""
    value_grad = jax.value_and_grad(fun)
    f, g = value_grad(x0, *args)
    finite = lambda x, f, g: jnp.all(jnp.isfinite(x)) & jnp.isfinite(f) & jnp.all(jnp.isfinite(g))
    state = (x0, f, g, transform.init(x0), jnp.asarray(0), finite(x0, f, g))

    def cond(s):
        return s[5] & (s[4] < max_iter) & (jnp.linalg.norm(s[2]) > tol)

    def step(s):
        import optax
        x, _, gradient, opt_state, iteration, _ = s
        updates, opt_state = transform.update(gradient, opt_state, x)
        x = optax.apply_updates(x, updates)
        f, gradient = value_grad(x, *args)
        return x, f, gradient, opt_state, iteration+1, finite(x, f, gradient)

    x, _, gradient, _, iterations, valid = jax.lax.while_loop(cond, step, state)
    success = valid & (jnp.linalg.norm(gradient) <= tol)
    return {'x': x, 'iterations': iterations, 'success': success,
            'status': jnp.where(success, 0, jnp.where(valid, 1, 3))}


def solvers(case, group):
    """Return callable/config pairs. Every timed solve starts from fresh state."""
    if group == 'linear':
        import lineax as lx

        def native(A, b):
            # This is exactly the numerical solve used for optinpy's Newton
            # direction; it is not presented as a separate optinpy LP solver.
            x = jnp.linalg.solve(A, b)
            valid = jnp.all(jnp.isfinite(x))
            return {'x': x, 'iterations': jnp.asarray(-1), 'success': valid,
                    'status': jnp.where(valid, 0, 1)}

        def wrapped(solver):
            def solve(A, b):
                tags = lx.positive_semidefinite_tag if case.positive_definite else ()
                result = lx.linear_solve(lx.MatrixLinearOperator(A, tags=tags), b, solver, throw=False)
                return {'x': result.value, 'iterations': jnp.asarray(result.stats.get('num_steps', -1)),
                        'success': result.result == lx.RESULTS.successful, 'status': result.result}
            return solve

        result = {'jax-newton-kernel': (native, {'kernel': 'jax.numpy.linalg.solve', 'factorization_reused': False}),
                  'lineax-lu': (wrapped(lx.LU()), {'solver': 'LU', 'factorization_reused': False})}
        if case.positive_definite:
            result['lineax-cholesky'] = (wrapped(lx.Cholesky()), {'solver': 'Cholesky', 'factorization_reused': False})
            result['lineax-cg'] = (wrapped(lx.CG(rtol=1e-10, atol=1e-10, max_steps=10*case.expected.size)),
                                    {'solver': 'CG', 'rtol': 1e-10, 'atol': 1e-10,
                                     'max_steps': 10*case.expected.size, 'matrix_free': False})
        return result

    fun = objective(case.kind)
    result = {}
    methods = ('adam', 'sgd') if group == 'first-order' else ('bfgs', 'dfp', 'lbfgs')
    for method in methods:
        settings = {'method': method, 'tol': 1e-6,
                    'max_iter': 10000 if group == 'first-order' else 2000}
        if group == 'first-order':
            settings['learning_rate'] = .01
        if method == 'adam':
            settings.update(beta1=.9, beta2=.999, adam_epsilon=1e-8)
        if method == 'lbfgs':
            settings['memory_size'] = 10
        def native(x, *args, settings=settings):
            r = optinpy.minimize(fun, x, args=args, **settings)
            return {key: r[key] for key in ('x', 'iterations', 'success', 'status')}
        result[f'optinpy-{method}'] = (native, dict(settings, linesearch='none' if group == 'first-order' else 'strong-wolfe'))

        if group == 'first-order':
            import optax
            tx = optax.adam(.01, b1=.9, b2=.999, eps=1e-8, eps_root=0.) if method == 'adam' else optax.sgd(.01)
            def reference(x, *args, tx=tx):
                return optax_solve(fun, x, args, tx)
            configuration = {'method': method, 'learning_rate': .01, 'tol': 1e-6, 'max_iter': 10000}
            if method == 'adam':
                configuration.update(beta1=.9, beta2=.999, epsilon=1e-8, eps_root=0.)
            result[f'optax-{method}'] = (reference, configuration)
        else:
            import optimistix as ox
            cls = {'bfgs': ox.BFGS, 'dfp': ox.DFP, 'lbfgs': ox.LBFGS}[method]
            options = {'history_length': 10} if method == 'lbfgs' else {}
            solver = cls(rtol=0., atol=1e-10, norm=ox.two_norm, **options)
            def reference(x, *args, solver=solver):
                r = ox.minimise(lambda y, data: fun(y, *data), solver, x, args=args,
                                max_steps=2000, throw=False)
                return {'x': r.value, 'iterations': r.stats['num_steps'],
                        'success': r.result == ox.RESULTS.successful, 'status': r.result}
            result[f'optimistix-{method}'] = (reference, {'method': method, 'rtol': 0., 'atol': 1e-10,
                                                         'norm': 'two_norm', 'max_steps': 2000,
                                                         'linesearch': 'default BacktrackingArmijo', **options})
    return result

"""Compare JAX derivative modes on identical dynamic inputs and analytic answers.

Warm calls are interleaved after compilation; each trial uses a fresh CPU
process. This measures derivatives, not complete solver performance.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
from statistics import median
import subprocess
import sys
import time


def worker(args):
    import jax
    import jax.numpy as jnp
    import numpy as np
    jax.config.update('jax_enable_x64', True)
    rng = np.random.default_rng(20260914)
    records = []

    def measure(case, functions, inputs, expected):
        jax.block_until_ready(inputs)
        names = list(functions)
        if args.worker_trial % 2:
            names.reverse()
        compiled, rows = {}, {}
        for name in names:
            started = time.perf_counter_ns()
            lowered = jax.jit(functions[name]).lower(*inputs)
            middle = time.perf_counter_ns()
            execute = lowered.compile()
            end = time.perf_counter_ns()
            result = jax.block_until_ready(execute(*inputs))
            ready = time.perf_counter_ns()
            np.testing.assert_allclose(result, expected, atol=1e-9, rtol=1e-9)
            compiled[name] = execute
            rows[name] = {'case': case, 'mode': name, 'samples_ms': [],
                          'lower_ms': (middle-started)/1e6,
                          'compile_ms': (end-middle)/1e6,
                          'first_execution_ms': (ready-end)/1e6,
                          'first_staged_call_ms': (ready-started)/1e6,
                          'max_absolute_error': float(np.max(np.abs(result-expected)))}
            for _ in range(3):
                jax.block_until_ready(execute(*inputs))
        for sample in range(args.repeats):
            offset = sample % len(names)
            for name in names[offset:] + names[:offset]:
                started = time.perf_counter_ns()
                result = jax.block_until_ready(compiled[name](*inputs))
                elapsed = (time.perf_counter_ns()-started)/1e6
                np.testing.assert_allclose(result, expected, atol=1e-9, rtol=1e-9)
                rows[name]['samples_ms'].append(elapsed)
        records.extend(rows.values())

    def scalar(x, A):
        return jnp.sum(jax.nn.softplus(A @ x)) + .15*jnp.sum(x*x)

    for n in args.sizes:
        A = rng.normal(size=(2*n, n))/np.sqrt(n)
        x = np.linspace(-.5, .5, n)
        p = 1/(1+np.exp(-(A @ x)))
        gradient = A.T @ p + .3*x
        H = A.T @ ((p*(1-p))[:, None]*A) + .3*np.eye(n)
        inputs = (jnp.asarray(x), jnp.asarray(A))
        measure(f'gradient/{n}', {'grad': jax.grad(scalar), 'jacrev': jax.jacrev(scalar),
                                 'jacfwd': jax.jacfwd(scalar)}, inputs, gradient)
        measure(f'hessian/{n}', {
            'fwd-over-rev': jax.jacfwd(jax.jacrev(scalar)),
            'rev-over-fwd': jax.jacrev(jax.jacfwd(scalar)),
            'rev-over-rev': jax.jacrev(jax.jacrev(scalar)),
            'fwd-over-fwd': jax.jacfwd(jax.jacfwd(scalar))}, inputs, H)
        v = np.linspace(.1, 1., n)
        measure(f'hessian-vector/{n}', {
            'jvp-grad': lambda x, A, v: jax.jvp(lambda y: jax.grad(scalar)(y, A), (x,), (v,))[1],
            'grad-dot-grad': lambda x, A, v: jax.grad(lambda y: jnp.vdot(jax.grad(scalar)(y, A), v))(x),
            'dense-hessian-product': lambda x, A, v: jax.hessian(scalar)(x, A) @ v},
            (*inputs, jnp.asarray(v)), H @ v)

    vector = lambda x, A: jnp.sin(A @ x)
    for outputs, inputs_count in ((8, 128), (128, 8), (64, 64)):
        A = rng.normal(size=(outputs, inputs_count))/np.sqrt(inputs_count)
        x = np.linspace(-.5, .5, inputs_count)
        J = np.cos(A @ x)[:, None]*A
        inputs = (jnp.asarray(x), jnp.asarray(A))
        case = f'jacobian/{outputs}x{inputs_count}'
        measure(case, {'jacfwd': jax.jacfwd(vector), 'jacrev': jax.jacrev(vector)}, inputs, J)
        cotangent = np.linspace(.1, 1., outputs)
        measure(f'transpose-product/{outputs}x{inputs_count}', {
            'vjp': lambda x, A, w: jax.vjp(lambda y: vector(y, A), x)[1](w)[0],
            'dense-jacrev-product': lambda x, A, w: jax.jacrev(vector)(x, A).T @ w},
            (*inputs, jnp.asarray(cotangent)), J.T @ cotangent)
    return {'trial': args.worker_trial, 'jax': jax.__version__,
            'jaxlib': jax.lib.__version__, 'devices': list(map(str, jax.devices())), 'records': records}


def report(data, raw_name):
    import numpy as np
    lines = ['# Forward and reverse autodiff: warm execution', '',
             'CPU float64, with dynamic matrices, points and tangent/cotangent vectors. '
             'Scalar objectives are coupled softplus sums plus a quadratic term; rectangular '
             'Jacobians use sin(Ax). Every result is checked against analytic derivatives. '
             'Different modes compute the same requested derivative or product.', '',
             f'{data["metadata"]["trials"]} fresh processes, three warmups, '
             f'{data["metadata"]["repeats"]} interleaved timed calls per mode in each process. '
             'Compilation order reverses between trials; warm order rotates. '
             'All calls synchronize and no samples are discarded. Persistent caching is disabled. '
             'Inputs, device initialization and imports are outside the timers. '
             'First staged call includes tracing/lowering, compilation and first execution; '
             'it is not total process startup. P5–P95 is the observed middle 90%, not a confidence interval.', '',
             '| Case | Mode | Lower (ms) | Compile (ms) | First staged call (ms) | Warm median (µs) | P5–P95 (µs) |',
             '| --- | --- | ---: | ---: | ---: | ---: | --- |']
    keys = [(r['case'], r['mode']) for r in data['raw_trials'][0]['records']]
    for case, mode in keys:
        rows = [r for t in data['raw_trials'] for r in t['records'] if (r['case'], r['mode']) == (case, mode)]
        warm = [v*1000 for r in rows for v in r['samples_ms']]
        timings = [median(r[k] for r in rows) for k in ('lower_ms', 'compile_ms', 'first_staged_call_ms')]
        lines.append(f'| {case} | {mode} | '+ ' | '.join(f'{v:.3f}' for v in timings) +
                     f' | {median(warm):.3f} | {np.percentile(warm,5):.3f}–{np.percentile(warm,95):.3f} |')
    lines += ['', 'These are derivative microbenchmarks on this CPU. They do not establish a '
              'universal fastest mode, whole-solver speedup, or GPU/memory claim. '
              'The JAX compiler may simplify dense-product expressions.', '',
              'Reproduce with `python -m benchmarks.autodiff_modes --trials '+str(data['metadata']['trials'])+
              ' --repeats '+str(data['metadata']['repeats'])+' --sizes '+
              ' '.join(map(str,data['metadata']['sizes']))+'`.', '',
              f'[Raw samples and environment]({raw_name}). '
              '[JAX autodiff guidance](https://docs.jax.dev/en/latest/301/cookbook.html).', '',
              '```json', json.dumps(data['metadata'], indent=2), '```', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--sizes', type=int, nargs='+', default=[16, 64, 128])
    parser.add_argument('--worker-trial', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--output', type=Path, default=Path('docs/autodiff-modes'))
    args = parser.parse_args()
    if min(args.trials, args.repeats, *args.sizes) < 1:
        parser.error('Trials, repeats and sizes must be positive')
    if args.worker_trial is not None:
        print(json.dumps(worker(args), allow_nan=False))
        return
    trials = []
    env = dict(os.environ, JAX_PLATFORMS='cpu', JAX_ENABLE_COMPILATION_CACHE='false')
    for trial in range(args.trials):
        result = subprocess.run([sys.executable, '-m', 'benchmarks.autodiff_modes',
                                 '--worker-trial', str(trial), '--repeats', str(args.repeats),
                                 '--sizes', *map(str,args.sizes)], env=env, text=True,
                                capture_output=True, check=True, timeout=300)
        trials.append(json.loads(result.stdout))
        print(f'Completed autodiff trial {trial+1}/{args.trials}', flush=True)
    data = {'metadata': {'utc': datetime.now(timezone.utc).isoformat(), 'platform': platform.platform(),
                         'python': platform.python_version(), 'trials': args.trials, 'repeats': args.repeats,
                         'sizes': args.sizes, 'persistent_compilation_cache': False,
                         'jax': trials[0]['jax'], 'jaxlib': trials[0]['jaxlib'], 'devices': trials[0]['devices'],
                         'benchmark_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
            'raw_trials': trials}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.json').write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    args.output.with_suffix('.md').write_text(report(data, args.output.with_suffix('.json').name))


if __name__ == '__main__':
    main()

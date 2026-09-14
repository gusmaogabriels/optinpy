"""Synchronized CPU benchmarks for native JAX algorithms and API usage.

The optional baseline is trusted local Git history. Each revision/trial runs
in its own process; both revisions receive the same objectives and settings.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
from statistics import median
import subprocess
import sys
import tarfile
import tempfile
import time


METHODS = ('gradient', 'newton', 'modified-newton', 'conjugate-gradient',
           'hessian-conjugate-gradient', 'fletcher-reeves', 'bfgs', 'dfp', 'lbfgs', 'adam', 'sgd')


def worker(args):
    if args.package_root:
        sys.path.insert(0, args.package_root)
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optinpy as op
    from optinpy.nonlinear.constrained import constrained
    jax.config.update('jax_enable_x64', True)
    records = []

    def measure(name, function, inputs, validate, *, staged=True):
        # Arrays and device initialization are outside these kernel/API timers.
        # Whole fresh-process startup is measured by time_simplex separately.
        jax.block_until_ready(inputs)
        jax.clear_caches()
        row = {'name': name, 'staged': staged, 'samples_ms': [], 'iterations': []}
        if staged:
            started = time.perf_counter_ns()
            lowered = jax.jit(function).lower(*inputs)
            row['lower_ms'] = (time.perf_counter_ns()-started)/1e6
            started = time.perf_counter_ns()
            execute = lowered.compile()
            row['compile_ms'] = (time.perf_counter_ns()-started)/1e6
        else:
            execute = function
        for index in range(1 + (args.repeats if staged else args.eager_repeats)):
            started = time.perf_counter_ns()
            result = jax.block_until_ready(execute(*inputs))
            elapsed = (time.perf_counter_ns()-started)/1e6
            validate(result)
            if index == 0:
                row['first_execution_ms'] = elapsed
            else:
                row['samples_ms'].append(elapsed)
            if isinstance(result, dict) and 'iterations' in result:
                row['iterations'].append(np.asarray(result['iterations']).tolist())
        records.append(row)

    def optimum(expected, tolerance=2e-4):
        expected = np.asarray(expected)
        def check(result):
            assert np.all(result['success']), (expected, result)
            np.testing.assert_allclose(result['x'], expected, atol=tolerance, rtol=0)
            if 'constraint_violation' in result:
                assert float(result['constraint_violation']) <= tolerance, result
        return check

    def quadratic(x, target, weights):
        return jnp.sum(weights*(x-target)**2)

    if args.group in ('all', 'unconstrained'):
        x0, target, weights = jnp.zeros(8), jnp.linspace(-1., 1., 8), jnp.linspace(1., 3., 8)
        for method in METHODS:
            solve = lambda x, t, w: op.minimize(quadratic, x, args=(t, w), method=method,
                                               tol=1e-5, max_iter=3000, learning_rate=0.03)
            measure('minimize/'+method, solve, (x0, target, weights), optimum(target))
        for n in (64, 256):
            inputs = (jnp.zeros(n), jnp.linspace(-1., 1., n), jnp.linspace(1., 10., n))
            solve = lambda x, t, w: op.minimize(quadratic, x, args=(t, w), method='bfgs', tol=1e-6)
            measure(f'bfgs/{n}', solve, inputs, optimum(inputs[1]))

    if args.group in ('all', 'kernels'):
        x0 = jnp.linspace(-1., 1., 16)
        fun = lambda x: jnp.sum(x*x)
        for name in ('backtracking', 'interp23', 'strong_wolfe', 'golden_section', 'unimodality'):
            def check(result):
                assert bool(result['success']), result
                assert float(result['f']) < float(fun(x0)), result
                np.testing.assert_allclose(result['f'], fun(result['x']), atol=1e-10)
            kwargs = {'key': jax.random.key(19)} if name == 'unimodality' else {}
            measure('linesearch/'+name, lambda x: getattr(op.linesearch, name)(fun, x, **kwargs), (x0,), check)
        for algorithm in ('autodiff', 'central', 'forward', 'backward'):
            for name, expected, tolerance in (('jacobian', 2*x0, 1e-5),
                                               ('hessian', 2*jnp.eye(16), 1e-4)):
                measure(name+'/'+algorithm,
                        lambda x: getattr(op.finitediff, name)(fun, x, algorithm=algorithm), (x0,),
                        lambda result: np.testing.assert_allclose(result, expected, atol=tolerance, rtol=0))
        # Dynamic coupled objectives expose probe-layout costs beyond the tiny
        # quadratic whose derivatives XLA can simplify especially aggressively.
        rng = np.random.default_rng(20260914)
        for n in (16, 64, 128):
            A = rng.normal(size=(2*n, n))/np.sqrt(n)
            x = np.linspace(-.5, .5, n)
            probability = 1/(1+np.exp(-(A @ x)))
            expected = A.T @ ((probability*(1-probability))[:, None]*A) + .3*np.eye(n)
            def evaluate(x, A):
                fun = lambda y: jnp.sum(jax.nn.softplus(A @ y)) + .15*jnp.sum(y*y)
                return op.finitediff.hessian(fun, x, algorithm='central')
            measure(f'hessian/central-coupled/{n}', evaluate, (jnp.asarray(x), jnp.asarray(A)),
                    lambda result: np.testing.assert_allclose(result, expected, atol=1e-4, rtol=0))

    if args.group in ('all', 'constrained'):
        solver = constrained(deepcopy(op.params), op.unconstrained)
        n = 16
        weights = jnp.linspace(1., 3., n)
        fun = lambda x: jnp.sum(weights*(x-2)**2)
        # Symmetric target, sum(x)<=n: analytic KKT multiplier yields x=2-k/w.
        expected = 2 - n/(jnp.sum(1/weights)*weights)
        measure('constrained/projected-gradient',
                lambda x: solver.fmincon(fun, x, A=jnp.ones((1, n)), b=jnp.array([n]), threshold=1e-10),
                (jnp.zeros(n),), optimum(expected), staged=False)
        for method in ('penalty', 'barrier', 'log-barrier'):
            solver.params['fminnlcon']['method'] = method
            measure('constrained/'+method,
                    lambda x: solver.fminnlcon(lambda y: (y[0]-2)**2, x,
                                               [lambda y: y[0]-1], threshold=1e-8),
                    (jnp.zeros(1),), optimum([1.]), staged=False)

    if args.group in ('all', 'usage'):
        options = dict(method='bfgs', tol=1e-6)
        x0, target, weights = jnp.zeros(8), jnp.linspace(-1., 1., 8), jnp.linspace(1., 3., 8)
        direct = lambda x, t, w: op.minimize(quadratic, x, args=(t, w), **options)
        # The baseline already supports an outer jit. Do not credit the factory
        # for a capability available through that wrapper in the older version.
        reusable = (op.compile_minimizer(quadratic, **options) if hasattr(op, 'compile_minimizer')
                    else jax.jit(direct))
        measure('usage/direct', direct, (x0, target, weights), optimum(target), staged=False)
        measure('usage/reusable', reusable, (x0, target, weights), optimum(target), staged=False)
        targets = jnp.stack([target + i/16 for i in range(16)])
        def serial(x, ts, w):
            rows = [reusable(x, ts[i], w) for i in range(len(ts))]
            return {'x': jnp.stack([row['x'] for row in rows]),
                    'success': jnp.stack([row['success'] for row in rows])}
        measure('batch/16-serial', serial, (x0, targets, weights), optimum(targets), staged=False)
        measure('batch/16-vmap', jax.vmap(direct, in_axes=(None, 0, None)),
                (x0, targets, weights), optimum(targets))

    root = Path(op.__file__).parent
    return {'records': records, 'versions': {name: importlib.metadata.version(name)
                                            for name in ('jax', 'jaxlib', 'numpy')},
            'devices': [str(device) for device in jax.devices()],
            'source_sha256': {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in sorted(root.rglob('*.py'))}}


def run(args):
    trials = []
    env = dict(os.environ, JAX_PLATFORMS='cpu', JAX_ENABLE_COMPILATION_CACHE='false')
    with tempfile.TemporaryDirectory(prefix='optinpy-jax-baseline-') as tmp:
        baseline = None
        if args.baseline_ref:
            baseline = subprocess.run(['git', 'rev-parse', args.baseline_ref], check=True,
                                      capture_output=True, text=True).stdout.strip()
            archive = subprocess.run(['git', 'archive', '--format=tar', baseline, 'optinpy'],
                                     check=True, capture_output=True).stdout
            with tarfile.open(fileobj=io.BytesIO(archive)) as source:
                source.extractall(tmp, filter='data')
        for trial in range(args.trials):
            revisions = ['current', 'baseline'] if baseline else ['current']
            if trial % 2:
                revisions.reverse()
            for revision in revisions:
                command = [sys.executable, '-m', 'benchmarks.jax_audit', '--worker',
                           '--group', args.group, '--repeats', str(args.repeats),
                           '--eager-repeats', str(args.eager_repeats)]
                if revision == 'baseline':
                    command.extend(['--package-root', tmp])
                result = subprocess.run(command, capture_output=True, text=True,
                                        env=env, timeout=600)
                if result.returncode:
                    raise RuntimeError(f'{revision} benchmark failed:\n{result.stderr}')
                trials.append(dict(trial=trial, revision=revision, **json.loads(result.stdout)))
                print(f'Completed trial {trial+1}/{args.trials}: {revision}', flush=True)
    return {'metadata': {'utc': datetime.now(timezone.utc).isoformat(), 'platform': platform.platform(),
                         'python': platform.python_version(), 'trials': args.trials,
                         'repeats': args.repeats, 'eager_repeats': args.eager_repeats,
                         'group': args.group, 'baseline_ref': baseline,
                         'persistent_compilation_cache': False}, 'raw_trials': trials}


def report(data, raw_name='jax-performance.json'):
    import numpy as np
    meta = data['metadata']
    lines = ['# Native JAX performance audit: measurements', '',
             'CPU float64 measurements on fixed analytic problems. Both revisions use the same inputs, '
             'objectives, settings, and JAX environment. All results synchronize before stopping the timer; '
             'every sample is checked against an analytic answer (line searches check descent and the '
             'returned objective). No samples are discarded. Timings are informational.', '',
             'Each revision/trial uses a fresh process, alternating revision order between trials. '
             'JAX caches are cleared before each case and persistent caching is disabled. '
             'Prepared input arrays and device initialization are outside these timers. Staged cases '
             'separately measure tracing/lowering, compilation, and calls to the compiled executable. '
             'API cases measure the ordinary Python call, including any compilation it causes: '
             'their repeated timings are not necessarily free of compilation. The first execution '
             'is excluded from the repeated distribution. P5–P95 is an observed range, not a confidence interval.', '',
             '| Case | Revision | Timer | Lower (ms) | Compile (ms) | First execution (ms) | Repeats | Median (ms) | P5–P95 (ms) |',
             '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |']
    names = [r['name'] for r in data['raw_trials'][0]['records']]
    for name in names:
        for revision in ('current', 'baseline'):
            rows = [r for t in data['raw_trials'] if t['revision'] == revision
                    for r in t['records'] if r['name'] == name]
            if not rows:
                continue
            values = [v for row in rows for v in row['samples_ms']]
            optional = lambda key: f'{median(row[key] for row in rows):.3f}' if key in rows[0] else '—'
            lines.append(f'| {name} | {revision} | {"AOT" if rows[0]["staged"] else "API"} | '
                         f'{optional("lower_ms")} | {optional("compile_ms")} | '
                         f'{optional("first_execution_ms")} | {len(values)} | {median(values):.3f} | '
                         f'{np.percentile(values, 5):.3f}–{np.percentile(values, 95):.3f} |')
    lines += ['', 'The reusable API uses `compile_minimizer` in the current revision and an equivalent '
              'outer `jax.jit` in the baseline. Batching compares 16 complete solves in a Python loop '
              'with one compiled `vmap`; both include result collection. These are small dense test '
              'problems, not a representative nonlinear optimization corpus or GPU measurements. '
              'First execution in an AOT row excludes the preceding lowering/compilation. See '
              '[the separate process timing report](simplex-timing.md) for end-to-end startup.', '',
              '## Reproduction', '', '```bash',
              f'python -m benchmarks.jax_audit --group {meta["group"]} --trials {meta["trials"]} '
              f'--repeats {meta["repeats"]} --eager-repeats {meta["eager_repeats"]}' +
              (f' --baseline-ref {meta["baseline_ref"]}' if meta['baseline_ref'] else ''), '```', '',
              'The optional baseline requires the trusted revision in local Git history. Omit it '
              'for a source distribution or shallow checkout. [Raw samples, versions, devices, '
              f'iteration counts, and hashes of all runtime source files]({raw_name}).', '',
              '```json', json.dumps(meta, indent=2), '```', '',
              'Method: [JAX benchmarking](https://docs.jax.dev/en/latest/benchmarking.html) and '
              '[explicit compilation stages](https://docs.jax.dev/en/latest/aot.html).', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=('all', 'unconstrained', 'kernels', 'constrained', 'usage'), default='all')
    parser.add_argument('--trials', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--eager-repeats', type=int, default=3)
    parser.add_argument('--baseline-ref')
    parser.add_argument('--output', type=Path, default=Path('docs/jax-performance'))
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--package-root', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.trials, args.repeats, args.eager_repeats) < 1:
        parser.error('trials and repeat counts must be positive')
    if args.worker:
        print(json.dumps(worker(args), allow_nan=False))
        return
    data = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.json').write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    args.output.with_suffix('.md').write_text(report(data, args.output.with_suffix('.json').name))
    print(args.output.with_suffix('.md'))


if __name__ == '__main__':
    main()

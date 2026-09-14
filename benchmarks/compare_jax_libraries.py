"""Compare native optinpy with optional Lineax, Optimistix, and Optax references.

Run from the checkout with ``python -m benchmarks.compare_jax_libraries``.
Timings are informational; --strict enforces numerical acceptance only.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
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
    from .jax_library_cases import assess, cases
    from .jax_library_solvers import solvers

    jax.config.update('jax_enable_x64', True)
    records = []
    for group in args.groups:
        for case in cases(group, args.quick):
            inputs = jax.block_until_ready(tuple(jnp.asarray(v) for v in case.inputs))
            adapters = solvers(case, group)
            names = list(adapters)
            if args.worker_trial % 2:
                names.reverse()
            compiled, rows = {}, {}
            for name in names:
                fun, configuration = adapters[name]
                row = {'group': group, 'case': case.name, 'solver': name,
                       'configuration': configuration, 'samples': [], 'warmup_checks': []}
                rows[name] = row
                stage = 'lower'
                try:
                    start = time.perf_counter_ns()
                    lowered = jax.jit(fun).lower(*inputs)
                    lower_end = time.perf_counter_ns()
                    row['lower_ms'] = (lower_end-start)/1e6
                    stage = 'compile'
                    execute = lowered.compile()
                    compile_end = time.perf_counter_ns()
                    row['compile_ms'] = (compile_end-lower_end)/1e6
                    stage = 'first_execution'
                    result = jax.block_until_ready(execute(*inputs))
                    ready = time.perf_counter_ns()
                    row['first_execution_ms'] = (ready-compile_end)/1e6
                    row['first_staged_call_ms'] = (ready-start)/1e6
                    row['first_check'] = assess(case, result)
                    compiled[name] = execute
                    stage = 'warmup'
                    for _ in range(args.warmup):
                        result = jax.block_until_ready(execute(*inputs))
                        row['warmup_checks'].append(assess(case, result))
                except Exception as exc:
                    row['error'] = {'stage': stage, 'message': f'{type(exc).__name__}: {exc}'}
                    compiled.pop(name, None)
            for sample in range(args.repeats):
                offset = sample % len(names)
                for name in names[offset:] + names[:offset]:
                    if name not in compiled:
                        continue
                    try:
                        start = time.perf_counter_ns()
                        result = jax.block_until_ready(compiled[name](*inputs))
                        elapsed = (time.perf_counter_ns()-start)/1e6
                        rows[name]['samples'].append({'elapsed_ms': elapsed, **assess(case, result)})
                    except Exception as exc:
                        rows[name]['error'] = {'stage': 'warm_execution',
                                               'message': f'{type(exc).__name__}: {exc}'}
                        compiled.pop(name)
            records.extend(rows.values())
    return {'trial': args.worker_trial, 'devices': list(map(str, jax.devices())),
            'versions': {name: version(name) for name in
                         ('jax', 'jaxlib', 'numpy', 'lineax', 'optimistix', 'optax', 'equinox')},
            'records': records}


def valid(row, repeats):
    checks = [row.get('first_check', {'passed': False}), *row['warmup_checks'], *row['samples']]
    return (not row.get('error') and len(row['samples']) == repeats
            and all(check['passed'] for check in checks))


def report(data, raw_name):
    import numpy as np
    meta = data['metadata']
    lines = ['# JAX library comparison', '',
             'CPU float64, deterministic dense problems with known solutions. '
             'The matrices, targets, right-hand sides and starting points are dynamic inputs. '
             'Every solve starts from fresh optimizer state; direct linear solvers factor the matrix '
             'on every call. This small suite is not a ranking across all optimization workloads.', '',
             '- **Lineax:** LU, Cholesky and CG against `jax.numpy.linalg.solve`, the kernel used '
             'inside optinpy Newton. These rows measure a linear-system component, not a complete '
             'optimizer or linear-programming solver. Cholesky/CG receive the known SPD structure; '
             'LU is the general dense comparison. CG is given a dense matrix, not a matrix-free operator.',
             '- **Optimistix:** BFGS, DFP and L-BFGS against the matching optinpy methods, each '
             'with a 2,000-step budget. optinpy uses gradient-norm tolerance 1e-6 and strong Wolfe; '
             'Optimistix uses its default Armijo search and change-based stopping with rtol=0, '
             'atol=1e-10 and two-norm. Both L-BFGS methods store ten pairs. '
             'These are whole-solve comparisons at common output accuracy, not identical trajectories. '
             'Optimistix counts search steps; optinpy counts outer iterations.',
             '- **Optax:** Adam and SGD inside a benchmark-owned compiled full-objective loop. '
             'Both libraries use learning rate 0.01, gradient-norm tolerance 1e-6 and a 10,000-step '
             'budget. Adam uses beta1=0.9, beta2=0.999 and epsilon=1e-8 outside the square root '
             '(Optax eps_root=0); SGD has no momentum. This is not a minibatch-training benchmark.', '',
             'Acceptance requires backend success and independent NumPy checks on every first, '
             'warmup and timed result. Linear systems require relative residual ≤1e-8 and relative '
             'solution error ≤1e-7. Minimization requires gradient two-norm ≤1e-5, relative objective '
             'error ≤1e-7 and relative solution error ≤1e-4 against the known optimum. '
             'Relative errors divide by 1 plus the reference norm or absolute objective. '
             'Nonlinear cases include coupled quadratics, regularized softplus sums and Rosenbrock chains.', '',
             f'{meta["trials"]} fresh processes, {meta["warmup"]} warmups and {meta["repeats"]} '
             'interleaved timed calls per solver/case/process. Compilation order reverses between '
             'processes; warm order rotates. All calls synchronize; no samples are discarded. '
             'Persistent compilation caching is disabled. Imports, input construction and device '
             'initialization are outside per-solve timers. First staged call includes lowering, '
             'compilation and first execution, but not process startup. Whole-trial process wall time '
             'is recorded separately in the raw data. P5–P95 describes observed spread, not a confidence interval.', '',
             '**FAIL rows retain their timings but cannot support a speed claim for an accurate solve.** '
             'The acceptance column checks all stages; warm pass counts count timed samples only.', '',
             '| Group / case | Solver | Accepted | Warm passes | Lower (ms) | Compile (ms) | First staged call (ms) | Warm median (µs) | P5–P95 (µs) | Steps |',
             '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |']
    keys = sorted({(r['group'], r['case'], r['solver']) for t in data['raw_trials'] for r in t['records']})
    failures = []
    for group, case, solver in keys:
        rows = [r for t in data['raw_trials'] for r in t['records']
                if (r['group'], r['case'], r['solver']) == (group, case, solver)]
        samples = [s for r in rows for s in r['samples']]
        warm = [s['elapsed_ms']*1000 for s in samples]
        passed = all(valid(r, meta['repeats']) for r in rows)
        timing = []
        for key in ('lower_ms', 'compile_ms', 'first_staged_call_ms'):
            values = [r[key] for r in rows if key in r]
            timing.append(f'{median(values):.3f}' if values else '—')
        warm_median = f'{median(warm):.3f}' if warm else '—'
        spread = f'{np.percentile(warm,5):.3f}–{np.percentile(warm,95):.3f}' if warm else '—'
        steps = sorted({s['iterations'] for s in samples if s['iterations'] >= 0})
        steps_text = ', '.join(map(str, steps)) or '—'
        lines.append(f'| {group} / {case} | {solver} | {"pass" if passed else "FAIL"} | '
                     f'{sum(s["passed"] for s in samples)}/{len(samples)} | '+ ' | '.join(timing)+
                     f' | {warm_median} | {spread} | {steps_text} |')
        if not passed:
            failures.append({'group': group, 'case': case, 'solver': solver,
                             'trials': [{'trial': i, 'error': r.get('error'),
                                         'first_check': r.get('first_check')} for i, r in enumerate(rows)]})
    if failures:
        lines += ['', 'Rejected results (including solver status and independent accuracy):', '',
                  '```json', json.dumps(failures, indent=2, allow_nan=False), '```']
    command = ('python -m benchmarks.compare_jax_libraries --groups '+' '.join(meta['groups'])+
               f' --trials {meta["trials"]} --repeats {meta["repeats"]} --warmup {meta["warmup"]}'+
               (' --quick' if meta['quick'] else ''))
    lines += ['', f'Reproduce with `{command}` after `python -m pip install -e ".[benchmark]"`. '
              'References are optional and are never imported by optinpy runtime code.', '',
              f'[Raw samples, checks and environment]({raw_name}). '
              '[Lineax](https://docs.kidger.site/lineax/), '
              '[Optimistix](https://docs.kidger.site/optimistix/), '
              '[Optax](https://optax.readthedocs.io/en/stable/api/optimizers.html).', '',
              '```json', json.dumps(meta, indent=2), '```', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--groups', nargs='+', choices=['linear', 'nonlinear', 'first-order'],
                        default=['linear', 'nonlinear', 'first-order'])
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--strict', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('docs/jax-library-comparison'))
    parser.add_argument('--worker-trial', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.trials, args.repeats, args.warmup) < 1:
        parser.error('Trials, repeats and warmups must be positive')
    if args.worker_trial is not None:
        print(json.dumps(worker(args), allow_nan=False))
        return
    trials = []
    env = dict(os.environ, JAX_PLATFORMS='cpu', JAX_ENABLE_COMPILATION_CACHE='false')
    for trial in range(args.trials):
        start = time.perf_counter()
        command = [sys.executable, '-m', 'benchmarks.compare_jax_libraries',
                   '--worker-trial', str(trial), '--repeats', str(args.repeats),
                   '--warmup', str(args.warmup), '--groups', *args.groups]
        if args.quick:
            command.append('--quick')
        result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=900)
        if result.returncode:
            raise RuntimeError(f'Benchmark worker failed: {result.stderr}\n{result.stdout}')
        data = json.loads(result.stdout)
        data['process_wall_ms'] = (time.perf_counter()-start)*1000
        data['stderr'] = result.stderr
        trials.append(data)
        print(f'Completed JAX library trial {trial+1}/{args.trials}', flush=True)
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root/'optinpy').rglob('*.py')) + sorted((root/'benchmarks').glob('*jax_librar*.py'))
    metadata = {'utc': datetime.now(timezone.utc).isoformat(), 'platform': platform.platform(),
                'python': platform.python_version(), 'groups': args.groups, 'quick': args.quick,
                'trials': args.trials, 'repeats': args.repeats, 'warmup': args.warmup,
                'dtype': 'float64', 'persistent_compilation_cache': False,
                'versions': trials[0]['versions'], 'devices': trials[0]['devices'],
                'source_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    data = {'metadata': metadata, 'raw_trials': trials}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.json').write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    args.output.with_suffix('.md').write_text(report(data, args.output.with_suffix('.json').name))
    rows = [r for t in trials for r in t['records']]
    rejected = sum(not valid(r, args.repeats) for r in rows)
    print(f'{len(rows)-rejected}/{len(rows)} solver/case/trial rows accepted; reports: {args.output}.*')
    if args.strict and rejected:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

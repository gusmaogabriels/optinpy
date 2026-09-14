"""Check warm timing spread with identical LPs and interleaved solver runs.

Each trial uses a fresh CPU process. Solvers are warmed separately, then
measured sequentially in balanced rotating order. All raw samples are saved.
An optional baseline ref loads optinpy's earlier simplex from local Git history.
"""
import argparse
from datetime import datetime, timezone
from functools import partial
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
from statistics import mean, median, pstdev
import subprocess
import sys
import time
import types


def native_solve(cls, lp):
    import jax
    import numpy as np
    from .simplex_cases import violation
    from .simplex_solvers import STATUS
    solver = cls(lp.A, lp.b, lp.c, lb=lp.lb, ub=lp.ub, mode=lp.mode)
    result = jax.block_until_ready(solver.solve(max_iter=10000))
    x = np.asarray(jax.device_get(result['x']), dtype=float)
    return {'status': STATUS[result['status']], 'objective': float(lp.c @ x),
            'violation': violation(lp, x), 'iterations': result['iterations']}


def validate(result, expected):
    if (result['status'] not in ('optimal', 'approximate') or
            abs(result['objective']-expected) > 1e-7*(1+abs(expected)) or
            result['violation'] > 1e-7):
        raise RuntimeError(f'Solver result failed validation: {result}')


def worker(args):
    import jax
    import numpy as np
    from optinpy import simplex
    from .simplex_cases import normalize, random_cases
    from .simplex_solvers import highs_solve, clarabel_solve
    jax.config.update('jax_enable_x64', True)
    solvers = {'optinpy': partial(native_solve, simplex),
               'highs-ds': partial(highs_solve, method='highs-ds'),
               'highs-ipm': partial(highs_solve, method='highs-ipm'),
               'clarabel': clarabel_solve}
    baseline = None
    if args.baseline_ref:
        # Benchmark a trusted revision of this repository, not a different
        # dependency/environment. Git is invoked without a shell.
        source = subprocess.run(['git', 'show', f'{args.baseline_ref}:optinpy/simplex/base.py'],
                                check=True, capture_output=True, text=True).stdout
        module = types.ModuleType('optinpy_benchmark_baseline')
        exec(compile(source, f'<simplex at {args.baseline_ref}>', 'exec'), module.__dict__)
        solvers['optinpy-before'] = partial(native_solve, module.simplex)
        baseline = {'ref': args.baseline_ref, 'sha256': hashlib.sha256(source.encode()).hexdigest()}
    rng = np.random.default_rng(args.seed + args.worker_trial)
    cases = random_cases(max(args.case_indices)+1)
    records = []
    for index in rng.permutation(args.case_indices):
        lp = normalize(cases[index])
        expected = highs_solve(lp)['objective']
        names = list(solvers)
        for _ in range(args.warmup):
            for name in rng.permutation(names):
                validate(solvers[name](lp), expected)
        values = {name: {'samples_ms': [], 'iterations': [],
                         'max_objective_error': 0., 'max_violation': 0.} for name in names}
        orders = []
        for sample in range(args.repeats):
            if sample % len(names) == 0:
                order = list(rng.permutation(names))
            offset = sample % len(names)
            current_order = order[offset:] + order[:offset]
            orders.append(current_order)
            for name in current_order:
                started = time.perf_counter_ns()
                result = solvers[name](lp)
                elapsed = (time.perf_counter_ns()-started)/1e6
                validate(result, expected)
                values[name]['samples_ms'].append(elapsed)
                values[name]['iterations'].append(int(result['iterations']))
                values[name]['max_objective_error'] = max(values[name]['max_objective_error'],
                                                          abs(result['objective']-expected)/(1+abs(expected)))
                values[name]['max_violation'] = max(values[name]['max_violation'], result['violation'])
        records.append({'index': int(index), 'name': lp.name, 'variables': len(lp.c),
                        'constraints': len(lp.b), 'expected_objective': expected,
                        'solvers': values, 'measurement_order': orders})
    return {'trial': args.worker_trial, 'baseline': baseline, 'cases': records,
            'versions': {name: importlib.metadata.version(name)
                         for name in ('jax', 'jaxlib', 'numpy', 'scipy', 'clarabel', 'optinpy')},
            'devices': [str(device) for device in jax.devices()]}


def distribution(samples):
    import numpy as np
    return {'n': len(samples), 'median_ms': median(samples),
            'p05_ms': float(np.percentile(samples, 5)), 'p95_ms': float(np.percentile(samples, 95)),
            'min_ms': min(samples), 'max_ms': max(samples), 'mean_ms': mean(samples),
            'stdev_ms': pstdev(samples), 'cv_percent': 100*pstdev(samples)/mean(samples)}


def run(args):
    trials = []
    env = dict(os.environ, JAX_PLATFORMS='cpu', JAX_ENABLE_COMPILATION_CACHE='false')
    for trial in range(args.trials):
        command = [sys.executable, '-m', 'benchmarks.repeat_simplex', '--worker-trial', str(trial),
                   '--case-indices', *map(str, args.case_indices), '--repeats', str(args.repeats),
                   '--warmup', str(args.warmup), '--seed', str(args.seed)]
        if args.baseline_ref:
            command.extend(['--baseline-ref', args.baseline_ref])
        result = subprocess.run(command, check=True, capture_output=True, text=True, env=env, timeout=300)
        trials.append(json.loads(result.stdout))
        print(f'Completed trial {trial+1}/{args.trials}', flush=True)
    combined = []
    for index in args.case_indices:
        cases = [next(case for case in trial['cases'] if case['index'] == index) for trial in trials]
        row = {key: cases[0][key] for key in ('name', 'index', 'variables', 'constraints')}
        row['solvers'] = {}
        for name in cases[0]['solvers']:
            records = [case['solvers'][name] for case in cases]
            row['solvers'][name] = distribution([value for record in records for value in record['samples_ms']])
            row['solvers'][name].update(trial_medians_ms=[median(record['samples_ms']) for record in records],
                                        pivot_counts=sorted({count for record in records for count in record['iterations']}),
                                        max_objective_error=max(record['max_objective_error'] for record in records),
                                        max_violation=max(record['max_violation'] for record in records))
        combined.append(row)
    return {'metadata': {'utc': datetime.now(timezone.utc).isoformat(), 'platform': platform.platform(),
                         'python': platform.python_version(), 'versions': trials[0]['versions'],
                         'devices': trials[0]['devices'], 'seed': args.seed, 'trials': args.trials,
                         'repeats': args.repeats, 'warmup': args.warmup, 'case_indices': args.case_indices,
                         'persistent_compilation_cache': False, 'baseline': trials[0]['baseline'],
                         'simplex_sha256': hashlib.sha256(Path('optinpy/simplex/base.py').read_bytes()).hexdigest()},
            'cases': combined, 'raw_trials': trials}


def report(data):
    meta = data['metadata']
    lines = ['# Repeated-input simplex timing', '',
             f'Each LP is identical across all solvers and all {meta["trials"]*meta["repeats"]} measured '
             f'samples per solver ({meta["trials"]} fresh processes, {meta["repeats"]} samples per process). '
             f'Each solver receives {meta["warmup"]} untimed warmups per LP in each process. '
             'Measurements run sequentially in balanced rotating order; problem order changes between processes. '
             'Persistent compilation caching is disabled. Every native solve synchronizes all JAX results.', '',
             'Times include a fresh model, complete solve and adapter result conversion/residual calculation. '
             'They exclude warmup and initial compilation. Objective and feasibility checks against HiGHS '
             'run outside the timer for every sample. All reported samples passed `1e-7` relative-objective '
             'and maximum-feasibility checks. No samples are discarded.', '',
             '**A range across different LPs is not run-to-run variance.** The earlier 0.85–10.17 ms range '
             'combined four problems, from 4 variables/2 pivots to 24 variables/247 pivots. '
             'This report measures spread separately for each fixed input. P5–P95 describes the middle 90% '
             'of observed timings, not a confidence interval.', '',
             '| Variables | Solver | Samples | Median (ms) | P5–P95 (ms) | Min–max (ms) | CV (%) | Iterations |',
             '| ---: | --- | ---: | ---: | --- | --- | ---: | --- |']
    for case in data['cases']:
        for name, value in case['solvers'].items():
            lines.append(f'| {case["variables"]} | {name} | {value["n"]} | {value["median_ms"]:.3f} | '
                         f'{value["p05_ms"]:.3f}–{value["p95_ms"]:.3f} | {value["min_ms"]:.3f}–{value["max_ms"]:.3f} | '
                         f'{value["cv_percent"]:.1f} | {value["pivot_counts"]} |')
    lines.extend(['', '## Relative warm performance on identical inputs', '',
                  'Ratios divide the named solver’s median by the current optinpy median. '
                  'A ratio below 1 means the named solver is faster. Iteration counts are algorithm-specific '
                  'and should not be compared as equal units of work across different solver libraries.', '',
                  '| Variables | HiGHS simplex / optinpy | HiGHS IPM / optinpy | Clarabel / optinpy | Earlier optinpy / current |',
                  '| ---: | ---: | ---: | ---: | ---: |'])
    for case in data['cases']:
        own = case['solvers']['optinpy']['median_ms']
        ratios = [f'{case["solvers"][name]["median_ms"]/own:.2f}×' if name in case['solvers'] else '—'
                  for name in ('highs-ds', 'highs-ipm', 'clarabel', 'optinpy-before')]
        lines.append(f'| {case["variables"]} | '+ ' | '.join(ratios)+' |')
    lines.extend(['', '## Process-to-process medians', '', '| Variables | Solver | Median in each process (ms) |',
                  '| ---: | --- | --- |'])
    for case in data['cases']:
        for name, value in case['solvers'].items():
            lines.append(f'| {case["variables"]} | {name} | '+', '.join(f'{v:.3f}' for v in value['trial_medians_ms'])+' |')
    command = (f'python -m benchmarks.repeat_simplex --trials {meta["trials"]} --repeats {meta["repeats"]} '
               f'--warmup {meta["warmup"]} --seed {meta["seed"]} --case-indices '+ ' '.join(map(str, meta['case_indices'])))
    if meta['baseline']:
        command += ' --baseline-ref '+meta['baseline']['ref']
    lines.extend(['', '## Environment and reproduction', '', '```json', json.dumps(meta, indent=2), '```', '',
                  'The optional earlier implementation is loaded from local Git history into the same environment. '
                  'It needs the referenced commit to be present. A source distribution can run the other solvers '
                  'by omitting `--baseline-ref`.', '', '```bash', "python -m pip install -e '.[dev,benchmark]'",
                  command, '```', '', '[Raw samples and measurement orders](simplex-repeatability.json).', ''])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-indices', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20260913)
    parser.add_argument('--baseline-ref')
    parser.add_argument('--worker-trial', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--output', type=Path, default=Path('docs/simplex-repeatability'))
    args = parser.parse_args()
    if min(args.case_indices) < 0 or min(args.trials, args.repeats, args.warmup) < 1:
        parser.error('case indices must be nonnegative; trials, repeats and warmup must be positive')
    if args.worker_trial is not None:
        print(json.dumps(worker(args), allow_nan=False))
        return
    data = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.json').write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    args.output.with_suffix('.md').write_text(report(data))
    print(args.output.with_suffix('.md'))


if __name__ == '__main__':
    main()

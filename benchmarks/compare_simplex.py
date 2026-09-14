"""Run with python -m benchmarks.compare_simplex --output docs/simplex-comparison.

All solvers receive identical row/objective-normalized LPs. Times include
adapter conversion, model construction and solve, with JAX synchronization.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from functools import partial
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
from statistics import median
import time

import numpy as np

from .simplex_cases import curated_cases, random_cases, normalize
from .simplex_solvers import optinpy_solve, highs_solve, clarabel_solve


def compare(lp, result, reference, tolerance):
    status_match = result['status'] == lp.expected_status
    if lp.expected_status != 'optimal':
        return status_match, None
    status_match = result['status'] in ('optimal', 'approximate')
    if not status_match or reference['status'] != 'optimal':
        return False, None
    expected = reference['objective'] if lp.expected_objective is None else lp.expected_objective
    relative_error = abs(result['objective'] - expected) / (1. + abs(expected))
    return status_match and relative_error <= tolerance and result['violation'] <= tolerance, relative_error


def run(random_count, repeats, seed, include_float32):
    solvers = {
        'optinpy-float64': partial(optinpy_solve, dtype='float64'),
        'highs-ds': partial(highs_solve, method='highs-ds'),
        'highs-ipm': partial(highs_solve, method='highs-ipm'),
        'clarabel': clarabel_solve,
    }
    if include_float32:
        solvers['optinpy-float32-default'] = partial(optinpy_solve, dtype='float32')
        solvers['optinpy-float32-tol1e-4'] = partial(optinpy_solve, dtype='float32', tol=1e-4)
    results = []
    cases = [normalize(lp) for lp in curated_cases() + random_cases(random_count, seed)]
    for index, lp in enumerate(cases):
        record = {'name': lp.name, 'category': lp.category, 'variables': len(lp.c),
                  'constraints': len(lp.b), 'expected_status': lp.expected_status, 'solvers': {}}
        for name, solve in solvers.items():
            try:
                start = time.perf_counter()
                first = solve(lp)
                cold = time.perf_counter() - start
                timings = []
                for _ in range(repeats):
                    start = time.perf_counter()
                    result = solve(lp)
                    timings.append(time.perf_counter() - start)
                if result['status'] != first['status']:
                    raise RuntimeError('Solver status changed across repeated solves')
                result.update(first_seconds=cold, median_seconds=median(timings))
            except Exception as exc:
                result = {'status': 'exception', 'error': f'{type(exc).__name__}: {exc}'}
            record['solvers'][name] = result
        reference = record['solvers']['highs-ds']
        for name, result in record['solvers'].items():
            tolerance = 2e-4 if 'float32' in name else 1e-7
            passed, error = compare(lp, result, reference, tolerance)
            result.update(passed=bool(passed), relative_objective_error=error)
        results.append(record)
        if (index+1) % 10 == 0 or index+1 == len(cases):
            print(f'Compared {index+1}/{len(cases)} LPs', flush=True)
    import jax
    source = Path('optinpy/simplex/base.py')
    versions = {pkg: importlib.metadata.version(pkg) for pkg in ['optinpy', 'jax', 'jaxlib', 'numpy', 'scipy', 'clarabel']}
    return {'metadata': {'utc': datetime.now(timezone.utc).isoformat(),
                         'platform': platform.platform(), 'processor': platform.processor(),
                         'python': platform.python_version(), 'versions': versions,
                         'jax_devices': [str(d) for d in jax.devices()],
                         'simplex_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                         'seed': seed, 'random_cases': random_count, 'warm_repeats': repeats,
                         'solver_options': {name: getattr(solve, 'keywords', {}) for name, solve in solvers.items()},
                         'common_row_objective_normalization': True}, 'cases': results}


def report(data):
    names = list(data['cases'][0]['solvers'])
    lines = ['# Simplex comparison', '',
             'Reproducible local CPU comparison of optinpy’s native tableau implementation, '
             'HiGHS dual revised simplex, HiGHS interior point with crossover, and Clarabel. '
             'The two HiGHS algorithms share one solver library; Clarabel is a separate implementation.', '',
             'All LPs receive identical row/objective normalization before timing. '
             'Optinpy’s unit-scaling regressions separately test unnormalized coefficients. '
             'Timings include adapter/model construction and solving, and synchronize JAX results. '
             'The first call is recorded separately; warm times are medians of repeated fresh solves. '
             'Later first calls may reuse compiled kernels. These are not fresh-process cold timings. '
             'See [startup and JIT timing](simplex-timing.md) for separate process, compilation and execution measurements. '
             'See [repeated-input measurements](simplex-repeatability.md) for timing spread on identical LPs. '
             'There is no sparse/large-scale performance claim.', '',
             'A Clarabel dual-infeasibility certificate is disambiguated with a zero-objective '
             'feasibility solve, included in its timing. This distinguishes an unbounded primal '
             'from simultaneous primal/dual infeasibility. Reduced-accuracy solution statuses '
             'are retained and must still pass the same objective/residual checks.', '',
             'Float32 is measured both with the default stopping tolerance and with an explicit '
             '`tol=1e-4`. The default uses a stricter pivot/feasibility test and may return numerical '
             'failure as tableau roundoff accumulates. This comparison keeps both outcomes visible; '
             'use float64 for reliable scientific LP solves.', '',
             f"Cases: {len(data['cases'])}; seeded random LPs: {data['metadata']['random_cases']}; "
             f"seed: {data['metadata']['seed']}; repeats: {data['metadata']['warm_repeats']}.", '',
             '| Solver | Passed | Approximate status | Median warm time (ms) | Max relative objective error | Max feasibility violation |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name in names:
        records = [case['solvers'][name] for case in data['cases']]
        times = [r['median_seconds'] for r in records if 'median_seconds' in r]
        errors = [r['relative_objective_error'] for r in records if r.get('relative_objective_error') is not None]
        violations = [r['violation'] for r in records if r.get('violation') is not None and r['status'] in ('optimal', 'approximate')]
        lines.append(f"| {name} | {sum(r['passed'] for r in records)}/{len(records)} | "
                     f"{sum(r['status']=='approximate' for r in records)} | "
                     f"{median(times)*1000:.3f} | {max(errors, default=0):.3e} | {max(violations, default=0):.3e} |")
    lines.extend(['', 'For an optimum, passing requires an optimal or approximate-solution status, relative objective error '
                  '`abs(f-f_ref)/(1+abs(f_ref)) <= 1e-7`, and maximum constraint/bound violation '
                  '`<= 1e-7` in the common normalized problem. Float32 uses `2e-4`. '
                  'Analytic objectives are used where available; otherwise HiGHS dual simplex supplies '
                  'the reference objective. Different optimal vertices are allowed. Infeasible and '
                  'unbounded cases are checked against their constructed classification.', '',
                  '## First calls in the shared benchmark process', '',
                  'These medians include compilation when needed, with cache reuse from earlier cases. '
                  'They include adapter imports when first encountered. They must not be interpreted as '
                  'isolated compilation or fresh-process startup time.', '',
                  '| Solver | Median first call (ms) |', '| --- | ---: |'])
    for name in names:
        first = [case['solvers'][name]['first_seconds'] for case in data['cases']
                 if 'first_seconds' in case['solvers'][name]]
        lines.append(f'| {name} | {median(first)*1000:.3f} |')
    if 'baseline' in data:
        lines.extend(['', '## Change from the earlier implementation', '',
                      'The baseline is an earlier recorded run, rather than a simultaneous A/B experiment. '
                      'This table compares warm timings of the same passing float64 cases; speedup is the ratio '
                      'of medians. Baseline metadata and per-case times are retained in the JSON. '
                      'A separate [controlled comparison](simplex-repeatability.md) runs both implementations '
                      'in the same environment with identical inputs and alternating measurement order.', '',
                      '| Cases | Earlier median (ms) | Current median (ms) | Speedup |', '| --- | ---: | ---: | ---: |'])
        for group in ('all', 'dense_4', 'dense_8', 'dense_16', 'dense_24'):
            matched = [case for case in data['cases']
                       if (group == 'all' or case['category'] == group)
                       and case['name'] in data['baseline']['cases']
                       and data['baseline']['cases'][case['name']]['passed']
                       and case['solvers']['optinpy-float64']['passed']]
            if matched:
                previous = median(data['baseline']['cases'][case['name']]['median_seconds'] for case in matched)
                current = median(case['solvers']['optinpy-float64']['median_seconds'] for case in matched)
                lines.append(f'| {group} ({len(matched)}) | {previous*1000:.3f} | {current*1000:.3f} | {previous/current:.1f}× |')
    lines.extend(['', '## Timing by problem size', '', '| Solver | Variables | Median warm time (ms) |', '| --- | ---: | ---: |'])
    for name in names:
        groups = defaultdict(list)
        for case in data['cases']:
            r = case['solvers'][name]
            if case['category'].startswith('dense') and 'median_seconds' in r:
                groups[case['variables']].append(r['median_seconds']*1000)
        for size, values in sorted(groups.items()):
            lines.append(f'| {name} | {size} | {median(values):.3f} |')
    failures = [(case['name'], name, result) for case in data['cases']
                for name, result in case['solvers'].items() if not result['passed']]
    lines.extend(['', '## Disagreements', ''])
    if not failures:
        lines.append('None in this suite. This is finite validation evidence, not a proof of correctness for all LPs.')
    else:
        for case, name, r in failures:
            lines.append((f"- `{case}` / `{name}`: status `{r['status']}`, objective error "
                          f"`{r.get('relative_objective_error')}`, violation `{r.get('violation')}`. {r.get('error','')}").rstrip())
    lines.extend(['', '## Environment', '', '```json', json.dumps(data['metadata'], indent=2), '```', '',
                  '## Reproduce', '', '```bash', "python -m pip install -e '.[dev,benchmark]'",
                  f"python -m benchmarks.compare_simplex --random-cases {data['metadata']['random_cases']} "
                  f"--repeats {data['metadata']['warm_repeats']} --seed {data['metadata']['seed']} --float32",
                  '```', '',
                  'Methods: [HiGHS dual simplex](https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs-ds.html), '
                  '[HiGHS interior point](https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs-ipm.html), '
                  '[Clarabel](https://clarabel.org/stable/python/getting_started_py/).', ''])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--random-cases', type=int, default=80)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20260913)
    parser.add_argument('--float32', action='store_true')
    parser.add_argument('--baseline', type=Path, help='Earlier comparison JSON for a recorded warm-time comparison')
    parser.add_argument('--output', type=Path, default=Path('docs/simplex-comparison'))
    args = parser.parse_args()
    if args.random_cases < 0 or args.repeats < 1:
        parser.error('random-cases must be nonnegative and repeats must be positive')
    data = run(args.random_cases, args.repeats, args.seed, args.float32)
    if args.baseline:
        previous = json.loads(args.baseline.read_text())
        data['baseline'] = {'metadata': previous['metadata'],
                            'cases': {case['name']: {key: case['solvers']['optinpy-float64'][key]
                                                    for key in ('median_seconds', 'passed')}
                                      for case in previous['cases']}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.json').write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    args.output.with_suffix('.md').write_text(report(data))
    print(args.output.with_suffix('.md'))
    # Float32/reference disagreements remain visible in the report. Fail the
    # command if our primary float64 implementation fails any validation case.
    if any(not case['solvers']['optinpy-float64']['passed'] for case in data['cases']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()

"""Separate fresh-process startup, complete warm solves, and JIT phase costs.

Run: python -m benchmarks.time_simplex --output docs/simplex-timing
Each case uses isolated CPU subprocesses with the persistent JAX cache disabled.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
from statistics import median
import subprocess
import sys
import time


def worker(index, repeats, mode):
    started = time.perf_counter()
    import jax
    import jax.numpy as jnp
    from optinpy import simplex
    from optinpy.simplex.base import _primal_phase
    from .simplex_cases import normalize, random_cases
    jax.config.update('jax_enable_x64', True)
    imported = time.perf_counter()
    lp = normalize(random_cases(index + 1)[index])

    def solve():
        start = time.perf_counter()
        solver = simplex(lp.A, lp.b, lp.c, lb=lp.lb, ub=lp.ub, mode=lp.mode)
        jax.block_until_ready(solver.tableau)
        constructed = time.perf_counter()
        result = jax.block_until_ready(solver.solve(max_iter=10000))
        finished = time.perf_counter()
        return result, {'construction_seconds': constructed-start,
                        'solve_seconds': finished-constructed,
                        'total_seconds': finished-start}

    result, first = solve()
    record = {'case': lp.name, 'variables': len(lp.c), 'constraints': len(lp.b),
              'import_seconds': imported-started, 'first': first,
              'status': result['status'], 'objective': float(result['f']),
              'x': result['x'].tolist(), 'iterations': result['iterations']}
    if mode == 'cold':
        return record
    timings = [solve()[1] for _ in range(repeats)]
    record['warm'] = {key: median(t[key] for t in timings) for key in first}

    # A separate experiment on prepared tableaus. Clear in-memory caches
    # before lowering each phase; the process-level persistent cache is off.
    # These measurements are not subtracted from the full-solve timings.
    solver = simplex(lp.A, lp.b, lp.c, lb=lp.lb, ub=lp.ub, mode=lp.mode)
    phases = []
    while True:
        args = jax.block_until_ready((solver.tableau, jnp.asarray(solver.basic, dtype=jnp.int32),
                                     solver._cost, jnp.asarray(solver.tol), jnp.asarray(10000),
                                     jnp.asarray(solver._stalled), jnp.asarray(solver._force_bland)))
        jax.clear_caches()
        start = time.perf_counter()
        lowered = _primal_phase.lower(*args)
        traced = time.perf_counter()
        executable = lowered.compile()
        compiled = time.perf_counter()
        phase_result = jax.block_until_ready(executable(*args))
        executed = time.perf_counter()
        timings = []
        for _ in range(repeats):
            t = time.perf_counter()
            jax.block_until_ready(executable(*args))
            timings.append(time.perf_counter()-t)
        phases.append({'phase': solver.phase, 'tableau_shape': list(solver.tableau.shape),
                       'lower_seconds': traced-start, 'compile_seconds': compiled-traced,
                       'first_execution_seconds': executed-compiled,
                       'warm_execution_seconds': median(timings),
                       'pivots': int(phase_result[2]), 'status': int(phase_result[3])})
        if int(phase_result[3]) != 0:
            raise RuntimeError(f'Phase did not reach an optimum: {phases[-1]}')
        if solver.phase == 2:
            break
        solver.tableau = phase_result[0]
        solver.basic = phase_result[1].tolist()
        solver._stalled, solver._force_bland = int(phase_result[4]), bool(phase_result[5])
        for _ in range(len(solver.basic) + 1):
            solver._phase_two()
            if solver.phase == 2 or solver.status is not None:
                break
        if solver.phase != 2:
            raise RuntimeError('Phase-I cleanup failed in timing fixture')
    record['phases'] = phases
    record['versions'] = {name: importlib.metadata.version(name)
                          for name in ('jax', 'jaxlib', 'numpy', 'optinpy')}
    record['devices'] = [str(device) for device in jax.devices()]
    return record


def run(indices, repeats):
    from .simplex_cases import normalize, random_cases, violation
    from .simplex_solvers import highs_solve
    cases = random_cases(max(indices)+1)
    records = []
    env = dict(os.environ, JAX_PLATFORMS='cpu', JAX_ENABLE_COMPILATION_CACHE='false')
    for index in indices:
        lp = normalize(cases[index])
        reference = highs_solve(lp)
        record = {}
        for mode in ('cold', 'profile'):
            command = [sys.executable, '-m', 'benchmarks.time_simplex', '--worker', mode,
                       '--case-indices', str(index), '--repeats', str(repeats)]
            start = time.perf_counter()
            process = subprocess.run(command, capture_output=True, text=True, env=env, timeout=180)
            elapsed = time.perf_counter()-start
            if process.returncode:
                raise RuntimeError(process.stderr or process.stdout)
            result = json.loads(process.stdout)
            error = abs(result['objective']-reference['objective']) / (1+abs(reference['objective']))
            if result['status'] != 0 or error > 1e-7 or violation(lp, result['x']) > 1e-7:
                raise RuntimeError(f'Timing fixture failed independent validation: {result}')
            if mode == 'cold':
                result['process_wall_seconds'] = elapsed
            record[mode] = result
        records.append(record)
        print(f'Timed {len(lp.c)} variables: cold {record["cold"]["first"]["total_seconds"]:.3f}s, '
              f'warm {record["profile"]["warm"]["total_seconds"]*1000:.3f}ms', flush=True)
    return {'metadata': {'platform': platform.platform(), 'python': platform.python_version(),
                         'persistent_compilation_cache': False, 'repeats': repeats,
                         'indices': indices, 'seed': 20260913,
                         'simplex_sha256': hashlib.sha256(Path('optinpy/simplex/base.py').read_bytes()).hexdigest()},
            'cases': records}


def report(data):
    lines = ['# Simplex startup and JIT timing', '',
             'Each cold measurement runs in a new CPU process with the persistent JAX compilation cache disabled. '
             'Process wall time includes Python startup, imports, fixture generation, the first solve, JSON output, '
             'and process exit. The first complete solve includes model construction, device initialization that '
             'has not already occurred, compilation and execution. It is not a measurement of compilation alone.', '',
             'Warm measurements use fresh solver objects with the same LP after one complete solve in a separate '
             'process. They include construction, phase-I cleanup and final residual checks. Every timed JAX '
             'operation is synchronized. Independent HiGHS objective and feasibility checks run outside these times.', '',
             'The rows below are different LPs, with different sizes and pivot counts. Their combined range '
             'does not measure run-to-run variance. See [repeated-input measurements](simplex-repeatability.md) '
             'for distributions on identical inputs and comparisons in balanced solver order.', '',
             '| Variables | Process wall (s) | Imports (s) | First complete solve (s) | Warm construction (ms) | Warm solve (ms) | Warm total (ms) |',
             '| ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for case in data['cases']:
        cold, warm = case['cold'], case['profile']['warm']
        lines.append(f'| {cold["variables"]} | {cold["process_wall_seconds"]:.3f} | {cold["import_seconds"]:.3f} | '
                     f'{cold["first"]["total_seconds"]:.3f} | {warm["construction_seconds"]*1000:.3f} | '
                     f'{warm["solve_seconds"]*1000:.3f} | {warm["total_seconds"]*1000:.3f} |')
    lines.extend(['', '## Compilation of prepared phases', '',
                  'This separate experiment prepares each phase’s tableau, clears JAX’s in-memory caches, '
                  'then measures `.lower()`, `.compile()`, and synchronized execution separately. '
                  'It excludes construction, phase transitions and residual checks. These measurements must '
                  'not be added to or subtracted from the complete-solve measurements above. '
                  'Shapes and dtypes determine compiled variants; changing bound structure or the number '
                  'of artificial variables can require another compilation.', '',
                  '| Variables | Phase | Tableau shape | Pivots | Trace/lower (ms) | Compile (ms) | First execution (ms) | Warm execution (ms) |',
                  '| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |'])
    for case in data['cases']:
        for phase in case['profile']['phases']:
            lines.append(f'| {case["cold"]["variables"]} | {phase["phase"]} | {phase["tableau_shape"]} | '
                         f'{phase["pivots"]} | {phase["lower_seconds"]*1000:.3f} | {phase["compile_seconds"]*1000:.3f} | '
                         f'{phase["first_execution_seconds"]*1000:.3f} | {phase["warm_execution_seconds"]*1000:.3f} |')
    environment = dict(data['metadata'], versions=data['cases'][0]['profile']['versions'],
                       devices=data['cases'][0]['profile']['devices'])
    lines.extend(['', '## Environment and reproduction', '', '```json', json.dumps(environment, indent=2),
                  '```', '', '```bash', "python -m pip install -e '.[dev,benchmark]'",
                  f'python -m benchmarks.time_simplex --case-indices {" ".join(map(str, data["metadata"]["indices"]))} '
                  f'--repeats {data["metadata"]["repeats"]}', '```', '',
                  'Method: [JAX benchmarking guidance](https://docs.jax.dev/en/latest/benchmarking.html) and '
                  '[explicit lowering and compilation](https://docs.jax.dev/en/latest/aot.html).', ''])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-indices', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--output', type=Path, default=Path('docs/simplex-timing'))
    parser.add_argument('--worker', choices=['cold', 'profile'], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.case_indices) < 0 or args.repeats < 1:
        parser.error('case indices must be nonnegative and repeats positive')
    if args.worker:
        print(json.dumps(worker(args.case_indices[0], args.repeats, args.worker), allow_nan=False))
        return
    data = run(args.case_indices, args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.json').write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    args.output.with_suffix('.md').write_text(report(data))
    print(args.output.with_suffix('.md'))


if __name__ == '__main__':
    main()

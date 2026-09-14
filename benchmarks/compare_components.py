"""Interleave pre/post-refactor executables and check outputs and compiled operations.

The baseline is trusted local Git history. Each trial uses a fresh CPU process,
loading both full packages under separate module names before compiling either.
"""
import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
from statistics import median
import subprocess
import sys
import tarfile
import tempfile
import time


def normalized_hlo(text):
    # Ignore only debug tables/metadata and symbolic instruction/argument names.
    # Keep operation order, shapes, constants, operands and backend configuration.
    text = text[text.index('\n%'):]
    text = re.sub(r', metadata=\{[^\n]*?\}', '', text)
    names = {}
    text = re.sub(r'%[\w.-]+', lambda m: names.setdefault(m.group(), f'%v{len(names)}'), text)
    return re.sub(r'\b[A-Za-z_]\w*\.\d+(?=:)', 'argument', text)


def worker(args):
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optinpy
    jax.config.update('jax_enable_x64', True)
    name = '_optinpy_component_baseline'
    spec = importlib.util.spec_from_file_location(name, Path(args.package_root)/'optinpy/__init__.py')
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[name] = baseline
    spec.loader.exec_module(baseline)
    packages = {'baseline': baseline, 'current': optinpy}
    order = ['baseline', 'current'] if args.worker_trial % 2 == 0 else ['current', 'baseline']
    records = []
    for method, n in [('modified-newton', 8), ('bfgs', 8), ('dfp', 8), ('lbfgs', 8), ('bfgs', 64), ('bfgs', 256)]:
        objective = lambda x, target, weights: jnp.sum(weights*(x-target)**2)
        inputs = (jnp.zeros(n), jnp.linspace(-1., 1., n), jnp.linspace(1., 3. if n == 8 else 10., n))
        jax.block_until_ready(inputs)
        row = {'case': f'{method}/{n}', 'timings': {}, 'samples_ms': {rev: [] for rev in order}}
        compiled, results, hlo = {}, {}, {}
        for revision in order:
            solve = lambda x, t, w: packages[revision].minimize(
                objective, x, args=(t, w), method=method, tol=1e-5 if n == 8 else 1e-6,
                max_iter=3000 if n == 8 else 1000, learning_rate=.03)
            start = time.perf_counter_ns()
            lower = jax.jit(solve).lower(*inputs)
            lowered = time.perf_counter_ns()
            compiled[revision] = lower.compile()
            ready = time.perf_counter_ns()
            results[revision] = jax.block_until_ready(compiled[revision](*inputs))
            executed = time.perf_counter_ns()
            row['timings'][revision] = dict(lower_ms=(lowered-start)/1e6, compile_ms=(ready-lowered)/1e6,
                                             first_execution_ms=(executed-ready)/1e6,
                                             first_staged_call_ms=(executed-start)/1e6)
            hlo[revision] = normalized_hlo(compiled[revision].as_text())
        row['identical_compiled_operations'] = hlo['baseline'] == hlo['current']
        row['identical_initial_outputs'] = all(np.array_equal(results['baseline'][k], results['current'][k], equal_nan=True)
                                                for k in results['baseline'])
        row['iterations'] = {rev: int(result['iterations']) for rev, result in results.items()}
        for i in range(args.repeats+5):
            for revision in (order if i % 2 == 0 else order[::-1]):
                start = time.perf_counter_ns()
                result = jax.block_until_ready(compiled[revision](*inputs))
                elapsed = (time.perf_counter_ns()-start)/1e6
                assert bool(result['success']), (method, n, revision, result)
                np.testing.assert_allclose(result['x'], inputs[1], atol=2e-4, rtol=0)
                assert all(np.array_equal(result[k], results['baseline'][k], equal_nan=True) for k in result)
                if i >= 5:
                    row['samples_ms'][revision].append(elapsed)
        records.append(row)
    return {'trial': args.worker_trial, 'jax': jax.__version__, 'jaxlib': jax.lib.__version__,
            'devices': list(map(str, jax.devices())), 'records': records,
            'source_sha256': {rev: {str(p.relative_to(Path(package.__file__).parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted(Path(package.__file__).parent.rglob('*.py'))}
                              for rev, package in packages.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-ref', help='Trusted commit available in local Git history')
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=80)
    parser.add_argument('--worker-trial', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--package-root', help=argparse.SUPPRESS)
    parser.add_argument('--output', type=Path, default=Path('docs/component-interleaved'))
    args = parser.parse_args()
    if min(args.trials, args.repeats) < 1:
        parser.error('Trials and repeats must be positive')
    if args.worker_trial is not None:
        print(json.dumps(worker(args), allow_nan=False))
        return
    if not args.baseline_ref:
        parser.error('--baseline-ref must name a trusted commit available in local Git history')
    baseline = subprocess.run(['git', 'rev-parse', args.baseline_ref], capture_output=True, text=True, check=True).stdout.strip()
    archive = subprocess.run(['git', 'archive', '--format=tar', baseline, 'optinpy'], capture_output=True, check=True).stdout
    trials = []
    with tempfile.TemporaryDirectory(prefix='optinpy-components-') as tmp:
        with tarfile.open(fileobj=io.BytesIO(archive)) as source:
            source.extractall(tmp, filter='data')
        for trial in range(args.trials):
            result = subprocess.run([sys.executable, '-m', 'benchmarks.compare_components', '--worker-trial', str(trial),
                                     '--package-root', tmp, '--repeats', str(args.repeats)],
                                    env=dict(os.environ, JAX_PLATFORMS='cpu', JAX_ENABLE_COMPILATION_CACHE='false'),
                                    text=True, capture_output=True, timeout=300)
            if result.returncode:
                raise RuntimeError(result.stderr)
            trials.append(json.loads(result.stdout))
            print(f'Completed interleaved component trial {trial+1}/{args.trials}', flush=True)
    data = {'baseline_ref': baseline, 'trials': args.trials, 'repeats': args.repeats,
            'benchmark_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'raw_trials': trials}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.json').write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    import numpy as np
    lines = ['# Interleaved component refactor check', '',
             'CPU float64; both revisions use identical dynamic inputs and settings. '
             'Five untimed warmups per executable, followed by alternating synchronized calls. '
             'Both full packages are loaded before tracing. Compile order reverses between fresh processes. '
             'Persistent caching is disabled, but within-process JAX caches can be shared. '
             'Every execution must converge and match every baseline result field exactly. '
             'Compiled-operation equality ignores debug metadata and symbolic names only. '
             'No timed samples are discarded; P5–P95 is the observed spread.', '',
             '| Case | Same compiled operations, all trials | Baseline warm median (µs) | Current warm median (µs) | Baseline P5–P95 (µs) | Current P5–P95 (µs) |',
             '| --- | --- | ---: | ---: | --- | --- |']
    for name in [r['case'] for r in trials[0]['records']]:
        rows = [r for t in trials for r in t['records'] if r['case'] == name]
        samples = {rev: [s*1000 for r in rows for s in r['samples_ms'][rev]] for rev in ('baseline', 'current')}
        same = all(r['identical_compiled_operations'] for r in rows)
        lines.append(f'| {name} | {same} | {median(samples["baseline"]):.3f} | {median(samples["current"]):.3f} | '+
                     ' | '.join(f'{np.percentile(v,5):.3f}–{np.percentile(v,95):.3f}' for v in samples.values())+' |')
    lines += ['', f'Reproduce: `python -m benchmarks.compare_components --baseline-ref {baseline} '
              f'--trials {args.trials} --repeats {args.repeats}`.', '',
              f'[Raw samples, compilation timings, output checks and source hashes]({args.output.with_suffix(".json").name}). '
              '[The separate-process benchmark](component-performance.md) covers all eleven minimizers. '
              'First staged call in the raw data includes lowering, compilation and first execution; '
              'it excludes imports and device initialization. This check addresses refactor overhead on '
              'these inputs, not the performance of every possible component combination.', '']
    args.output.with_suffix('.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()

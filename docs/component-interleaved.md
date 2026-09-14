# Interleaved component refactor check

> Historical comparison: baseline commits predate the consolidated branch history. See [baseline availability and reproduction](benchmark-history.md).

CPU float64; both revisions use identical dynamic inputs and settings. Five untimed warmups per executable, followed by alternating synchronized calls. Both full packages are loaded before tracing. Compile order reverses between fresh processes. Persistent caching is disabled, but within-process JAX caches can be shared. Every execution must converge and match every baseline result field exactly. Compiled-operation equality ignores debug metadata and symbolic names only. No timed samples are discarded; P5–P95 is the observed spread.

| Case | Same compiled operations, all trials | Baseline warm median (µs) | Current warm median (µs) | Baseline P5–P95 (µs) | Current P5–P95 (µs) |
| --- | --- | ---: | ---: | --- | --- |
| modified-newton/8 | True | 29.333 | 29.291 | 20.750–38.058 | 21.188–38.587 |
| bfgs/8 | True | 36.146 | 36.375 | 29.033–49.631 | 28.985–46.304 |
| dfp/8 | True | 36.625 | 36.959 | 31.269–57.487 | 30.927–58.040 |
| lbfgs/8 | True | 61.625 | 62.000 | 55.581–81.310 | 55.269–79.977 |
| bfgs/64 | True | 418.458 | 419.458 | 402.789–462.637 | 401.533–462.616 |
| bfgs/256 | True | 6968.312 | 6971.479 | 6576.504–7453.297 | 6124.010–7350.534 |

Reproduce: `python -m benchmarks.compare_components --baseline-ref 24668ce44f8acc0d3fd078ce015651f3a8f83acb --trials 3 --repeats 80`.

[Raw samples, compilation timings, output checks and source hashes](component-interleaved.json). [The separate-process benchmark](component-performance.md) covers all eleven minimizers. First staged call in the raw data includes lowering, compilation and first execution; it excludes imports and device initialization. This check addresses refactor overhead on these inputs, not the performance of every possible component combination.

# Native JAX performance audit: measurements

> Historical comparison: baseline commits predate the consolidated branch history. See [baseline availability and reproduction](benchmark-history.md).

CPU float64 measurements on fixed analytic problems. Both revisions use the same inputs, objectives, settings, and JAX environment. All results synchronize before stopping the timer; every sample is checked against an analytic answer (line searches check descent and the returned objective). No samples are discarded. Timings are informational.

Each revision/trial uses a fresh process, alternating revision order between trials. JAX caches are cleared before each case and persistent caching is disabled. Prepared input arrays and device initialization are outside these timers. Staged cases separately measure tracing/lowering, compilation, and calls to the compiled executable. API cases measure the ordinary Python call, including any compilation it causes: their repeated timings are not necessarily free of compilation. The first execution is excluded from the repeated distribution. P5–P95 is an observed range, not a confidence interval.

| Case | Revision | Timer | Lower (ms) | Compile (ms) | First execution (ms) | Repeats | Median (ms) | P5–P95 (ms) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| minimize/gradient | current | AOT | 26.863 | 47.368 | 0.394 | 120 | 0.025 | 0.023–0.060 |
| minimize/gradient | baseline | AOT | 26.100 | 50.494 | 0.403 | 120 | 0.026 | 0.024–0.038 |
| minimize/newton | current | AOT | 140.641 | 57.207 | 0.457 | 120 | 0.018 | 0.017–0.029 |
| minimize/newton | baseline | AOT | 145.211 | 59.787 | 0.413 | 120 | 0.019 | 0.017–0.061 |
| minimize/modified-newton | current | AOT | 61.959 | 62.091 | 0.479 | 120 | 0.030 | 0.021–0.087 |
| minimize/modified-newton | baseline | AOT | 64.949 | 59.957 | 0.432 | 120 | 0.024 | 0.019–0.059 |
| minimize/conjugate-gradient | current | AOT | 26.901 | 50.093 | 0.361 | 120 | 0.027 | 0.026–0.041 |
| minimize/conjugate-gradient | baseline | AOT | 28.513 | 49.577 | 0.427 | 120 | 0.026 | 0.026–0.037 |
| minimize/hessian-conjugate-gradient | current | AOT | 31.694 | 52.424 | 0.382 | 120 | 0.028 | 0.027–0.036 |
| minimize/hessian-conjugate-gradient | baseline | AOT | 30.114 | 51.695 | 0.405 | 120 | 0.028 | 0.026–0.051 |
| minimize/fletcher-reeves | current | AOT | 28.324 | 47.108 | 0.384 | 120 | 0.031 | 0.029–0.044 |
| minimize/fletcher-reeves | baseline | AOT | 27.965 | 53.298 | 0.423 | 120 | 0.031 | 0.029–0.047 |
| minimize/bfgs | current | AOT | 31.024 | 55.286 | 0.412 | 120 | 0.043 | 0.026–0.074 |
| minimize/bfgs | baseline | AOT | 30.574 | 57.486 | 0.373 | 120 | 0.037 | 0.025–0.098 |
| minimize/dfp | current | AOT | 28.677 | 59.738 | 0.454 | 120 | 0.039 | 0.026–0.142 |
| minimize/dfp | baseline | AOT | 27.865 | 55.655 | 0.422 | 120 | 0.032 | 0.024–0.058 |
| minimize/lbfgs | current | AOT | 47.754 | 72.295 | 0.613 | 120 | 0.080 | 0.054–0.151 |
| minimize/lbfgs | baseline | AOT | 39.572 | 68.935 | 0.685 | 120 | 0.066 | 0.049–0.136 |
| minimize/adam | current | AOT | 16.928 | 29.327 | 0.294 | 120 | 0.044 | 0.042–0.078 |
| minimize/adam | baseline | AOT | 19.039 | 28.889 | 0.292 | 120 | 0.043 | 0.040–0.051 |
| minimize/sgd | current | AOT | 15.492 | 25.340 | 0.257 | 120 | 0.025 | 0.024–0.032 |
| minimize/sgd | baseline | AOT | 16.717 | 23.614 | 0.247 | 120 | 0.025 | 0.024–0.050 |
| bfgs/64 | current | AOT | 31.710 | 74.914 | 0.905 | 120 | 0.451 | 0.417–0.526 |
| bfgs/64 | baseline | AOT | 29.696 | 79.016 | 0.898 | 120 | 0.455 | 0.420–0.514 |
| bfgs/256 | current | AOT | 31.116 | 68.852 | 9.056 | 120 | 6.679 | 5.689–7.902 |
| bfgs/256 | baseline | AOT | 30.471 | 65.379 | 6.780 | 120 | 6.978 | 6.163–8.463 |

The reusable API uses `compile_minimizer` in the current revision and an equivalent outer `jax.jit` in the baseline. Batching compares 16 complete solves in a Python loop with one compiled `vmap`; both include result collection. These are small dense test problems, not a representative nonlinear optimization corpus or GPU measurements. First execution in an AOT row excludes the preceding lowering/compilation. See [the separate process timing report](simplex-timing.md) for end-to-end startup.

## Reproduction

```bash
python -m benchmarks.jax_audit --group unconstrained --trials 3 --repeats 40 --eager-repeats 3 --baseline-ref 24668ce44f8acc0d3fd078ce015651f3a8f83acb
```

The optional baseline requires the trusted revision in local Git history. Omit it for a source distribution or shallow checkout. [Raw samples, versions, devices, iteration counts, and hashes of all runtime source files](component-performance.json).

```json
{
  "utc": "2026-09-14T20:05:49.376720+00:00",
  "platform": "macOS-26.2-arm64-arm-64bit",
  "python": "3.12.2",
  "trials": 3,
  "repeats": 40,
  "eager_repeats": 3,
  "group": "unconstrained",
  "baseline_ref": "24668ce44f8acc0d3fd078ce015651f3a8f83acb",
  "persistent_compilation_cache": false
}
```

Method: [JAX benchmarking](https://docs.jax.dev/en/latest/benchmarking.html) and [explicit compilation stages](https://docs.jax.dev/en/latest/aot.html).

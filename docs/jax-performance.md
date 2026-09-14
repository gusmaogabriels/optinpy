# Native JAX performance audit: measurements

> Historical comparison: baseline commits predate the consolidated branch history. See [baseline availability and reproduction](benchmark-history.md).

These measurements record the runtime at performance-audit commit `49c4978`.
Subsequent changes restore the original Hessian-conjugate method and five-point
central Hessian diagonal. These historical timings do not measure those changes
and do not establish their speed. Current LP measurements are in the
[simplex report](simplex-comparison.md).

CPU float64 measurements on fixed analytic problems. Both revisions use the same inputs, objectives, settings, and JAX environment. All results synchronize before stopping the timer; every sample is checked against an analytic answer (line searches check descent and the returned objective). No samples are discarded. Timings are informational.

Each revision/trial uses a fresh process, alternating revision order between trials. JAX caches are cleared before each case and persistent caching is disabled. Prepared input arrays and device initialization are outside these timers. Staged cases separately measure tracing/lowering, compilation, and calls to the compiled executable. API cases measure the ordinary Python call, including any compilation it causes: their repeated timings are not necessarily free of compilation. The first execution is excluded from the repeated distribution. P5–P95 is an observed range, not a confidence interval.

| Case | Revision | Timer | Lower (ms) | Compile (ms) | First execution (ms) | Repeats | Median (ms) | P5–P95 (ms) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| minimize/gradient | current | AOT | 24.402 | 40.465 | 0.310 | 20 | 0.026 | 0.024–0.172 |
| minimize/gradient | baseline | AOT | 26.098 | 39.996 | 0.309 | 20 | 0.031 | 0.024–0.247 |
| minimize/newton | current | AOT | 136.669 | 49.754 | 0.360 | 20 | 0.021 | 0.017–0.061 |
| minimize/newton | baseline | AOT | 129.690 | 49.075 | 0.380 | 20 | 0.019 | 0.017–0.064 |
| minimize/modified-newton | current | AOT | 56.712 | 54.317 | 0.628 | 20 | 0.031 | 0.023–0.077 |
| minimize/modified-newton | baseline | AOT | 58.637 | 52.427 | 0.348 | 20 | 0.032 | 0.028–0.080 |
| minimize/conjugate-gradient | current | AOT | 26.437 | 43.419 | 0.362 | 20 | 0.028 | 0.025–0.067 |
| minimize/conjugate-gradient | baseline | AOT | 27.515 | 42.055 | 0.319 | 20 | 0.029 | 0.026–0.074 |
| minimize/fletcher-reeves | current | AOT | 25.802 | 42.213 | 0.358 | 20 | 0.032 | 0.030–0.062 |
| minimize/fletcher-reeves | baseline | AOT | 26.726 | 41.271 | 0.322 | 20 | 0.032 | 0.027–0.071 |
| minimize/bfgs | current | AOT | 28.495 | 48.321 | 0.374 | 20 | 0.036 | 0.023–0.065 |
| minimize/bfgs | baseline | AOT | 30.051 | 45.994 | 0.347 | 20 | 0.051 | 0.044–0.089 |
| minimize/dfp | current | AOT | 27.594 | 50.594 | 0.340 | 20 | 0.039 | 0.033–0.064 |
| minimize/dfp | baseline | AOT | 29.013 | 48.037 | 0.349 | 20 | 0.037 | 0.034–0.076 |
| minimize/lbfgs | current | AOT | 37.440 | 59.395 | 0.454 | 20 | 0.074 | 0.058–0.132 |
| minimize/lbfgs | baseline | AOT | 39.307 | 61.097 | 0.346 | 20 | 0.054 | 0.040–0.089 |
| minimize/adam | current | AOT | 17.901 | 25.367 | 0.300 | 20 | 0.045 | 0.042–0.059 |
| minimize/adam | baseline | AOT | 17.316 | 26.367 | 0.268 | 20 | 0.042 | 0.037–0.067 |
| minimize/sgd | current | AOT | 15.628 | 22.800 | 0.233 | 20 | 0.028 | 0.026–0.063 |
| minimize/sgd | baseline | AOT | 15.565 | 22.448 | 0.243 | 20 | 0.025 | 0.022–0.046 |
| bfgs/64 | current | AOT | 28.191 | 66.582 | 0.761 | 20 | 0.420 | 0.401–0.478 |
| bfgs/64 | baseline | AOT | 29.893 | 58.701 | 2.310 | 20 | 1.996 | 1.945–2.104 |
| bfgs/256 | current | AOT | 28.358 | 57.825 | 7.692 | 20 | 7.056 | 6.324–7.418 |
| bfgs/256 | baseline | AOT | 29.363 | 54.075 | 33.513 | 20 | 32.276 | 31.896–33.916 |
| linesearch/backtracking | current | AOT | 11.635 | 20.822 | 0.195 | 20 | 0.012 | 0.011–0.033 |
| linesearch/backtracking | baseline | AOT | 12.077 | 21.396 | 0.226 | 20 | 0.012 | 0.011–0.033 |
| linesearch/interp23 | current | AOT | 16.696 | 28.177 | 0.216 | 20 | 0.014 | 0.013–0.036 |
| linesearch/interp23 | baseline | AOT | 16.676 | 27.217 | 0.225 | 20 | 0.016 | 0.014–0.045 |
| linesearch/strong_wolfe | current | AOT | 14.942 | 28.431 | 0.239 | 20 | 0.014 | 0.012–0.042 |
| linesearch/strong_wolfe | baseline | AOT | 15.166 | 26.295 | 0.233 | 20 | 0.015 | 0.011–0.039 |
| linesearch/golden_section | current | AOT | 16.608 | 26.526 | 0.203 | 20 | 0.013 | 0.013–0.040 |
| linesearch/golden_section | baseline | AOT | 18.534 | 35.289 | 0.237 | 20 | 0.015 | 0.013–0.038 |
| linesearch/unimodality | current | AOT | 27.497 | 46.751 | 0.311 | 20 | 0.027 | 0.024–0.052 |
| linesearch/unimodality | baseline | AOT | 42.584 | 46.354 | 0.339 | 20 | 0.029 | 0.025–0.054 |
| jacobian/autodiff | current | AOT | 4.140 | 8.916 | 0.139 | 20 | 0.005 | 0.004–0.015 |
| jacobian/autodiff | baseline | AOT | 3.791 | 9.048 | 0.123 | 20 | 0.005 | 0.004–0.013 |
| hessian/autodiff | current | AOT | 9.588 | 11.214 | 0.117 | 20 | 0.005 | 0.004–0.013 |
| hessian/autodiff | baseline | AOT | 9.382 | 10.856 | 0.116 | 20 | 0.005 | 0.004–0.012 |
| jacobian/central | current | AOT | 6.325 | 13.026 | 0.130 | 20 | 0.020 | 0.017–0.032 |
| jacobian/central | baseline | AOT | 6.275 | 12.700 | 0.154 | 20 | 0.022 | 0.020–0.031 |
| hessian/central | current | AOT | 10.762 | 21.207 | 0.125 | 20 | 0.022 | 0.019–0.036 |
| hessian/central | baseline | AOT | 10.530 | 20.698 | 0.116 | 20 | 0.022 | 0.020–0.032 |
| jacobian/forward | current | AOT | 6.930 | 13.467 | 0.127 | 20 | 0.020 | 0.017–0.027 |
| jacobian/forward | baseline | AOT | 6.708 | 13.654 | 0.123 | 20 | 0.019 | 0.014–0.025 |
| hessian/forward | current | AOT | 11.463 | 17.353 | 0.130 | 20 | 0.022 | 0.019–0.027 |
| hessian/forward | baseline | AOT | 11.314 | 17.474 | 0.131 | 20 | 0.021 | 0.018–0.029 |
| jacobian/backward | current | AOT | 6.501 | 13.794 | 0.145 | 20 | 0.020 | 0.014–0.044 |
| jacobian/backward | baseline | AOT | 6.508 | 13.342 | 0.130 | 20 | 0.020 | 0.017–0.028 |
| hessian/backward | current | AOT | 12.090 | 18.272 | 0.144 | 20 | 0.021 | 0.020–0.024 |
| hessian/backward | baseline | AOT | 11.317 | 17.359 | 0.123 | 20 | 0.022 | 0.019–0.025 |
| constrained/projected-gradient | current | API | — | — | 420.552 | 6 | 47.204 | 46.050–48.171 |
| constrained/projected-gradient | baseline | API | — | — | 1136.416 | 6 | 386.351 | 368.355–398.643 |
| constrained/penalty | current | API | — | — | 197.734 | 6 | 107.791 | 102.787–112.859 |
| constrained/penalty | baseline | API | — | — | 912.504 | 6 | 533.106 | 510.603–544.187 |
| constrained/barrier | current | API | — | — | 230.167 | 6 | 111.535 | 108.320–115.603 |
| constrained/barrier | baseline | API | — | — | 1437.761 | 6 | 1009.554 | 959.072–1024.386 |
| constrained/log-barrier | current | API | — | — | 217.143 | 6 | 112.914 | 108.614–115.940 |
| constrained/log-barrier | baseline | API | — | — | 981.358 | 6 | 484.506 | 467.256–505.418 |
| usage/direct | current | API | — | — | 341.924 | 6 | 65.434 | 64.197–65.940 |
| usage/direct | baseline | API | — | — | 334.276 | 6 | 63.256 | 61.411–65.043 |
| usage/reusable | current | API | — | — | 80.067 | 6 | 0.045 | 0.040–0.083 |
| usage/reusable | baseline | API | — | — | 76.280 | 6 | 0.053 | 0.047–0.080 |
| batch/16-serial | current | API | — | — | 120.499 | 6 | 1.534 | 1.381–1.674 |
| batch/16-serial | baseline | API | — | — | 119.836 | 6 | 1.419 | 1.266–1.560 |
| batch/16-vmap | current | AOT | 68.988 | 77.255 | 0.543 | 20 | 0.152 | 0.142–0.218 |
| batch/16-vmap | baseline | AOT | 70.434 | 75.170 | 0.570 | 20 | 0.192 | 0.183–0.241 |

The reusable API uses `compile_minimizer` in the current revision and an equivalent outer `jax.jit` in the baseline. Batching compares 16 complete solves in a Python loop with one compiled `vmap`; both include result collection. These are small dense test problems, not a representative nonlinear optimization corpus or GPU measurements. First execution in an AOT row excludes the preceding lowering/compilation. See [the separate process timing report](simplex-timing.md) for end-to-end startup.

## Reproduction

```bash
python -m benchmarks.jax_audit --group all --trials 2 --repeats 10 --eager-repeats 3 --baseline-ref 6a0f69a5bfd059604604a8e04c5d0c71e91cb8dc
```

The optional baseline requires the trusted revision in local Git history. Omit it for a source distribution or shallow checkout. [Raw samples, versions, devices, iteration counts, and hashes of all runtime source files](jax-performance.json).

```json
{
  "utc": "2026-09-13T20:07:18.494151+00:00",
  "platform": "macOS-26.2-arm64-arm-64bit",
  "python": "3.12.2",
  "trials": 2,
  "repeats": 10,
  "eager_repeats": 3,
  "group": "all",
  "baseline_ref": "6a0f69a5bfd059604604a8e04c5d0c71e91cb8dc",
  "persistent_compilation_cache": false
}
```

Method: [JAX benchmarking](https://docs.jax.dev/en/latest/benchmarking.html) and [explicit compilation stages](https://docs.jax.dev/en/latest/aot.html).

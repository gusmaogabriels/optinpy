# Native JAX performance audit: measurements

> Historical comparison: baseline commits predate the consolidated branch history. See [baseline availability and reproduction](benchmark-history.md).

CPU float64 measurements on fixed analytic problems. Both revisions use the same inputs, objectives, settings, and JAX environment. All results synchronize before stopping the timer; every sample is checked against an analytic answer (line searches check descent and the returned objective). No samples are discarded. Timings are informational.

Each revision/trial uses a fresh process, alternating revision order between trials. JAX caches are cleared before each case and persistent caching is disabled. Prepared input arrays and device initialization are outside these timers. Staged cases separately measure tracing/lowering, compilation, and calls to the compiled executable. API cases measure the ordinary Python call, including any compilation it causes: their repeated timings are not necessarily free of compilation. The first execution is excluded from the repeated distribution. P5–P95 is an observed range, not a confidence interval.

| Case | Revision | Timer | Lower (ms) | Compile (ms) | First execution (ms) | Repeats | Median (ms) | P5–P95 (ms) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| linesearch/backtracking | current | AOT | 14.896 | 41.412 | 0.338 | 60 | 0.012 | 0.011–0.025 |
| linesearch/backtracking | baseline | AOT | 12.597 | 22.905 | 0.228 | 60 | 0.012 | 0.011–0.016 |
| linesearch/interp23 | current | AOT | 19.015 | 34.223 | 0.226 | 60 | 0.015 | 0.013–0.031 |
| linesearch/interp23 | baseline | AOT | 19.267 | 30.385 | 0.241 | 60 | 0.014 | 0.014–0.021 |
| linesearch/strong_wolfe | current | AOT | 16.809 | 32.427 | 0.257 | 60 | 0.013 | 0.013–0.020 |
| linesearch/strong_wolfe | baseline | AOT | 16.735 | 32.385 | 0.445 | 60 | 0.013 | 0.012–0.019 |
| linesearch/golden_section | current | AOT | 18.719 | 31.529 | 0.211 | 60 | 0.014 | 0.013–0.019 |
| linesearch/golden_section | baseline | AOT | 17.971 | 29.915 | 0.222 | 60 | 0.014 | 0.013–0.017 |
| linesearch/unimodality | current | AOT | 28.991 | 59.311 | 0.350 | 60 | 0.028 | 0.026–0.039 |
| linesearch/unimodality | baseline | AOT | 28.068 | 51.774 | 0.363 | 60 | 0.028 | 0.027–0.036 |
| jacobian/autodiff | current | AOT | 4.259 | 10.851 | 0.133 | 60 | 0.005 | 0.004–0.009 |
| jacobian/autodiff | baseline | AOT | 4.049 | 9.357 | 0.133 | 60 | 0.005 | 0.005–0.009 |
| hessian/autodiff | current | AOT | 11.717 | 12.589 | 0.126 | 60 | 0.005 | 0.004–0.013 |
| hessian/autodiff | baseline | AOT | 10.273 | 11.374 | 0.135 | 60 | 0.005 | 0.004–0.008 |
| jacobian/central | current | AOT | 6.745 | 13.027 | 0.136 | 60 | 0.023 | 0.014–0.037 |
| jacobian/central | baseline | AOT | 7.106 | 13.450 | 0.182 | 60 | 0.020 | 0.015–0.053 |
| hessian/central | current | AOT | 19.126 | 31.961 | 0.188 | 60 | 0.030 | 0.019–0.068 |
| hessian/central | baseline | AOT | 27.692 | 59.602 | 0.347 | 60 | 0.024 | 0.015–0.199 |
| jacobian/forward | current | AOT | 6.946 | 13.840 | 0.140 | 60 | 0.023 | 0.010–0.050 |
| jacobian/forward | baseline | AOT | 7.023 | 15.669 | 0.155 | 60 | 0.023 | 0.017–0.080 |
| hessian/forward | current | AOT | 12.011 | 18.772 | 0.152 | 60 | 0.029 | 0.016–0.083 |
| hessian/forward | baseline | AOT | 12.101 | 20.865 | 0.168 | 60 | 0.021 | 0.017–0.043 |
| jacobian/backward | current | AOT | 6.896 | 14.994 | 0.133 | 60 | 0.026 | 0.013–0.083 |
| jacobian/backward | baseline | AOT | 7.020 | 15.284 | 0.164 | 60 | 0.019 | 0.013–0.061 |
| hessian/backward | current | AOT | 26.242 | 20.912 | 0.159 | 60 | 0.026 | 0.012–0.068 |
| hessian/backward | baseline | AOT | 12.152 | 20.373 | 0.181 | 60 | 0.025 | 0.019–0.105 |
| hessian/central-coupled/16 | current | AOT | 32.070 | 147.250 | 1.022 | 60 | 0.144 | 0.118–0.249 |
| hessian/central-coupled/16 | baseline | AOT | 36.119 | 150.636 | 0.697 | 60 | 0.158 | 0.137–0.369 |
| hessian/central-coupled/64 | current | AOT | 32.170 | 74.017 | 3.464 | 60 | 1.748 | 1.245–3.337 |
| hessian/central-coupled/64 | baseline | AOT | 39.426 | 99.132 | 3.835 | 60 | 2.064 | 1.672–2.871 |
| hessian/central-coupled/128 | current | AOT | 44.835 | 75.434 | 21.292 | 60 | 16.789 | 13.316–27.787 |
| hessian/central-coupled/128 | baseline | AOT | 39.949 | 92.616 | 25.196 | 60 | 17.101 | 14.484–27.915 |

The reusable API uses `compile_minimizer` in the current revision and an equivalent outer `jax.jit` in the baseline. Batching compares 16 complete solves in a Python loop with one compiled `vmap`; both include result collection. These are small dense test problems, not a representative nonlinear optimization corpus or GPU measurements. First execution in an AOT row excludes the preceding lowering/compilation. See [the separate process timing report](simplex-timing.md) for end-to-end startup.

## Reproduction

```bash
python -m benchmarks.jax_audit --group kernels --trials 2 --repeats 30 --eager-repeats 3 --baseline-ref 8ddfe01fb879d70db55dd885a1ab36eca52a1659
```

The optional baseline requires the trusted revision in local Git history. Omit it for a source distribution or shallow checkout. [Raw samples, versions, devices, iteration counts, and hashes of all runtime source files](warm-kernels.json).

```json
{
  "utc": "2026-09-14T17:56:56.225815+00:00",
  "platform": "macOS-26.2-arm64-arm-64bit",
  "python": "3.12.2",
  "trials": 2,
  "repeats": 30,
  "eager_repeats": 3,
  "group": "kernels",
  "baseline_ref": "8ddfe01fb879d70db55dd885a1ab36eca52a1659",
  "persistent_compilation_cache": false
}
```

Method: [JAX benchmarking](https://docs.jax.dev/en/latest/benchmarking.html) and [explicit compilation stages](https://docs.jax.dev/en/latest/aot.html).

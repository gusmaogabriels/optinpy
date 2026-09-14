# Original-kernel restoration: measurements

> Historical comparison: baseline commits predate the consolidated branch history. See [baseline availability and reproduction](benchmark-history.md).

CPU float64 measurements on fixed analytic problems. Both revisions use the same inputs, objectives, settings, and JAX environment. All results synchronize before stopping the timer; every sample is checked against an analytic answer (line searches check descent and the returned objective). No samples are discarded. Timings are informational.

Each revision/trial uses a fresh process, alternating revision order between trials. JAX caches are cleared before each case and persistent caching is disabled. Prepared input arrays and device initialization are outside these timers. Staged cases separately measure tracing/lowering, compilation, and calls to the compiled executable. API cases measure the ordinary Python call, including any compilation it causes: their repeated timings are not necessarily free of compilation. The first execution is excluded from the repeated distribution. P5–P95 is an observed range, not a confidence interval.

| Case | Revision | Timer | Lower (ms) | Compile (ms) | First execution (ms) | Repeats | Median (ms) | P5–P95 (ms) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| linesearch/backtracking | current | AOT | 13.482 | 28.965 | 0.290 | 40 | 0.012 | 0.010–0.094 |
| linesearch/backtracking | baseline | AOT | 12.894 | 22.649 | 0.227 | 40 | 0.012 | 0.011–0.026 |
| linesearch/interp23 | current | AOT | 18.317 | 29.744 | 0.306 | 40 | 0.015 | 0.014–0.031 |
| linesearch/interp23 | baseline | AOT | 18.562 | 28.545 | 0.248 | 40 | 0.016 | 0.013–0.047 |
| linesearch/strong_wolfe | current | AOT | 16.208 | 32.817 | 0.297 | 40 | 0.013 | 0.011–0.023 |
| linesearch/strong_wolfe | baseline | AOT | 16.350 | 30.974 | 0.361 | 40 | 0.013 | 0.012–0.040 |
| linesearch/golden_section | current | AOT | 17.549 | 30.175 | 0.234 | 40 | 0.016 | 0.014–0.040 |
| linesearch/golden_section | baseline | AOT | 17.252 | 27.520 | 0.194 | 40 | 0.014 | 0.012–0.033 |
| linesearch/unimodality | current | AOT | 28.239 | 51.056 | 0.355 | 40 | 0.027 | 0.026–0.035 |
| linesearch/unimodality | baseline | AOT | 29.244 | 47.403 | 0.338 | 40 | 0.027 | 0.024–0.056 |
| jacobian/autodiff | current | AOT | 4.188 | 9.704 | 0.151 | 40 | 0.005 | 0.004–0.008 |
| jacobian/autodiff | baseline | AOT | 4.229 | 9.408 | 0.135 | 40 | 0.005 | 0.004–0.012 |
| hessian/autodiff | current | AOT | 12.148 | 12.173 | 0.136 | 40 | 0.005 | 0.004–0.016 |
| hessian/autodiff | baseline | AOT | 10.417 | 11.924 | 0.129 | 40 | 0.005 | 0.004–0.008 |
| jacobian/central | current | AOT | 6.583 | 12.447 | 0.170 | 40 | 0.020 | 0.016–0.028 |
| jacobian/central | baseline | AOT | 6.416 | 13.299 | 0.149 | 40 | 0.022 | 0.018–0.029 |
| hessian/central | current | AOT | 27.610 | 46.063 | 0.265 | 40 | 0.029 | 0.023–0.038 |
| hessian/central | baseline | AOT | 11.697 | 21.977 | 0.139 | 40 | 0.023 | 0.020–0.035 |
| jacobian/forward | current | AOT | 7.591 | 13.805 | 0.136 | 40 | 0.021 | 0.018–0.027 |
| jacobian/forward | baseline | AOT | 7.069 | 14.705 | 0.131 | 40 | 0.022 | 0.017–0.026 |
| hessian/forward | current | AOT | 11.678 | 18.191 | 0.145 | 40 | 0.023 | 0.019–0.048 |
| hessian/forward | baseline | AOT | 13.163 | 19.474 | 0.161 | 40 | 0.021 | 0.019–0.037 |
| jacobian/backward | current | AOT | 6.929 | 14.407 | 0.130 | 40 | 0.020 | 0.015–0.025 |
| jacobian/backward | baseline | AOT | 7.003 | 14.998 | 0.141 | 40 | 0.026 | 0.017–0.223 |
| hessian/backward | current | AOT | 12.047 | 18.251 | 0.159 | 40 | 0.021 | 0.018–0.037 |
| hessian/backward | baseline | AOT | 11.993 | 18.968 | 0.138 | 40 | 0.017 | 0.010–0.024 |

The reusable API uses `compile_minimizer` in the current revision and an equivalent outer `jax.jit` in the baseline. Batching compares 16 complete solves in a Python loop with one compiled `vmap`; both include result collection. These are small dense test problems, not a representative nonlinear optimization corpus or GPU measurements. First execution in an AOT row excludes the preceding lowering/compilation. See [the separate process timing report](simplex-timing.md) for end-to-end startup.

The restored central Hessian uses the original higher-order five-point diagonal;
the baseline uses a different three-point diagonal. This comparison records the
compatibility change, not an equivalent-algorithm speedup. See
[compatibility notes](original-compatibility.md).

## Reproduction

```bash
python -m benchmarks.jax_audit --group kernels --trials 2 --repeats 20 --eager-repeats 3 --baseline-ref 74f2ce47800adf66723c8977da0fdc54901dccd5 --output docs/original-kernel-performance
```

The optional baseline requires the trusted revision in local Git history. Omit it for a source distribution or shallow checkout. [Raw samples, versions, devices, iteration counts, and hashes of all runtime source files](original-kernel-performance.json).

```json
{
  "utc": "2026-09-14T13:24:04.371476+00:00",
  "platform": "macOS-26.2-arm64-arm-64bit",
  "python": "3.12.2",
  "trials": 2,
  "repeats": 20,
  "eager_repeats": 3,
  "group": "kernels",
  "baseline_ref": "74f2ce47800adf66723c8977da0fdc54901dccd5",
  "persistent_compilation_cache": false
}
```

Method: [JAX benchmarking](https://docs.jax.dev/en/latest/benchmarking.html) and [explicit compilation stages](https://docs.jax.dev/en/latest/aot.html).

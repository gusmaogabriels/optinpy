# Forward and reverse autodiff: warm execution

CPU float64, with dynamic matrices, points and tangent/cotangent vectors. Scalar objectives are coupled softplus sums plus a quadratic term; rectangular Jacobians use sin(Ax). Every result is checked against analytic derivatives. Different modes compute the same requested derivative or product.

2 fresh processes, three warmups, 30 interleaved timed calls per mode in each process. Compilation order reverses between trials; warm order rotates. All calls synchronize and no samples are discarded. Persistent caching is disabled. Inputs, device initialization and imports are outside the timers. First staged call includes tracing/lowering, compilation and first execution; it is not total process startup. P5–P95 is the observed middle 90%, not a confidence interval.

| Case | Mode | Lower (ms) | Compile (ms) | First staged call (ms) | Warm median (µs) | P5–P95 (µs) |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| gradient/16 | grad | 13.283 | 18.261 | 31.741 | 12.312 | 9.496–21.979 |
| gradient/16 | jacrev | 10.725 | 16.799 | 27.715 | 18.084 | 9.617–22.631 |
| gradient/16 | jacfwd | 16.559 | 57.431 | 74.856 | 19.959 | 10.121–27.179 |
| hessian/16 | fwd-over-rev | 19.801 | 55.643 | 75.636 | 20.500 | 11.290–33.858 |
| hessian/16 | rev-over-fwd | 24.644 | 55.876 | 80.719 | 23.021 | 13.786–36.590 |
| hessian/16 | rev-over-rev | 21.590 | 54.835 | 76.626 | 19.958 | 10.956–27.950 |
| hessian/16 | fwd-over-fwd | 17.613 | 56.329 | 74.132 | 26.896 | 16.542–37.558 |
| hessian-vector/16 | jvp-grad | 6.421 | 17.585 | 24.187 | 19.104 | 16.079–25.925 |
| hessian-vector/16 | grad-dot-grad | 9.543 | 16.876 | 26.594 | 19.187 | 15.998–25.662 |
| hessian-vector/16 | dense-hessian-product | 10.985 | 55.655 | 66.848 | 21.771 | 17.996–31.517 |
| gradient/64 | grad | 7.758 | 15.471 | 23.382 | 21.250 | 18.494–28.835 |
| gradient/64 | jacrev | 8.431 | 16.234 | 24.828 | 21.125 | 19.536–27.579 |
| gradient/64 | jacfwd | 11.722 | 26.026 | 37.974 | 52.354 | 48.167–62.716 |
| hessian/64 | fwd-over-rev | 21.098 | 30.942 | 52.322 | 107.500 | 78.383–165.066 |
| hessian/64 | rev-over-fwd | 30.571 | 56.583 | 88.087 | 413.667 | 345.900–678.101 |
| hessian/64 | rev-over-rev | 26.255 | 24.050 | 50.618 | 105.791 | 75.521–183.153 |
| hessian/64 | fwd-over-fwd | 17.802 | 28.539 | 47.010 | 335.145 | 285.846–446.921 |
| hessian-vector/64 | jvp-grad | 6.534 | 17.345 | 24.047 | 22.020 | 18.662–31.625 |
| hessian-vector/64 | grad-dot-grad | 9.692 | 16.662 | 26.535 | 21.209 | 18.492–33.281 |
| hessian-vector/64 | dense-hessian-product | 11.193 | 30.423 | 41.933 | 75.584 | 70.165–182.746 |
| gradient/128 | grad | 7.900 | 15.667 | 23.732 | 29.709 | 18.581–34.221 |
| gradient/128 | jacrev | 8.356 | 15.938 | 24.463 | 29.416 | 18.240–34.752 |
| gradient/128 | jacfwd | 11.379 | 27.793 | 39.809 | 123.124 | 88.666–157.371 |
| hessian/128 | fwd-over-rev | 20.272 | 31.730 | 52.643 | 234.521 | 178.233–322.109 |
| hessian/128 | rev-over-fwd | 24.275 | 39.463 | 69.777 | 2605.646 | 2398.950–3201.496 |
| hessian/128 | rev-over-rev | 20.554 | 26.254 | 47.247 | 241.334 | 190.790–315.842 |
| hessian/128 | fwd-over-fwd | 20.214 | 30.258 | 53.273 | 2308.479 | 2062.356–2482.152 |
| hessian-vector/128 | jvp-grad | 6.376 | 17.229 | 23.784 | 35.541 | 30.202–51.363 |
| hessian-vector/128 | grad-dot-grad | 9.973 | 18.449 | 28.686 | 34.333 | 29.705–53.211 |
| hessian-vector/128 | dense-hessian-product | 12.438 | 32.413 | 45.302 | 194.812 | 175.873–269.992 |
| jacobian/8x128 | jacfwd | 5.523 | 21.622 | 27.354 | 37.770 | 31.494–52.483 |
| jacobian/8x128 | jacrev | 5.712 | 12.801 | 18.657 | 20.291 | 18.625–26.618 |
| transpose-product/8x128 | vjp | 2.979 | 11.787 | 14.914 | 18.959 | 15.279–23.679 |
| transpose-product/8x128 | dense-jacrev-product | 5.042 | 12.656 | 17.848 | 20.791 | 19.159–26.759 |
| jacobian/128x8 | jacfwd | 4.849 | 13.054 | 18.057 | 20.688 | 17.998–26.631 |
| jacobian/128x8 | jacrev | 4.796 | 11.709 | 16.699 | 59.042 | 52.659–68.704 |
| transpose-product/128x8 | vjp | 2.840 | 12.584 | 15.569 | 19.646 | 17.577–37.295 |
| transpose-product/128x8 | dense-jacrev-product | 4.419 | 12.181 | 16.798 | 56.001 | 52.579–90.357 |
| jacobian/64x64 | jacfwd | 4.761 | 19.357 | 24.290 | 34.584 | 31.081–40.948 |
| jacobian/64x64 | jacrev | 5.105 | 13.466 | 18.757 | 37.812 | 34.748–46.875 |
| transpose-product/64x64 | vjp | 2.958 | 13.261 | 16.373 | 18.125 | 9.748–33.438 |
| transpose-product/64x64 | dense-jacrev-product | 4.605 | 14.074 | 18.864 | 40.520 | 28.542–62.270 |

These are derivative microbenchmarks on this CPU. They do not establish a universal fastest mode, whole-solver speedup, or GPU/memory claim. The JAX compiler may simplify dense-product expressions.

Reproduce with `python -m benchmarks.autodiff_modes --trials 2 --repeats 30 --sizes 16 64 128`.

[Raw samples and environment](autodiff-modes.json). [JAX autodiff guidance](https://docs.jax.dev/en/latest/301/cookbook.html).

```json
{
  "utc": "2026-09-14T17:55:51.863429+00:00",
  "platform": "macOS-26.2-arm64-arm-64bit",
  "python": "3.12.2",
  "trials": 2,
  "repeats": 30,
  "sizes": [
    16,
    64,
    128
  ],
  "persistent_compilation_cache": false,
  "jax": "0.8.2",
  "jaxlib": "0.8.2",
  "devices": [
    "TFRT_CPU_0"
  ],
  "benchmark_sha256": "8f1549f9ca8a08227397c345bf6e303fefd09e7449d2f1a847723c423a2496b8"
}
```

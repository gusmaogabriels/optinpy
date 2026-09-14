# Warm execution and autodiff choices

This pass preserves the original formulas, solver interfaces, pricing rules,
precision settings, and stopping tolerances. It reduces array copies and
dispatch overhead, and computes derivative products directly where a full
matrix is unnecessary. All solver code remains inside optinpy.

## Simplex

Tableau elimination uses broadcast selections to set the pivot row/column in
the elimination pass. Input validation runs in fused JAX kernels. The phase
transition removes appended artificial columns with a compiled slice and cost
reconstruction. When no artificial variable is basic, its mass is already zero
and no RHS transfer is necessary. Cleanup pivots still respect budgets.

The [matched comparison](simplex-repeatability.md) uses baseline `8ddfe01`,
three fresh processes and 90 interleaved warm solves per solver/input. Every
sample builds a fresh model, solves it, synchronizes, and collects the answer;
validation runs outside the timer. All 1,800 samples pass. No samples are dropped.

| Variables | Before (ms) | After (ms) | Ratio before/after | Pivots in both |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 0.773 | 0.639 | 1.21× | 2 |
| 8 | 1.760 | 1.000 | 1.76× | 17 |
| 16 | 2.711 | 1.834 | 1.48× | 52 |
| 24 | 4.862 | 3.884 | 1.25× | 93 |

The four-variable timings are particularly noisy: its current P5–P95 spans
0.502–1.213 ms. The 24-variable spread is 3.563–4.713 ms. These are ratios of
medians, not guarantees for individual calls or other problems. Current medians
are lower in each process at all four sizes. HiGHS and Clarabel remain faster
on most cases.

The separate [104-case comparison](simplex-comparison.md) passes every float64
case, with a 1.005 ms median warm time. Default float32 still passes 63/104;
explicit `tol=1e-4` passes 104/104 against its separate acceptance tolerance.
No accuracy threshold was relaxed to obtain the speed improvements.

Warm times exclude initial compilation. [Fresh-process measurements](simplex-timing.md)
record first complete solves of 0.393–0.510 s and process wall times of
0.757–0.807 s. Prepared phase compilation/execution is reported separately.

## Forward versus reverse autodiff

Reverse mode generally suits many inputs and few outputs; forward mode often
suits more outputs than inputs. Scalar objectives are the former case. JAX's
dense Hessian composes forward over reverse.
[JAX autodiff cookbook](https://docs.jax.dev/en/latest/301/cookbook.html).

The [mode benchmark](autodiff-modes.md) verifies analytic derivatives with
dynamic matrices, points and vectors, using 60 interleaved warm calls per mode:

| Requested result | Alternatives on the specified CPU case | Warm medians |
| --- | --- | --- |
| Scalar gradient, 128 inputs | `grad` / `jacfwd` | 29.7 / 123.1 µs |
| Jacobian, 128 outputs and 8 inputs | `jacfwd` / `jacrev` | 20.7 / 59.0 µs |
| Jacobian, 8 outputs and 128 inputs | `jacrev` / `jacfwd` | 20.3 / 37.8 µs |
| Hessian, 128 inputs | forward-over-reverse / reverse-over-forward | 234.5 / 2605.6 µs |
| Hessian-vector product, 128 inputs | `jvp(grad)` / dense Hessian product | 35.5 / 194.8 µs |
| Transpose product, 128 outputs and 8 inputs | `vjp` / dense Jacobian product | 19.6 / 56.0 µs |

Reverse-over-reverse Hessians were close to forward-over-reverse here. There
is no universal winning composition. These are derivative timings, not whole
optimizer speedups or peak-memory measurements.

Optinpy keeps reverse-mode scalar gradients and forward-over-reverse dense
Hessians as defaults. The original Hessian-conjugate solver already uses a
Hessian-vector product. Nonlinear KKT checks now use `vjp` for the same
stationarity term $J(x)^T\lambda$, avoiding full constraint Jacobians. Existing
`jac` and `hess` callbacks support explicit mode choices.

## Original finite differences

The central Hessian keeps the original five-point diagonal and mixed four-probe
stencil. Probe indices depend only on vector size, so they are now prepared
during tracing. Objective evaluations remain batched with `vmap`.

The [matched kernel report](warm-kernels.md) records mixed warm results:
the 64-variable coupled case decreased from 2.064 to 1.748 ms, while the tiny
quadratic increased from 0.024 to 0.030 ms, with broad overlapping ranges.
The 128-variable coupled case was nearly unchanged. The tiny case's lowering
plus compilation decreased from about 87 to 51 ms. There is no blanket
finite-difference speedup claim. Reports retain raw samples and source hashes.

`jacfwd`/`jacrev` are autodiff modes, distinct from `algorithm='forward'` or
`'backward'`, which select finite-difference stencils in optinpy.

## Validation

The full source suite passes 347 tests, including original Hessian formulas,
JIT/batching, simplex pricing/budgets/cycling, independent LP comparisons,
analytic KKT residuals for multiple nonlinear constraints, and actual README
examples. The example program passes. CI includes a bounded autodiff-mode
benchmark with analytic checks and informational timings. Hosted CI has not
run for these local changes.

The installed wheel, tested outside the checkout on JAX 0.4.38 and NumPy
1.26.4, passes 315 tests; 32 optional Clarabel comparisons are skipped there
and pass in the main environment. Every installed runtime file matches its
source. The bounded autodiff benchmark also passes on the minimum environment.
Both source distribution and wheel metadata pass `twine check`.

# JAX optimization audit

This audit covers every runtime module while preserving the original package
structure and native solver implementations. The changes target redundant
arithmetic, repeated compilation, and Python/device transfers. Tests cover
mathematical answers and JAX transformations; measurements are local CPU
results, with compilation reported separately. See the [measurement tables
and raw samples](jax-performance.md) and [independent LP comparisons](simplex-repeatability.md).

The latest [warm-execution pass](warm-execution.md) measures 1.25–1.76× faster
warm simplex solves on the 8–24-variable fixed cases against `8ddfe01`, with
identical pivot counts. It also compares [forward/reverse autodiff modes](autodiff-modes.md)
and replaces dense constraint Jacobians with VJPs in KKT checks.

The historical matched run of `49c4978` against commit `6a0f69a` measured these improvements:
256-variable BFGS, 32.276 to 7.056 ms (4.6×); 16-variable projected gradient,
386.351 to 47.204 ms (8.2×); penalty/barrier APIs, 4.3–9.1× on the one-variable
boundary cases. Both BFGS versions took 83 iterations on the 256-variable case.
These numbers describe the specified cases and timing boundaries, not all
objectives. Repeated constrained calls still include some compilation.

Small-kernel results are mixed. The eight-variable L-BFGS case regressed from
0.054 to 0.074 ms with the same ten iterations; the report retains both trials
and all samples. Many line-search and derivative timings are within overlapping
observed ranges. Reducing source-level evaluations does not guarantee faster
generated code on a tiny objective. This audit makes no L-BFGS or standalone
line-search speedup claim; that regression remains a profiling target.

## Module coverage

| Module or algorithms | Checked and implemented | Remaining limit |
| --- | --- | --- |
| BFGS | Expand the inverse-Hessian update into one matrix-vector product and rank-two updates, reducing update work from O(n³) to O(n²). Retain curvature and finite-value safeguards. | Dense O(n²) state; limited-memory BFGS is available. |
| Gradient, Newton, modified Newton, Hessian-conjugate gradient, PR+, Fletcher–Reeves, DFP, L-BFGS | Keep iterations in `lax.while_loop`; reuse the accepted strong-Wolfe value and gradient when it uses the solver's derivative. Line-search-only derivative overrides retain independent objective-gradient evaluation. All eleven minimizers support a reusable compiled callable and `vmap`. The original Hessian-conjugate formula now uses a matrix-free autodiff Hessian-vector product. | Newton forms a dense Hessian; modified Newton also decomposes it. Truncated Newton would be a separate algorithm addition. |
| Adam and SGD | Existing compiled iterations and full-gradient updates verified in float32/float64 and batches. | Flat arrays and full objectives; no PyTree/minibatch training interface. |
| Armijo and interpolation | Accept known initial function values and gradients; remove duplicate initial trial evaluation. | Standalone calls should be wrapped once in `jit` for repetition. |
| Strong Wolfe | Carry the gradient at the returned point, including failure; reuse it in the minimizer. Supplied derivatives remain honored at trial points. | Bracketing/zoom work depends on the objective. |
| Golden section and randomized interval reduction | Golden section retains one probe/value per reduction. The random variant keeps an explicit PRNG key; both reuse the initial objective at final selection. | Random search needs two probes per reduction. |
| Autodiff and finite differences | Automatic gradients/Hessians and `vmap` over finite-difference probes verified. Central Hessians retain the original five-point diagonal, evaluate one mixed-derivative triangle, and share the center value. | Dense finite-difference Hessians can have large intermediate memory; batched evaluation does not eliminate function-evaluation cost. The historical timing snapshot predates the restored stencil. |
| Projected gradient and feasibility | One SVD supplies nullspace geometry and multipliers. Cache it while the working set stays the same; compile projection, feasibility, step limits, and a reusable line search. Joint value/gradient evaluation avoids repeated objective work. | Active-set decisions and termination still synchronize with Python. Factors are cached within a solve, not between separate calls. |
| Penalty, reciprocal barrier, log barrier | Compile the inner minimizer once per outer solve; pass the changing weight as data. Compile KKT residual evaluation using VJP transpose products instead of full constraint Jacobians. | Repeated top-level calls can compile again; continuation and stopping remain in Python. |
| Simplex | Fixed-shape primal phases are compiled. Pricing retains a persistent Bland fallback. Fused validation, broadcast pivot updates, and compiled removal of artificial columns reduce overhead. | Dense tableau, no general presolve or sparse revised simplex. Python handles changing shapes during phase-I cleanup. |
| Minimum-cost flow | Uses the native simplex improvements; transfer the flow vector to Python once when assigning mutable arcs. | Dense Big-M formulation; a dedicated network simplex could exploit incidence structure better. |
| Graph, shortest paths, spanning trees | Reviewed object traversal, heaps, relaxation, and union-find implementations. Preserve Python algorithms and mixed-label graph interfaces. | A JAX array graph API would need a separate representation; replacing object traversal with individual JAX operations would add dispatch overhead. |
| Package exports and legacy configuration | Export `compile_minimizer` through both modern entry points. Avoid recording constrained histories unless requested. | Legacy parameter dictionaries remain shared mutable configuration. |

## Compilation and data lifetime

Create `compile_minimizer(fun, **options)` once, then call `solve(x0, *args)`.
The factory supplies an outer `jax.jit`; callers could already write an
equivalent wrapper. Matching shapes/dtypes and fixed solver configuration reuse
the compiled program. Pass targets, coefficients, and other changing numerical
data through objective arguments. A new function object or changed static
configuration can cause compilation; this is why weighted continuation now
uses one function and a dynamic scalar weight. The tests check trace reuse
when objective data changes. [JAX compilation caching](https://docs.jax.dev/en/latest/jit-compilation.html).

First-call time is part of wall time. The audit reports lowering and compilation
separately for staged kernels, and includes any compilation inside ordinary
API calls. Every timed result is synchronized. The [simplex startup report](simplex-timing.md)
also measures whole fresh-process wall time including imports; prepared-kernel
timings must not be presented as that total. [JAX benchmarking guidance](https://docs.jax.dev/en/latest/benchmarking.html).

The library does not enable a persistent disk cache globally. Applications may
configure one for reuse between processes; measurements disable it explicitly
so it cannot hide compilation costs. Shapes, dtypes, static options, and device
configuration still determine usable cache entries. [Persistent compilation cache](https://docs.jax.dev/en/latest/persistent_compilation_cache.html).

## Batching, memory, and precision

Use `jax.jit(jax.vmap(solve, in_axes=(None, 0)))` for independent targets and
a shared starting vector. The usage benchmark compares 16 collected solves
with a single batched solve. Batch size, differing convergence lengths, and
device memory can affect the tradeoff; there is no blanket batching speedup
claim. Simplex objects, constrained drivers, and mutable graphs are not
whole-solve `vmap` interfaces. [JAX transformations](https://docs.jax.dev/en/latest/101/transformations.html).

Optional `donate_x0=True` permits reuse of the initial array's buffer. It is
off by default because callers commonly reuse their starting point; donated
inputs must not be used after the call. Donation is a permission to reuse
memory, not a guaranteed speedup. No peak-memory improvement has been measured.
Histories require additional storage and use a Python driver. L-BFGS already
bounds curvature history; replacing its rolling buffer with a ring buffer
is a future profiling candidate. [Buffer donation](https://docs.jax.dev/en/latest/buffer_donation.html).

Keep data on the chosen device across repeated compiled solves. The mutable
simplex and constrained APIs need some host decisions; the audit reduces
their frequency but does not claim zero transfers. Mixed precision, GPU/TPU
kernels, Pallas/FFI, sharding, and distributed execution were considered but
are not implemented or benchmarked in this CPU audit. The current test corpus
does not justify hardware-specific tuning. [JAX data placement](https://docs.jax.dev/en/latest/201/placement.html).

Float64 remains recommended for demanding LP tolerances. Float32 outcomes at
both default and relaxed tolerances remain visible in the [LP report](simplex-comparison.md).
The package never changes global precision settings. JIT performance does not
justify relaxing correctness thresholds silently. [JAX default dtypes](https://docs.jax.dev/en/latest/101/default_dtypes.html).

Dynamic optimization loops support differentiation of the objective; reverse-mode
differentiation through the solver is not provided. Checkpointing would not by
itself add that capability. An implicit derivative API would need additional
optimality assumptions, linear solves, and tests. [JAX loop differentiation limits](https://docs.jax.dev/en/latest/_autosummary/jax.lax.while_loop.html).

## Next performance work

Priorities depend on workload: presolve and revised simplex for large/sparse
LPs, Hessian-vector methods for large nonlinear models, and a fixed-shape
compiled constrained interface for batches. Measure these against the actual
problem family before choosing one. This audit does not establish parity
with mature external solvers on industrial optimization workloads.

## Validation

The latest warm-execution pass passes 347 source tests and all 1,800 matched LP
timing samples. Its 104-case comparison retains the float64/float32 outcomes below.
The installed wheel passes 315 tests on JAX 0.4.38/NumPy 1.26.4 outside the
checkout, with 32 optional Clarabel checks skipped. Its runtime source hashes
match, the bounded autodiff benchmark passes, and distribution checks pass.

The subsequent original-formula restoration passes 344 source tests, including
the actual main README snippets, Hessian-conjugacy checks, and exact quartic
derivatives. Its [kernel timing check](original-kernel-performance.md) retains
the small central-Hessian regression; see [compatibility details](original-compatibility.md).
The 72 affected derivative, unconstrained, and README tests also pass against
the installed wheel outside the checkout on JAX 0.4.38. Every installed runtime
file matches its source hash, and both distribution metadata checks pass.

The preceding README-flow-fix snapshot passed 334 tests on JAX/JAXlib 0.8.2 with NumPy 2.1.0.
An installed wheel, imported outside the checkout and checked against every
runtime source file's hash, passes 302 tests on JAX/JAXlib 0.4.38 with NumPy
1.26.4. Its 32 Clarabel tests are skipped because that optional benchmark
dependency is absent there; they pass in the primary environment. The example
also runs from the installed wheel. Source distribution and wheel metadata
pass `twine check`. Hosted GitHub Actions has not been run for this local commit.

All 104 LP cases pass in float64, and all 1,800 timed repeated-input LP samples
pass their objective and feasibility checks. Default-tolerance float32 passes
63/104 cases; the explicit `tol=1e-4` run passes 104/104 against the separate
`2e-4` acceptance threshold. See the reports for exact inputs and tolerances.

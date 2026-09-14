# Preserving the original optinpy

The reference is the original implementation on `master`, not another solver's
API. Keep its module layout, public calling patterns, mathematical algorithms,
and README explanations recognizable. Use JAX for numerical kernels and
measure speed improvements without silently changing the requested algorithm.
New methods remain optional additions implemented inside optinpy.

This branch is a substantial numerical rewrite, not a line-for-line NumPy
translation. The current relationship to the original is:

| Original feature | Current implementation and differences |
| --- | --- |
| Package/module paths and parameter-based `fminunc`, `fmincon`, `fminnlcon` | Retained. JAX objectives and array results replace mutable NumPy calculations; `vectorized=True` still requests history. Defaults and stopping safeguards have changed, as documented in the README. |
| Gradient, Newton, BFGS, DFP | Retain the mathematical direction/update families, with JAX loops, linear solves instead of explicit inverses, gradient reuse, and numerical safeguards. BFGS retains the original expanded rank-two form. Quasi-Newton state updates fix bugs in the old driver. |
| Separate direction/update routines and interchangeable line searches | Restored as public JAX primitives in the existing `nonlinear/unconstrained.py` module, exported through `optinpy.nonlinear`. The complete driver composes the same functions. Original underscored class methods are not a supported compatibility API; public `fminunc` and parameter-based selection remain. `minimize` also accepts custom stateless direction and line-search callables. |
| Original Hessian-conjugate gradient | Restored for the original `params` selector; exposed as `hessian-conjugate-gradient` in `minimize`. Autodiff computes the Hessian-vector product directly; explicit Hessian callbacks and finite differences remain supported. PR+ is a separate modern option. |
| Modified Newton | Symmetric eigendecomposition and a linear solve replace the old SciPy factorization. The current method floors eigenvalues; the original took their absolute values first. This is a remaining algorithmic difference. |
| Central finite-difference Hessian | Original five-point diagonal and four-probe mixed derivative restored. Evaluate one triangle with `vmap`, reflect it, and share the central function value. Default derivatives use autodiff; finite differences remain explicitly selectable. |
| Backtracking, interpolation, interval searches | Original method families retained with compiled loops, reused probes, bounded searches, and failure reporting. Strong Wolfe is an additional option. |
| Projected gradient, penalty, reciprocal and log barriers | Original method families retained. Feasibility search, SVD projection, KKT checks, and continuation drivers were rewritten for correctness. Barrier/penalty inner solves default to modified Newton but now accept `inner_options` for method, line search and derivatives. Unlike the original driver, this explicit configuration is independent of `params['fminunc']`. |
| Simplex | Retains tableau pivots and the class interface. New two-phase setup, bound transformations, scaling, termination, and anti-cycling logic fix unfinished/incorrect behavior. Bounds were keyword arguments in the original constructor too; the old README's positional-bounds example was inaccurate. |
| Graph, shortest paths, spanning trees | Preserve mutable graph objects, mixed labels, and named algorithms. Traversal remains Python; these are not compiled JAX solvers. |
| Big-M minimum-cost flow | Public network-in/network-out interface retained. The solver currently builds a dense LP and uses optinpy's own simplex, instead of the original network-tree pivot loop. Restoring a dedicated network simplex is remaining work, not something this branch already provides. |
| README | Original explanations, equations, figures, and adapted usage examples stay in the main README. CI executes its Python examples. Historical plotted trajectories/timings and the archived README are not exact-output compatibility guarantees. |

The numerical regression tests check the restored conjugate directions for
Hessian orthogonality and the five-point stencil against exact quartic
derivatives, including compiled and batched execution. These distinguish the
original mathematics from merely reaching the same final minimum.

Performance reports must identify the measured revision and distinguish total
first-call time (including JIT) from synchronized warm execution. Mathematical
equivalence, fewer operations, or use of JAX alone does not establish a speedup.

The [restored-kernel timing check](original-kernel-performance.md) records two
fresh-process trials per revision, with 40 synchronized warm samples per case.
On the 16-variable quadratic, the restored central Hessian's median rose from
0.023 to 0.029 ms; lowering plus compilation rose from about 34 to 74 ms.
Warm ranges overlap. This is an accuracy/compatibility restoration, with no
speedup claim. Autodiff remains the default. The new Hessian-conjugate solver
has correctness/JIT/batching tests but no matched solver timing claim yet.

Subsequent [warm-execution work](warm-execution.md) preserves these restored
formulas while reducing simplex overhead, preparing finite-difference indices
during tracing, and using direct VJPs for nonlinear constraint stationarity.

The direction/update extraction restores independently callable components
without changing their equations. The [before/after audit](component-performance.md)
compares all eleven minimizers with commit `24668ce`; iteration counts match
in every trial. Some tiny cases measured 16–28% slower in separate processes.
The follow-up [interleaved check](component-interleaved.md) compiles both revisions
in each of three fresh processes and alternates their executions. Across its
six cases, all result fields and normalized compiled operations are identical,
and warm medians differ by less than 1%. Both timing distributions are retained;
these measurements support no material refactor overhead on these inputs,
not a universal performance guarantee. The refreshed library comparison also
matches all 62 preceding accuracy/status results, including the DFP failures.

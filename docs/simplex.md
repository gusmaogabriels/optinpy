# Simplex: bounds, numerical fixes, and independent validation

The old “BEING FIXED” warning described the original Python 2 implementation
and its lower/upper-bound handling. The current implementation is a native
JAX two-phase tableau simplex in `optinpy/simplex/base.py`. External LP solvers
are used only by the comparison tests and benchmark scripts.

## Solve a bounded LP

```python
import jax
jax.config.update('jax_enable_x64', True)
import optinpy

# Maximize 3*x[0] + 2*x[1], with linear inequalities and explicit bounds.
problem = optinpy.simplex(
    A=[[1., 1.], [1., 0.]], b=[4., 2.], c=[3., 2.],
    mode='max', lb=[0., 0.], ub=[3., 5.],
)
result = problem.solve(max_iter=1000)
assert result['success'], result['message']
print(result['x'], result['f'])  # [2., 2.], 10.
```

The problem is `min/max c @ x` subject to `A @ x <= b` and `lb <= x <= ub`.
Bounds are keyword arguments and can be scalars or vectors. Defaults are
`lb=0`, `ub=inf`. Negative lower bounds, upper-only bounds, free variables
(`lb=-inf`, `ub=inf`), and fixed variables (`lb=ub`) are supported. Supply
equalities as two opposite inequalities. Input arrays/lists are not modified.

Lower bounds are handled by a shift, upper-only bounds by a reflection, and
free variables by a difference of two nonnegative variables. Finite upper
bounds become explicit constraints, and fixed variables are eliminated.
Negative right-hand sides receive
artificial variables; phase I minimizes their sum before the original
objective is restored in phase II.

## What was corrected

- The original README's minimum-cost-flow network exposed roundoff-sized
  improving costs at the end of phase I. If no eligible pivot remains but
  artificial mass is already zero within tolerance, that mass certifies
  phase-I optimality. Both the compiled driver and individual primal steps
  now proceed to cleanup instead of reporting an apparent unbounded auxiliary
  problem as numerical failure. The original network is a regression fixture
  for all pricing rules and independent solver comparisons.
- The legacy code modified objective lists in place; maximization could erase
  the list. Array and list concatenation also behaved differently. The native
  formulation now handles both through immutable JAX arrays.
- Lower/upper bounds are part of a consistent standard-form transformation,
  including objective offsets and reconstruction of the original variables.
- Equivalent constraint/objective units no longer make small coefficients
  disappear under the pivot tolerance. Implied variable ranges, row and
  decision-column scaling, and objective normalization balance the tableau.
- Reduced costs are recomputed from the current basis with a separate
  cancellation tolerance for each column. A small improving coefficient
  is not ignored merely because another cost coefficient is much larger.
- A near-tie in the minimum-ratio test previously allowed a step past the
  tightest constraint. The solver now takes the actual minimum ratio and uses
  Bland's variable ordering only for tied minima.
- An optimum reached on the last permitted pivot is now reported as optimal.
  An initially optimal point succeeds with a zero-pivot budget.
- Phase-I cleanup respects the same pivot budget. Solves can resume with a new
  budget, and return both total and per-call pivot counts.
- Nonfinite tableaus and excessive final primal residuals produce an explicit
  numerical-failure status. Exactly contradictory zero rows are infeasible.
- Pivot arithmetic runs inside compiled JAX phases, and the pivot column is set
  exactly to its canonical basis vector to reduce accumulation of roundoff.

## Results and pivot interfaces

`solve(max_iter=N)` allows at most **N additional pivots**. Optimality checks
and phase transitions that do not pivot are free. `iterations` is cumulative;
`iterations_this_solve` counts the current call. A solve stopped by its budget
can resume by calling `solve` again.

| Status | Meaning |
| --- | --- |
| 0 | Optimal within the selected numerical tolerance |
| 1 | Pivot budget exhausted; resume or increase the budget |
| 2 | Unbounded after establishing feasibility |
| 3 | Infeasible |
| 4 | Numerical failure; inspect the data, scaling, and precision |

The result includes `x`, `f`, `success`, `status`, `message`,
`primal_violation`, and `scaled_primal_violation`. The first violation is the
largest positive constraint/bound residual in the original units. The second
divides linear-constraint residuals by their maximum absolute row coefficient;
variable-bound residuals remain in the original variable units. An unsuccessful
result's `x` is a candidate, not a certified solution.

`primal()` performs one pivot or a phase-management operation. `dual()` performs
one dual-simplex pivot and requires nonnegative reduced costs; it is not a
complete alternative driver. `pivot(i, j)` exposes a single validated
Gauss–Jordan step. The main `solve()` driver uses primal two-phase simplex.

## Precision and practical limits

Use **float64** for scientific LP solves. Optinpy does not change your global
JAX precision configuration. The default tolerance is
`max(1e-9, 10 * machine_epsilon)` in the normalized problem. Float32 can report
numerical failure as pivot errors accumulate; `tol=1e-4` is also measured in
the comparison, with a separate acceptance threshold. Relaxing tolerance
trades accuracy for robustness and is not a substitute for double precision.

This is a dense tableau solver. Each fixed-shape primal phase runs in a JIT
compiled `jax.lax.while_loop`, including reduced-cost evaluation, pivot
selection, budget checks, and numerical checks. Python manages bound metadata
and phase-I cleanup, where rows and artificial columns can be removed.
Construction, equilibration, and final residual checks also use compiled
kernels. There is no Python/device synchronization per ordinary pivot.

`pricing='dantzig'` is the default: select the most negative improving reduced
cost. `pricing='steepest-edge'` divides reduced costs by the exact tableau edge
norm; its extra column reductions can outweigh fewer pivots. `pricing='bland'`
retains first-improving-variable selection. All rules use Bland ordering for
exact minimum-ratio ties. After eight consecutive steps of length at most
`tol`, an instance permanently switches to Bland pricing. This safeguard
persists across phase transitions and resumed `solve` budgets; the result's
`bland_fallback` flag exposes it. Regression tests include the classical cycling
tableau and comparisons of all three rules against independent solutions.

The individual `primal`, `dual`, and `pivot` methods remain available, and
regression tests compare individual primal steps with the compiled driver.
This is not a sparse revised-simplex engine, a differentiable optimization
layer, or a whole-object `jax.jit` interface. It has no general presolve or basis
refactorization. The comparison covers small dense LPs and edge cases, not
large industrial models, and includes actual runtime and float32 limitations.

## Compilation and wall time

The solver comparison records each case's first call separately from the
median of subsequent fresh solves. A first call can reuse compilation from
earlier cases. The warm median excludes the initial compilation, but includes
construction, cleanup, device synchronization, and final checks.

The [startup and JIT timing report](simplex-timing.md) uses fresh CPU processes
with the persistent compilation cache disabled. It records process wall time,
the first complete solve, and warm construction/solve times separately. A
second experiment times explicit lowering, compilation, and execution of
prepared phases. These kernel-only measurements exclude the Python wrapper.
Changing shapes, dtypes, or the number of artificial variables can trigger
compilation; changing numerical data and the pivot budget preserves a cached
variant when shapes and dtypes match.
Changing the pricing rule selects a separate compiled variant.

The [repeated-input comparison](simplex-repeatability.md) measures timing spread
for identical LPs across multiple processes, with warmups and balanced solver
order. It preserves every raw sample and compares current/earlier optinpy,
HiGHS simplex, HiGHS IPM, and Clarabel. A range across different problem sizes
must not be interpreted as repeatability of one solve.

## Compare and reproduce

See the [measured comparison](simplex-comparison.md) and
[per-case results](simplex-comparison.json). They include HiGHS dual revised
simplex, HiGHS interior point with crossover, and Clarabel. Both HiGHS methods
share a library; Clarabel provides a separate implementation.

```bash
python -m pip install -e '.[dev,benchmark]'
python -m pytest -q tests/test_simplex.py tests/test_simplex_comparison.py
python -m benchmarks.compare_simplex --random-cases 80 --repeats 3 --float32
python -m benchmarks.time_simplex --repeats 5
```

The same normalized mathematical problem is given to every solver. Comparisons
check statuses, objective values, and original-variable constraint/bound
residuals, allowing different vertices when the optimum is nonunique. Fixtures
include the old bounded example, negative and infinite bounds, fixed/free
variables, redundant equalities, infeasibility, unboundedness, zero rows,
near-ratio ties, the classical cycling example, a Klee–Minty example, and
seeded dense LPs of 4, 8, 16, and 24 variables. Separate analytic tests exercise
unnormalized data and iteration-limit behavior.

CI runs the independent comparison tests and uploads a fresh comparison report.
Timing is informational; objective and feasibility failures block the build.
Optional comparison dependencies do not enter optinpy's runtime dependency list.

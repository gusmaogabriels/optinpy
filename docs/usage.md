# Use each optinpy algorithm

These examples use the current JAX implementation and the original module
layout. Run the setup first, then any task below. Each task defines its own
problem. All optimization algorithms are implemented inside optinpy.

```python
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import optinpy
```

## Unconstrained optimization

Pass a scalar objective and a one-dimensional starting point. Select the
algorithm with `method`; this example uses BFGS.

```python
objective = lambda x: jnp.sum((x - jnp.array([1., -2.]))**2)
result = optinpy.minimize(objective, [0., 0.], method='bfgs', tol=1e-6)
assert result['success'], result['status']
print(result['x'])  # Approximately [1., -2.]
```

Other method names are `gradient`, `newton`, `modified-newton`,
`conjugate-gradient` (Polak–Ribiere+), `hessian-conjugate-gradient` (original
Hessian-conjugate formula), `fletcher-reeves`, `dfp`, `lbfgs`,
`adam`, and `sgd`. Newton methods form a dense Hessian. BFGS/DFP store a dense
inverse-Hessian approximation; L-BFGS uses `memory_size` stored pairs. Adam
and SGD use `learning_rate` and the full objective gradient. Method settings
and iteration budgets can need adjustment for a different problem.

Check `success`, `status`, and `gradient_norm` before using the answer. See the
[README status table](../README.md#algorithms-and-module-structure).

## Original parameters and history

The original entry point still selects its method through `params`.

```python
objective = lambda x: jnp.sum((x - jnp.array([1., -2.]))**2)
optinpy.nonlinear.params['fminunc']['method'] = 'bfgs'
optinpy.nonlinear.params['linesearch']['method'] = 'backtracking'
result = optinpy.nonlinear.unconstrained.fminunc(
    objective, [0., 0.], threshold=1e-12, vectorized=True)
assert result['success'], result['status']
print(result['x'][-1])  # Final point; result['x'] holds the full history
```

`threshold=1e-12` bounds the squared gradient norm, corresponding to
`minimize(tol=1e-6)`. Legacy `vectorized=True` means iteration history.
For independent batches use [JAX batching](../README.md#jax-compilation-and-batching).
The parameter dictionary is shared mutable configuration; its changes persist
for later legacy calls. It does not configure the modern `minimize` API.
Selecting `conjugate-gradient` in this original parameter dictionary uses the
original Hessian-conjugate formula; the modern API exposes it explicitly as
`hessian-conjugate-gradient`.

## Derivatives

The historical name `jacobian` returns the gradient of a scalar objective.
Use either automatic differentiation or a finite-difference algorithm directly.

```python
objective = lambda x: 3*x[0]**2 + x[1]**2
x = jnp.array([1., 2.])
gradient = optinpy.finitediff.jacobian(objective, x)
hessian = optinpy.finitediff.hessian(objective, x)
central = optinpy.finitediff.jacobian(objective, x, algorithm='central')
print(gradient)  # [6., 4.]
print(hessian)   # [[6., 0.], [0., 2.]]
```

Both functions accept `algorithm='autodiff'`, `'central'`, `'forward'`, or
`'backward'`. Finite differences also accept an explicit `epsilon`.

JAX's `jacfwd`/`jacrev` select **autodiff modes**; they are different from
`algorithm='forward'`/`'backward'`, which select finite-difference stencils here.
For `minimize`, pass a desired autodiff function through `jac` or `hess`, such as
`jac=jax.jacrev(objective)` and `hess=jax.jacfwd(jax.grad(objective))`.
The defaults already use reverse-mode scalar gradients and forward-over-reverse
Hessians. See the [mode comparisons](autodiff-modes.md) for measured tradeoffs.

## Line search

A line search returns a step along one direction; it does not run a full
multidimensional minimization. Supply the direction or let it default to the
negative gradient.

```python
objective = lambda x: jnp.sum(x*x)
x = jnp.array([1., 2.])
gradient = optinpy.finitediff.jacobian(objective, x)
step = optinpy.linesearch.strong_wolfe(
    objective, x, -gradient, gradient=gradient, value=objective(x))
assert step['success']
print(step['alpha'], step['x'])  # 0.5, approximately [0., 0.]
```

Standalone functions are `backtracking`, `interp23`, `strong_wolfe`,
`golden_section`, and `unimodality`. Known `gradient` and `value` arguments
avoid reevaluating the starting point. `unimodality` accepts an explicit JAX
random key; golden section accepts an interval endpoint `b`.
When selecting a line search through `minimize`, use hyphenated names such
as `linesearch='strong-wolfe'` or `'golden-section'`.

## Mix and match components

Independent algorithm selection was part of the original package. The JAX
implementation retains those modules and also exposes its direction/update
primitives through `optinpy.nonlinear`. No external solver supplies these updates.

| Function | Inputs and result |
| --- | --- |
| `gradient_direction(g)`, `sgd_direction(g)` | Return `-g`; SGD's learning rate is applied by the driver. |
| `newton_direction(g, Q)` | Solve `Q @ d = -g` for direction `d`. |
| `modified_newton_direction(g, Q, sigma=...)` | Symmetrize Q, floor eigenvalues at sigma, solve for d. |
| `conjugate_gradient_direction(g, old_g, old_d, restart=...)` | Modern Polak–Ribiere+ direction. |
| `fletcher_reeves_direction(g, old_g, old_d, restart=...)` | Fletcher–Reeves direction. |
| `hessian_conjugate_gradient_direction(g, old_d, Qd)` | Original Q-conjugate recurrence; supply `Q @ old_d` or an equivalent Hessian-vector product. |
| `quasi_newton_direction(g, inverse)` | Return `-inverse @ g`; pairs with either inverse-Hessian update below. |
| `bfgs_update(inverse, p, y)`, `dfp_update(inverse, p, y)` | Return an updated inverse Hessian, with `p=x_new-x`, `y=g_new-g`. |
| `lbfgs_direction(g, steps, changes, count)` | Limited-memory two-loop inverse-Hessian product. |
| `lbfgs_update(steps, changes, count, p, y)` | Return updated histories and count, keeping valid pairs newest first. |
| `adam_direction(g, m, v, iteration, beta1=..., beta2=..., epsilon=...)` | Return `(d, m_new, v_new)`; the driver applies the learning rate. |

All directions/updates operate on real vectors and precomputed derivatives.
They preserve float32/float64 and work inside `jax.jit` and `jax.vmap`.
Hyperparameters such as sigma and Adam's beta values are static under JIT;
derivatives, matrices, histories, restart flags and iteration counters can be
dynamic. Allocate L-BFGS histories as zero arrays `(memory_size, len(x))` with
`count=0`; populate them with `lbfgs_update`. Initialize Adam moments to zero
and pass the number of **completed** updates (zero for the first call).
Restart conjugate-gradient methods by selecting `gradient_direction` or their
`restart=True` option; `minimize` restarts every `len(x)` iterations.

The driver handles iteration budgets, stopping, line searches, nonfinite results,
and fallback to steepest descent for an unusable direction. A primitive alone
does not perform a complete minimization or all those driver checks. In particular,
a singular `newton_direction` can be nonfinite. BFGS/DFP/L-BFGS updates skip
insufficient curvature; dense updates also reject nonfinite candidates.

Pass a named method with a standalone search directly:

```python
objective = lambda x: jnp.sum((x-jnp.array([1., -2.]))**2)
result = optinpy.minimize(
    objective, [0., 0.], method='bfgs',
    linesearch=optinpy.linesearch.interp23,
    jac=lambda x: optinpy.finitediff.jacobian(objective, x, algorithm='central'))
assert result['success']
```

A custom **stateless** direction has signature `(x, gradient, *args) -> d`.
A custom line search takes `(fun, x, d, *, gradient, value, **options)` and
returns a dictionary with `x`, scalar `success`, and scalar `iterations`.
The objective given to the search already has its problem arguments bound.
For example, supply a diagonal preconditioner while retaining optinpy's
line-search and convergence checks:

```python
def component_objective(x, target, weights):
    return .5*jnp.sum(weights*(x-target)**2)

def diagonal_direction(x, gradient, target, weights):
    return -gradient/weights

component_solve = optinpy.compile_minimizer(
    component_objective, method=diagonal_direction,
    linesearch=optinpy.linesearch.backtracking)
component_result = component_solve(
    jnp.zeros(2), jnp.array([1., -2.]), jnp.array([2., 5.]))
assert component_result['success']
```

Custom searches have their final objective/gradient recomputed using the chosen
derivative, so an unrelated derivative inside a search cannot determine solver
convergence. The built-in strong Wolfe callable reuses its accepted gradient when
it matches the configured derivative. Stateful custom algorithms can assemble
the same public primitives in their own loop, as in the [README example](../README.md#algorithms-and-module-structure).
Adam/SGD selected by name retain their fixed-learning-rate behavior; their
standalone direction primitives can be combined with a search in a custom driver.

## Linear programming

The simplex convention is `A @ x <= b`, with nonnegative variables unless
bounds are supplied. This example minimizes `-3*x[0] - 2*x[1]`.

```python
problem = optinpy.simplex(
    [[1., 1.]], [4.], [-3., -2.],
    lb=[0., 0.], ub=[2., 3.], pricing='dantzig')
result = problem.solve()
assert result['success'], result['status']
print(result['x'], result['f'])  # Approximately [2., 2.], -10.
```

Use `mode='max'` for maximization. Individual `primal()`, `dual()`, and
`pivot(i, j)` operations remain available with the preconditions explained
in the [simplex guide](simplex.md). Simplex status codes differ from nonlinear
minimizer codes; inspect the appropriate guide when `success` is false.

## Linear constraints

`fmincon` uses projected gradient for `A @ x <= b` and optional equalities
`Aeq @ x == beq`. It imposes no implicit nonnegativity bounds.

```python
objective = lambda x: jnp.sum((x - jnp.array([2., 1.]))**2)
result = optinpy.nonlinear.constrained.fmincon(
    objective, [0., 0.], A=[[1., 1.]], b=[1.], threshold=1e-12)
assert result['success'], result['status']
print(result['x'])  # Approximately [1., 0.]
```

The solver can search for a feasible start. Status 4 means its feasibility
projection budget was exhausted, rather than proving the constraints infeasible.

## Nonlinear constraints

Supply scalar inequality functions `g_i(x) <= 0`. This example minimizes
`(x-2)**2` subject to `x <= 1`, using the penalty method.

```python
objective = lambda x: (x[0] - 2)**2
optinpy.nonlinear.params['fminnlcon']['method'] = 'penalty'
result = optinpy.nonlinear.constrained.fminnlcon(
    objective, [0.], [lambda x: x[0] - 1], threshold=1e-8)
assert result['success'], result['status']
print(result['x'], result['constraint_violation'])  # Near [1.], small violation
```

Choose `barrier` or `log-barrier` through the same parameter dictionary.
Both require a strictly feasible starting point; `[0.]` satisfies that
requirement here. These are experimental local continuation methods. Check
`success`, the KKT error `err`, and `constraint_violation`; a small constraint
violation alone does not establish convergence.

The outer penalty/barrier method can also use a selected inner optimizer:
pass `inner_options={'method': 'bfgs', 'linesearch': 'backtracking'}` to
`fminnlcon`. These options override `minimize` settings for the weighted inner
objective; the default remains modified Newton with strong Wolfe. A supplied
inner derivative takes `(x, weight)`, and a custom direction takes
`(x, gradient, weight)`. This explicit configuration is independent of the
legacy `params['fminunc']` dictionary. Changing it may affect convergence and
requires checking the outer KKT result.

## Graphs, shortest paths, and spanning trees

Graph interfaces retain Python objects and node labels. An added connection is
directed unless `bidirectional=True`; spanning-tree algorithms treat the input
connections as undirected. Node `0` is reserved for artificial arcs.

```python
network = optinpy.graph()
for edge in [('A', 'B', 2), ('B', 'C', 1), ('A', 'C', 4)]:
    network.add_connection(*edge)

predecessor, distance = optinpy.sp.dijkstra(network, 'A')
print(distance['C'], predecessor['C'])  # 3., 'B'
tree = optinpy.mst.kruskal(network)
print(sum(network.arcs[u][v].cost for u, v in tree))  # 3
```

`optinpy.sp.fmb(network, source)` runs Bellman–Ford and supports negative
edges while detecting reachable negative cycles. Dijkstra requires finite,
nonnegative costs. Other spanning-tree entry points are
`optinpy.mst.prim(network, start)` and `optinpy.mst.boruvka(network)`.
Disconnected spanning-tree requests raise `ValueError`.

## Minimum-cost flow

Positive node balances supply flow; negative balances consume it. Balances
must sum to zero. The Big-M method uses optinpy's own simplex and writes
optimal flows to the network's arcs.

```python
network = optinpy.graph()
network.add_node(['source', 'middle', 'sink'], [3, 0, -3])
for edge in [('source', 'middle', 1), ('middle', 'sink', 2), ('source', 'sink', 5)]:
    network.add_connection(*edge)
optinpy.mcfp.bigm(network, 100)
print(network.arcs['source']['middle'].flow)  # Approximately 3.
print(network.solution['f'])                 # Approximately 9.
```

This interface handles nonnegative, uncapacitated flows. It raises an error
if the LP fails or artificial flow remains, which can indicate an infeasible
network or a Big-M cost that is too small. The entry point is
`optinpy.mcfp.bigm`, not the historical top-level `optinpy.bigm` example.

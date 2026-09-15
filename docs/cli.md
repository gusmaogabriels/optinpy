# Local command-line interface

Install Optinpy on your own computer with Python 3.11 or newer. The JAX version
is available as the `2.0.0a2` alpha prerelease on [PyPI](https://pypi.org/project/optinpy/2.0.0a2/):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install "optinpy==2.0.0a2"
optinpy --help
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. From an
existing checkout, `python -m pip install -e .` also installs the `optinpy`
command. `python -m optinpy` exposes the same interface. Source archives and checksums
are attached to the [release](https://github.com/gusmaogabriels/optinpy/releases/tag/v2.0.0a2).

The CLI executes locally using Optinpy's own Python API. It has no server,
account, hosted solver or network API. Objectives, problem files and results
remain on the machine running it. User-provided Python objectives can perform
their own side effects, just as when called directly from Python.

## Discover commands

```bash
optinpy --version
optinpy methods
optinpy methods --json
optinpy minimize --help
```

`methods --json` returns the version, schema version, built-in objectives,
minimizers, line searches, differentiation algorithms and simplex pricing
rules. Shell scripts and local tools can inspect this catalog before invoking
a method. A website or MCP discovery entry can link to these instructions;
it does not need to install or execute Optinpy itself.

## Minimize an objective

```bash
optinpy minimize rosenbrock --x0 '[-1.2, 1]' --method lbfgs --json
optinpy minimize quadratic --x0 '[0, 0]' --method bfgs --linesearch interp23
```

`quadratic` is the sum of `(x - 2)**2`; its minimum is at all coordinates equal
to 2. `rosenbrock` is the chained Rosenbrock function and needs at least two
coordinates; its minimum is at all coordinates equal to 1.

Select any named minimizer and line search from `methods`. Adam and SGD use
`--learning-rate` instead of line search. `--tol` sets the gradient-norm
stopping tolerance, `--max-iter` the iteration budget, and `--memory-size`
the L-BFGS history size. `--linesearch-options` accepts a JSON object of the
selected search's options, for example `'{"alpha": 0.5}'`.

For your own objective, create `objective.py`:

```python
import jax.numpy as jnp

def loss(x):
    return jnp.sum((x - jnp.array([3., -4.]))**2)
```

```bash
optinpy minimize objective.py:loss --x0 '[0, 0]' --method bfgs --json
```

An installed/importable `module:function` is also supported. The callable takes
one real JAX vector and returns a real scalar; use a closure or module constants
for extra problem data. Local Python files/modules execute as ordinary Python
code, so use objectives you trust. Expressions such as `"x*x"` are not evaluated.
For custom direction/update compositions, batching or repeated warm solves,
use the [Python component API](usage.md#mix-and-match-components).

## Run an individual line search or derivative

```bash
optinpy line-search quadratic --x '[0, 0]' --method strong-wolfe --json
optinpy line-search quadratic --x '[0, 0]' --direction '[1, 1]' --method backtracking --options '{"alpha": 2}'
optinpy differentiate quadratic --x '[0, 1]' --order 1 --json
optinpy differentiate quadratic --x '[0, 1]' --order 2 --algorithm central --json
```

A line search takes one step along a direction, defaulting to the negative
gradient. Its options are passed to that standalone search. Differentiation
returns a gradient (`--order 1`) or Hessian (`--order 2`). `forward`, `backward`
and `central` select finite differences; `autodiff` is the default. `--epsilon`
can set the finite-difference step explicitly.

## Solve a linear program

Save this as `problem.json`:

```json
{
  "A": [[1, 1]],
  "b": [3],
  "c": [1, 2],
  "mode": "max"
}
```

```bash
optinpy simplex problem.json --json
```

The result is `x = [0, 3]`, `f = 6`. The convention is `A @ x <= b`, with
`mode` equal to `min` (default) or `max`. `A`, `b` and `c` are required;
`lb` and `ub` are optional scalars or vectors. Omitted bounds mean
`0 <= x < infinity`. A JSON `null` in `lb` means negative infinity, and in
`ub` means positive infinity. Use `A: []` and `b: []` for bounds-only problems.
Unknown keys and nonstandard JSON values such as `NaN`/`Infinity` are rejected.

`optinpy simplex - --json` reads the same object from standard input.
`--pricing` selects `dantzig`, `bland` or `steepest-edge`; `--max-iter` limits
pivots. `--tol` applies to normalized feasibility and reduced costs.

## Precision, results and timing

Numerical commands default to `--dtype float64`. Use `--dtype float32` when
appropriate and choose a tolerance suited to that precision. These CLI
settings do not change the behavior of importing the Python package.

With `--json`, stdout contains one JSON object with `schema_version: 1`,
`package_version`, `command`, `dtype`, `elapsed_seconds` and `result`. The result
contains the underlying algorithm's fields, including `success` and solver
`status` where available. Nonfinite result values become JSON `null`; their
failure status is retained. Input errors and prints from Python objectives go
to stderr. Parse/input errors use a textual diagnostic, even with `--json`.

| Process exit code | Meaning |
| --- | --- |
| 0 | Successful result, catalog, help or version |
| 1 | Solver/search did not succeed, or a derivative was nonfinite |
| 2 | Invalid arguments, input or objective interface |

`elapsed_seconds` includes solver construction, JIT compilation, execution
and synchronization/result transfer. It excludes interpreter startup, package
imports, objective-file loading, input parsing and JSON formatting. Each CLI
invocation is a new process; this value is **not** a warm-execution benchmark.
Measure whole-process time externally if you need all startup costs. For warm
execution, retain a compiled callable through the [Python API](warm-execution.md).

## Opt-in local usage counts

The source checkout adds local usage accounting after the existing 2.0.0a2 release.
Install the checkout with `python -m pip install -e .` to use it. Choose a database
path to enable counting:

```bash
export OPTINPY_USAGE_DB="$HOME/optinpy-usage.sqlite3"
optinpy methods --json
optinpy usage --json
```

PowerShell: `$env:OPTINPY_USAGE_DB = "$HOME/optinpy-usage.sqlite3"`.
The JSON report contains daily invocation counts by version, command and exit
code, plus command duration excluding interpreter startup and imports. It stores
no objective names, arguments, file paths, model contents or user/device identifiers. Nothing
is transmitted. `usage` does not count itself; concurrent processes update counts
transactionally. Unset the environment variable to stop recording, or remove the
selected database to clear it. Storage failures preserve solver output and exit
codes. These are local invocation counts; public download statistics and MCP
request counts remain separate and do not automatically receive these records.

# Contributing

Keep algorithm implementations in the existing optinpy submodules. JAX supplies
array operations, differentiation, and compilation; solver logic belongs here.
Do not replace algorithms with calls to external optimization packages.

Use the original implementation on `master` as the baseline for module layout,
public calling patterns, and each named algorithm's mathematical update.
Port those updates to JAX and optimize their execution; give alternative
algorithms explicit names instead of silently replacing an original method.
Preserve readable links between code and the README equations. Document
necessary correctness fixes and changes required by JAX tracing or immutable
arrays. Do not restore a known bug for source similarity. Measure speed changes
on matching inputs, including compilation in first-call wall time and reporting
synchronized warm execution separately.

Keep direction and Hessian-update algorithms individually callable in the
existing nonlinear module. The full optimizer must compose those same kernels;
do not duplicate their formulas in its iteration loop. Select built-in methods
and callbacks during setup/tracing. Test mathematical identities for the
standalone primitives and combinations of solver, line search and derivative
method, including the original parameter-based API.
For a composition refactor, `python -m benchmarks.compare_components
--baseline-ref <trusted-commit>` alternates precompiled baseline/current calls,
checks every output against the baseline, and compares normalized compiled
operations. Preserve the raw timing spread even when compiled operations match.
Baseline IDs in stored reports predate the consolidated branch history; see
[historical benchmark baselines](docs/benchmark-history.md). Choose an available
local revision for a new comparison instead of assuming those IDs exist in a
fresh clone.

Keep the algorithm explanations, mathematical derivations, examples, and
original figures in the main README. Update API details alongside them when
implementations change, and label historical plot settings. The historical
copy in `docs/legacy.md` supplements those chapters; it does not replace them.

Create a virtual environment, then install `python -m pip install -e '.[dev]'`.
Run `python -m pytest -q`, `python main.py`, `python -m build`, and
`python -m twine check dist/*` before proposing a release. Avoid relying on the
checkout for installation verification: install the built wheel in a separate
virtual environment and import it from another directory.

Numerical tests should compare with analytic answers, mathematical conditions,
or independent formulations. Exercise nonconvergence and invalid inputs, not
just successful runs. New JAX kernels should work with explicit float32 and
float64 arrays without changing global precision settings on import.

For simplex changes, install the optional comparison dependencies with
`python -m pip install -e '.[dev,benchmark]'` and run
`python -m pytest -q tests/test_simplex.py tests/test_simplex_comparison.py`.
Reproduce measurements with `python -m benchmarks.compare_simplex --float32`.
Use `python -m benchmarks.time_simplex` for fresh-process wall time and
separate JIT compilation/execution measurements. Run timing commands without
concurrent tests or other heavy work. The first call in a shared process is
not necessarily cold; always synchronize JAX results before stopping timers.
Use `python -m benchmarks.repeat_simplex` to measure distributions on identical
inputs with interleaved solvers. Its optional `--baseline-ref` compares a
trusted commit from this repository in the same Python environment.
The comparison libraries belong in `benchmarks/` and tests, never in runtime
solver implementations. Compare objectives and feasibility, not particular
vertices of a nonunique optimum. Record discrepancies and precision settings.

Use `python -m benchmarks.compare_jax_libraries` for optional Lineax linear-system,
Optimistix BFGS/DFP/L-BFGS, and Optax Adam/SGD comparisons. Install `.[benchmark]`
first; Lineax 0.1 requires JAX >=0.6.1, while the core package still supports
JAX >=0.4.38. Run `python -m pytest -q tests/test_jax_library_comparison.py` to
check the adapters. `--quick --trials 1 --repeats 3 --strict` is the bounded CI
benchmark; strict mode fails on numerical discrepancies, never on timing.
The full report retains failures and records raw synchronized samples,
tracing/lowering, compilation, first execution and fresh-process trial wall time.
Compare accepted solves at the same output accuracy and disclose differences
in stopping rules and line searches. See [the library comparison](docs/jax-library-comparison.md).

Run `python -m benchmarks.jax_audit --baseline-ref <trusted-commit>` for
matched native JAX kernel/API measurements across revisions. Omit the baseline
for a shallow checkout. `--group usage --trials 1 --repeats 3 --eager-repeats 1`
is the bounded CI run. Staged kernels report lowering, compilation, and
synchronized execution separately; eager APIs include any repeated compilation.
Review [the module audit](docs/jax-audit.md) before adding caching, donation,
host transfers, or new derivative representations. Performance assertions must
not depend on a particular CI runner's speed.

Use `python -m benchmarks.autodiff_modes` to compare forward/reverse derivatives
and matrix-free products on dynamic inputs against analytic answers. Choose
derivative modes from the needed output (gradient, full Jacobian/Hessian, or
product); avoid building a matrix when only its product is used. Preserve
the original finite-difference stencils when optimizing their probe layout.

To release, update the version in `pyproject.toml` and `optinpy/__init__.py`,
create the matching `v<version>` tag, and run the Publish package workflow on
that tag. First configure a `testpypi` or `pypi` GitHub environment and its Trusted
Publisher on the corresponding registry (owner `gusmaogabriels`, repository
`optinpy`, workflow `publish.yml`, matching environment name). Add environment
reviewers if desired. Publishing defaults to TestPyPI and requires the full
reusable CI workflow to pass. A local branch alone does not activate these workflows.

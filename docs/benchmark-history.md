# Historical benchmark baselines

The JAX development changes were consolidated into one commit. Earlier commit
IDs in the stored benchmark reports identify the implementations that were
measured before that consolidation. A fresh clone of the consolidated branch
does not include those intermediate commits.

The recorded timings, outcomes, environments and source hashes are retained
as measured. Historical baseline IDs have not been replaced with the new
commit: doing so would attribute measurements to a different implementation.

Commands with a historical `--baseline-ref` require that original commit to
be available locally. They cannot reproduce the historical comparison from
the consolidated branch alone. For a new comparison, explicitly choose a
trusted available revision with the APIs required by the benchmark.

Current implementation measurements do not require those earlier commits:

```bash
python -m benchmarks.compare_jax_libraries
python -m benchmarks.compare_simplex --float32
python -m benchmarks.time_simplex
python -m benchmarks.repeat_simplex
python -m benchmarks.jax_audit
python -m benchmarks.autodiff_modes
```

`compare_components` always compares two implementations and therefore requires
an explicit `--baseline-ref`. Its command-line default no longer names an
intermediate development commit. Stored benchmark-script hashes describe the
script used for the recorded run, including its defaults at that time.

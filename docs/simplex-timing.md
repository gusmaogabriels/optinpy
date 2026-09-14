# Simplex startup and JIT timing

Each cold measurement runs in a new CPU process with the persistent JAX compilation cache disabled. Process wall time includes Python startup, imports, fixture generation, the first solve, JSON output, and process exit. The first complete solve includes model construction, device initialization that has not already occurred, compilation and execution. It is not a measurement of compilation alone.

Warm measurements use fresh solver objects with the same LP after one complete solve in a separate process. They include construction, phase-I cleanup and final residual checks. Every timed JAX operation is synchronized. Independent HiGHS objective and feasibility checks run outside these times.

The rows below are different LPs, with different sizes and pivot counts. Their combined range does not measure run-to-run variance. See [repeated-input measurements](simplex-repeatability.md) for distributions on identical inputs and comparisons in balanced solver order.

| Variables | Process wall (s) | Imports (s) | First complete solve (s) | Warm construction (ms) | Warm solve (ms) | Warm total (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 0.807 | 0.318 | 0.393 | 0.346 | 0.125 | 0.471 |
| 8 | 0.757 | 0.183 | 0.476 | 0.430 | 0.391 | 0.820 |
| 16 | 0.759 | 0.191 | 0.466 | 0.407 | 1.136 | 1.526 |
| 24 | 0.801 | 0.187 | 0.510 | 0.522 | 3.509 | 3.952 |

## Compilation of prepared phases

This separate experiment prepares each phase’s tableau, clears JAX’s in-memory caches, then measures `.lower()`, `.compile()`, and synchronized execution separately. It excludes construction, phase transitions and residual checks. These measurements must not be added to or subtracted from the complete-solve measurements above. Shapes and dtypes determine compiled variants; changing bound structure or the number of artificial variables can require another compilation.

| Variables | Phase | Tableau shape | Pivots | Trace/lower (ms) | Compile (ms) | First execution (ms) | Warm execution (ms) |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 2 | [17, 21] | 2 | 32.441 | 63.242 | 0.326 | 0.050 |
| 8 | 1 | [35, 49] | 7 | 30.010 | 59.686 | 0.477 | 0.090 |
| 8 | 2 | [35, 45] | 10 | 40.170 | 62.191 | 0.511 | 0.125 |
| 16 | 1 | [69, 100] | 23 | 29.599 | 66.186 | 0.737 | 0.522 |
| 16 | 2 | [69, 89] | 29 | 40.348 | 58.676 | 0.787 | 0.441 |
| 24 | 1 | [103, 149] | 46 | 29.732 | 62.038 | 1.939 | 1.547 |
| 24 | 2 | [103, 133] | 47 | 39.855 | 61.934 | 1.821 | 1.453 |

## Environment and reproduction

```json
{
  "platform": "macOS-26.2-arm64-arm-64bit",
  "python": "3.12.2",
  "persistent_compilation_cache": false,
  "repeats": 5,
  "indices": [
    0,
    1,
    2,
    3
  ],
  "seed": 20260913,
  "simplex_sha256": "ec97d07d10b3c369f8d6109efd58ba145c0675b26b7fe6a65a67b7200070d35c",
  "versions": {
    "jax": "0.8.2",
    "jaxlib": "0.8.2",
    "numpy": "2.1.0",
    "optinpy": "2.0.0a1"
  },
  "devices": [
    "TFRT_CPU_0"
  ]
}
```

```bash
python -m pip install -e '.[dev,benchmark]'
python -m benchmarks.time_simplex --case-indices 0 1 2 3 --repeats 5
```

Method: [JAX benchmarking guidance](https://docs.jax.dev/en/latest/benchmarking.html) and [explicit lowering and compilation](https://docs.jax.dev/en/latest/aot.html).

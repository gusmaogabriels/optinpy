# Repeated-input simplex timing

> Historical comparison: baseline commits predate the consolidated branch history. See [baseline availability and reproduction](benchmark-history.md).

Each LP is identical across all solvers and all 90 measured samples per solver (3 fresh processes, 30 samples per process). Each solver receives 3 untimed warmups per LP in each process. Measurements run sequentially in balanced rotating order; problem order changes between processes. Persistent compilation caching is disabled. Every native solve synchronizes all JAX results.

Times include a fresh model, complete solve and adapter result conversion/residual calculation. They exclude warmup and initial compilation. Objective and feasibility checks against HiGHS run outside the timer for every sample. All reported samples passed `1e-7` relative-objective and maximum-feasibility checks. No samples are discarded.

**A range across different LPs is not run-to-run variance.** The earlier 0.85–10.17 ms range combined four problems, from 4 variables/2 pivots to 24 variables/247 pivots. This report measures spread separately for each fixed input. P5–P95 describes the middle 90% of observed timings, not a confidence interval.

| Variables | Solver | Samples | Median (ms) | P5–P95 (ms) | Min–max (ms) | CV (%) | Iterations |
| ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 4 | optinpy | 90 | 0.639 | 0.502–1.213 | 0.486–2.233 | 38.9 | [2] |
| 4 | highs-ds | 90 | 0.617 | 0.496–1.118 | 0.485–2.381 | 39.7 | [2] |
| 4 | highs-ipm | 90 | 0.670 | 0.552–1.039 | 0.540–1.774 | 25.9 | [6] |
| 4 | clarabel | 90 | 0.158 | 0.124–0.251 | 0.121–2.473 | 130.9 | [8] |
| 4 | optinpy-before | 90 | 0.773 | 0.552–2.066 | 0.539–2.829 | 49.7 | [2] |
| 8 | optinpy | 90 | 1.000 | 0.760–1.269 | 0.732–1.439 | 17.0 | [17] |
| 8 | highs-ds | 90 | 0.692 | 0.580–0.884 | 0.552–1.529 | 18.2 | [11] |
| 8 | highs-ipm | 90 | 0.815 | 0.717–1.017 | 0.700–1.182 | 11.4 | [8] |
| 8 | clarabel | 90 | 0.216 | 0.176–0.289 | 0.173–0.326 | 16.4 | [8] |
| 8 | optinpy-before | 90 | 1.760 | 1.405–2.217 | 1.373–5.691 | 27.1 | [17] |
| 16 | optinpy | 90 | 1.834 | 1.609–2.046 | 1.510–2.718 | 9.3 | [52] |
| 16 | highs-ds | 90 | 0.888 | 0.723–1.074 | 0.695–1.148 | 12.7 | [20] |
| 16 | highs-ipm | 90 | 1.201 | 1.008–1.421 | 0.972–2.468 | 14.6 | [10] |
| 16 | clarabel | 90 | 0.448 | 0.401–0.516 | 0.382–0.590 | 8.5 | [11] |
| 16 | optinpy-before | 90 | 2.711 | 2.282–3.429 | 2.221–3.836 | 11.8 | [52] |
| 24 | optinpy | 90 | 3.884 | 3.563–4.713 | 3.426–6.327 | 10.6 | [93] |
| 24 | highs-ds | 90 | 1.142 | 0.993–1.330 | 0.941–2.563 | 15.6 | [28] |
| 24 | highs-ipm | 90 | 1.911 | 1.653–2.175 | 1.610–2.303 | 7.9 | [11] |
| 24 | clarabel | 90 | 0.785 | 0.735–0.878 | 0.715–1.015 | 5.8 | [11] |
| 24 | optinpy-before | 90 | 4.862 | 4.496–5.367 | 4.266–6.361 | 6.1 | [93] |

## Relative warm performance on identical inputs

Ratios divide the named solver’s median by the current optinpy median. A ratio below 1 means the named solver is faster. Iteration counts are algorithm-specific and should not be compared as equal units of work across different solver libraries.

| Variables | HiGHS simplex / optinpy | HiGHS IPM / optinpy | Clarabel / optinpy | Earlier optinpy / current |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 0.97× | 1.05× | 0.25× | 1.21× |
| 8 | 0.69× | 0.82× | 0.22× | 1.76× |
| 16 | 0.48× | 0.65× | 0.24× | 1.48× |
| 24 | 0.29× | 0.49× | 0.20× | 1.25× |

## Process-to-process medians

| Variables | Solver | Median in each process (ms) |
| ---: | --- | --- |
| 4 | optinpy | 0.797, 0.542, 0.616 |
| 4 | highs-ds | 0.755, 0.521, 0.534 |
| 4 | highs-ipm | 0.828, 0.588, 0.637 |
| 4 | clarabel | 0.195, 0.138, 0.142 |
| 4 | optinpy-before | 0.959, 0.649, 0.697 |
| 8 | optinpy | 1.192, 0.808, 0.999 |
| 8 | highs-ds | 0.780, 0.646, 0.700 |
| 8 | highs-ipm | 0.865, 0.800, 0.853 |
| 8 | clarabel | 0.248, 0.188, 0.222 |
| 8 | optinpy-before | 1.994, 1.464, 1.771 |
| 16 | optinpy | 1.912, 1.676, 1.866 |
| 16 | highs-ds | 0.959, 0.824, 0.888 |
| 16 | highs-ipm | 1.283, 1.082, 1.227 |
| 16 | clarabel | 0.471, 0.432, 0.446 |
| 16 | optinpy-before | 2.857, 2.460, 2.738 |
| 24 | optinpy | 3.934, 3.766, 3.909 |
| 24 | highs-ds | 1.149, 1.130, 1.160 |
| 24 | highs-ipm | 1.918, 1.844, 1.955 |
| 24 | clarabel | 0.786, 0.776, 0.806 |
| 24 | optinpy-before | 4.998, 4.804, 4.927 |

## Environment and reproduction

```json
{
  "utc": "2026-09-14T17:59:40.407505+00:00",
  "platform": "macOS-26.2-arm64-arm-64bit",
  "python": "3.12.2",
  "versions": {
    "jax": "0.8.2",
    "jaxlib": "0.8.2",
    "numpy": "2.1.0",
    "scipy": "1.16.3",
    "clarabel": "0.11.1",
    "optinpy": "2.0.0a1"
  },
  "devices": [
    "TFRT_CPU_0"
  ],
  "seed": 20260913,
  "trials": 3,
  "repeats": 30,
  "warmup": 3,
  "case_indices": [
    0,
    1,
    2,
    3
  ],
  "persistent_compilation_cache": false,
  "baseline": {
    "ref": "8ddfe01",
    "sha256": "2312f7d7336a6c208541679638295b89ea05247646ea11f3fb4c616c49821bf5"
  },
  "simplex_sha256": "ec97d07d10b3c369f8d6109efd58ba145c0675b26b7fe6a65a67b7200070d35c"
}
```

The optional earlier implementation is loaded from local Git history into the same environment. It needs the referenced commit to be present. A source distribution can run the other solvers by omitting `--baseline-ref`.

```bash
python -m pip install -e '.[dev,benchmark]'
python -m benchmarks.repeat_simplex --trials 3 --repeats 30 --warmup 3 --seed 20260913 --case-indices 0 1 2 3 --baseline-ref 8ddfe01
```

[Raw samples and measurement orders](simplex-repeatability.json).

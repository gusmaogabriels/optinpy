# Simplex comparison

Reproducible local CPU comparison of optinpy’s native tableau implementation, HiGHS dual revised simplex, HiGHS interior point with crossover, and Clarabel. The two HiGHS algorithms share one solver library; Clarabel is a separate implementation.

All LPs receive identical row/objective normalization before timing. Optinpy’s unit-scaling regressions separately test unnormalized coefficients. Timings include adapter/model construction and solving, and synchronize JAX results. The first call is recorded separately; warm times are medians of repeated fresh solves. Later first calls may reuse compiled kernels. These are not fresh-process cold timings. See [startup and JIT timing](simplex-timing.md) for separate process, compilation and execution measurements. See [repeated-input measurements](simplex-repeatability.md) for timing spread on identical LPs. There is no sparse/large-scale performance claim.

A Clarabel dual-infeasibility certificate is disambiguated with a zero-objective feasibility solve, included in its timing. This distinguishes an unbounded primal from simultaneous primal/dual infeasibility. Reduced-accuracy solution statuses are retained and must still pass the same objective/residual checks.

Float32 is measured both with the default stopping tolerance and with an explicit `tol=1e-4`. The default uses a stricter pivot/feasibility test and may return numerical failure as tableau roundoff accumulates. This comparison keeps both outcomes visible; use float64 for reliable scientific LP solves.

Cases: 104; seeded random LPs: 80; seed: 20260913; repeats: 3.

| Solver | Passed | Approximate status | Median warm time (ms) | Max relative objective error | Max feasibility violation |
| --- | ---: | ---: | ---: | ---: | ---: |
| optinpy-float64 | 104/104 | 0 | 1.005 | 5.634e-15 | 2.309e-14 |
| highs-ds | 104/104 | 0 | 0.636 | 2.500e-10 | 5.000e-10 |
| highs-ipm | 104/104 | 0 | 0.680 | 2.500e-10 | 5.000e-10 |
| clarabel | 104/104 | 1 | 0.161 | 7.894e-10 | 1.703e-10 |
| optinpy-float32-default | 63/104 | 0 | 0.915 | 6.145e-07 | 1.183e-06 |
| optinpy-float32-tol1e-4 | 104/104 | 0 | 0.800 | 1.683e-06 | 7.931e-06 |

For an optimum, passing requires an optimal or approximate-solution status, relative objective error `abs(f-f_ref)/(1+abs(f_ref)) <= 1e-7`, and maximum constraint/bound violation `<= 1e-7` in the common normalized problem. Float32 uses `2e-4`. Analytic objectives are used where available; otherwise HiGHS dual simplex supplies the reference objective. Different optimal vertices are allowed. Infeasible and unbounded cases are checked against their constructed classification.

## First calls in the shared benchmark process

These medians include compilation when needed, with cache reuse from earlier cases. They include adapter imports when first encountered. They must not be interpreted as isolated compilation or fresh-process startup time.

| Solver | Median first call (ms) |
| --- | ---: |
| optinpy-float64 | 165.789 |
| highs-ds | 0.989 |
| highs-ipm | 0.758 |
| clarabel | 0.255 |
| optinpy-float32-default | 167.932 |
| optinpy-float32-tol1e-4 | 0.859 |

## Timing by problem size

| Solver | Variables | Median warm time (ms) |
| --- | ---: | ---: |
| optinpy-float64 | 4 | 0.623 |
| optinpy-float64 | 8 | 0.966 |
| optinpy-float64 | 16 | 1.831 |
| optinpy-float64 | 24 | 4.119 |
| highs-ds | 4 | 0.544 |
| highs-ds | 8 | 0.640 |
| highs-ds | 16 | 0.754 |
| highs-ds | 24 | 0.923 |
| highs-ipm | 4 | 0.567 |
| highs-ipm | 8 | 0.680 |
| highs-ipm | 16 | 0.993 |
| highs-ipm | 24 | 1.531 |
| clarabel | 4 | 0.109 |
| clarabel | 8 | 0.163 |
| clarabel | 16 | 0.359 |
| clarabel | 24 | 0.717 |
| optinpy-float32-default | 4 | 0.603 |
| optinpy-float32-default | 8 | 0.915 |
| optinpy-float32-default | 16 | 1.301 |
| optinpy-float32-default | 24 | 3.043 |
| optinpy-float32-tol1e-4 | 4 | 0.585 |
| optinpy-float32-tol1e-4 | 8 | 0.800 |
| optinpy-float32-tol1e-4 | 16 | 1.304 |
| optinpy-float32-tol1e-4 | 24 | 2.990 |

## Disagreements

- `random_002` / `optinpy-float32-default`: status `numerical`, objective error `None`, violation `1.238919284496376e-06`.
- `random_003` / `optinpy-float32-default`: status `numerical`, objective error `None`, violation `5.080837403248495e-06`.
- `random_006` / `optinpy-float32-default`: status `numerical`, objective error `None`, violation `2.800599549345322`.
- `random_007` / `optinpy-float32-default`: status `numerical`, objective error `None`, violation `2.438984266994737e-06`.
- `random_009` / `optinpy-float32-default`: status `numerical`, objective error `None`, violation `1.579101259086002e-06`.
- `random_010` / `optinpy-float32-default`: status `numerical`, objective error `None`, violation `3.024253312999825e-06`.
- `random_011` / `optinpy-float32-default`: status `numerical`, objective error `None`, violation `5.617722850459472e-06`.
- `random_015` / `optinpy-float32-default`: status `numerical`, objective error `None`, violation `2.100635421723921`.
- `random_018` / `optinpy-float32-default`: status `numerical`, objective error `None`, violation `2.821205394409887e-06`.
- `random_019` / `optinpy-float32-default`: status `numerical`, objective error `None`, violation `5.143150740849478e-06`.
- `random_022` / `optinpy-float32-default`: status `numerical`, objective error `None`, violation `1.2712592105668108e-06`.
- `random_023` / `optinpy-float32-default`: status `numerical`, objective error `None`, violation `4.90217587101327e-06`.
- `random_026` / `optinpy-float32-default`: status `numerical`, objective error `None`, violation `2.0670666585065334e-06`.
- `random_027` / `optinpy-float32-default`: status `numerical`, objective error `None`, violation `6.769125418415456e-06`.
- `random_030` / `optinpy-float32-default`: status `numerical`, objective error `None`, violation `2.987628907602513e-06`.
- `random_031` / `optinpy-float32-default`: status `numerical`, objective error `None`, violation `0.07165564280366077`.
- `random_034` / `optinpy-float32-default`: status `numerical`, objective error `None`, violation `1.4857112450861187e-06`.
- `random_035` / `optinpy-float32-default`: status `numerical`, objective error `None`, violation `5.2970061505952515e-06`.
- `random_038` / `optinpy-float32-default`: status `numerical`, objective error `None`, violation `3.1080009295081368e-06`.
- `random_039` / `optinpy-float32-default`: status `numerical`, objective error `None`, violation `7.930533694633368e-06`.
- `random_042` / `optinpy-float32-default`: status `numerical`, objective error `None`, violation `2.448693939102742e-06`.
- `random_043` / `optinpy-float32-default`: status `numerical`, objective error `None`, violation `3.271468445475989e-06`.
- `random_046` / `optinpy-float32-default`: status `numerical`, objective error `None`, violation `1.9669316775328127e-06`.
- `random_047` / `optinpy-float32-default`: status `numerical`, objective error `None`, violation `4.229342879025211e-06`.
- `random_049` / `optinpy-float32-default`: status `numerical`, objective error `None`, violation `2.3254641616787808e-06`.
- `random_050` / `optinpy-float32-default`: status `numerical`, objective error `None`, violation `3.720295594256129e-06`.
- `random_051` / `optinpy-float32-default`: status `numerical`, objective error `None`, violation `4.281634224057029e-06`.
- `random_054` / `optinpy-float32-default`: status `numerical`, objective error `None`, violation `2.291668766585264e-06`.
- `random_055` / `optinpy-float32-default`: status `numerical`, objective error `None`, violation `3.7762234335048106e-06`.
- `random_058` / `optinpy-float32-default`: status `numerical`, objective error `None`, violation `2.258090060047735e-06`.
- `random_059` / `optinpy-float32-default`: status `numerical`, objective error `None`, violation `5.15656699562328e-06`.
- `random_062` / `optinpy-float32-default`: status `numerical`, objective error `None`, violation `2.660403527698918e-06`.
- `random_063` / `optinpy-float32-default`: status `numerical`, objective error `None`, violation `2.6874847116609146e-06`.
- `random_066` / `optinpy-float32-default`: status `numerical`, objective error `None`, violation `1.985573911511551e-06`.
- `random_067` / `optinpy-float32-default`: status `numerical`, objective error `None`, violation `3.2157037419544565e-06`.
- `random_070` / `optinpy-float32-default`: status `numerical`, objective error `None`, violation `1.7222847854547751e-06`.
- `random_071` / `optinpy-float32-default`: status `numerical`, objective error `None`, violation `3.9263552753521225e-06`.
- `random_074` / `optinpy-float32-default`: status `numerical`, objective error `None`, violation `1.7382419778000369e-06`.
- `random_075` / `optinpy-float32-default`: status `numerical`, objective error `None`, violation `3.7567716311137644e-06`.
- `random_078` / `optinpy-float32-default`: status `numerical`, objective error `None`, violation `2.604057590449571e-06`.
- `random_079` / `optinpy-float32-default`: status `numerical`, objective error `None`, violation `2.8251909585108592e-06`.

## Environment

```json
{
  "utc": "2026-09-14T18:00:13.342020+00:00",
  "platform": "macOS-26.2-arm64-arm-64bit",
  "processor": "arm",
  "python": "3.12.2",
  "versions": {
    "optinpy": "2.0.0a1",
    "jax": "0.8.2",
    "jaxlib": "0.8.2",
    "numpy": "2.1.0",
    "scipy": "1.16.3",
    "clarabel": "0.11.1"
  },
  "jax_devices": [
    "TFRT_CPU_0"
  ],
  "simplex_sha256": "ec97d07d10b3c369f8d6109efd58ba145c0675b26b7fe6a65a67b7200070d35c",
  "seed": 20260913,
  "random_cases": 80,
  "warm_repeats": 3,
  "solver_options": {
    "optinpy-float64": {
      "dtype": "float64"
    },
    "highs-ds": {
      "method": "highs-ds"
    },
    "highs-ipm": {
      "method": "highs-ipm"
    },
    "clarabel": {},
    "optinpy-float32-default": {
      "dtype": "float32"
    },
    "optinpy-float32-tol1e-4": {
      "dtype": "float32",
      "tol": 0.0001
    }
  },
  "common_row_objective_normalization": true
}
```

## Reproduce

```bash
python -m pip install -e '.[dev,benchmark]'
python -m benchmarks.compare_simplex --random-cases 80 --repeats 3 --seed 20260913 --float32
```

Methods: [HiGHS dual simplex](https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs-ds.html), [HiGHS interior point](https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs-ipm.html), [Clarabel](https://clarabel.org/stable/python/getting_started_py/).

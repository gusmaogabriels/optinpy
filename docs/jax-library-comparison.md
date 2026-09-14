# JAX library comparison

CPU float64, deterministic dense problems with known solutions. The matrices, targets, right-hand sides and starting points are dynamic inputs. Every solve starts from fresh optimizer state; direct linear solvers factor the matrix on every call. This small suite is not a ranking across all optimization workloads.

- **Lineax:** LU, Cholesky and CG against `jax.numpy.linalg.solve`, the kernel used inside optinpy Newton. These rows measure a linear-system component, not a complete optimizer or linear-programming solver. Cholesky/CG receive the known SPD structure; LU is the general dense comparison. CG is given a dense matrix, not a matrix-free operator.
- **Optimistix:** BFGS, DFP and L-BFGS against the matching optinpy methods, each with a 2,000-step budget. optinpy uses gradient-norm tolerance 1e-6 and strong Wolfe; Optimistix uses its default Armijo search and change-based stopping with rtol=0, atol=1e-10 and two-norm. Both L-BFGS methods store ten pairs. These are whole-solve comparisons at common output accuracy, not identical trajectories. Optimistix counts search steps; optinpy counts outer iterations.
- **Optax:** Adam and SGD inside a benchmark-owned compiled full-objective loop. Both libraries use learning rate 0.01, gradient-norm tolerance 1e-6 and a 10,000-step budget. Adam uses beta1=0.9, beta2=0.999 and epsilon=1e-8 outside the square root (Optax eps_root=0); SGD has no momentum. This is not a minibatch-training benchmark.

Acceptance requires backend success and independent NumPy checks on every first, warmup and timed result. Linear systems require relative residual ≤1e-8 and relative solution error ≤1e-7. Minimization requires gradient two-norm ≤1e-5, relative objective error ≤1e-7 and relative solution error ≤1e-4 against the known optimum. Relative errors divide by 1 plus the reference norm or absolute objective. Nonlinear cases include coupled quadratics, regularized softplus sums and Rosenbrock chains.

3 fresh processes, 3 warmups and 20 interleaved timed calls per solver/case/process. Compilation order reverses between processes; warm order rotates. All calls synchronize; no samples are discarded. Persistent compilation caching is disabled. Imports, input construction and device initialization are outside per-solve timers. First staged call includes lowering, compilation and first execution, but not process startup. Whole-trial process wall time is recorded separately in the raw data. P5–P95 describes observed spread, not a confidence interval.

**FAIL rows retain their timings but cannot support a speed claim for an accurate solve.** The acceptance column checks all stages; warm pass counts count timed samples only.

| Group / case | Solver | Accepted | Warm passes | Lower (ms) | Compile (ms) | First staged call (ms) | Warm median (µs) | P5–P95 (µs) | Steps |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| first-order / quadratic_16 | optax-adam | pass | 60/60 | 12.164 | 34.568 | 48.478 | 690.625 | 635.933–788.898 | 1296 |
| first-order / quadratic_16 | optax-sgd | pass | 60/60 | 8.924 | 28.605 | 37.735 | 428.563 | 389.440–492.622 | 919 |
| first-order / quadratic_16 | optinpy-adam | pass | 60/60 | 14.372 | 41.307 | 57.133 | 673.312 | 603.429–755.664 | 1296 |
| first-order / quadratic_16 | optinpy-sgd | pass | 60/60 | 11.102 | 33.529 | 46.399 | 433.688 | 396.301–508.510 | 919 |
| first-order / quadratic_64 | optax-adam | pass | 60/60 | 11.708 | 38.527 | 52.458 | 1105.417 | 1051.229–1212.421 | 893 |
| first-order / quadratic_64 | optax-sgd | pass | 60/60 | 7.967 | 32.964 | 42.492 | 1389.521 | 1336.386–1526.189 | 1239 |
| first-order / quadratic_64 | optinpy-adam | pass | 60/60 | 14.168 | 42.702 | 56.939 | 1201.625 | 1147.838–1316.035 | 893 |
| first-order / quadratic_64 | optinpy-sgd | pass | 60/60 | 11.923 | 37.968 | 51.764 | 1468.688 | 1415.609–1586.215 | 1239 |
| first-order / softplus_16 | optax-adam | pass | 60/60 | 17.213 | 83.809 | 101.661 | 289.646 | 258.723–359.502 | 264 |
| first-order / softplus_16 | optax-sgd | pass | 60/60 | 13.007 | 71.691 | 87.493 | 2620.729 | 2468.350–2989.843 | 3583 |
| first-order / softplus_16 | optinpy-adam | pass | 60/60 | 16.726 | 76.099 | 93.430 | 290.250 | 261.735–388.410 | 264 |
| first-order / softplus_16 | optinpy-sgd | pass | 60/60 | 15.527 | 82.333 | 102.466 | 2699.354 | 2509.499–3091.707 | 3583 |
| linear / indefinite_64 | jax-newton-kernel | pass | 60/60 | 8.370 | 40.056 | 49.666 | 55.917 | 37.362–90.551 | — |
| linear / indefinite_64 | lineax-lu | pass | 60/60 | 14.536 | 43.719 | 57.810 | 55.645 | 36.360–87.024 | — |
| linear / spd_16 | jax-newton-kernel | pass | 60/60 | 132.612 | 26.774 | 159.720 | 17.604 | 12.957–44.191 | — |
| linear / spd_16 | lineax-cg | pass | 60/60 | 55.754 | 40.552 | 92.799 | 45.729 | 32.910–76.094 | 18 |
| linear / spd_16 | lineax-cholesky | pass | 60/60 | 15.319 | 23.486 | 42.030 | 17.812 | 13.985–49.727 | — |
| linear / spd_16 | lineax-lu | pass | 60/60 | 48.675 | 26.228 | 75.192 | 19.980 | 14.701–55.101 | — |
| linear / spd_256 | jax-newton-kernel | pass | 60/60 | 13.597 | 33.386 | 46.294 | 273.375 | 226.865–374.388 | — |
| linear / spd_256 | lineax-cg | pass | 60/60 | 43.934 | 37.003 | 80.031 | 563.937 | 534.425–629.916 | 64 |
| linear / spd_256 | lineax-cholesky | pass | 60/60 | 11.493 | 30.475 | 41.842 | 164.166 | 134.319–248.120 | — |
| linear / spd_256 | lineax-lu | pass | 60/60 | 18.163 | 37.575 | 56.484 | 275.042 | 224.212–390.725 | — |
| linear / spd_64 | jax-newton-kernel | pass | 60/60 | 17.786 | 39.364 | 57.476 | 49.041 | 39.130–81.191 | — |
| linear / spd_64 | lineax-cg | pass | 60/60 | 31.675 | 40.129 | 72.184 | 79.458 | 68.288–110.415 | 53 |
| linear / spd_64 | lineax-cholesky | pass | 60/60 | 13.184 | 33.772 | 46.176 | 38.562 | 30.216–74.111 | — |
| linear / spd_64 | lineax-lu | pass | 60/60 | 21.410 | 38.960 | 61.231 | 48.896 | 41.025–96.194 | — |
| nonlinear / quadratic_16 | optimistix-bfgs | pass | 60/60 | 44.395 | 43.553 | 88.472 | 87.541 | 65.194–162.799 | 61 |
| nonlinear / quadratic_16 | optimistix-dfp | pass | 60/60 | 35.377 | 42.780 | 78.583 | 82.062 | 61.812–163.421 | 50 |
| nonlinear / quadratic_16 | optimistix-lbfgs | pass | 60/60 | 77.547 | 65.224 | 143.340 | 158.938 | 130.137–249.296 | 66 |
| nonlinear / quadratic_16 | optinpy-bfgs | pass | 60/60 | 30.907 | 65.228 | 95.792 | 88.104 | 64.939–232.819 | 23 |
| nonlinear / quadratic_16 | optinpy-dfp | pass | 60/60 | 22.351 | 70.409 | 93.349 | 88.354 | 66.046–177.777 | 22 |
| nonlinear / quadratic_16 | optinpy-lbfgs | pass | 60/60 | 40.498 | 80.504 | 121.532 | 102.333 | 78.031–170.696 | 33 |
| nonlinear / quadratic_64 | optimistix-bfgs | pass | 60/60 | 38.403 | 50.125 | 90.021 | 484.020 | 442.142–809.452 | 197 |
| nonlinear / quadratic_64 | optimistix-dfp | pass | 60/60 | 37.292 | 44.343 | 85.885 | 394.834 | 359.808–570.303 | 149 |
| nonlinear / quadratic_64 | optimistix-lbfgs | pass | 60/60 | 65.675 | 67.955 | 139.021 | 334.646 | 314.811–519.554 | 87 |
| nonlinear / quadratic_64 | optinpy-bfgs | pass | 60/60 | 60.368 | 84.108 | 140.435 | 551.583 | 507.523–746.497 | 53 |
| nonlinear / quadratic_64 | optinpy-dfp | pass | 60/60 | 21.597 | 70.374 | 92.397 | 661.645 | 613.127–819.121 | 49 |
| nonlinear / quadratic_64 | optinpy-lbfgs | pass | 60/60 | 33.697 | 95.856 | 131.280 | 391.208 | 300.598–574.621 | 47 |
| nonlinear / rosenbrock_2 | optimistix-bfgs | pass | 60/60 | 44.156 | 38.293 | 78.856 | 26.020 | 24.994–48.711 | 56 |
| nonlinear / rosenbrock_2 | optimistix-dfp | pass | 60/60 | 36.210 | 36.993 | 72.925 | 29.770 | 28.617–49.436 | 71 |
| nonlinear / rosenbrock_2 | optimistix-lbfgs | pass | 60/60 | 67.182 | 57.187 | 125.114 | 337.438 | 325.429–423.810 | 703 |
| nonlinear / rosenbrock_2 | optinpy-bfgs | pass | 60/60 | 36.706 | 52.331 | 93.216 | 27.895 | 26.289–56.975 | 34 |
| nonlinear / rosenbrock_2 | optinpy-dfp | pass | 60/60 | 28.845 | 56.653 | 83.522 | 28.291 | 26.867–52.806 | 35 |
| nonlinear / rosenbrock_2 | optinpy-lbfgs | pass | 60/60 | 42.813 | 60.603 | 103.787 | 37.229 | 35.244–60.707 | 34 |
| nonlinear / rosenbrock_8 | optimistix-bfgs | pass | 60/60 | 41.930 | 39.895 | 81.002 | 88.979 | 65.346–127.598 | 153 |
| nonlinear / rosenbrock_8 | optimistix-dfp | FAIL | 0/60 | 38.409 | 39.386 | 78.440 | 637.062 | 600.883–759.968 | 2000 |
| nonlinear / rosenbrock_8 | optimistix-lbfgs | pass | 60/60 | 74.886 | 62.820 | 131.989 | 167.812 | 148.375–231.861 | 92 |
| nonlinear / rosenbrock_8 | optinpy-bfgs | pass | 60/60 | 36.382 | 58.167 | 93.651 | 94.812 | 71.871–144.590 | 71 |
| nonlinear / rosenbrock_8 | optinpy-dfp | FAIL | 0/60 | 26.566 | 54.787 | 82.817 | 1039.583 | 989.565–1143.615 | 2000 |
| nonlinear / rosenbrock_8 | optinpy-lbfgs | pass | 60/60 | 74.267 | 71.465 | 146.237 | 136.041 | 110.483–192.738 | 62 |
| nonlinear / softplus_16 | optimistix-bfgs | pass | 60/60 | 50.274 | 81.237 | 142.904 | 98.083 | 76.778–145.060 | 42 |
| nonlinear / softplus_16 | optimistix-dfp | pass | 60/60 | 51.269 | 80.300 | 127.502 | 98.270 | 76.860–140.639 | 39 |
| nonlinear / softplus_16 | optimistix-lbfgs | pass | 60/60 | 72.244 | 93.954 | 164.634 | 133.250 | 104.998–177.004 | 57 |
| nonlinear / softplus_16 | optinpy-bfgs | pass | 60/60 | 34.643 | 107.535 | 142.710 | 74.084 | 51.956–131.257 | 14 |
| nonlinear / softplus_16 | optinpy-dfp | pass | 60/60 | 26.265 | 104.190 | 130.919 | 77.188 | 55.596–133.187 | 15 |
| nonlinear / softplus_16 | optinpy-lbfgs | pass | 60/60 | 39.967 | 113.006 | 150.054 | 85.063 | 61.742–150.889 | 14 |
| nonlinear / softplus_64 | optimistix-bfgs | pass | 60/60 | 41.492 | 57.625 | 100.069 | 251.292 | 239.452–329.113 | 56 |
| nonlinear / softplus_64 | optimistix-dfp | pass | 60/60 | 78.608 | 50.835 | 129.064 | 255.562 | 239.031–299.198 | 57 |
| nonlinear / softplus_64 | optimistix-lbfgs | pass | 60/60 | 72.955 | 73.379 | 147.042 | 213.480 | 195.188–261.831 | 44 |
| nonlinear / softplus_64 | optinpy-bfgs | pass | 60/60 | 32.589 | 88.871 | 125.611 | 172.771 | 159.159–244.971 | 15 |
| nonlinear / softplus_64 | optinpy-dfp | pass | 60/60 | 26.572 | 86.448 | 113.691 | 260.375 | 236.581–315.936 | 17 |
| nonlinear / softplus_64 | optinpy-lbfgs | pass | 60/60 | 39.256 | 99.084 | 143.613 | 174.896 | 151.926–254.608 | 15 |

Rejected results (including solver status and independent accuracy):

```json
[
  {
    "group": "nonlinear",
    "case": "rosenbrock_8",
    "solver": "optimistix-dfp",
    "trials": [
      {
        "trial": 0,
        "error": null,
        "first_check": {
          "backend_success": false,
          "status": "optimistix._solution.RESULTS<The maximum number of steps was reached in the nonlinear solver. The problem may not be solveable (e.g., a root-find on a function that has no roots), or you may need to increase `max_steps`.>",
          "iterations": 2000,
          "finite": true,
          "passed": false,
          "relative_solution_error": 0.017916981762971962,
          "objective": 0.7828225816938944,
          "relative_objective_error": 0.7828225816938944,
          "gradient_norm": 46.783516950326955
        }
      },
      {
        "trial": 1,
        "error": null,
        "first_check": {
          "backend_success": false,
          "status": "optimistix._solution.RESULTS<The maximum number of steps was reached in the nonlinear solver. The problem may not be solveable (e.g., a root-find on a function that has no roots), or you may need to increase `max_steps`.>",
          "iterations": 2000,
          "finite": true,
          "passed": false,
          "relative_solution_error": 0.017916981762971962,
          "objective": 0.7828225816938944,
          "relative_objective_error": 0.7828225816938944,
          "gradient_norm": 46.783516950326955
        }
      },
      {
        "trial": 2,
        "error": null,
        "first_check": {
          "backend_success": false,
          "status": "optimistix._solution.RESULTS<The maximum number of steps was reached in the nonlinear solver. The problem may not be solveable (e.g., a root-find on a function that has no roots), or you may need to increase `max_steps`.>",
          "iterations": 2000,
          "finite": true,
          "passed": false,
          "relative_solution_error": 0.017916981762971962,
          "objective": 0.7828225816938944,
          "relative_objective_error": 0.7828225816938944,
          "gradient_norm": 46.783516950326955
        }
      }
    ]
  },
  {
    "group": "nonlinear",
    "case": "rosenbrock_8",
    "solver": "optinpy-dfp",
    "trials": [
      {
        "trial": 0,
        "error": null,
        "first_check": {
          "backend_success": false,
          "status": "1",
          "iterations": 2000,
          "finite": true,
          "passed": false,
          "relative_solution_error": 0.21202413082206806,
          "objective": 1.8205350123712085,
          "relative_objective_error": 1.8205350123712085,
          "gradient_norm": 48.66939313451471
        }
      },
      {
        "trial": 1,
        "error": null,
        "first_check": {
          "backend_success": false,
          "status": "1",
          "iterations": 2000,
          "finite": true,
          "passed": false,
          "relative_solution_error": 0.21202413082206806,
          "objective": 1.8205350123712085,
          "relative_objective_error": 1.8205350123712085,
          "gradient_norm": 48.66939313451471
        }
      },
      {
        "trial": 2,
        "error": null,
        "first_check": {
          "backend_success": false,
          "status": "1",
          "iterations": 2000,
          "finite": true,
          "passed": false,
          "relative_solution_error": 0.21202413082206806,
          "objective": 1.8205350123712085,
          "relative_objective_error": 1.8205350123712085,
          "gradient_norm": 48.66939313451471
        }
      }
    ]
  }
]
```

Reproduce with `python -m benchmarks.compare_jax_libraries --groups linear nonlinear first-order --trials 3 --repeats 20 --warmup 3` after `python -m pip install -e ".[benchmark]"`. References are optional and are never imported by optinpy runtime code.

[Raw samples, checks and environment](jax-library-comparison.json). [Lineax](https://docs.kidger.site/lineax/), [Optimistix](https://docs.kidger.site/optimistix/), [Optax](https://optax.readthedocs.io/en/stable/api/optimizers.html).

```json
{
  "utc": "2026-09-14T20:06:55.239807+00:00",
  "platform": "macOS-26.2-arm64-arm-64bit",
  "python": "3.12.2",
  "groups": [
    "linear",
    "nonlinear",
    "first-order"
  ],
  "quick": false,
  "trials": 3,
  "repeats": 20,
  "warmup": 3,
  "dtype": "float64",
  "persistent_compilation_cache": false,
  "versions": {
    "jax": "0.8.2",
    "jaxlib": "0.8.2",
    "numpy": "2.1.0",
    "lineax": "0.1.0",
    "optimistix": "0.1.0",
    "optax": "0.2.3",
    "equinox": "0.13.4"
  },
  "devices": [
    "TFRT_CPU_0"
  ],
  "source_sha256": {
    "optinpy/__init__.py": "761451018bde1e3b45dc7200f93d9aa4ff162fc4c0372dc375c3cd5a0fdead32",
    "optinpy/finitediff/__init__.py": "95de5c0a13d3857d75151a9cd10cf8ba39cf9d2abdc007d54b2203c0bc46f8ce",
    "optinpy/finitediff/finitediff.py": "a556580dd9b8143f7701c9cdaa5e79c05a367474cddb78aae772d51f5074c9ba",
    "optinpy/graph/__init__.py": "57e1bdd5299e92d48507bc0e8fc89cad46b32aa6f166fd2323c9c125e9250a07",
    "optinpy/graph/base.py": "bec60f902313f853be7c9f654b20cbb776f06eac80c79328879054261d6492b4",
    "optinpy/linesearch/__init__.py": "412f1f5e523ad9d33c63ab266ce38974e5bc3b629e6eefdbbf59844b22e0af7c",
    "optinpy/linesearch/linesearch.py": "c856c6693199c9aef41cea4817bd6e17a4d520fd4c168d8c02ad4f7cc0ceb7e0",
    "optinpy/mcfp/__init__.py": "a430e57ba214d600a7e56c5f7c05a2a4cef0f05424c1ddbaa9fa6dfa7daf9bcf",
    "optinpy/mcfp/mcfp.py": "fa89ac115343cb73b5fc61c9728f0da0f2aa55dc6fdf263d0207c6594bd6f610",
    "optinpy/mst/__init__.py": "8d7848de376cf4875ad2db775a435ce1342789d70838e6cf303485e2e6a003ac",
    "optinpy/mst/mst.py": "9ceeb019eeafc9f15884efa750b6738a0ad2004daab1a5ca9d1927c93f0a7c92",
    "optinpy/nonlinear/__init__.py": "8bd07c01ae6abef6f621eef32c6231d71a0ff104235308c49bd203c644331538",
    "optinpy/nonlinear/constrained.py": "f79f67b45aa87384a8d22b5ac2e4292bf6e47fa551fc414fcd0812851640e49d",
    "optinpy/nonlinear/unconstrained.py": "92eeda9fa4473d0b2bcae9a1fc006e9ecc6992efe6936d4c0e2b095750ce176d",
    "optinpy/simplex/__init__.py": "e9afcae690c8c85deb9a1a9d553d0995fd30484480bf6370b45124d960dec34a",
    "optinpy/simplex/base.py": "ec97d07d10b3c369f8d6109efd58ba145c0675b26b7fe6a65a67b7200070d35c",
    "optinpy/sp/__init__.py": "bec7d0b29abfe8d86e635638674490ff9221b7d4a94ad853c048575008a56e58",
    "optinpy/sp/sp.py": "9c48b1e30793e06f2006262e7e717a56750289ff7ed9e936135faaf1d586aff7",
    "benchmarks/compare_jax_libraries.py": "1b0efedfd39c0175f61cfb1850660ae8a22b86161bab673fb6832ed1a7054fbc",
    "benchmarks/jax_library_cases.py": "9853f9ea0b65abf880796b45e96cd6bd6387cda488cf263447d1d624613fc376",
    "benchmarks/jax_library_solvers.py": "f62caca143cc5ff27e4d2c06ded3d18fc3cb2f244a3156e25925a40e44d69741"
  }
}
```

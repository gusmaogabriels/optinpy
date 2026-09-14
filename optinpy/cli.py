"""Command-line access to optinpy's own solvers and standalone algorithms."""
import argparse
from contextlib import redirect_stdout
import importlib
import json
import math
from pathlib import Path
import runpy
import sys
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from . import __version__, compile_minimizer, finitediff, simplex
from .nonlinear.unconstrained import _METHODS, _SEARCHES


def _positive(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def _budget(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return value


def _reject_constant(value):
    raise ValueError(f"{value} is not a JSON number; use null for an infinite LP bound")


def _json(value):
    return json.loads(value, parse_constant=_reject_constant)


def _vector(value, name, dtype):
    data = np.asarray(_json(value))
    if data.ndim != 1 or not data.size or data.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a nonempty JSON array of real numbers")
    result = jnp.asarray(data, dtype=dtype)
    if not bool(jnp.all(jnp.isfinite(result))):
        raise ValueError(f"{name} must be finite in the selected dtype")
    return result


def _objective(spec):
    if spec == "quadratic":
        return lambda x: jnp.sum((x - 2.) ** 2)
    if spec == "rosenbrock":
        def rosenbrock(x):
            if x.size < 2:
                raise ValueError("rosenbrock requires at least two coordinates")
            return jnp.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)
        return rosenbrock
    source, separator, name = spec.rpartition(":")
    if not separator or not source or not name.isidentifier():
        raise ValueError("objective must be quadratic, rosenbrock, module:function or file.py:function")
    if source.endswith(".py"):
        namespace = runpy.run_path(str(Path(source).expanduser()))
        if name not in namespace:
            raise ValueError(f"objective function {name!r} is missing from {source}")
        fun = namespace[name]
    else:
        fun = getattr(importlib.import_module(source), name, None)
    if not callable(fun):
        raise ValueError(f"{spec!r} does not identify a callable")
    return fun


def _options(value):
    result = _json(value)
    if not isinstance(result, dict):
        raise ValueError("line-search options must be a JSON object")
    reserved = {"fun", "x0", "x", "d", "gradient", "value"} & result.keys()
    if reserved:
        raise ValueError(f"line-search options cannot override {sorted(reserved)}")
    return result


def _bounds(value, lower):
    # An omitted bound keeps the Python API default; null entries mean infinity.
    infinity = -math.inf if lower else math.inf
    if isinstance(value, list):
        return [infinity if item is None else item for item in value]
    return infinity if value is None else value


def _lp_problem(path):
    source = sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")
    data = _json(source)
    if not isinstance(data, dict):
        raise ValueError("LP input must be a JSON object")
    missing = {"A", "b", "c"} - data.keys()
    unknown = data.keys() - {"A", "b", "c", "lb", "ub", "mode"}
    if missing:
        raise ValueError(f"LP input is missing {sorted(missing)}")
    if unknown:
        raise ValueError(f"unknown LP fields: {sorted(unknown)}")
    for key in ("A", "b", "c"):
        if np.asarray(data[key]).dtype.kind not in "iuf":
            raise ValueError(f"LP {key} must contain real numbers")
    for key in ("lb", "ub"):
        if key in data:
            data[key] = _bounds(data[key], key == "lb")
            if np.asarray(data[key]).dtype.kind not in "iuf":
                raise ValueError(f"LP {key} must contain real numbers or null")
    return data


def _plain(value):
    """Produce strict JSON, retaining solver status when a result is nonfinite."""
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (np.ndarray, np.generic)):
        return _plain(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _parser():
    parser = argparse.ArgumentParser(
        prog="optinpy", description="Composable optimization algorithms implemented in JAX.")
    parser.add_argument("--version", action="version", version=f"optinpy {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    catalog = commands.add_parser("methods", help="list available algorithms")
    catalog.add_argument("--json", action="store_true", help="emit a machine-readable catalog")

    def numerical(name, help_text):
        child = commands.add_parser(name, help=help_text)
        child.add_argument("--dtype", choices=("float64", "float32"), default="float64",
                           help="arithmetic precision (default: float64)")
        child.add_argument("--json", action="store_true", help="emit strict JSON to stdout")
        return child

    def objective(child):
        child.add_argument("objective", help="quadratic, rosenbrock, module:function or file.py:function")

    minimize = numerical("minimize", "minimize a scalar JAX objective")
    objective(minimize)
    minimize.add_argument("--x0", required=True, help="initial vector as JSON, e.g. '[0, 0]'")
    minimize.add_argument("--method", choices=sorted(_METHODS), default="bfgs")
    minimize.add_argument("--linesearch", choices=sorted(_SEARCHES), default="strong-wolfe")
    minimize.add_argument("--linesearch-options", default="{}", help="line-search options as JSON")
    minimize.add_argument("--tol", type=_positive, default=1e-6)
    minimize.add_argument("--max-iter", type=_budget, default=1000)
    minimize.add_argument("--learning-rate", type=_positive, default=0.01,
                          help="Adam/SGD step size; these methods do not use line search")
    minimize.add_argument("--memory-size", type=int, default=10, help="L-BFGS history size")

    search = numerical("line-search", "run one independently selected line search")
    objective(search)
    search.add_argument("--x", required=True, help="starting vector as JSON")
    search.add_argument("--direction", help="direction as JSON; defaults to negative gradient")
    search.add_argument("--method", choices=sorted(_SEARCHES), default="strong-wolfe")
    search.add_argument("--options", default="{}", help="line-search options as JSON")

    derivative = numerical("differentiate", "evaluate a gradient or Hessian")
    objective(derivative)
    derivative.add_argument("--x", required=True, help="evaluation vector as JSON")
    derivative.add_argument("--order", type=int, choices=(1, 2), default=1)
    derivative.add_argument("--algorithm", choices=("autodiff", "central", "forward", "backward"),
                            default="autodiff")
    derivative.add_argument("--epsilon", type=_positive, help="finite-difference step")

    lp = numerical("simplex", "solve a linear program from JSON")
    lp.add_argument("problem", help="JSON file, or - to read stdin")
    lp.add_argument("--tol", type=_positive, help="normalized feasibility/optimality tolerance")
    lp.add_argument("--max-iter", type=_budget, default=1000)
    lp.add_argument("--pricing", choices=("dantzig", "steepest-edge", "bland"), default="dantzig")
    return parser


def _run(args):
    dtype = jnp.float64 if args.dtype == "float64" else jnp.float32
    if args.command == "simplex":
        data = _lp_problem(args.problem)
        start = perf_counter()
        result = simplex(**data, tol=args.tol, pricing=args.pricing).solve(max_iter=args.max_iter)
    else:
        fun = _objective(args.objective)
        x = _vector(args.x0 if args.command == "minimize" else args.x, "x", dtype)
        if args.command == "minimize":
            options = _options(args.linesearch_options)
            start = perf_counter()
            solve = compile_minimizer(fun, method=args.method, linesearch=args.linesearch,
                                      linesearch_options=options, tol=args.tol,
                                      max_iter=args.max_iter, learning_rate=args.learning_rate,
                                      memory_size=args.memory_size)
            result = solve(x)
        elif args.command == "line-search":
            direction = None if args.direction is None else _vector(args.direction, "direction", dtype)
            if direction is not None and direction.shape != x.shape:
                raise ValueError("direction must have the same shape as x")
            options = _options(args.options)
            start = perf_counter()
            result = jax.jit(lambda point: _SEARCHES[args.method](fun, point, direction, **options))(x)
        else:
            derivative = finitediff.jacobian if args.order == 1 else finitediff.hessian
            def evaluate(point):
                value = jnp.asarray(fun(point))
                if value.ndim != 0:
                    raise ValueError("objective must return a scalar")
                result = derivative(fun, point, algorithm=args.algorithm, epsilon=args.epsilon)
                return {"value": value, "derivative": result,
                        "success": jnp.isfinite(value) & jnp.all(jnp.isfinite(result))}
            start = perf_counter()
            result = jax.jit(evaluate)(x)
    # device_get waits for execution and copies every result to the host.
    result = jax.device_get(result)
    elapsed = perf_counter() - start
    if args.command == "minimize":
        messages = {0: "Gradient tolerance reached", 1: "Iteration budget exhausted",
                    2: "Line search failed or stagnated", 3: "Nonfinite objective or gradient"}
        result["message"] = messages[int(result["status"])]
    return {"schema_version": 1, "package_version": __version__, "command": args.command,
            "dtype": args.dtype, "elapsed_seconds": elapsed, "result": _plain(result)}


def main(argv=None):
    """Return 0 on success, 1 on an unsuccessful result, or 2 for invalid input."""
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "methods":
        catalog = {"schema_version": 1, "package_version": __version__,
                   "minimize": sorted(_METHODS), "line-search": sorted(_SEARCHES),
                   "differentiate": ["autodiff", "central", "forward", "backward"],
                   "simplex": ["dantzig", "steepest-edge", "bland"],
                   "objectives": ["quadratic", "rosenbrock"]}
        if args.json:
            print(json.dumps(catalog, allow_nan=False))
        else:
            for key, value in catalog.items():
                if isinstance(value, list):
                    print(f"{key}: {', '.join(value)}")
        return 0
    previous_x64 = jax.config.jax_enable_x64
    try:
        # Precision belongs to this CLI invocation, not to package import.
        jax.config.update("jax_enable_x64", args.dtype == "float64")
        # Objective imports and user print statements must not corrupt JSON stdout.
        with redirect_stdout(sys.stderr):
            payload = _run(args)
    except (ValueError, TypeError, OSError, ImportError, AttributeError) as error:
        parser.error(str(error))
    finally:
        jax.config.update("jax_enable_x64", previous_x64)
    if args.json:
        print(json.dumps(payload, allow_nan=False))
    else:
        print(f"optinpy {__version__} | {args.command} | {args.dtype}")
        for key, value in payload["result"].items():
            print(f"{key}: {value}")
        print(f"elapsed_seconds (includes JIT and synchronization): {payload['elapsed_seconds']:.6f}")
    return 0 if payload["result"]["success"] else 1

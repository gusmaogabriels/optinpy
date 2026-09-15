"""Exercise local CLI results, input validation and shell exit semantics."""
import io
import json
import subprocess
import sys

import jax
import numpy as np
import pytest

from optinpy.cli import main


def invoke(capsys, *args):
    code = main([*args, "--json"])
    output = capsys.readouterr()
    data = json.loads(output.out, parse_constant=lambda value: pytest.fail(value))
    return code, data


def test_module_entrypoint():
    process = subprocess.run([sys.executable, "-m", "optinpy", "--version"],
                             capture_output=True, text=True, check=True)
    assert process.stdout.strip() == "optinpy 2.0.0a2"
    process = subprocess.run([sys.executable, "-m", "optinpy", "methods", "--json"],
                             capture_output=True, text=True, check=True)
    catalog = json.loads(process.stdout)
    assert {"newton", "bfgs", "lbfgs", "adam"} <= set(catalog["minimize"])
    assert {"backtracking", "strong-wolfe"} <= set(catalog["line-search"])


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_minimize_and_precision_scope(capsys, dtype):
    before = jax.config.jax_enable_x64
    code, data = invoke(capsys, "minimize", "quadratic", "--x0", "[0, -1]", "--dtype", dtype)
    assert code == 0 and data["result"]["success"]
    assert data["dtype"] == dtype and data["schema_version"] == 1
    assert data["elapsed_seconds"] > 0
    np.testing.assert_allclose(data["result"]["x"], [2, 2], atol=1e-6)
    assert jax.config.jax_enable_x64 == before


def test_rosenbrock(capsys):
    code, data = invoke(capsys, "minimize", "rosenbrock", "--x0", "[-1.2, 1]", "--method", "lbfgs")
    assert code == 0
    np.testing.assert_allclose(data["result"]["x"], [1, 1], atol=1e-5)


def test_custom_local_objective(capsys, tmp_path):
    path = tmp_path / "my objective.py"
    path.write_text("import jax.numpy as jnp\nprint('loading objective')\n"
                    "def loss(x):\n    return jnp.sum((x-jnp.array([3., -4.]))**2)\n", encoding="utf-8")
    code, data = invoke(capsys, "minimize", f"{path}:loss", "--x0", "[0, 0]")
    assert code == 0
    np.testing.assert_allclose(data["result"]["x"], [3, -4], atol=1e-7)


def test_importable_objective(capsys):
    # An importable scalar objective uses precisely the same path as local files.
    code, data = invoke(capsys, "differentiate", "jax.numpy:sum", "--x", "[1, 2]")
    assert code == 0
    np.testing.assert_array_equal(data["result"]["derivative"], [1, 1])


@pytest.mark.parametrize("method", ["backtracking", "interp23", "strong-wolfe", "golden-section", "unimodality"])
def test_standalone_line_search(capsys, method):
    code, data = invoke(capsys, "line-search", "quadratic", "--x", "[0,0]", "--method", method)
    assert code == 0 and data["result"]["success"]
    assert data["result"]["f"] < 8
    assert data["result"]["alpha"] > 0


def test_custom_search_direction_and_options(capsys):
    code, data = invoke(capsys, "line-search", "quadratic", "--x", "[0,0]",
                        "--direction", "[1,1]", "--method", "backtracking", "--options", '{"alpha": 2}')
    assert code == 0
    np.testing.assert_allclose(data["result"]["x"], [2, 2])


@pytest.mark.parametrize("algorithm", ["autodiff", "central", "forward", "backward"])
@pytest.mark.parametrize("order", [1, 2])
def test_standalone_derivatives(capsys, algorithm, order):
    code, data = invoke(capsys, "differentiate", "quadratic", "--x", "[0,1]",
                        "--order", str(order), "--algorithm", algorithm)
    assert code == 0 and data["result"]["value"] == 5
    expected = [-4, -2] if order == 1 else [[2, 0], [0, 2]]
    np.testing.assert_allclose(data["result"]["derivative"], expected, atol=1e-4)


def test_lp_file_and_stdin(capsys, tmp_path, monkeypatch):
    source = '{"A": [[1,1]], "b": [3], "c": [1,2], "mode": "max"}'
    path = tmp_path / "problem.json"
    path.write_text(source, encoding="utf-8")
    code, data = invoke(capsys, "simplex", str(path))
    assert code == 0 and data["result"]["f"] == pytest.approx(6)
    np.testing.assert_allclose(data["result"]["x"], [0, 3])
    monkeypatch.setattr(sys, "stdin", io.StringIO(source))
    code, piped = invoke(capsys, "simplex", "-")
    assert code == 0 and piped["result"] == data["result"]


def test_lp_infinite_bounds(capsys, tmp_path):
    path = tmp_path / "bounds.json"
    path.write_text('{"A": [], "b": [], "c": [-1, 1], "lb": [null, -2], "ub": [3, null]}')
    code, data = invoke(capsys, "simplex", str(path))
    assert code == 0
    np.testing.assert_allclose(data["result"]["x"], [3, -2])


def test_unsolved_exit_codes(capsys, tmp_path):
    code, data = invoke(capsys, "minimize", "quadratic", "--x0", "[0,0]", "--max-iter", "0")
    assert code == 1 and data["result"]["status"] == 1
    path = tmp_path / "unbounded.json"
    path.write_text('{"A": [], "b": [], "c": [-1]}')
    code, data = invoke(capsys, "simplex", str(path))
    assert code == 1 and data["result"]["status"] == 2


def test_nonfinite_result_is_strict_json(capsys, tmp_path):
    path = tmp_path / "nonfinite.py"
    path.write_text("import jax.numpy as jnp\ndef loss(x):\n    return jnp.log(x[0])\n")
    code, data = invoke(capsys, "minimize", f"{path}:loss", "--x0", "[-1]")
    assert code == 1 and data["result"]["status"] == 3
    assert data["result"]["f"] is None


@pytest.mark.parametrize("args, message", [
    (["minimize", "quadratic", "--x0", "[]"], "nonempty"),
    (["minimize", "quadratic", "--x0", "[NaN]"], "JSON number"),
    (["minimize", "quadratic", "--x0", "[1e400]"], "finite"),
    (["minimize", "quadratic", "--x0", '["1"]'], "real numbers"),
    (["minimize", "quadratic", "--x0", "[true]"], "real numbers"),
    (["minimize", "quadratic", "--x0", "[0]", "--tol", "nan"], "finite and positive"),
    (["minimize", "quadratic", "--x0", "[0]", "--memory-size", "0"], "positive integer"),
    (["minimize", "x*x", "--x0", "[0]"], "objective must be"),
    (["minimize", "rosenbrock", "--x0", "[0]"], "at least two"),
    (["minimize", "jax.numpy:pi", "--x0", "[0]"], "callable"),
    (["line-search", "quadratic", "--x", "[0,0]", "--direction", "[1]"], "same shape"),
    (["line-search", "quadratic", "--x", "[0]", "--options", "[]"], "JSON object"),
    (["line-search", "quadratic", "--x", "[0]", "--options", '{"d": [1]}'], "cannot override"),
])
def test_invalid_cli_inputs(capsys, args, message):
    before = jax.config.jax_enable_x64
    with pytest.raises(SystemExit) as exc:
        main([*args, "--json"])
    assert exc.value.code == 2
    output = capsys.readouterr()
    assert not output.out and message in output.err
    assert jax.config.jax_enable_x64 == before


@pytest.mark.parametrize("source", ['[]', '{"c": [1]}',
                                   '{"A": [], "b": [], "c": [1], "oops": 1}',
                                   '{"A": [], "b": [], "c": [null]}',
                                   '{"A": [[1,2]], "b": [1], "c": [1]}'])
def test_invalid_lp_inputs(capsys, tmp_path, source):
    path = tmp_path / "bad.json"
    path.write_text(source)
    with pytest.raises(SystemExit) as exc:
        main(["simplex", str(path), "--json"])
    assert exc.value.code == 2
    output = capsys.readouterr()
    assert not output.out and "error:" in output.err

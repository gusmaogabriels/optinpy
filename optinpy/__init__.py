"""Optinpy: native optimization algorithms with JAX numerical kernels."""
__author__ = {"Gabriel S. Gusmao": "gusmaogabriels@gmail.com"}
__version__ = "2.0.0a1"

# NumPy is retained for the legacy graph/simplex interfaces. Numerical
# optimization, differentiation, and line searches import JAX directly.
import numpy as np
from .graph import graph, node
from .simplex import simplex
from .mcfp import mcfp
from .sp import sp
from .mst import mst
from . import finitediff, linesearch, nonlinear
from .nonlinear import minimize, compile_minimizer, fminunc, fmincon, fminnlcon, params, unconstrained, constrained

__all__ = ["graph", "node", "simplex", "mcfp", "sp", "mst", "finitediff",
           "linesearch", "nonlinear", "minimize", "compile_minimizer", "fminunc", "fmincon", "fminnlcon",
           "params", "unconstrained", "constrained", "__version__"]

"""Native JAX nonlinear optimizers in the original optinpy module layout."""
from .unconstrained import minimize, compile_minimizer, unconstrained as _Unconstrained
from .unconstrained import (
    gradient_direction, sgd_direction, newton_direction, modified_newton_direction,
    conjugate_gradient_direction, fletcher_reeves_direction,
    hessian_conjugate_gradient_direction, quasi_newton_direction,
    bfgs_update, dfp_update, lbfgs_direction, lbfgs_update, adam_direction,
)
from .constrained import constrained as _Constrained

params = {
    "fminunc": {"method": "bfgs", "params": {
        "gradient": {"max_iter": 1000}, "newton": {"max_iter": 1000},
        "modified-newton": {"sigma": 1e-4, "max_iter": 1000},
        "conjugate-gradient": {"max_iter": 1000}, "fletcher-reeves": {"max_iter": 1000},
        "quasi-newton": {"max_iter": 1000, "hessian_update": "bfgs"},
        "bfgs": {"max_iter": 1000}, "dfp": {"max_iter": 1000},
        "lbfgs": {"max_iter": 1000, "memory_size": 10},
        "adam": {"max_iter": 1000, "learning_rate": 0.01},
        "sgd": {"max_iter": 1000, "learning_rate": 0.01},
    }},
    "fmincon": {"method": "projected-gradient", "params": {
        "projected-gradient": {"max_iter": 1000}}},
    "fminnlcon": {"method": "penalty", "params": {
        "penalty": {"max_iter": 20}, "barrier": {"max_iter": 20},
        "log-barrier": {"max_iter": 20}}},
    "jacobian": {"algorithm": "autodiff", "epsilon": None},
    "hessian": {"algorithm": "autodiff", "epsilon": None, "initial": None},
    "linesearch": {"method": "backtracking", "params": {
        "strong-wolfe": {"alpha": 1., "c1": 1e-4, "c2": 0.9, "max_iter": 60},
        "backtracking": {"alpha": 1., "rho": 0.6, "c": 1e-4, "max_iter": 50},
        "interp23": {"alpha": 1., "rho": 0.5, "c": 1e-4, "alpha_min": 0.1, "max_iter": 50},
        "golden-section": {"b": 1., "threshold": 1e-5, "max_iter": 100},
        "unimodality": {"b": 1., "threshold": 1e-5, "max_iter": 200},
    }},
}

unconstrained = _Unconstrained(params)
constrained = _Constrained(params, unconstrained)
fminunc = unconstrained.fminunc
fmincon = constrained.fmincon
fminnlcon = constrained.fminnlcon

__all__ = ["minimize", "compile_minimizer", "fminunc", "fmincon", "fminnlcon", "params", "unconstrained", "constrained",
           "gradient_direction", "sgd_direction", "newton_direction", "modified_newton_direction",
           "conjugate_gradient_direction", "fletcher_reeves_direction", "hessian_conjugate_gradient_direction",
           "quasi_newton_direction", "bfgs_update", "dfp_update", "lbfgs_direction", "lbfgs_update", "adam_direction"]

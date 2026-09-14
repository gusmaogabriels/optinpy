"""JAX automatic differentiation and optional finite differences."""
from .finitediff import jacobian, hessian

__all__ = ["jacobian", "hessian"]

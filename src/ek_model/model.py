"""Economic objects and numerical conventions shared by both solution routes.

Rows n are importers; columns i are exporters. There is no industry axis.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
SOLVER_TOL = 1e-13
RESIDUAL_TOL = 1e-11
MAX_NFEV = 2000


def positive_array(value, name: str) -> Array:
    """Copy to float64 and reject nonfinite or nonpositive economic data."""
    array = np.array(value, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(array)) or np.any(array <= 0):
        raise ValueError(f"{name} must be finite and strictly positive")
    return array


@dataclass(frozen=True)
class Primitives:
    """Technology, labor and iceberg costs; gamma is a fixed price multiplier."""

    T: Array
    L: Array
    d: Array
    theta: float
    gamma: float = 1.0

    def __post_init__(self):
        T = positive_array(self.T, "T")
        L = positive_array(self.L, "L")
        d = positive_array(self.d, "d")
        if T.ndim != 1 or T.size < 2 or L.shape != T.shape:
            raise ValueError("T and L must have shape (N,), with N >= 2")
        if d.shape != (T.size, T.size):
            raise ValueError("d must have shape (N, N): importer rows, exporter columns")
        if np.any(d < 1) or not np.all(np.diag(d) == 1):
            raise ValueError("iceberg costs must be >= 1 with unit domestic costs")
        for name in ("theta", "gamma"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.ndim != 0 or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive scalar")
            object.__setattr__(self, name, float(value))
        for name, value in (("T", T), ("L", L), ("d", d)):
            value.setflags(write=False)
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class Diagnostics:
    converged: bool
    solver_success: bool
    status: int
    message: str
    nfev: int
    residual_norm: float


@dataclass(frozen=True)
class Equilibrium:
    primitives: Primitives
    wages: Array
    prices: Array
    shares: Array
    incomes: Array
    real_wages: Array
    residual: Array
    diagnostics: Diagnostics


@dataclass(frozen=True)
class HatEquilibrium:
    wage_hats: Array
    price_hats: Array
    shares: Array
    incomes: Array
    real_wage_hats: Array
    residual: Array
    diagnostics: Diagnostics


def normalized_wages(log_free_wages: Array) -> Array:
    """Country 0 is the numeraire exactly, not a post-solve rescaling."""
    return np.exp(np.concatenate(([0.0], log_free_wages)))


def market_clearing(shares: Array, incomes: Array) -> Array:
    """Sum importer spending down rows to obtain each exporter's sales."""
    sales = shares.T @ incomes
    return (sales - incomes) / incomes


def make_diagnostics(result, residual: Array, *outputs: Array) -> Diagnostics:
    """Optimizer success alone never certifies economic convergence."""
    finite = all(np.all(np.isfinite(x)) for x in (residual, *outputs))
    norm = float(np.max(np.abs(residual))) if finite else float("inf")
    return Diagnostics(
        converged=bool(result.success and finite and norm < RESIDUAL_TOL),
        solver_success=bool(result.success),
        status=int(result.status),
        message=str(result.message),
        nfev=int(result.nfev),
        residual_norm=norm,
    )

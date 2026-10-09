"""Multi-industry economics. Axes are importer n, exporter i, industry j."""
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
ITERATION_TOL = 1e-13
RESIDUAL_TOL = 1e-11
DEFAULT_DAMPING = 0.2
MAX_ITER = 10_000

def positive_array(value, name: str) -> Array:
    """Copy economic inputs to float64; reject nonfinite or nonpositive data."""
    array = np.array(value, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(array)) or np.any(array <= 0):
        raise ValueError(f"{name} must be finite and strictly positive")
    return array

@dataclass(frozen=True)
class Primitives:
    """One mobile domestic labor factor; fixed final expenditure shares."""
    T: Array
    L: Array
    d: Array
    theta: Array
    alpha: Array
    gamma: Array | float = 1.0

    def __post_init__(self):
        values = {name: positive_array(getattr(self, name), name)
                  for name in ("T", "L", "d", "theta", "alpha", "gamma")}
        T = values["T"]
        if T.ndim != 2 or T.shape[0] < 2 or T.shape[1] < 1:
            raise ValueError("T must have shape (N,J), N >= 2 and J >= 1")
        n, j = T.shape
        shapes = {"L": (n,), "d": (n, n, j), "theta": (j,), "alpha": (n, j)}
        for name, shape in shapes.items():
            if values[name].shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
        d = values["d"]
        if np.any(d < 1) or not np.all(d[np.arange(n), np.arange(n), :] == 1):
            raise ValueError("iceberg costs must be >= 1 with unit domestic costs")
        if not np.allclose(values["alpha"].sum(axis=1), 1, rtol=0, atol=1e-12):
            raise ValueError("alpha rows must sum to one; inputs are not renormalized")
        if values["gamma"].ndim == 0:
            values["gamma"] = np.full(j, float(values["gamma"]))
        elif values["gamma"].shape != (j,):
            raise ValueError("gamma must be a positive scalar or have shape (J,)")
        for name, value in values.items():
            value.setflags(write=False)
            object.__setattr__(self, name, value)

@dataclass(frozen=True)
class IterationRecord:
    iteration: int
    wages: tuple[float, ...]
    residual: tuple[float, ...]
    residual_norm: float

@dataclass(frozen=True)
class Diagnostics:
    converged: bool
    status: str
    message: str
    niter: int
    nfev: int
    residual_norm: float
    damping: float
    tolerance: float
    max_iter: int
    history: tuple[IterationRecord, ...]

@dataclass(frozen=True)
class EconomicState:
    """Evaluated state shared by the wage loop; prices can be levels or hats."""
    wages: Array
    prices: Array
    shares: Array
    incomes: Array
    cost_of_living: Array
    expenditures: Array

    @property
    def real_wages(self):
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            return self.wages / self.cost_of_living

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
    cost_of_living: Array
    expenditures: Array

@dataclass(frozen=True)
class HatEquilibrium:
    wage_hats: Array
    price_hats: Array
    shares: Array
    incomes: Array
    real_wage_hats: Array
    residual: Array
    diagnostics: Diagnostics
    cost_of_living_hats: Array
    expenditures: Array

def normalized_wages(log_wages: Array) -> Array:
    """Fix country 0 exactly after every simultaneous update."""
    return np.exp(log_wages - log_wages[0])

def exporter_sales(shares: Array, expenditures: Array) -> Array:
    """Sum importer spending across importers and industries, leaving exporter i."""
    return np.einsum("nij,nj->i", shares, expenditures)

def market_clearing(shares: Array, incomes: Array, expenditures: Array) -> Array:
    """Full country residuals, including country 0."""
    sales = exporter_sales(shares, expenditures)
    return (sales - incomes) / incomes

def row_logsumexp(log_weights: Array) -> Array:
    """Stable exporter-axis log sum: (N,N,J) -> (N,J)."""
    row_max = np.max(log_weights, axis=1, keepdims=True)
    shifted_sum = np.sum(np.exp(log_weights - row_max), axis=1)
    return np.squeeze(row_max, axis=1) + np.log(shifted_sum)

def aggregate_price(prices: Array, alpha: Array) -> Array:
    """Normalized Cobb–Douglas unit expenditure: no expenditure-share constants."""
    if prices.shape[1] == 1 and np.all(alpha == 1):
        return prices[:, 0].copy()
    return np.exp(np.sum(alpha * np.log(prices), axis=1))

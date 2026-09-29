"""Solve the levels economy directly from its primitives."""

import numpy as np
from scipy.optimize import least_squares
from scipy.special import logsumexp

from .model import (
    MAX_NFEV, SOLVER_TOL, Equilibrium, Primitives,
    make_diagnostics, market_clearing, normalized_wages,
)


def levels_objects(primitives: Primitives, log_free_wages):
    """Delivered unit costs determine importer-row shares and price indices."""
    wages = normalized_wages(log_free_wages)
    log_costs = np.log(wages)[None, :] + np.log(primitives.d)
    log_weights = np.log(primitives.T)[None, :] - primitives.theta * log_costs
    log_denominator = logsumexp(log_weights, axis=1)
    shares = np.exp(log_weights - log_denominator[:, None])
    prices = primitives.gamma * np.exp(-log_denominator / primitives.theta)
    incomes = wages * primitives.L
    return wages, prices, shares, incomes


def levels_residual(log_free_wages, primitives: Primitives):
    """Drop only the numeraire country's equation from the optimizer system."""
    _, _, shares, incomes = levels_objects(primitives, log_free_wages)
    return market_clearing(shares, incomes)[1:]


def solve_levels(primitives: Primitives) -> Equilibrium:
    """Start from unit wages and retain full, normalized residual diagnostics."""
    result = least_squares(
        levels_residual, np.zeros(primitives.T.size - 1), args=(primitives,),
        ftol=SOLVER_TOL, xtol=SOLVER_TOL, gtol=SOLVER_TOL, max_nfev=MAX_NFEV,
    )
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        wages, prices, shares, incomes = levels_objects(primitives, result.x)
        real_wages = wages / prices
        residual = market_clearing(shares, incomes)
    diagnostics = make_diagnostics(
        result, residual, wages, prices, shares, incomes, real_wages,
    )
    return Equilibrium(
        primitives, wages, prices, shares, incomes, real_wages, residual, diagnostics,
    )

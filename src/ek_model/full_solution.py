"""Solve the levels economy directly from its primitives."""

import numpy as np
from .iteration import iterate_wages
from .model import (
    DEFAULT_DAMPING, ITERATION_TOL, MAX_ITER, Equilibrium, Primitives,
    normalized_wages, row_logsumexp,
)


def levels_objects(primitives: Primitives, log_wages):
    """Delivered unit costs determine importer-row shares and price indices."""
    wages = normalized_wages(log_wages)
    log_costs = (log_wages - log_wages[0])[None, :] + np.log(primitives.d)
    log_weights = np.log(primitives.T)[None, :] - primitives.theta * log_costs
    log_denominator = row_logsumexp(log_weights)
    shares = np.exp(log_weights - log_denominator[:, None])
    prices = primitives.gamma * np.exp(-log_denominator / primitives.theta)
    incomes = wages * primitives.L
    return wages, prices, shares, incomes


def solve_levels(primitives: Primitives, *, damping=DEFAULT_DAMPING,
                 max_iter=MAX_ITER, tol=ITERATION_TOL) -> Equilibrium:
    """Start from unit wages and adjust using exporter sales divided by income."""
    objects, residual, diagnostics = iterate_wages(
        lambda log_wages: levels_objects(primitives, log_wages), primitives.T.size,
        damping=damping, max_iter=max_iter, tol=tol,
    )
    wages, prices, shares, incomes = objects
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        real_wages = wages / prices
    return Equilibrium(
        primitives, wages, prices, shares, incomes, real_wages, residual, diagnostics,
    )

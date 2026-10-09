"""Levels economy directly from multi-industry primitives."""
import numpy as np
from .iteration import iterate_wages
from .model import (DEFAULT_DAMPING, ITERATION_TOL, MAX_ITER, EconomicState,
                    Equilibrium, Primitives, aggregate_price, normalized_wages, row_logsumexp)

def levels_objects(primitives: Primitives, log_wages):
    """Exporter competitiveness normalizes separately within each importer/industry."""
    wages = normalized_wages(log_wages)
    log_costs = (log_wages - log_wages[0])[None, :, None] + np.log(primitives.d)
    log_weights = np.log(primitives.T)[None, :, :] - primitives.theta[None, None, :] * log_costs
    log_denominator = row_logsumexp(log_weights)
    shares = np.exp(log_weights - log_denominator[:, None, :])
    prices = primitives.gamma * np.exp(-log_denominator / primitives.theta)
    incomes = wages * primitives.L
    expenditures = primitives.alpha * incomes[:, None]
    cost_of_living = aggregate_price(prices, primitives.alpha)
    return EconomicState(wages, prices, shares, incomes, cost_of_living, expenditures)

def solve_levels(primitives: Primitives, *, damping=DEFAULT_DAMPING,
                 max_iter=MAX_ITER, tol=ITERATION_TOL) -> Equilibrium:
    """Start at unit wages; return economic objects at the final evaluated state."""
    objects, residual, diagnostics = iterate_wages(
        lambda log_wages: levels_objects(primitives, log_wages), primitives.T.shape[0],
        industries=primitives.T.shape[1], damping=damping, max_iter=max_iter, tol=tol,
    )
    return Equilibrium(primitives, objects.wages, objects.prices, objects.shares,
                       objects.incomes, objects.real_wages, residual, diagnostics,
                       objects.cost_of_living, objects.expenditures)

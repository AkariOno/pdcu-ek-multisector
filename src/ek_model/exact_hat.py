"""Independent multi-industry hats using baseline shares and country incomes."""
import numpy as np
from .iteration import iterate_wages
from .model import (DEFAULT_DAMPING, ITERATION_TOL, MAX_ITER, RESIDUAL_TOL,
                    EconomicState, Equilibrium, HatEquilibrium, Primitives,
                    aggregate_price, market_clearing, normalized_wages, positive_array, row_logsumexp)

def validate_hat_inputs(baseline: Equilibrium, d_hat):
    """A converged baseline and admissible trade-cost-only shock are required."""
    if not baseline.diagnostics.converged or baseline.diagnostics.status != "converged":
        raise ValueError("exact hats require a converged baseline")
    p = baseline.primitives
    n, j = p.T.shape
    shares = positive_array(baseline.shares, "baseline shares")
    incomes = positive_array(baseline.incomes, "baseline incomes")
    if shares.shape != (n, n, j) or incomes.shape != (n,):
        raise ValueError("baseline shares/incomes have invalid dimensions")
    if not np.allclose(shares.sum(axis=1), 1, rtol=0, atol=1e-12):
        raise ValueError("baseline exporter share sums must equal one")
    expenditures = p.alpha * incomes[:, None]
    if np.max(np.abs(market_clearing(shares, incomes, expenditures))) >= RESIDUAL_TOL:
        raise ValueError("baseline shares and incomes must clear markets")
    shock = positive_array(d_hat, "d_hat")
    if shock.shape != (n, n, j):
        raise ValueError("d_hat must have shape (N,N,J)")
    Primitives(p.T, p.L, p.d * shock, p.theta, p.alpha, p.gamma)
    return shock

def hat_objects(baseline: Equilibrium, d_hat, log_hats):
    """Reweight baseline shares; technology competitiveness is never recomputed."""
    p = baseline.primitives
    wage_hats = normalized_wages(log_hats)
    log_a = np.log(baseline.shares) - p.theta[None, None, :] * (
        (log_hats - log_hats[0])[None, :, None] + np.log(d_hat)
    )
    log_row_sum = row_logsumexp(log_a)
    shares = np.exp(log_a - log_row_sum[:, None, :])
    price_hats = np.exp(-log_row_sum / p.theta)
    incomes = baseline.incomes * wage_hats
    expenditures = p.alpha * incomes[:, None]
    cost_of_living_hats = aggregate_price(price_hats, p.alpha)
    return EconomicState(wage_hats, price_hats, shares, incomes, cost_of_living_hats, expenditures)

def solve_exact_hat(baseline: Equilibrium, d_hat, *, damping=DEFAULT_DAMPING,
                    max_iter=MAX_ITER, tol=ITERATION_TOL) -> HatEquilibrium:
    """Unit initial hats, country 0 fixed; no call to the levels route."""
    shock = validate_hat_inputs(baseline, d_hat)
    objects, residual, diagnostics = iterate_wages(
        lambda log_hats: hat_objects(baseline, shock, log_hats), baseline.incomes.size,
        industries=baseline.primitives.T.shape[1], damping=damping, max_iter=max_iter, tol=tol,
    )
    real_wage_hats = objects.real_wages
    return HatEquilibrium(objects.wages, objects.prices, objects.shares, objects.incomes,
                          real_wage_hats, residual, diagnostics,
                          objects.cost_of_living, objects.expenditures)

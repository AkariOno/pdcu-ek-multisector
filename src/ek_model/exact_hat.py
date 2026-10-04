"""Independent exact-hat route: baseline expenditure shares and incomes suffice.

Primitives are consulted only to validate the counterfactual's admissibility
and read theta. No levels solver or levels price/share calculation is called.
"""

import numpy as np
from .iteration import iterate_wages
from .model import (
    DEFAULT_DAMPING, ITERATION_TOL, MAX_ITER, RESIDUAL_TOL,
    Equilibrium, HatEquilibrium, Primitives,
    market_clearing, normalized_wages, positive_array, row_logsumexp,
)


def validate_hat_inputs(baseline: Equilibrium, d_hat):
    """Require a converged baseline and an admissible trade-cost-only shock."""
    if not baseline.diagnostics.converged or baseline.diagnostics.status != "converged":
        raise ValueError("exact hats require a converged baseline")
    n = baseline.primitives.T.size
    shares = positive_array(baseline.shares, "baseline shares")
    incomes = positive_array(baseline.incomes, "baseline incomes")
    if shares.shape != (n, n) or incomes.shape != (n,):
        raise ValueError("baseline shares/incomes have invalid dimensions")
    if not np.allclose(shares.sum(axis=1), 1, rtol=0, atol=1e-12):
        raise ValueError("baseline share rows must sum to one")
    if np.max(np.abs(market_clearing(shares, incomes))) >= RESIDUAL_TOL:
        raise ValueError("baseline shares and incomes must clear markets")
    shock = positive_array(d_hat, "d_hat")
    if shock.shape != (n, n):
        raise ValueError("d_hat must have shape (N, N)")
    p = baseline.primitives
    Primitives(p.T, p.L, p.d * shock, p.theta, p.gamma)
    return shock


def hat_objects(baseline: Equilibrium, d_hat, log_hats):
    """The log row sum is log(sum_i pi0[n,i]*(w_hat[i]*d_hat[n,i])^-theta)."""
    wage_hats = normalized_wages(log_hats)
    log_a = np.log(baseline.shares) - baseline.primitives.theta * (
        (log_hats - log_hats[0])[None, :] + np.log(d_hat)
    )
    log_row_sum = row_logsumexp(log_a)
    shares = np.exp(log_a - log_row_sum[:, None])
    price_hats = np.exp(-log_row_sum / baseline.primitives.theta)
    incomes = baseline.incomes * wage_hats
    return wage_hats, price_hats, shares, incomes


def solve_exact_hat(baseline: Equilibrium, d_hat, *, damping=DEFAULT_DAMPING,
                    max_iter=MAX_ITER, tol=ITERATION_TOL) -> HatEquilibrium:
    """Adjust wage hats independently, starting from unit hats and fixing hat[0]."""
    shock = validate_hat_inputs(baseline, d_hat)
    objects, residual, diagnostics = iterate_wages(
        lambda log_hats: hat_objects(baseline, shock, log_hats), baseline.incomes.size,
        damping=damping, max_iter=max_iter, tol=tol,
    )
    wage_hats, price_hats, shares, incomes = objects
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        real_wage_hats = wage_hats / price_hats
    return HatEquilibrium(
        wage_hats, price_hats, shares, incomes, real_wage_hats, residual, diagnostics,
    )

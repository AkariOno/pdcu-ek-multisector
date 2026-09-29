"""Independent exact-hat route: baseline expenditure shares and incomes suffice.

Primitives are consulted only to validate the counterfactual's admissibility
and read theta. No levels solver or levels price/share calculation is called.
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.special import logsumexp

from .model import (
    MAX_NFEV, RESIDUAL_TOL, SOLVER_TOL, Equilibrium, HatEquilibrium, Primitives,
    make_diagnostics, market_clearing, normalized_wages, positive_array,
)


def validate_hat_inputs(baseline: Equilibrium, d_hat):
    """Require a converged baseline and an admissible trade-cost-only shock."""
    if not baseline.diagnostics.converged:
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


def hat_objects(baseline: Equilibrium, d_hat, log_free_hats):
    """The log row sum is log(sum_i pi0[n,i]*(w_hat[i]*d_hat[n,i])^-theta)."""
    wage_hats = normalized_wages(log_free_hats)
    log_a = np.log(baseline.shares) - baseline.primitives.theta * (
        np.log(wage_hats)[None, :] + np.log(d_hat)
    )
    log_row_sum = logsumexp(log_a, axis=1)
    shares = np.exp(log_a - log_row_sum[:, None])
    price_hats = np.exp(-log_row_sum / baseline.primitives.theta)
    incomes = baseline.incomes * wage_hats
    return wage_hats, price_hats, shares, incomes


def hat_residual(log_free_hats, baseline: Equilibrium, d_hat):
    """New sales must equal baseline income multiplied by each wage hat."""
    _, _, shares, incomes = hat_objects(baseline, d_hat, log_free_hats)
    return market_clearing(shares, incomes)[1:]


def solve_exact_hat(baseline: Equilibrium, d_hat) -> HatEquilibrium:
    """Solve log wage changes independently, fixing country 0's wage hat to 1."""
    shock = validate_hat_inputs(baseline, d_hat)
    result = least_squares(
        hat_residual, np.zeros(baseline.incomes.size - 1), args=(baseline, shock),
        ftol=SOLVER_TOL, xtol=SOLVER_TOL, gtol=SOLVER_TOL, max_nfev=MAX_NFEV,
    )
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        wage_hats, price_hats, shares, incomes = hat_objects(baseline, shock, result.x)
        real_wage_hats = wage_hats / price_hats
        residual = market_clearing(shares, incomes)
    diagnostics = make_diagnostics(
        result, residual, wage_hats, price_hats, shares, incomes, real_wage_hats,
    )
    return HatEquilibrium(
        wage_hats, price_hats, shares, incomes, real_wage_hats, residual, diagnostics,
    )

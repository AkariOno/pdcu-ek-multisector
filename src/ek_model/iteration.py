"""An explicit, fixed-damping wage adjustment; no optimizer or line search.

Both economic routes supply their own shares, incomes and prices. This loop
only follows the sales/income wage rule and checks every country's residual.
"""

from numbers import Integral
from dataclasses import astuple

import numpy as np

from .model import (
    DEFAULT_DAMPING, ITERATION_TOL, MAX_ITER, Diagnostics, IterationRecord,
    EconomicState, exporter_sales, market_clearing, normalized_wages,
)


def validate_controls(damping, max_iter, tol):
    """Reject invalid controls before any equilibrium evaluation."""
    for name, value in (("damping", damping), ("tol", tol)):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be a numeric scalar")
        scalar = np.asarray(value, dtype=float)
        if scalar.ndim != 0 or not np.isfinite(scalar) or scalar <= 0:
            raise ValueError(f"{name} must be a finite positive scalar")
    if float(damping) > 1:
        raise ValueError("damping must be <= 1")
    if isinstance(max_iter, (bool, np.bool_)) or not isinstance(max_iter, Integral) or max_iter <= 0:
        raise ValueError("max_iter must be a positive integer")


def damped_wage_step(log_wages, sales_income_ratio, damping):
    """Simultaneous multiplicative update, expressed in normalized log wages."""
    log_w_next = log_wages + damping * np.log(sales_income_ratio)
    log_w_next -= log_w_next[0]
    return log_w_next


def iterate_wages(evaluate, n, *, industries=1, damping=DEFAULT_DAMPING, max_iter=MAX_ITER,
                  tol=ITERATION_TOL):
    """Evaluate → test the full residual → update all wages → normalize → repeat.

    evaluate(log_wages) returns an EconomicState, including industry spending
    and aggregate cost of living. There are at most max_iter updates and
    max_iter+1 evaluations. Failed states remain in history for diagnosis.
    """
    validate_controls(damping, max_iter, tol)
    damping, tol, max_iter = float(damping), float(tol), int(max_iter)
    log_wages = np.zeros(n)
    history = []
    for iteration in range(max_iter + 1):
        with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
            try:
                objects = evaluate(log_wages)
            except (FloatingPointError, OverflowError):
                objects = EconomicState(normalized_wages(log_wages), np.full((n, industries), np.nan),
                                        np.full((n, n, industries), np.nan), np.full(n, np.nan),
                                        np.full(n, np.nan), np.full((n, industries), np.nan))
            wages, incomes = objects.wages, objects.incomes
            sales = exporter_sales(objects.shares, objects.expenditures)
            q = sales / incomes
            residual = market_clearing(objects.shares, incomes, objects.expenditures)
            real_wages = objects.real_wages
        finite_positive = all(
            np.all(np.isfinite(values)) and np.all(values > 0)
            for values in (*astuple(objects), q, real_wages)
        ) and np.all(np.isfinite(residual))
        norm = float(np.max(np.abs(residual))) if finite_positive else float("inf")
        history.append(IterationRecord(
            iteration, tuple(float(w) for w in wages),
            tuple(float(r) for r in residual), norm,
        ))
        if not finite_positive:
            status, message = "numerical_failure", "Nonfinite or nonpositive numerical state."
            break
        if norm < tol:
            status, message = "converged", "Full normalized market-clearing residual is below tolerance."
            break
        if iteration == max_iter:
            status, message = "max_iterations", "Wage-update limit reached before convergence."
            break
        log_wages = damped_wage_step(log_wages, q, damping)
    diagnostics = Diagnostics(
        converged=status == "converged", status=status, message=message,
        niter=iteration, nfev=len(history), residual_norm=norm,
        damping=damping, tolerance=tol, max_iter=max_iter, history=tuple(history),
    )
    return objects, residual, diagnostics

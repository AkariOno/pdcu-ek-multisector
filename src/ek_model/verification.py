"""The pre-specified acceptance rule, shared by tests and the viewer builder."""

from dataclasses import asdict
import platform

import numpy as np
import scipy

from .exact_hat import solve_exact_hat
from .full_solution import solve_levels
from .model import MAX_NFEV, RESIDUAL_TOL, SOLVER_TOL, Primitives

ERROR_TOL = 1e-9


def fixture():
    """Asymmetric three-country economy; a symmetric proportional bilateral cut."""
    baseline = Primitives(
        T=[1.0, 1.3, 0.8], L=[1.0, 1.2, 0.9],
        d=[[1.0, 1.4, 1.8], [1.3, 1.0, 1.6], [1.7, 1.5, 1.0]], theta=4.0,
    )
    d_hat = np.ones((3, 3))
    d_hat[0, 1] = d_hat[1, 0] = 0.9
    return baseline, d_hat


def compare_changes(reference, candidate):
    """Both strict bounds must pass; relative errors use the levels-route ratio."""
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    valid = (
        reference.shape == candidate.shape and reference.size > 0
        and np.all(np.isfinite(reference)) and np.all(np.isfinite(candidate))
        and np.all(reference != 0)
    )
    if not valid:
        return {"max_absolute_error": None, "max_relative_error": None,
                "tolerance": ERROR_TOL, "passed": False}
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        difference = np.abs(candidate - reference)
        absolute = float(difference.max())
        relative = float((difference / np.abs(reference)).max())
    finite = np.isfinite(absolute) and np.isfinite(relative)
    return {
        "max_absolute_error": absolute if np.isfinite(absolute) else None,
        "max_relative_error": relative if np.isfinite(relative) else None,
        "tolerance": ERROR_TOL,
        "passed": bool(finite and absolute < ERROR_TOL and relative < ERROR_TOL),
    }


def diagnostics_record(equilibrium):
    """JSON has no Infinity; null records a nonfinite residual as a failure."""
    record = asdict(equilibrium.diagnostics)
    if not np.isfinite(record["residual_norm"]):
        record["residual_norm"] = None
        record["converged"] = False
    elif record["residual_norm"] >= RESIDUAL_TOL:
        record["converged"] = False
    record["converged"] = bool(record["converged"] and record["solver_success"])
    return record


def compare_equilibria(e0, e1, hats):
    """Compare multiplicative changes, including all nine bilateral share ratios."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        pairs = {
            "wages": (e1.wages / e0.wages, hats.wage_hats),
            "prices": (e1.prices / e0.prices, hats.price_hats),
            "shares": (e1.shares / e0.shares, hats.shares / e0.shares),
            "real_wages": (e1.real_wages / e0.real_wages, hats.real_wage_hats),
        }
    comparisons = {name: compare_changes(*pair) for name, pair in pairs.items()}
    solvers = {name: diagnostics_record(eq) for name, eq in
               (("E0", e0), ("E1", e1), ("exact_hat", hats))}
    passed = all(row["passed"] for row in comparisons.values()) and all(
        row["converged"] for row in solvers.values()
    )
    return {"passed": passed, "comparisons": comparisons, "solvers": solvers}


def run_verification():
    """Always run the two levels solves and independent hat route afresh."""
    p, shock = fixture()
    certificate = {
        "schema_version": 1,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__},
        "fixture": {"T": p.T.tolist(), "L": p.L.tolist(), "d": p.d.tolist(),
                    "theta": p.theta, "gamma": p.gamma, "d_hat": shock.tolist()},
        "contract": {"error_tolerance": ERROR_TOL, "residual_tolerance": RESIDUAL_TOL,
                     "solver_tolerance": SOLVER_TOL, "max_nfev": MAX_NFEV,
                     "numeraire": "w[0] = w_hat[0] = 1",
                     "relative_denominator": "absolute levels-route ratio",
                     "residual": "max_i abs((sales[i]-income[i])/income[i])"},
        "passed": False, "comparisons": {}, "solvers": {}, "failure": None,
    }
    try:
        e0 = solve_levels(p)
        certificate["solvers"]["E0"] = diagnostics_record(e0)
        e1 = solve_levels(Primitives(p.T, p.L, p.d * shock, p.theta, p.gamma))
        certificate["solvers"]["E1"] = diagnostics_record(e1)
        hats = solve_exact_hat(e0, shock)
        certificate.update(compare_equilibria(e0, e1, hats))
        if not certificate["passed"]:
            certificate["failure"] = "Convergence or strict equivalence criteria failed."
    except (ValueError, RuntimeError, FloatingPointError, OverflowError) as exc:
        certificate["failure"] = f"{type(exc).__name__}: {exc}"
    return certificate

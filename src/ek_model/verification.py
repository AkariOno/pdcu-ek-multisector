"""The pre-specified acceptance rule, shared by tests and the viewer builder."""

from dataclasses import asdict, replace
import platform

import numpy as np

from .exact_hat import solve_exact_hat
from .full_solution import solve_levels
from .model import DEFAULT_DAMPING, ITERATION_TOL, MAX_ITER, RESIDUAL_TOL, Primitives

ERROR_TOL = 1e-9


def fixture():
    """Three countries, two industries; bilateral cost cut in industry 0 only."""
    baseline = Primitives(
        T=[[1.0, 0.9], [1.3, 1.1], [0.8, 1.5]], L=[1.0, 1.2, 0.9],
        d=np.stack(([[1.0, 1.4, 1.8], [1.3, 1.0, 1.6], [1.7, 1.5, 1.0]],
                    [[1.0, 1.7, 1.4], [1.5, 1.0, 1.8], [1.6, 1.4, 1.0]]), axis=2),
        theta=[4.0, 6.0], alpha=[[.60, .40], [.35, .65], [.50, .50]],
    )
    d_hat = np.ones((3, 3, 2))
    d_hat[0, 1, 0] = d_hat[1, 0, 0] = 0.9
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
    """Recheck the full residual; encode nonfinite failure evidence as JSON null."""
    record = asdict(equilibrium.diagnostics)
    actual_norm = float(np.max(np.abs(equilibrium.residual)))
    record["converged"] = bool(
        record["converged"] and record["status"] == "converged"
        and np.isfinite(actual_norm) and actual_norm < RESIDUAL_TOL
        and np.isfinite(record["residual_norm"]) and record["residual_norm"] < RESIDUAL_TOL
    )
    return json_safe(record)


def json_safe(value):
    """History may contain a failed numerical state; preserve it without NaN/Inf."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def compare_equilibria(e0, e1, hats):
    """Compare multiplicative changes, including every bilateral industry share ratio."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        pairs = {
            "wages": (e1.wages / e0.wages, hats.wage_hats),
            "prices": (e1.prices / e0.prices, hats.price_hats),
            "cost_of_living": (e1.cost_of_living / e0.cost_of_living, hats.cost_of_living_hats),
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
        "schema_version": 3,
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "fixture": {"T": p.T.tolist(), "L": p.L.tolist(), "d": p.d.tolist(),
                    "theta": p.theta.tolist(), "gamma": p.gamma.tolist(),
                    "alpha": p.alpha.tolist(), "d_hat": shock.tolist()},
        "contract": {"error_tolerance": ERROR_TOL, "residual_tolerance": RESIDUAL_TOL,
                     "iteration_tolerance": ITERATION_TOL, "max_iter": MAX_ITER,
                     "damping": DEFAULT_DAMPING, "method": "fixed-damping multiplicative wage iteration",
                     "numeraire": "w[0] = w_hat[0] = 1",
                     "utility": "U[n] = product_j (c[n,j]/alpha[n,j])**alpha[n,j]",
                     "aggregate_price": "C[n] = product_j P[n,j]**alpha[n,j]",
                     "relative_denominator": "absolute levels-route ratio",
                     "residual": "max_i abs((sales[i]-income[i])/income[i])"},
        "passed": False, "comparisons": {}, "solvers": {}, "failure": None,
    }
    try:
        e0 = solve_levels(p)
        certificate["solvers"]["E0"] = diagnostics_record(e0)
        e1 = solve_levels(replace(p, d=p.d * shock))
        certificate["solvers"]["E1"] = diagnostics_record(e1)
        hats = solve_exact_hat(e0, shock)
        certificate.update(compare_equilibria(e0, e1, hats))
        from .regression import run_j1_regression
        certificate["j1_regression"] = run_j1_regression()
        certificate["passed"] = certificate["passed"] and certificate["j1_regression"]["passed"]
        if not certificate["passed"]:
            certificate["failure"] = "Convergence or strict equivalence criteria failed."
    except (ValueError, RuntimeError, FloatingPointError, OverflowError) as exc:
        certificate["passed"] = False
        certificate["failure"] = f"{type(exc).__name__}: {exc}"
    return certificate

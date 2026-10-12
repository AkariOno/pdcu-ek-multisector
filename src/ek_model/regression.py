"""Data-only oracle from accepted PR #4; no historical production solver."""
from dataclasses import replace
import hashlib
from importlib.resources import files
import json
import numpy as np
from .model import Primitives
from .full_solution import solve_levels
from .exact_hat import solve_exact_hat

REFERENCE_SHA256 = "a0e1c68038dac764d51acefada95b1f60580ca1fc8fb930b1a9f1d562a163ae9"
REFERENCE_COMMIT = "9eb5b52bcefc6a5e4055acc5dfb2a6f6a184596b"

def reference_data():
    data = files("ek_model").joinpath("data/one_industry_reference.json").read_bytes()
    if hashlib.sha256(data).hexdigest() != REFERENCE_SHA256:
        raise ValueError("Immutable one-industry reference hash mismatch")
    reference = json.loads(data)
    if reference["source_commit"] != REFERENCE_COMMIT:
        raise ValueError("Reference must originate at the accepted PR #4 revision")
    return reference

def run_j1_regression():
    """Compare E0, E1 and independent hats, including levels and multiplicative hats."""
    from .verification import compare_changes, diagnostics_record
    reference = reference_data()
    f = reference["fixture"]
    p = Primitives(np.array(f["T"])[:, None], f["L"], np.array(f["d"])[:, :, None],
                   [f["theta"]], np.ones((len(f["L"]), 1)), [f["gamma"]])
    shock = np.array(reference["d_hat"])[:, :, None]
    e0 = solve_levels(p)
    e1 = solve_levels(replace(p, d=p.d * shock))
    hats = solve_exact_hat(e0, shock)
    comparisons = {}
    for name, eq in (("E0", e0), ("E1", e1), ("exact_hat", hats)):
        old = reference[name]
        hat = name == "exact_hat"
        fields = ("wage_hats", "price_hats", "shares", "incomes", "real_wage_hats") if hat else (
            "wages", "prices", "shares", "incomes", "real_wages")
        for field in fields:
            candidate = getattr(eq, field)
            if field in ("prices", "price_hats"):
                candidate = candidate[:, 0]
            elif field == "shares":
                candidate = candidate[:, :, 0]
            comparisons[f"{name}.{field}"] = compare_changes(old[field], candidate)
        cost = eq.cost_of_living_hats if hat else eq.cost_of_living
        price = eq.price_hats if hat else eq.prices
        comparisons[f"{name}.cost_of_living"] = compare_changes(
            old["price_hats" if hat else "prices"], cost)
        # A separate identity guards the exact J=1 aggregation contract.
        if not np.array_equal(cost, price[:, 0]):
            raise ValueError("J=1 aggregate cost of living must equal the sole price")
    solvers = {name: diagnostics_record(eq) for name, eq in (
        ("E0", e0), ("E1", e1), ("exact_hat", hats))}
    return {"source_commit": REFERENCE_COMMIT, "reference_sha256": REFERENCE_SHA256,
            "source_hashes": reference["source_hashes"],
            "comparisons": comparisons, "solvers": solvers,
            "passed": all(c["passed"] for c in comparisons.values()) and
                      all(d["converged"] for d in solvers.values())}

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from ek_model import Primitives, solve_exact_hat, solve_levels
from ek_model import exact_hat, full_solution, verification
from ek_model.model import RESIDUAL_TOL, make_diagnostics, market_clearing
from ek_model.verification import compare_changes, compare_equilibria, fixture, run_verification


@pytest.fixture(scope="module")
def solutions():
    p, shock = fixture()
    e0 = solve_levels(p)
    e1 = solve_levels(replace(p, d=p.d * shock))
    return e0, e1, solve_exact_hat(e0, shock)


def test_prespecified_equivalence():
    certificate = run_verification()
    assert certificate["passed"], certificate
    assert set(certificate["solvers"]) == {"E0", "E1", "exact_hat"}
    assert set(certificate["comparisons"]) == {"wages", "prices", "shares", "real_wages"}
    for row in certificate["comparisons"].values():
        assert row["max_absolute_error"] < 1e-9
        assert row["max_relative_error"] < 1e-9


def test_equations_and_diagnostics(solutions):
    e0, e1, hats = solutions
    for eq in solutions:
        assert eq.diagnostics.converged
        assert 0 < eq.diagnostics.nfev <= 2000
        np.testing.assert_allclose(eq.shares.sum(axis=1), 1, rtol=0, atol=1e-14)
        actual = market_clearing(eq.shares, eq.incomes)
        np.testing.assert_array_equal(eq.residual, actual)
        assert eq.diagnostics.residual_norm == max(abs(actual)) < RESIDUAL_TOL
    assert e0.wages[0] == e1.wages[0] == hats.wage_hats[0] == 1.0
    for eq in (e0, e1):
        p = eq.primitives
        # Direct equations provide an oracle independent of log-sum-exp code.
        weights = p.T[None, :] * (eq.wages[None, :] * p.d) ** (-p.theta)
        np.testing.assert_allclose(eq.shares, weights / weights.sum(axis=1)[:, None],
                                   rtol=1e-13, atol=0)
        np.testing.assert_allclose(eq.prices, p.gamma * weights.sum(axis=1) ** (-1 / p.theta),
                                   rtol=1e-13, atol=0)


def test_no_shock(solutions):
    e0 = solutions[0]
    hats = solve_exact_hat(e0, np.ones_like(e0.shares))
    assert hats.diagnostics.converged
    for values in (hats.wage_hats, hats.price_hats, hats.real_wage_hats):
        np.testing.assert_allclose(values, 1, rtol=0, atol=1e-13)
    np.testing.assert_allclose(hats.shares, e0.shares, rtol=0, atol=1e-14)


def test_gamma_changes_levels_not_hats(solutions):
    p, shock = fixture()
    scaled = solve_levels(replace(p, gamma=2.0))
    hats = solve_exact_hat(scaled, shock)
    np.testing.assert_allclose(scaled.prices, 2 * solutions[0].prices, rtol=1e-13)
    np.testing.assert_allclose(hats.price_hats, solutions[2].price_hats, rtol=1e-13)


@pytest.mark.parametrize("changes", [
    {"T": [1, 0, 2]}, {"T": [1, np.nan, 2]}, {"T": [[1, 2, 3]]},
    {"L": [1, 2]}, {"L": [1, -1, 1]}, {"d": np.ones((2, 2))},
    {"d": [[1, .9, 2], [2, 1, 2], [2, 2, 1]]},
    {"d": np.full((3, 3), 2)}, {"d": np.full((3, 3), np.inf)},
    {"theta": 0}, {"theta": [4]}, {"gamma": -1}, {"gamma": np.nan},
])
def test_invalid_primitives(changes):
    with pytest.raises(ValueError):
        replace(fixture()[0], **changes)


@pytest.mark.parametrize("shock", [np.ones((2, 2)), np.zeros((3, 3)),
                                    np.full((3, 3), np.nan), np.full((3, 3), .5)])
def test_invalid_shock(solutions, shock):
    with pytest.raises(ValueError):
        solve_exact_hat(solutions[0], shock)


def test_baseline_must_be_valid_and_converged(solutions):
    e0 = solutions[0]
    for bad in (
        replace(e0, diagnostics=replace(e0.diagnostics, converged=False)),
        replace(e0, shares=np.full((3, 3), np.nan)),
        replace(e0, shares=e0.shares.T),
        replace(e0, incomes=e0.incomes * [1, 2, 1]),
    ):
        with pytest.raises(ValueError):
            solve_exact_hat(bad, np.ones((3, 3)))


def test_hat_does_not_call_levels_solver(solutions, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("hat route called levels route")
    monkeypatch.setattr(full_solution, "solve_levels", forbidden)
    monkeypatch.setattr(full_solution, "levels_objects", forbidden)
    assert solve_exact_hat(solutions[0], fixture()[1]).diagnostics.converged


def test_error_rules_are_separate_and_strict():
    assert not compare_changes([1e-6], [1e-6 + 1e-10])["passed"]  # relative fails
    assert not compare_changes([1e6], [1e6 + 1e-5])["passed"]  # absolute fails
    assert not compare_changes([1e-9], [2e-9])["passed"]  # absolute equals boundary
    assert compare_changes([1], [1 + 1e-12])["passed"]
    for reference, candidate in (([1], [np.nan]), ([np.inf], [1]), ([0], [0]),
                                 ([1], [1, 1]), ([], [])):
        assert not compare_changes(reference, candidate)["passed"]


def test_numeraire_residual_cannot_be_hidden():
    result = SimpleNamespace(success=True, status=1, message="ok", nfev=1)
    diagnostics = make_diagnostics(result, np.array([1e-4, 0, 0]), np.ones(3))
    assert not diagnostics.converged
    assert diagnostics.residual_norm == 1e-4


@pytest.mark.parametrize("module", [full_solution, exact_hat])
@pytest.mark.parametrize("success,logs", [(False, [0, 0]), (True, [np.nan, 0])])
def test_failed_and_nonfinite_optimizer_results(module, success, logs, monkeypatch, solutions):
    result = SimpleNamespace(success=success, status=0, message="injected failure",
                             nfev=2000, x=np.array(logs))
    monkeypatch.setattr(module, "least_squares", lambda *args, **kwargs: result)
    if module is full_solution:
        eq = module.solve_levels(fixture()[0])
    else:
        eq = module.solve_exact_hat(solutions[0], fixture()[1])
    assert not eq.diagnostics.converged


def test_comparison_rejects_failed_solver_even_with_identical_values(solutions):
    e0, e1, hats = solutions
    failed = replace(hats, diagnostics=replace(hats.diagnostics, converged=False))
    assert not compare_equilibria(e0, e1, failed)["passed"]
    bad = replace(hats, price_hats=np.full(3, np.nan))
    assert not compare_equilibria(e0, e1, bad)["passed"]


def test_verification_retains_failed_baseline_diagnostics(solutions, monkeypatch):
    failed = replace(solutions[0], diagnostics=replace(solutions[0].diagnostics, converged=False))
    monkeypatch.setattr(verification, "solve_levels", lambda p: failed)
    certificate = run_verification()
    assert not certificate["passed"]
    assert not certificate["solvers"]["E0"]["converged"]
    assert "converged baseline" in certificate["failure"]

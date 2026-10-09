from dataclasses import replace

import numpy as np
import pytest

from ek_model import Primitives, solve_exact_hat, solve_levels
from ek_model import exact_hat, full_solution, verification
from ek_model.iteration import damped_wage_step, iterate_wages
from ek_model.model import ITERATION_TOL, RESIDUAL_TOL, EconomicState, aggregate_price, exporter_sales, market_clearing, row_logsumexp
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
    assert set(certificate["comparisons"]) == {"wages", "prices", "cost_of_living", "shares", "real_wages"}
    for row in certificate["comparisons"].values():
        assert row["max_absolute_error"] < 1e-9
        assert row["max_relative_error"] < 1e-9


def test_equations_and_diagnostics(solutions):
    e0, e1, hats = solutions
    for eq in solutions:
        assert eq.diagnostics.converged
        assert eq.diagnostics.status == "converged"
        assert 0 <= eq.diagnostics.niter <= 10_000
        assert eq.diagnostics.nfev == eq.diagnostics.niter + 1
        assert len(eq.diagnostics.history) == eq.diagnostics.nfev
        np.testing.assert_allclose(eq.shares.sum(axis=1), 1, rtol=0, atol=1e-14)
        actual = market_clearing(eq.shares, eq.incomes, eq.expenditures)
        np.testing.assert_array_equal(eq.residual, actual)
        assert eq.diagnostics.residual_norm == max(abs(actual)) < RESIDUAL_TOL
        assert eq.diagnostics.residual_norm < ITERATION_TOL
        history = eq.diagnostics.history
        np.testing.assert_array_equal(history[0].wages, np.ones(3))
        assert [row.iteration for row in history] == list(range(eq.diagnostics.nfev))
        assert all(row.wages[0] == 1 for row in history)
        assert all(row.residual_norm == max(abs(np.array(row.residual))) for row in history)
        np.testing.assert_array_equal(history[-1].residual, eq.residual)
        np.testing.assert_array_equal(history[-1].wages, eq.wages if hasattr(eq, "wages") else eq.wage_hats)
    assert e0.wages[0] == e1.wages[0] == hats.wage_hats[0] == 1.0
    for eq in (e0, e1):
        p = eq.primitives
        # Direct equations provide an oracle independent of log-sum-exp code.
        weights = p.T[None, :, :] * (eq.wages[None, :, None] * p.d) ** (-p.theta)
        np.testing.assert_allclose(eq.shares, weights / weights.sum(axis=1)[:, None, :],
                                   rtol=1e-13, atol=0)
        np.testing.assert_allclose(eq.prices, p.gamma * weights.sum(axis=1) ** (-1 / p.theta),
                                   rtol=1e-13, atol=0)


def test_no_shock(solutions):
    e0 = solutions[0]
    hats = solve_exact_hat(e0, np.ones_like(e0.shares))
    assert hats.diagnostics.converged
    assert hats.diagnostics.niter == 0
    assert hats.diagnostics.nfev == 1
    for values in (hats.wage_hats, hats.price_hats, hats.cost_of_living_hats, hats.real_wage_hats):
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
        replace(e0, shares=np.full((3, 3, 2), np.nan)),
        replace(e0, shares=np.swapaxes(e0.shares, 0, 1)),
        replace(e0, incomes=e0.incomes * [1, 2, 1]),
    ):
        with pytest.raises(ValueError):
            solve_exact_hat(bad, np.ones((3, 3, 2)))


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
    # Non-numeraire residual is < .01, but country's 0 residual is .1.
    def evaluate(log_wages):
        return EconomicState(np.ones(2), np.ones((2,1)), np.array([[.99, .01], [.0011, .9989]])[:,:,None], np.array([1., 100.]), np.ones(2), np.array([[1.], [100.]]))
    _, residual, diagnostics = iterate_wages(evaluate, 2, tol=.01, max_iter=1)
    assert abs(residual[1]) < .01 < abs(residual[0])
    assert not diagnostics.converged
    assert diagnostics.status == "max_iterations"
    assert diagnostics.residual_norm == max(abs(residual))


@pytest.mark.parametrize("module", [full_solution, exact_hat])
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, 0., -1.])
def test_nonfinite_or_nonpositive_numerical_states(module, bad_value, monkeypatch, solutions):
    def bad_objects(*args):
        return EconomicState(np.ones(3), np.full((3,2), bad_value), np.ones((3,3,2))/3, np.ones(3), np.full(3,bad_value), np.ones((3,2))/2)
    symbol = "levels_objects" if module is full_solution else "hat_objects"
    monkeypatch.setattr(module, symbol, bad_objects)
    if module is full_solution:
        eq = module.solve_levels(fixture()[0])
    else:
        eq = module.solve_exact_hat(solutions[0], fixture()[1])
    assert not eq.diagnostics.converged
    assert eq.diagnostics.status == "numerical_failure"
    assert eq.diagnostics.nfev == 1
    assert np.isinf(eq.diagnostics.residual_norm)


def test_update_matches_multiplicative_wage_rule():
    wages = np.array([2., 3., 5.])
    q = np.array([1.4, .8, 1.1])
    raw = wages * q ** .2
    np.testing.assert_allclose(np.exp(damped_wage_step(np.log(wages), q, .2)),
                               raw / raw[0], rtol=1e-14, atol=0)
    assert damped_wage_step(np.log(wages), q, .2)[0] == 0


@pytest.mark.parametrize("damping", [.1, .2, .3])
def test_damping_changes_path_not_equilibrium(damping, solutions):
    p, shock = fixture()
    e0 = solve_levels(p, damping=damping)
    e1 = solve_levels(replace(p, d=p.d * shock), damping=damping)
    hats = solve_exact_hat(e0, shock, damping=damping)
    assert compare_equilibria(e0, e1, hats)["passed"]
    for eq, reference in zip((e0, e1, hats), solutions):
        values = eq.wages if hasattr(eq, "wages") else eq.wage_hats
        expected = reference.wages if hasattr(reference, "wages") else reference.wage_hats
        np.testing.assert_allclose(values, expected, rtol=0, atol=1e-11)
        assert eq.diagnostics.damping == damping
        assert eq.diagnostics.history[-1].residual_norm < 1e-13


@pytest.mark.parametrize("kwargs", [
    {"damping": 0}, {"damping": -1}, {"damping": 1.1}, {"damping": np.nan},
    {"damping": np.inf}, {"damping": [0.2]}, {"damping": True},
    {"tol": 0}, {"tol": -1}, {"tol": np.nan}, {"tol": np.inf}, {"tol": [1e-13]},
    {"max_iter": 0}, {"max_iter": -1}, {"max_iter": 1.5}, {"max_iter": True},
])
def test_invalid_controls_rejected_by_both_routes(kwargs, solutions):
    with pytest.raises(ValueError):
        solve_levels(fixture()[0], **kwargs)
    with pytest.raises(ValueError):
        solve_exact_hat(solutions[0], fixture()[1], **kwargs)


@pytest.mark.parametrize("route", ["levels", "hats"])
def test_iteration_limit_returns_last_evaluated_state(route, solutions):
    p, shock = fixture()
    eq = solve_levels(p, max_iter=1) if route == "levels" else solve_exact_hat(solutions[0], shock, max_iter=1)
    assert not eq.diagnostics.converged
    assert eq.diagnostics.status == "max_iterations"
    assert eq.diagnostics.niter == 1 and eq.diagnostics.nfev == 2
    wages = np.array(eq.diagnostics.history[-1].wages)
    # Re-evaluate the final economic objects; do not return stale pre-update shares.
    expected = (full_solution.levels_objects(p, np.log(wages)) if route == "levels"
                else exact_hat.hat_objects(solutions[0], shock, np.log(wages)))
    # log(exp(x)) has a floating-point round trip; stale objects differ materially.
    np.testing.assert_allclose(eq.shares, expected.shares, rtol=1e-14, atol=0)
    np.testing.assert_allclose(eq.incomes, expected.incomes, rtol=1e-14, atol=0)
    np.testing.assert_allclose(eq.residual, market_clearing(expected.shares, expected.incomes, expected.expenditures), rtol=0, atol=1e-14)


def test_tiny_wage_step_is_not_convergence():
    eq = solve_levels(fixture()[0], damping=1e-300, max_iter=1)
    assert not eq.diagnostics.converged
    assert eq.diagnostics.status == "max_iterations"
    np.testing.assert_array_equal(eq.diagnostics.history[0].wages, eq.diagnostics.history[-1].wages)


def test_numerical_exception_reports_failure():
    def broken(log_wages):
        raise FloatingPointError("injected evaluation failure")
    _, _, diagnostics = iterate_wages(broken, 3)
    assert diagnostics.status == "numerical_failure" and not diagnostics.converged
    assert diagnostics.nfev == 1


def test_row_logsumexp_is_stable_and_axis_is_exporters():
    values = np.array([[1000., 1001., 999.], [-1000., -999., -1001.]])
    expected = np.array([1001., -999.]) + np.log(1 + np.exp(-1) + np.exp(-2))
    np.testing.assert_allclose(row_logsumexp(values), expected, rtol=0, atol=1e-13)


def test_verification_sanitizes_nonfinite_history(solutions):
    from ek_model.model import IterationRecord
    eq = solutions[0]
    bad_record = IterationRecord(0, (np.nan, 1., 1.), (np.inf, 0., 0.), np.inf)
    failed = replace(eq, diagnostics=replace(eq.diagnostics, status="numerical_failure", converged=False,
                     residual_norm=np.inf, history=(bad_record,)))
    record = verification.diagnostics_record(failed)
    assert record["residual_norm"] is None
    assert record["history"][0]["wages"][0] is None
    assert record["history"][0]["residual"][0] is None


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

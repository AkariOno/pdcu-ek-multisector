from dataclasses import replace
import numpy as np
import pytest
from ek_model import Primitives, solve_levels, solve_exact_hat
from ek_model.model import aggregate_price, exporter_sales, EconomicState
from ek_model.regression import run_j1_regression, reference_data, REFERENCE_COMMIT
from ek_model.verification import fixture, run_verification
from ek_model import regression, full_solution, exact_hat

@pytest.fixture(scope="module")
def equilibria():
    p, shock = fixture()
    e0 = solve_levels(p)
    e1 = solve_levels(replace(p, d=p.d * shock))
    return e0, e1, solve_exact_hat(e0, shock)

def test_primitives_copy_and_do_not_renormalize():
    p, _ = fixture()
    alpha = p.alpha.copy()
    alpha[0, 0] += 1e-13
    q = replace(p, alpha=alpha)
    np.testing.assert_array_equal(q.alpha, alpha)
    alpha[0, 0] = 100
    assert q.alpha[0, 0] < 1
    for name in ("T", "L", "d", "theta", "alpha", "gamma"):
        assert not getattr(q, name).flags.writeable
    np.testing.assert_array_equal(replace(p, gamma=2).gamma, [2, 2])

@pytest.mark.parametrize("changes", [
    {"T": np.ones((3,0))}, {"T": np.ones((1,2))}, {"T": np.ones((3,2,1))},
    {"alpha": np.ones((3,2))}, {"alpha": [[1,0],[.4,.6],[.5,.5]]},
    {"alpha": [[.6,.4],[np.nan,.65],[.5,.5]]}, {"alpha": np.ones((3,1))},
    {"alpha": [[.6,.40000000001],[.35,.65],[.5,.5]]},
    {"theta": [4,-6]}, {"theta": [4,np.inf]}, {"theta": [[4,6]]},
    {"gamma": [1,0]}, {"gamma": [1,2,3]}, {"gamma": [[1,2]]},
    {"L": np.ones((3,1))}, {"d": np.ones((3,3,3))},
])
def test_invalid_multi_industry_inputs(changes):
    with pytest.raises(ValueError):
        replace(fixture()[0], **changes)

def test_domestic_costs_and_shocks_in_every_industry(equilibria):
    p, shock = fixture()
    wrong = p.d.copy()
    wrong[2,2,1] = 1.01
    with pytest.raises(ValueError):
        replace(p, d=wrong)
    for idx, value in (((1,1,1), 1.01), ((0,1,1), .1), ((2,0,0), 0)):
        wrong = shock.copy()
        wrong[idx] = value
        with pytest.raises(ValueError):
            solve_exact_hat(equilibria[0], wrong)

def test_expenditure_budgets_and_industry_prices(equilibria):
    for eq in equilibria:
        assert eq.shares.shape == (3,3,2)
        assert eq.expenditures.shape == (3,2)
        np.testing.assert_allclose(eq.expenditures.sum(axis=1), eq.incomes, rtol=0, atol=1e-14)
        manual = np.array([sum(eq.shares[n,i,j]*eq.expenditures[n,j]
                                   for n in range(3) for j in range(2)) for i in range(3)])
        np.testing.assert_allclose(exporter_sales(eq.shares, eq.expenditures), manual, rtol=1e-14)
    for eq in equilibria[:2]:
        p = eq.primitives
        direct_C = np.prod(eq.prices ** p.alpha, axis=1)
        np.testing.assert_allclose(eq.cost_of_living, direct_C, rtol=1e-14)
        np.testing.assert_allclose(eq.real_wages, eq.wages/direct_C, rtol=1e-14)

def test_normalized_utility_and_superseded_convention(equilibria):
    e0, e1, hats = equilibria
    alpha = e0.primitives.alpha
    K = np.prod(alpha**alpha, axis=1)
    for eq in (e0,e1):
        consumption = eq.expenditures/eq.prices
        utility = np.prod((consumption/alpha)**alpha, axis=1)
        np.testing.assert_allclose(utility, eq.incomes/eq.cost_of_living, rtol=1e-14)
        old_C = np.prod((eq.prices/alpha)**alpha, axis=1)
        np.testing.assert_allclose(eq.cost_of_living, K*old_C, rtol=1e-14)
        assert np.all(old_C > eq.cost_of_living)
    old_real0 = e0.wages/(e0.cost_of_living/K)
    old_real1 = e1.wages/(e1.cost_of_living/K)
    np.testing.assert_allclose(old_real1/old_real0, hats.real_wage_hats, rtol=1e-12)
    np.testing.assert_allclose(e1.cost_of_living/e0.cost_of_living,
                               np.prod(hats.price_hats**alpha, axis=1), rtol=1e-12)

def test_gamma_by_industry_changes_only_price_levels(equilibria):
    e0, _, hats = equilibria
    gamma = np.array([2.,3.])
    p = replace(e0.primitives, gamma=gamma)
    new = solve_levels(p)
    np.testing.assert_allclose(new.prices, e0.prices*gamma, rtol=1e-14)
    factor = np.prod(gamma**p.alpha, axis=1)
    np.testing.assert_allclose(new.cost_of_living, e0.cost_of_living*factor, rtol=1e-14)
    np.testing.assert_array_equal(new.wages, e0.wages)
    other = solve_exact_hat(new, fixture()[1])
    np.testing.assert_array_equal(other.cost_of_living_hats, hats.cost_of_living_hats)

def test_shock_restrictions_and_cross_industry_response(equilibria):
    p, shock = fixture()
    np.testing.assert_array_equal(shock[:,:,1], np.ones((3,3)))
    assert np.count_nonzero(shock != 1) == 2
    assert np.min(p.d*shock) >= 1
    e0, e1, hats = equilibria
    assert np.max(abs(e1.shares[:,:,1]-e0.shares[:,:,1])) > 1e-6
    assert np.max(abs(hats.price_hats[:,1]-1)) > 1e-6
    assert abs(hats.real_wage_hats[2]-1) > 1e-6

def test_frozen_j1_regression():
    reference = reference_data()
    assert reference["source_commit"] == REFERENCE_COMMIT
    record = run_j1_regression()
    assert record["passed"]
    assert len(record["comparisons"]) == 18
    for row in record["comparisons"].values():
        assert row["max_absolute_error"] < 1e-9 and row["max_relative_error"] < 1e-9
    assert run_verification()["j1_regression"]["passed"]

def test_j1_oracle_integrity_is_enforced(monkeypatch):
    monkeypatch.setattr(regression, "REFERENCE_SHA256", "invalid")
    with pytest.raises(ValueError, match="hash mismatch"):
        reference_data()
    certificate = run_verification()
    assert not certificate["passed"] and "hash mismatch" in certificate["failure"]

@pytest.mark.parametrize("attribute", ["prices","shares","incomes","cost_of_living","expenditures"])
@pytest.mark.parametrize("module", [full_solution, exact_hat])
def test_each_economic_object_is_checked(attribute, module, equilibria, monkeypatch):
    original = module.levels_objects if module is full_solution else module.hat_objects
    def broken(*args):
        state = original(*args)
        return replace(state, **{attribute: np.full_like(getattr(state, attribute), np.nan)})
    monkeypatch.setattr(module, "levels_objects" if module is full_solution else "hat_objects", broken)
    eq = (module.solve_levels(fixture()[0]) if module is full_solution else
          module.solve_exact_hat(equilibria[0], fixture()[1]))
    assert eq.diagnostics.status == "numerical_failure" and not eq.diagnostics.converged

def test_exporter_logsumexp_for_three_dimensional_arrays():
    from ek_model.model import row_logsumexp
    x = np.array([[[1000,-1000],[1001,-999],[999,-1001]],
                  [[1002,-998],[1000,-1000],[1001,-999]]])
    result = row_logsumexp(x)
    assert result.shape == (2,2)
    for n in range(2):
        for j in range(2):
            m = max(x[n,:,j])
            assert abs(result[n,j]-(m+np.log(sum(np.exp(x[n,:,j]-m))))) < 1e-12

@pytest.mark.parametrize("module", [full_solution, exact_hat])
def test_numerical_exception_preserves_industry_dimensions(module, equilibria, monkeypatch):
    def broken(*args):
        raise FloatingPointError("failed evaluation")
    monkeypatch.setattr(module, "levels_objects" if module is full_solution else "hat_objects", broken)
    eq = (module.solve_levels(fixture()[0]) if module is full_solution else
          module.solve_exact_hat(equilibria[0], fixture()[1]))
    assert eq.diagnostics.status == "numerical_failure"
    assert eq.shares.shape == (3, 3, 2)
    assert eq.expenditures.shape == (3, 2)
    prices = eq.prices if module is full_solution else eq.price_hats
    assert prices.shape == (3, 2)

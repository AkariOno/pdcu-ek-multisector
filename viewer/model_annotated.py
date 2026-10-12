# GENERATED READING COMPANION — not a standalone solver.
# Each excerpt is verbatim production source; comments and ordering are explanatory.
# Imports and other context may be omitted. Repeated functions illustrate different steps.
# SHA-256 source provenance is recorded in manifest.json and verification.json.

# LEVELS / 01 · Primitives & dimensions
# src/ek_model/model.py:19-51
# Explicit industry axes; positive primitives and preference shares. Demand-share rows sum to one without renormalization. Domestic costs are one in every industry.
@dataclass(frozen=True)
class Primitives:
    """One mobile domestic labor factor; fixed final expenditure shares."""
    T: Array
    L: Array
    d: Array
    theta: Array
    alpha: Array
    gamma: Array | float = 1.0

    def __post_init__(self):
        values = {name: positive_array(getattr(self, name), name)
                  for name in ("T", "L", "d", "theta", "alpha", "gamma")}
        T = values["T"]
        if T.ndim != 2 or T.shape[0] < 2 or T.shape[1] < 1:
            raise ValueError("T must have shape (N,J), N >= 2 and J >= 1")
        n, j = T.shape
        shapes = {"L": (n,), "d": (n, n, j), "theta": (j,), "alpha": (n, j)}
        for name, shape in shapes.items():
            if values[name].shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
        d = values["d"]
        if np.any(d < 1) or not np.all(d[np.arange(n), np.arange(n), :] == 1):
            raise ValueError("iceberg costs must be >= 1 with unit domestic costs")
        if not np.allclose(values["alpha"].sum(axis=1), 1, rtol=0, atol=1e-12):
            raise ValueError("alpha rows must sum to one; inputs are not renormalized")
        if values["gamma"].ndim == 0:
            values["gamma"] = np.full(j, float(values["gamma"]))
        elif values["gamma"].shape != (j,):
            raise ValueError("gamma must be a positive scalar or have shape (J,)")
        for name, value in values.items():
            value.setflags(write=False)
            object.__setattr__(self, name, value)

# LEVELS / 02 · Delivered unit costs
# src/ek_model/full_solution.py:7-18
# Labor moves between domestic industries, so one exporter wage applies to every industry. Labor does not move internationally.
def levels_objects(primitives: Primitives, log_wages):
    """Exporter competitiveness normalizes separately within each importer/industry."""
    wages = normalized_wages(log_wages)
    log_costs = (log_wages - log_wages[0])[None, :, None] + np.log(primitives.d)
    log_weights = np.log(primitives.T)[None, :, :] - primitives.theta[None, None, :] * log_costs
    log_denominator = row_logsumexp(log_weights)
    shares = np.exp(log_weights - log_denominator[:, None, :])
    prices = primitives.gamma * np.exp(-log_denominator / primitives.theta)
    incomes = wages * primitives.L
    expenditures = primitives.alpha * incomes[:, None]
    cost_of_living = aggregate_price(prices, primitives.alpha)
    return EconomicState(wages, prices, shares, incomes, cost_of_living, expenditures)

# LEVELS / 03 · Trade shares
# src/ek_model/full_solution.py:7-18
# Sum over exporter axis 1, independently within each importer and industry. The (N,J) log denominator broadcasts back across exporters.
def levels_objects(primitives: Primitives, log_wages):
    """Exporter competitiveness normalizes separately within each importer/industry."""
    wages = normalized_wages(log_wages)
    log_costs = (log_wages - log_wages[0])[None, :, None] + np.log(primitives.d)
    log_weights = np.log(primitives.T)[None, :, :] - primitives.theta[None, None, :] * log_costs
    log_denominator = row_logsumexp(log_weights)
    shares = np.exp(log_weights - log_denominator[:, None, :])
    prices = primitives.gamma * np.exp(-log_denominator / primitives.theta)
    incomes = wages * primitives.L
    expenditures = primitives.alpha * incomes[:, None]
    cost_of_living = aggregate_price(prices, primitives.alpha)
    return EconomicState(wages, prices, shares, incomes, cost_of_living, expenditures)

# LEVELS / 04 · Industry prices
# src/ek_model/full_solution.py:7-18
# Each industry's elasticity and fixed gamma determine its price index. Gamma is unchanged by the counterfactual.
def levels_objects(primitives: Primitives, log_wages):
    """Exporter competitiveness normalizes separately within each importer/industry."""
    wages = normalized_wages(log_wages)
    log_costs = (log_wages - log_wages[0])[None, :, None] + np.log(primitives.d)
    log_weights = np.log(primitives.T)[None, :, :] - primitives.theta[None, None, :] * log_costs
    log_denominator = row_logsumexp(log_weights)
    shares = np.exp(log_weights - log_denominator[:, None, :])
    prices = primitives.gamma * np.exp(-log_denominator / primitives.theta)
    incomes = wages * primitives.L
    expenditures = primitives.alpha * incomes[:, None]
    cost_of_living = aggregate_price(prices, primitives.alpha)
    return EconomicState(wages, prices, shares, incomes, cost_of_living, expenditures)

# LEVELS / 05 · Aggregate cost of living
# src/ek_model/model.py:132-136
# Normalized Cobb–Douglas utility gives a weighted geometric mean of industry prices. There are no additional preference constants in C. Real wages are w/C. For J=1, alpha=1 returns the sole industry price exactly.
def aggregate_price(prices: Array, alpha: Array) -> Array:
    """Normalized Cobb–Douglas unit expenditure: no expenditure-share constants."""
    if prices.shape[1] == 1 and np.all(alpha == 1):
        return prices[:, 0].copy()
    return np.exp(np.sum(alpha * np.log(prices), axis=1))

# LEVELS / 06 · Market clearing
# src/ek_model/model.py:117-119
# Aggregate importer spending over all industries to determine each exporter's sales. Balanced trade applies to country totals, not separately to each industry.
def exporter_sales(shares: Array, expenditures: Array) -> Array:
    """Sum importer spending across importers and industries, leaving exporter i."""
    return np.einsum("nij,nj->i", shares, expenditures)

# LEVELS / 07 · Levels entry point
# src/ek_model/full_solution.py:20-29
# The levels route supplies its own shares, prices and incomes. The shared wage loop starts at unit wages and retains the final evaluated state.
def solve_levels(primitives: Primitives, *, damping=DEFAULT_DAMPING,
                 max_iter=MAX_ITER, tol=ITERATION_TOL) -> Equilibrium:
    """Start at unit wages; return economic objects at the final evaluated state."""
    objects, residual, diagnostics = iterate_wages(
        lambda log_wages: levels_objects(primitives, log_wages), primitives.T.shape[0],
        industries=primitives.T.shape[1], damping=damping, max_iter=max_iter, tol=tol,
    )
    return Equilibrium(primitives, objects.wages, objects.prices, objects.shares,
                       objects.incomes, objects.real_wages, residual, diagnostics,
                       objects.cost_of_living, objects.expenditures)

# LEVELS / 08 · Explicit iteration loop
# src/ek_model/iteration.py:39-88
# Every evaluated state is recorded. The loop stops only when the full residual is below the target, the update limit is reached, or the numerical state is invalid. Defaults: damping 0.2, target 1e-13, maximum 10,000 updates.
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

# LEVELS / 09 · Damping & normalization
# src/ek_model/iteration.py:32-36
# Fixed damping 0.2 is a numerical control distinct from demand shares alpha[n,j]. Update wages simultaneously, then normalize country 0.
def damped_wage_step(log_wages, sales_income_ratio, damping):
    """Simultaneous multiplicative update, expressed in normalized log wages."""
    log_w_next = log_wages + damping * np.log(sales_income_ratio)
    log_w_next -= log_w_next[0]
    return log_w_next

# HATS / 01 · Baseline statistics & shock
# src/ek_model/exact_hat.py:8-27
# A converged baseline supplies shares and country incomes. Preferences and elasticities stay fixed. Shocked cost levels must remain admissible in every industry.
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

# HATS / 02 · Counterfactual shares
# src/ek_model/exact_hat.py:29-42
# Reweight each industry's baseline shares. This route never calls levels competitiveness or reads counterfactual levels wages.
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

# HATS / 03 · Industry price changes
# src/ek_model/exact_hat.py:29-42
# The exponent applies to each importer/industry exporter sum. Fixed industry gamma cancels from price ratios.
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

# HATS / 04 · Aggregate price changes
# src/ek_model/exact_hat.py:29-42
# Fixed preferences imply that the ratio of aggregate levels indices is this product of industry price hats. Demand and market clearing are unchanged by utility normalization.
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

# HATS / 05 · Country spending
# src/ek_model/exact_hat.py:29-42
# Fixed labor makes wage hats scale country incomes. Allocate spending using fixed alpha before summing exporter sales over importers and industries.
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

# HATS / 06 · Hat equilibrium solver
# src/ek_model/exact_hat.py:44-55
# The shared wage loop uses independently calculated hat economics. Welfare changes use aggregate cost-of-living hats.
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

# HATS / 07 · Full convergence check
# src/ek_model/iteration.py:39-88
# Every country participates in the residual check, including country 0. Small wage changes alone are not a stopping rule. Limits and invalid numerical states return explicit failure statuses.
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

# CHECK / 01 · Fix the experiment
# src/ek_model/verification.py:15-25
# Three countries, two heterogeneous industries; only industry 0's two bilateral costs fall. Industry 1's equilibrium still responds through wages.
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

# CHECK / 02 · Solve E0 and E1
# src/ek_model/verification.py:95-129
# Both solves start at unit wages. Technologies, labor, elasticities, preferences and gamma stay fixed.
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

# CHECK / 03 · Solve hats from E0
# src/ek_model/verification.py:95-129
# The same shock reaches the independent route through E0's expenditure shares and incomes.
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

# CHECK / 04 · Compare normalized changes
# src/ek_model/verification.py:76-92
# Five comparisons cover 3 wages, 6 industry prices, 3 aggregate prices, 18 bilateral industry share ratios, and 3 welfare changes. All three routes must converge.
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

# CHECK / 05 · Apply both strict bounds
# src/ek_model/verification.py:28-50
# The denominator is the levels-route ratio, not the change minus one. The two error tests are separate, so one cannot compensate for the other.
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

# CHECK / 06 · Record evidence
# src/ek_model/verification.py:95-129
# The builder runs this exact verification routine, embeds its certificate, and hashes the reviewed source. A completed run is not itself a pass.
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

# CHECK / 07 · J=1 regression
# src/ek_model/regression.py:23-60
# Compare 18 objects against immutable data captured before code changes at the accepted PR #4 revision. Separate absolute and relative bounds and convergence apply. There is no duplicate historical solver.
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

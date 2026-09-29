# GENERATED READING COMPANION — not a standalone solver.
# Each excerpt is verbatim production source; comments and ordering are explanatory.
# Imports and other context may be omitted. Repeated functions illustrate different steps.
# SHA-256 source provenance is recorded in manifest.json and verification.json.

# LEVELS / 01 · Primitives & dimensions
# src/ek_model/model.py:25-52
# Positive technology and labor; unit domestic iceberg costs. Rows always denote importers n and columns exporters i.
@dataclass(frozen=True)
class Primitives:
    """Technology, labor and iceberg costs; gamma is a fixed price multiplier."""

    T: Array
    L: Array
    d: Array
    theta: float
    gamma: float = 1.0

    def __post_init__(self):
        T = positive_array(self.T, "T")
        L = positive_array(self.L, "L")
        d = positive_array(self.d, "d")
        if T.ndim != 1 or T.size < 2 or L.shape != T.shape:
            raise ValueError("T and L must have shape (N,), with N >= 2")
        if d.shape != (T.size, T.size):
            raise ValueError("d must have shape (N, N): importer rows, exporter columns")
        if np.any(d < 1) or not np.all(np.diag(d) == 1):
            raise ValueError("iceberg costs must be >= 1 with unit domestic costs")
        for name in ("theta", "gamma"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.ndim != 0 or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive scalar")
            object.__setattr__(self, name, float(value))
        for name, value in (("T", T), ("L", L), ("d", d)):
            value.setflags(write=False)
            object.__setattr__(self, name, value)

# LEVELS / 02 · Delivered unit costs
# src/ek_model/full_solution.py:13-22
# One wage per exporter scales its production cost, then bilateral iceberg costs determine the delivered cost.
def levels_objects(primitives: Primitives, log_free_wages):
    """Delivered unit costs determine importer-row shares and price indices."""
    wages = normalized_wages(log_free_wages)
    log_costs = np.log(wages)[None, :] + np.log(primitives.d)
    log_weights = np.log(primitives.T)[None, :] - primitives.theta * log_costs
    log_denominator = logsumexp(log_weights, axis=1)
    shares = np.exp(log_weights - log_denominator[:, None])
    prices = primitives.gamma * np.exp(-log_denominator / primitives.theta)
    incomes = wages * primitives.L
    return wages, prices, shares, incomes

# LEVELS / 03 · Trade shares
# src/ek_model/full_solution.py:13-22
# Normalize competitiveness within each importer row. Log-sum-exp evaluates the denominator without directly raising costs to large negative powers.
def levels_objects(primitives: Primitives, log_free_wages):
    """Delivered unit costs determine importer-row shares and price indices."""
    wages = normalized_wages(log_free_wages)
    log_costs = np.log(wages)[None, :] + np.log(primitives.d)
    log_weights = np.log(primitives.T)[None, :] - primitives.theta * log_costs
    log_denominator = logsumexp(log_weights, axis=1)
    shares = np.exp(log_weights - log_denominator[:, None])
    prices = primitives.gamma * np.exp(-log_denominator / primitives.theta)
    incomes = wages * primitives.L
    return wages, prices, shares, incomes

# LEVELS / 04 · Price indices
# src/ek_model/full_solution.py:13-22
# The same denominator determines prices. gamma is a fixed positive multiplier and cancels from price changes.
def levels_objects(primitives: Primitives, log_free_wages):
    """Delivered unit costs determine importer-row shares and price indices."""
    wages = normalized_wages(log_free_wages)
    log_costs = np.log(wages)[None, :] + np.log(primitives.d)
    log_weights = np.log(primitives.T)[None, :] - primitives.theta * log_costs
    log_denominator = logsumexp(log_weights, axis=1)
    shares = np.exp(log_weights - log_denominator[:, None])
    prices = primitives.gamma * np.exp(-log_denominator / primitives.theta)
    incomes = wages * primitives.L
    return wages, prices, shares, incomes

# LEVELS / 05 · Market clearing
# src/ek_model/model.py:93-96
# Transpose the share matrix to sum importer spending into each exporter's sales. This returns all N normalized residuals.
def market_clearing(shares: Array, incomes: Array) -> Array:
    """Sum importer spending down rows to obtain each exporter's sales."""
    sales = shares.T @ incomes
    return (sales - incomes) / incomes

# LEVELS / 06 · Equilibrium solver
# src/ek_model/full_solution.py:31-46
# Optimize N−1 log wages from zero, enforcing the numeraire exactly. Successful termination is necessary but the full residual and finite outputs also determine convergence.
def solve_levels(primitives: Primitives) -> Equilibrium:
    """Start from unit wages and retain full, normalized residual diagnostics."""
    result = least_squares(
        levels_residual, np.zeros(primitives.T.size - 1), args=(primitives,),
        ftol=SOLVER_TOL, xtol=SOLVER_TOL, gtol=SOLVER_TOL, max_nfev=MAX_NFEV,
    )
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        wages, prices, shares, incomes = levels_objects(primitives, result.x)
        real_wages = wages / prices
        residual = market_clearing(shares, incomes)
    diagnostics = make_diagnostics(
        result, residual, wages, prices, shares, incomes, real_wages,
    )
    return Equilibrium(
        primitives, wages, prices, shares, incomes, real_wages, residual, diagnostics,
    )

# HATS / 01 · Baseline statistics & shock
# src/ek_model/exact_hat.py:17-35
# Require a valid, converged baseline and positive trade-cost ratios. The resulting level costs must still be admissible iceberg costs.
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

# HATS / 02 · Counterfactual shares
# src/ek_model/exact_hat.py:38-48
# Reweight baseline shares using wage and trade-cost changes. This route does not recompute levels competitiveness from technology.
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

# HATS / 03 · Price changes
# src/ek_model/exact_hat.py:38-48
# The exponent applies to the whole row sum. gamma cancels because it is held fixed across equilibria.
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

# HATS / 04 · Hat-market residual
# src/ek_model/exact_hat.py:51-54
# Labor is fixed, so wage hats also scale incomes. Demand and exporter sales must balance at the new shares.
def hat_residual(log_free_hats, baseline: Equilibrium, d_hat):
    """New sales must equal baseline income multiplied by each wage hat."""
    _, _, shares, incomes = hat_objects(baseline, d_hat, log_free_hats)
    return market_clearing(shares, incomes)[1:]

# HATS / 05 · Hat equilibrium solver
# src/ek_model/exact_hat.py:57-73
# Start from unit wage hats. The resulting changes are independently determined by market clearing, not copied from the levels counterfactual.
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

# HATS / 06 · Full convergence check
# src/ek_model/model.py:99-110
# Dropping one equation inside the optimizer does not permit dropping its residual from the final convergence check.
def make_diagnostics(result, residual: Array, *outputs: Array) -> Diagnostics:
    """Optimizer success alone never certifies economic convergence."""
    finite = all(np.all(np.isfinite(x)) for x in (residual, *outputs))
    norm = float(np.max(np.abs(residual))) if finite else float("inf")
    return Diagnostics(
        converged=bool(result.success and finite and norm < RESIDUAL_TOL),
        solver_success=bool(result.success),
        status=int(result.status),
        message=str(result.message),
        nfev=int(result.nfev),
        residual_norm=norm,
    )

# CHECK / 01 · Fix the experiment
# src/ek_model/verification.py:16-24
# The proportional cut is symmetric although initial bilateral costs are asymmetric. All other primitives are held fixed.
def fixture():
    """Asymmetric three-country economy; a symmetric proportional bilateral cut."""
    baseline = Primitives(
        T=[1.0, 1.3, 0.8], L=[1.0, 1.2, 0.9],
        d=[[1.0, 1.4, 1.8], [1.3, 1.0, 1.6], [1.7, 1.5, 1.0]], theta=4.0,
    )
    d_hat = np.ones((3, 3))
    d_hat[0, 1] = d_hat[1, 0] = 0.9
    return baseline, d_hat

# CHECK / 02 · Solve E0 and E1
# src/ek_model/verification.py:82-109
# Both levels runs begin from unit wages. Their difference is entirely the pre-specified trade-cost shock.
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

# CHECK / 03 · Solve hats from E0
# src/ek_model/verification.py:82-109
# The same shock reaches the independent route through E0's expenditure shares and incomes.
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

# CHECK / 04 · Compare normalized changes
# src/ek_model/verification.py:64-79
# Compare wages, prices, all bilateral share ratios, and real wages. The final decision also requires all three convergence flags.
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

# CHECK / 05 · Apply both strict bounds
# src/ek_model/verification.py:27-49
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
# src/ek_model/verification.py:82-109
# The builder runs this exact verification routine, embeds its certificate, and hashes the reviewed source. A completed run is not itself a pass.
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

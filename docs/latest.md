# Latest-state record: multi-industry Eaton–Kortum candidate

**Candidate, not formally accepted.** This branch implements the approved [Issue #5 Plan](https://github.com/AkariOno/pdcu-ek-multisector/issues/5) and records its automated Check evidence. Human review and learner explanation/quiz completion remain pending. The implementation PR is a draft; Issue #5 stays open. This document does not authorize a merge or certify completion of the multi-industry PDCU cycle.

The formally accepted Latest State remains the one-industry model on `main`, merged in [PR #2](https://github.com/AkariOno/pdcu-ek-multisector/pull/2), with acceptance documentation reconciled in [PR #4](https://github.com/AkariOno/pdcu-ek-multisector/pull/4). Implementation started from PR #4's accepted commit `9eb5b52bcefc6a5e4055acc5dfb2a6f6a184596b`. Earlier implementation and completed review/understanding evidence remain in Git and PR history, without duplicate historical production solvers.

## Economic contract

There are N ≥ 2 countries, J ≥ 1 industries and one factor, labor. Zero-based `n` is importer, `i` exporter and `j` industry. Labor is fully employed, mobile between industries within each country and immobile internationally. Each country has one wage. Trade is balanced at the country level; each industry's trade need not balance separately.

| Object | Shape | Meaning |
|---|---|---|
| `T` | `(N,J)` | Exporter technology |
| `L` | `(N,)` | Country labor endowment |
| `theta`, `gamma` | `(J,)` | Industry elasticity and price-unit multiplier |
| `d`, `pi` | `(N,N,J)` | Iceberg costs and conditional industry shares |
| `alpha`, `P`, `X` | `(N,J)` | Fixed final expenditure shares, industry prices and expenditure |
| `w`, `Y`, `C`, real wages | `(N,)` | Wages, country income, cost of living and welfare per worker |

Inputs must be finite and strictly positive. Demand-share rows sum to one within absolute tolerance `1e-12`, with no silent renormalization. Iceberg costs are at least one, and domestic costs equal one exactly in every industry. A positive scalar `gamma` broadcasts across industries; otherwise it has shape `(J,)`. All other industry axes must be explicit, including J=1.

Technology, labor, elasticities, demand shares and industry multipliers remain fixed across the trade-cost counterfactual. No intermediate inputs, tariffs, transfers, deficits or international labor mobility enter this cycle. Estimation, technology/preference shocks, optimizer fallbacks and performance optimization are outside scope.

The levels equations are:

```text
U[n] = product_j (c[n,j] / alpha[n,j])**alpha[n,j]
B[n,i,j] = T[i,j] * (w[i] * d[n,i,j])**(-theta[j])
S[n,j] = sum_i B[n,i,j]
pi[n,i,j] = B[n,i,j] / S[n,j]
P[n,j] = gamma[j] * S[n,j]**(-1/theta[j])
C[n] = product_j P[n,j]**alpha[n,j]
Y[n] = w[n] * L[n]
X[n,j] = alpha[n,j] * Y[n]
sales[i] = sum_n sum_j pi[n,i,j] * X[n,j]
sales[i] = Y[i]
real_wage[n] = w[n] / C[n]
```

The normalized Cobb–Douglas utility is the user's revised and approved choice; it supersedes ordinary `product c**alpha`. Demand still allocates `alpha[n,j]` of income to each industry. Substituting `c[n,j]=alpha[n,j]*Y[n]/P[n,j]` gives `U[n]=Y[n]/C[n]`, establishing the stated unit-expenditure index. The previous index `product(P/alpha)**alpha` differs by a fixed country-specific preference constant. Welfare levels change under the revised units; counterfactual welfare ratios, spending shares and market clearing remain unchanged when preferences are fixed. Industry `gamma` is unchanged.

## Independent exact hats

Baseline shares and country incomes are sufficient statistics for the trade-cost-only hat equations, together with fixed `theta`, `alpha` and the shock. Retained primitives validate shocked costs; technology competitiveness and the levels solver are not called by the hat route.

```text
a[n,i,j] = pi0[n,i,j] * (w_hat[i] * d_hat[n,i,j])**(-theta[j])
A[n,j] = sum_i a[n,i,j]
pi1[n,i,j] = a[n,i,j] / A[n,j]
P_hat[n,j] = A[n,j]**(-1/theta[j])
C_hat[n] = product_j P_hat[n,j]**alpha[n,j]
Y1[n] = Y0[n] * w_hat[n]
X1[n,j] = alpha[n,j] * Y1[n]
sales1[i] = sum_n sum_j pi1[n,i,j] * X1[n,j]
sales1[i] = Y1[i]
real_wage_hat[n] = w_hat[n] / C_hat[n]
```

The exponent in industry price hats applies to the sum over exporters. Fixed `gamma[j]` cancels from these ratios. Both routes use country 0 as numeraire: `w[0]=w_hat[0]=1`.

## Interfaces and wage iteration

```python
Primitives(T, L, d, theta, alpha, gamma=1.0)
solve_levels(primitives, *, damping=0.2, max_iter=10_000, tol=1e-13)
solve_exact_hat(baseline, d_hat, *, damping=0.2, max_iter=10_000, tol=1e-13)
```

The existing production modules are generalized in place. Levels outputs retain wages, industry prices, shares, incomes, real wages, full residuals and diagnostics; they add `cost_of_living` and industry `expenditures`. Hats retain wage hats, industry price hats, counterfactual shares/incomes, real-wage hats, residuals and diagnostics; they add `cost_of_living_hats` and counterfactual `expenditures`.

Both solvers start from unit wages or hats and share only the explicit wage-adjustment mechanics:

```text
q[i] = sales[i] / Y[i]
r[i] = (sales[i] - Y[i]) / Y[i]
log_w_next[i] = log_w[i] + damping * log(q[i])
log_w_next[:] -= log_w_next[0]
```

Evaluate every economic object, test the full infinity norm including country 0, update all wages simultaneously, normalize and reevaluate. Damping is fixed throughout each solve, with no backtracking or fallback. NumPy exporter-axis max shifting stabilizes log-sum-exp, and industry prices aggregate in log space. J=1 with `alpha=1` returns the sole price exactly.

Controls require finite `0 < damping <= 1`, finite positive tolerance and a positive integer update limit. Convergence requires finite positive economic objects and a full residual norm **strictly below** `tol`. Tiny wage changes alone do not establish convergence. Invalid numerical states and exhausted limits return explicit `numerical_failure` or `max_iterations` statuses; success is `converged`. Exceptions retain industry dimensions in failed outputs.

Diagnostics contain `converged`, status/message, `niter` (updates), `nfev` (evaluations), full residual norm, damping, tolerance, update limit and history. Evaluation counts include the initial state. History contains wages, every country's residual and norm for initial and final evaluated states. Returned economic objects belong to the final evaluated wages.

## Fixture and pre-specified Check

```python
T = [[1.0, 0.9], [1.3, 1.1], [0.8, 1.5]]
L = [1.0, 1.2, 0.9]
theta = [4.0, 6.0]
gamma = [1.0, 1.0]
alpha = [[0.60, 0.40], [0.35, 0.65], [0.50, 0.50]]
d[:, :, 0] = [[1.0, 1.4, 1.8], [1.3, 1.0, 1.6], [1.7, 1.5, 1.0]]
d[:, :, 1] = [[1.0, 1.7, 1.4], [1.5, 1.0, 1.8], [1.6, 1.4, 1.0]]
```

All trade-cost hats are one except `d_hat[0,1,0]=d_hat[1,0,0]=0.9`. The shocked costs are `1.26` and `1.17`, so every cost restriction is preserved. Industry 1 costs stay fixed, while shared wages can change its shares/prices and country 2's welfare.

Independently solve E0, shocked levels E1 and exact hats from E0, all with unit initial values. One verification routine serves pytest and certificate generation. It compares levels ratios with hats for wages, industry prices, aggregate cost of living, every bilateral industry share ratio and real wages. Share comparison is `(pi1/pi0)` between routes.

For every object, maximum absolute error and maximum elementwise relative error must **separately and strictly** be below `1e-9`. The relative denominator is the absolute levels-route ratio. All comparisons must be finite, all three solves converged, and all full residual norms strictly below `1e-11`; default solves additionally target `1e-13`. A combined `allclose` rule is not used for acceptance.

The generated Python 3.12.14 / NumPy 2.5.3 certificate records:

| Solve | Updates | Evaluations | Full residual norm |
|---|---:|---:|---:|
| E0 | 41 | 42 | 5.3974921964810676e-14 |
| E1 | 41 | 42 | 8.2655820161500400e-14 |
| Exact hats | 38 | 39 | 8.3148552323416900e-14 |

| Ratio object | Max absolute error | Max relative error |
|---|---:|---:|
| Wages | 4.440892098500626e-16 | 4.449319957590492e-16 |
| Industry prices | 2.220446049250313e-16 | 2.2235377419328625e-16 |
| Cost of living | 2.220446049250313e-16 | 2.2224564774844085e-16 |
| Industry trade shares | 1.887379141862766e-15 | 1.9032017626744743e-15 |
| Real wages | 4.440892098500626e-16 | 4.445295114559421e-16 |

## J=1 compatibility

Before production changes, baseline E0/E1/hats were captured from accepted commit `9eb5b52bcefc6a5e4055acc5dfb2a6f6a184596b`, after checking production bytes against that Git revision. The immutable [reference data](../src/ek_model/data/one_industry_reference.json) stores that source commit, SHA-256 hashes of its production sources, the original fixture/shock, environment and full equilibrium arrays. Its SHA-256 is `a0e1c68038dac764d51acefada95b1f60580ca1fc8fb930b1a9f1d562a163ae9`; regression rejects a changed reference.

The generalized solver uses explicit J=1 axes, `alpha[:,0]=1`, `theta=[4]`, `gamma=[1]` and the original costs/10% bilateral shock. Industry axes are squeezed only for reference comparisons. All 18 comparisons across E0, E1 and hats pass the same separate strict `1e-9` error bounds, and all three solves satisfy convergence requirements. Maximum absolute and relative errors are **zero in the certificate environment**. Aggregate cost of living equals the sole industry price exactly for all three routes. This is reference data, not a second historical production solver.

## Reproduction and provenance

Use Python 3.12 in a fresh virtual environment:

```sh
python -m pip install -r requirements-repro.txt
python -m pip install --no-deps -e .
python -c "import importlib.util; assert importlib.util.find_spec('scipy') is None"
python -m pytest -q
python scripts/build_model_viewer.py
python scripts/build_model_viewer.py --check
```

The clean installation and all 109 tests pass without SciPy. Tests cover direct equations and aggregate budgets, normalized utility, gamma, all-country market clearing, exporter-axis shares, fixed damping `0.1/0.2/0.3`, no shock, independent hats, invalid primitives/controls/shocks, update limits, nonfinite/nonpositive states, strict acceptance, J=1 integrity, failed builder evidence, stale source mappings and byte-identical repeated builds. [Validation record](validation.md) describes the checks actually run. Windows/Linux CI runs the same suite and freshness check; actual workflow outcomes are recorded in the linked implementation PR.

The builder reruns both verification routes and the J=1 regression; failures generate a FAIL certificate and exit nonzero. The [certificate](../viewer/verification.json) includes fixture, shock, five comparisons, tolerances, diagnostics, histories, Python/NumPy versions and J=1 provenance/evidence. The [manifest](../viewer/manifest.json) contains SHA-256 hashes of production sources, frozen data, builder inputs and generated artifacts. Freshness checks bind annotations and evidence to current sources, and rerun numerical verification. Identical rebuild bytes are required within the same environment; platform-dependent floating-point bytes need not match a different machine's certificate.

## Understand artifacts and remaining acceptance

The standalone [viewer](../viewer/model_viewer.html) embeds all scripts, styles, code and data with no external resources. It shows economic equations and two-route diagram; expandable levels/hats/verification code trees (9/7/7 steps); highlighted verbatim production excerpts with source locations; all five comparisons; per-solve residual plots and expandable wage traces; the immutable J=1 evidence; and eight understanding questions. [Annotated code](../viewer/model_annotated.py) is a reader-oriented arrangement of literal excerpts and explanatory comments, not a parallel executable implementation.

The quiz mixes five objective multiple-choice/code-selection questions with three written responses and nine self-assessment rubric items. It covers indexing, equations, aggregate closure and damping, numeraire, hats, equivalence/J=1, failure diagnosis and cross-industry welfare effects. Explanations and rubrics are revealed after submission. Answers remain session-only; reset/reload clears them. Automated quiz interaction tests do not certify learner understanding.

The economic specification was approved by the user. Implementation, automated checks and educational artifact verification are recorded as evidence in the draft PR and open Issue #5. **Human implementation review and the learner's explanation/eight-question quiz remain unchecked.** Formal acceptance and merge require the user's subsequent authorization.

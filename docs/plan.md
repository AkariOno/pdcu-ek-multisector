# Establish the one-industry Eaton–Kortum PDCU baseline

## Summary and provenance

Create the complete, tested one-industry baseline required by the later multi-industry tutorial. The starting checkout contains documentation only.

Provenance: https://github.com/yutawatabe/pdcu-ek-multisector/issues/1

## Model and numerical contract

- Python with NumPy and SciPy; follow the README package structure for model definitions, levels solver, and separate exact-hat solver.
- Countries n (importer) and i (exporter), one industry, one factor (labor). Balanced trade; no intermediates, tariffs, transfers, or deficits.
- Arrays T, L, wages, and prices have shape (N,); trade costs and shares have shape (N,N), with importer rows and exporter columns.
- Trade shares: pi[n,i] = T[i] (w[i] d[n,i])^(-theta) / sum_k T[k] (w[k] d[n,k])^(-theta).
- Prices: P[n] = gamma [sum_i T[i] (w[i] d[n,i])^(-theta)]^(-1/theta).
- Market clearing: w[i] L[i] = sum_n pi[n,i] w[n] L[n].
- gamma is a fixed positive multiplier, default 1. Technology, labor, elasticity, and gamma remain fixed in the counterfactual.
- Validate dimensions, finite inputs, positive technology/labor/elasticity, and iceberg costs at least one with unit domestic costs.
- Deterministic fixture:
  ```python
  T = [1.0, 1.3, 0.8]
  L = [1.0, 1.2, 0.9]
  d = [[1.0, 1.4, 1.8],
       [1.3, 1.0, 1.6],
       [1.7, 1.5, 1.0]]
  theta = 4.0
  gamma = 1.0
  ```
  Multiply only d[0,1] and d[1,0] by 0.9.
- Solve N-1 log wages with scipy.optimize.least_squares, fixing w[0]=1. Use log-sum-exp, solver tolerances 1e-13, and at most 2,000 function evaluations.
- The normalized residual is (sales[i] - income[i]) / income[i]. Solve countries 1 through N-1, but report the infinity norm over ALL countries. Convergence requires successful solver termination, finite results, and full residual norm below 1e-11.
- Expose solve_levels(primitives) and solve_exact_hat(baseline, d_hat). Return wages/wage hats, prices/price hats, counterfactual shares, real wages/real-wage hats, and diagnostics: status/message, evaluation count, and full residual norm.
- Exact hats use baseline shares and incomes without calling the levels solver:
  ```text
  a[n,i] = pi0[n,i] * (w_hat[i] * d_hat[n,i])**(-theta)
  pi1[n,i] = a[n,i] / sum_i a[n,i]
  P_hat[n] = (sum_i a[n,i])**(-1/theta)
  income1[i] = income0[i] * w_hat[i]
  ```
  Fix w_hat[0]=1 and use the same market-clearing residual and convergence rules.

## Pre-specified Check

1. Solve baseline E0 and shocked levels E1 from unit-wage initial guesses.
2. Independently solve hats from E0, starting from unit wage hats.
3. Compare multiplicative changes in wages, prices, bilateral shares, and real wages: levels ratios versus hat-route results. For shares compare pi1/pi0 between routes.
4. Report maximum absolute and maximum elementwise relative errors, using the absolute levels-route ratio as denominator.
5. Accept only when all three solves converge, all comparisons are finite, and BOTH errors for every object are strictly below 1e-9. Do not use the combined absolute/relative rule in allclose.
6. Add focused tests for share row sums, numeraire, all-country market clearing, no shock, invalid inputs, and rejection of failed/nonfinite solver results.
7. Use one verification routine for pytest and certificate generation.

## Understand artifacts and acceptance

- Standalone offline HTML using the README layout: model/equations and two-route diagram; expandable levels, hats, and verification code trees; generated certificate and quiz.
- Algorithm-step selection reveals and highlights corresponding code. Consistent importer/exporter labels. Embed all scripts, styles, code, and data; no network required.
- Reader-oriented annotated code from verbatim production excerpts, with source locations and explanatory comments.
- Reproducible builder and manifest with SHA-256 hashes of production sources, builder inputs, and generated artifacts.
- Generate the certificate from an actual verification run, including fixture, shock, diagnostics, errors, tolerances, and dependency versions. Failed checks produce a failure result and nonzero exit status.
- Eight quiz questions: indexing, equation mapping, closure, numeraire, hat inputs, equivalence logic, failure diagnosis, and economic interpretation. Mix multiple choice, code selection, and written answers; reveal explanations and self-assessment rubrics. Responses remain session-only.
- Verify offline interactions and identical rebuild output in the same environment. CI runs numerical tests and artifact freshness checks.
- Rewrite docs/latest.md for the accepted one-industry model, reproduction commands, diagnostics, and provenance. Correct README baseline-status claims to match delivered evidence.
- Record implementation and checks in one linked PR. Human review and learner understanding remain pending until actually completed.

## Acceptance checklist

- [ ] Economic specification and implementation agree.
- [ ] All three solves converge and all four comparisons meet both strict tolerances.
- [ ] Focused numerical and failure-path tests pass.
- [ ] Viewer builds reproducibly; provenance and artifact freshness checks pass.
- [ ] Offline viewer interactions and all eight quiz questions are verified.
- [ ] docs/latest.md and README accurately describe the baseline and review status.
- [ ] Human review completed.
- [ ] Learner explanation and understanding quiz completed.

## Deliberate non-goals

Multiple industries, intermediates, tariffs, transfers, deficits, estimation, and performance optimization remain outside this cycle.

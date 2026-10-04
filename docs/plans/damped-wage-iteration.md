# Replace SciPy with transparent damped wage iteration

## Goal and new issue

Create a new issue in `AkariOno/pdcu-ek-multisector` titled **“Plan: replace SciPy solvers with transparent damped wage iteration.”** Use this plan as its body, referencing [baseline issue #1](https://github.com/AkariOno/pdcu-ek-multisector/issues/1) and [draft PR #2](https://github.com/AkariOno/pdcu-ek-multisector/pull/2).

Revise PR #2 in place so the first accepted baseline uses NumPy and explicit wage iteration. Preserve the economics, fixture, shock, normalization, and equivalence test.

## Iteration algorithm

Use the chosen **multiplicative update with fixed damping**, independently for levels and exact hats.

Starting from unit wages or wage hats, evaluate shares, incomes, exporter sales, and all-country residuals:

```text
income[i] = w[i] * L[i]                 # levels
income[i] = income0[i] * w_hat[i]       # exact hats

sales[i] = sum_n pi[n,i] * income[n]
q[i] = sales[i] / income[i]
r[i] = (sales[i] - income[i]) / income[i]
```

If the full residual norm satisfies the stopping rule, return the current equilibrium. Otherwise update simultaneously:

```text
log_w_next[i] = log_w[i] + alpha * log(q[i])
log_w_next[:] -= log_w_next[0]
w_next[:] = exp(log_w_next[:])
```

Apply the identical rule to wage hats. This is equivalent to multiplying wages by `q**alpha` and normalizing country 0 after every update.

- Defaults: `alpha=0.2`, maximum **10,000 updates**, and full residual norm strictly below **`1e-13`**.
- Damping stays constant throughout each solve. Expose it as a parameter; do not introduce backtracking or optimizer fallback.
- Recompute the economic objects after each update. Return the objects belonging to the final evaluated wages.
- Stop with explicit failure on the iteration limit or nonfinite/nonpositive numerical state. Small wage changes alone never establish convergence.
- Replace SciPy’s log-sum-exp with a NumPy implementation using row maxima before exponentiation.

A NumPy-only feasibility check on the existing fixture converged in 40–43 updates; the largest comparison error was approximately `1.2e-14`. These results inform the defaults; implementation must generate fresh evidence.

## Interfaces, dependencies, and teaching artifacts

Keep the equilibrium output objects and expose:

```python
solve_levels(primitives, *, damping=0.2, max_iter=10_000, tol=1e-13)
solve_exact_hat(baseline, d_hat, *, damping=0.2, max_iter=10_000, tol=1e-13)
```

Validate finite `0 < damping <= 1`, positive tolerance, and a positive integer iteration limit.

- Diagnostics retain convergence, message, evaluation count, and full residual norm. Add update count, damping, and iteration history; replace optimizer-specific success and integer status with `converged`, `max_iterations`, or `numerical_failure` status.
- Count the initial equilibrium evaluation. History records evaluated wages, full residual vectors, and norms, including the initial and successful final states.
- Share the explicit wage-update mechanics while keeping levels and exact-hat economic calculations separate. The hat route must not call the levels solver.
- Remove SciPy imports and dependency declarations throughout runtime, verification, and build tooling. Record Python and NumPy versions in the regenerated certificate.
- Update the viewer’s source mappings and explanations to show the actual loop, damping, normalization, and stopping rule. Add per-solve residual histories and expandable wage traces.
- Retain eight quiz questions, revising their explanations to cover the wage-update direction, damping, and failure diagnosis.
- Regenerate annotated code, certificate, HTML, and hashes. Update `docs/latest.md`, README, and validation records; preserve the previous issue as historical evidence.

## Check and acceptance

Keep the original three-country fixture and symmetric proportional 10% bilateral cost cut.

- Independently solve `E0`, `E1`, and exact hats from unit initial values.
- Require all three routes to converge and their full residual norms to remain below the existing acceptance bound of `1e-11`.
- Compare wage, price, bilateral-share, and real-wage ratios. Every maximum absolute and relative error must separately be strictly below `1e-9`.
- Test the update equation and normalization directly, no-shock behavior, invalid controls, iteration-limit failure, numerical failure, and inclusion of country 0’s residual.
- Verify that damping values `0.1`, `0.2`, and `0.3` yield equivalent fixture equilibria without requiring monotonically decreasing residuals.
- Run tests, package installation, builder, and freshness checks in a clean environment where SciPy is absent. Retain Windows/Linux CI and byte-identical rebuild checks.
- Verify viewer histories, source highlighting, quiz interactions, and session-only answers. Failed verification must produce failure evidence and a nonzero exit status.

Human review and learner understanding remain unchecked until completed. Multi-industry economics, intermediates, tariffs, transfers, deficits, estimation, and performance optimization remain outside this cycle.

## Acceptance checklist

- [ ] NumPy-only runtime, installation, tests, and builder pass without SciPy.
- [ ] Both routes use the explicit fixed-damping update and report iterations and histories.
- [ ] All three solves converge and every comparison meets both strict bounds.
- [ ] Numerical failure, iteration limit, invalid controls, and normalization tests pass.
- [ ] Viewer histories, source mappings, quiz, reproducible artifacts, and documentation verified.
- [ ] Windows and Linux CI pass.
- [ ] Human review completed.
- [ ] Learner explanation and understanding quiz completed.

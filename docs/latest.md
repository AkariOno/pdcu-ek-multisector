# Latest state: one-industry Eaton–Kortum baseline

This is the Single Source of Truth for the implemented one-industry model. Numerical checks and generated evidence are complete; human acceptance and learner understanding remain pending in the linked review. The implementation branch does not itself establish that `main` has been reviewed or merged.

Current solver plan: [issue #3](https://github.com/AkariOno/pdcu-ek-multisector/issues/3) and [damped wage iteration specification](plans/damped-wage-iteration.md). The original baseline plan remains in [issue #1](https://github.com/AkariOno/pdcu-ek-multisector/issues/1) and [its historical document](plan.md), with provenance in [upstream issue #1](https://github.com/yutawatabe/pdcu-ek-multisector/issues/1). Implementation and review continue in [draft PR #2](https://github.com/AkariOno/pdcu-ek-multisector/pull/2). See the [validation record](validation.md). The later multi-industry exercise generalizes these production modules in place.

## Economic contract

There are N ≥ 2 countries, one industry, and one factor, labor. `n` is the importer (row); `i` is the exporter (column). `T`, `L`, wages, prices and incomes have shape `(N,)`; trade costs and expenditure shares have shape `(N,N)`. Technology and labor are strictly positive; iceberg costs are finite, at least one, and exactly one domestically. Elasticity `theta` and price multiplier `gamma` are positive scalars.

For positive wages:

```text
weights[n,i] = T[i] * (w[i] * d[n,i])**(-theta)
pi[n,i] = weights[n,i] / sum_k weights[n,k]
P[n] = gamma * (sum_i weights[n,i])**(-1/theta)
income[i] = w[i] * L[i]
sales[i] = sum_n pi[n,i] * income[n]
sales[i] = income[i]
real_wage[n] = w[n] / P[n]
```

Balanced trade means expenditure equals labor income. There are no intermediates, tariffs, transfers, deficits, estimation, or performance optimizations. `gamma=1` is a fixed price-unit convention in the fixture; this baseline does not introduce a separate CES elasticity to derive it. Changing a fixed gamma rescales price levels and real-wage levels, but cancels from counterfactual ratios.

## Solution routes and public interface

```python
from ek_model import Primitives, solve_levels, solve_exact_hat
from ek_model.verification import fixture

primitives, d_hat = fixture()
e0 = solve_levels(primitives)
e1 = solve_levels(Primitives(
    primitives.T, primitives.L, primitives.d * d_hat,
    primitives.theta, primitives.gamma,
))
hats = solve_exact_hat(e0, d_hat)
# Optional controls on either route:
# solve_levels(primitives, damping=0.1, max_iter=10_000, tol=1e-13)
```

`Primitives(T, L, d, theta, gamma=1.0)` validates and copies its inputs into read-only float64 arrays. Both solvers start from unit wages or unit wage hats and use only NumPy. They expose keyword controls `damping=0.2`, `max_iter=10_000`, and `tol=1e-13`. Damping and tolerance must be finite positive scalars, damping must not exceed one, and the update limit must be a positive integer. Damping remains fixed throughout the solve.

Each iteration computes current economic objects, all exporter sales, and all N residuals. If the full normalized residual norm is strictly below `tol`, return the current evaluated equilibrium. Otherwise update all log wages simultaneously and normalize country 0:

```text
q[i] = sales[i] / income[i]
log_w_next[i] = log_w[i] + damping * log(q[i])
log_w_next[:] -= log_w_next[0]
w_next[:] = exp(log_w_next[:])
```

Equivalently, multiply wages by `q**damping` and divide by country 0's new wage. Before normalization, sales exceeding income raise a wage; sales below income lower it. A fractional log adjustment dampens that change. The loop fixes `w[0]=1` or `w_hat[0]=1` exactly after every update and recomputes the economic objects before testing again. Small wage changes alone do not imply convergence. The loop does not require the residual to decrease monotonically and has no adaptive damping or optimizer fallback.

`solve_levels` returns an `Equilibrium` containing primitives, wages, prices, shares, incomes, real wages, the full residual vector, and diagnostics. `solve_exact_hat` returns a `HatEquilibrium` containing wage hats, price hats, counterfactual share **levels**, counterfactual incomes, real-wage hats, the full residual vector, and diagnostics.

Exact-hat equations use only baseline shares, incomes, theta, and trade-cost ratios; the retained primitives also validate the shocked iceberg costs. Technology, labor, theta, and gamma remain fixed:

```text
a[n,i] = pi0[n,i] * (w_hat[i] * d_hat[n,i])**(-theta)
pi1[n,i] = a[n,i] / sum_k a[n,k]
P_hat[n] = (sum_i a[n,i])**(-1/theta)
income1[i] = income0[i] * w_hat[i]
real_wage_hat[n] = w_hat[n] / P_hat[n]
```

The exponent applies to the complete row sum. The hat solver does not call the levels solver. Both routes compute share denominators with log-sum-exp.

Diagnostics expose `converged`, string `status`, `message`, `niter`, `nfev`, `residual_norm`, `damping`, `tolerance`, `max_iter`, and immutable `history`. Status is `converged`, `max_iterations`, or `numerical_failure`. This replaces the old optimizer-specific success flag and integer status. The residual vector is `(sales-income)/income`; its norm is the maximum absolute value over **all countries**. Convergence requires finite positive economic objects and a full norm strictly below the requested tolerance, default `1e-13`. Verification separately enforces its original `1e-11` acceptance bound, even if a caller selects a looser solve tolerance.

`niter` counts simultaneous wage updates. `nfev` counts evaluated economic states, including the initial unit-wage state: `nfev = niter + 1`. Each `IterationRecord` contains the update number, wages (or wage hats), the full residual vector, and its norm. History includes initial and terminal evaluated states; snapshots cannot change when later arrays change. The returned shares, prices, and incomes belong to the last evaluated wages, including on failure. Invalid inputs raise `ValueError`. Numerical exceptions or nonfinite/nonpositive evaluated states produce `numerical_failure`; hitting the limit produces `max_iterations`. Both have `converged=False`. Failed history values are recorded as JSON null in certificates, allowing failure evidence to be saved without NaN or Infinity.

## Pre-specified experiment and evidence

```python
T = [1.0, 1.3, 0.8]
L = [1.0, 1.2, 0.9]
d = [[1.0, 1.4, 1.8], [1.3, 1.0, 1.6], [1.7, 1.5, 1.0]]
theta = 4.0
gamma = 1.0
```

Only `d[0,1]` and `d[1,0]` are multiplied by `0.9`. This is a symmetric proportional cut, not equal initial bilateral cost levels. Two fresh levels solves yield `E0` and `E1`; the independent hat solve starts from `E0`.

For wages, prices, all nine bilateral shares, and real wages, the reference is the levels-route multiplicative ratio. The candidate is the hat result (or `hat_result.shares / E0.shares`). Compare:

```text
max_absolute_error = max(abs(candidate - reference))
max_relative_error = max(abs(candidate - reference) / abs(reference))
```

Both values must be finite and **strictly below `1e-9`**, separately for every object. All three solves must converge. Ratios, rather than ratios minus one, avoid an undefined relative error when a normalized wage does not change. A zero reference is rejected rather than patched with an arbitrary denominator floor.

The machine-readable [verification certificate](../viewer/verification.json), schema version 2, records the actual errors, all iteration diagnostics and histories, exact fixture, shock, thresholds, Python/NumPy versions, and source hashes. It is generated by the same acceptance routine used in pytest. The reference run uses 41 / 43 / 40 wage updates for E0 / E1 / hats, with residual norms below `1e-13`. Maximum comparison errors are below `1.3e-14`; use the certificate for exact values.

## Reproduction and provenance

Use Python 3.12. From the repository root:

```sh
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# POSIX shells: source .venv/bin/activate
python -m pip install -r requirements-repro.txt
python -m pip install --no-deps -e .
python -c "import importlib.util; assert importlib.util.find_spec('scipy') is None"
python -m pytest -q
python scripts/build_model_viewer.py --check
```

Regenerate after editing production code, quiz, step descriptions, template, or builder:

```sh
python scripts/build_model_viewer.py
python scripts/build_model_viewer.py --check
```

The [standalone viewer](../viewer/model_viewer.html) opens directly from disk without a server or network. Its hierarchy is model → expandable code routes → certificate → quiz, following the README specification. The external joint-BVP example was not supplied; no claim of matching that unseen file is made.

The builder creates the HTML, annotated Python reading companion, certificate, and manifest. Annotation consists of explanatory comments around verbatim AST-extracted source spans, including original locations; imports and context may be omitted and some functions repeat for different steps, so the companion is not a standalone solver. The manifest records SHA-256 for all production modules, builder inputs, and generated artifacts (excluding its own hash). There are no timestamps or absolute machine paths in generated artifacts.

Repeated builds in the same environment must be byte-identical. `--check` verifies recorded source hashes and reconstructs all rendered artifacts from the saved certificate, then reruns numerical acceptance. It does not demand identical floating-point bytes across operating systems or library versions. CI runs these checks on Windows and Linux. The manifest detects stale or modified files; it is provenance evidence, not a cryptographic signature of human approval.

## Understand and review

The viewer supplies three routes (eight levels steps, six hats steps, six verification steps), highlighted source lines, and keyboard-accessible selection. It shows the actual wage loop and update formula, per-solve residual plots, and expandable tables of every wage and full residual snapshot. The plot uses a logarithmic norm axis; a display floor for zero values does not affect acceptance.

Eight quiz questions cover indexing, equation mapping, closure and damped wage adjustment, numeraire, hat inputs, two-route logic, failure diagnosis, and interpretation. Five have objective answers; three written answers use self-assessment rubrics. Answers are page-session-only; reset or reload clears them. No answer storage or transmission occurs.

Acceptance requires a human to review the equations and evidence, close the viewer, explain the solution process, and complete the quiz. These actions must be recorded in the PR; passing automated tests does not mark them complete. After acceptance and merge, this document remains the one-industry Single Source of Truth until the next accepted PDCU cycle rewrites it.

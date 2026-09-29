# Latest state: one-industry Eaton–Kortum baseline

This is the Single Source of Truth for the implemented one-industry model. Numerical checks and generated evidence are complete; human acceptance and learner understanding remain pending in the linked review. The implementation branch does not itself establish that `main` has been reviewed or merged.

Plan: [fork issue #1](https://github.com/AkariOno/pdcu-ek-multisector/issues/1), with provenance in [upstream issue #1](https://github.com/yutawatabe/pdcu-ek-multisector/issues/1). The repository also retains [the cycle plan](plan.md). The subsequent multi-industry exercise generalizes these production modules in place; it must not introduce parallel historical implementations.

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
```

`Primitives(T, L, d, theta, gamma=1.0)` validates and copies its inputs into read-only float64 arrays. Both solvers start from zero log unknowns, impose `w[0]=1` or `w_hat[0]=1` exactly, and use SciPy `least_squares` with `ftol=xtol=gtol=1e-13` and `max_nfev=2000`. The optimizer uses market equations for countries `1…N−1`. The redundant country-0 equation remains part of the final diagnostic.

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

Diagnostics expose `converged`, `solver_success`, integer `status`, `message`, `nfev`, and `residual_norm`. The residual vector is `(sales-income)/income`; its reported norm is the maximum absolute value over **all countries**. Convergence requires optimizer success, finite outputs, and norm strictly below `1e-11`. `nfev` is the optimizer's evaluation count, not an iteration count; SciPy does not include finite-difference Jacobian calls in it. Invalid inputs raise `ValueError`. A failed optimizer result is returned with `converged=False`; callers must check it. The verification runner converts supported numerical failures into a failed certificate, and a failed builder exits nonzero.

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

The machine-readable [verification certificate](../viewer/verification.json) records the actual errors, all solver diagnostics, exact fixture, shock, thresholds, dependency versions, and source hashes. It is generated by the same acceptance routine used in pytest. The checked-in reference run passes with errors around machine precision; use the certificate for exact values.

## Reproduction and provenance

Use Python 3.12. From the repository root:

```sh
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# POSIX shells: source .venv/bin/activate
python -m pip install -r requirements-repro.txt
python -m pip install --no-deps -e .
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

The viewer supplies three routes with six expandable steps each, highlighted source lines, keyboard-accessible route selection, and eight questions. The quiz covers indexing, equation mapping, equilibrium closure, numeraire, hat inputs, two-route logic, failure diagnosis, and interpretation. Five questions have objective answers; three written answers use self-assessment rubrics. Answers are page-session-only; reset or reload clears them. No answer storage or transmission occurs.

Acceptance requires a human to review the equations and evidence, close the viewer, explain the solution process, and complete the quiz. These actions must be recorded in the PR; passing automated tests does not mark them complete. After acceptance and merge, this document remains the one-industry Single Source of Truth until the next accepted PDCU cycle rewrites it.

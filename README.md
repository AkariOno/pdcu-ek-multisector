# Learning the PDCU Cycle with a Coding Agent

## From One Industry to Many in the Eaton-Kortum Model

This repository is an educational tutorial on how to run a PDCU cycle with a coding agent. The Eaton-Kortum model is the worked example because it provides a realistic research task with a clear implementation target, a decisive verification test, and substantive code to understand. The objective is to learn the workflow, not international trade theory itself.

The tutorial begins from the accepted, tested one-industry Eaton-Kortum baseline, merged into `main` through [PR #2](https://github.com/AkariOno/pdcu-ek-multisector/pull/2). Human review and the learner explanation and understanding quiz have been completed. The baseline includes its numerical verification and Understand artifacts.

This feature branch implements the multi-industry candidate approved in [Issue #5](https://github.com/AkariOno/pdcu-ek-multisector/issues/5). The accepted starting state is `main` at [PR #4's reconciliation commit](https://github.com/AkariOno/pdcu-ek-multisector/commit/9eb5b52bcefc6a5e4055acc5dfb2a6f6a184596b). Automated checks pass; multi-industry human review and learner understanding remain pending. The candidate has not been accepted or merged.

Start with the [candidate specification and acceptance status](docs/latest.md), [validation record](docs/validation.md), and [offline code viewer](viewer/model_viewer.html). Download or open the HTML locally to use the interactions; GitHub's source view does not execute it. The accepted one-industry state remains available through Git history.

Both levels and independent exact hats retain the accepted NumPy-only loop: multiply each wage by `(sales/income)**0.2`, normalize country 0, and recompute until every country's residual meets the target. Exporter sales now sum final expenditure across importers and industries. Normalized Cobb–Douglas utility gives `C[n] = product_j P[n,j]**alpha[n,j]` and real wages `w/C`. The viewer exposes the actual loop, aggregation, and each solve's residual and wage history.

### Run the candidate

With Python 3.12 in an activated virtual environment:

```sh
python -m pip install -r requirements-repro.txt
python -m pip install --no-deps -e .
python -m pytest -q
python scripts/build_model_viewer.py --check
```

Regenerate the viewer with `python scripts/build_model_viewer.py`. Its [certificate](viewer/verification.json) records actual diagnostics, five comparisons, and J=1 regression evidence; the [manifest](viewer/manifest.json) binds generated artifacts and the frozen baseline reference to production source. See [reproduction details](docs/latest.md#reproduction-and-provenance).

In the clean Python 3.12.14 / NumPy 2.5.3 environment, **109 tests pass**. E0, E1 and hats converge in 41, 41 and 38 updates, with full residual norms below `1e-13`. Every absolute and relative comparison error is below `1.91e-15`. All 18 J=1 comparisons match the immutable accepted baseline exactly in this environment. Windows/Linux CI runs the same numerical, failure-path, freshness and reproducibility checks; its actual run results are recorded in the linked draft PR.

The learner's task is one coherent extension:

> Extend the existing one-industry Eaton-Kortum model to multiple industries, verify it by comparing two full solutions with exact-hat algebra, and build a code viewer and quiz that make the implementation understandable.

The coding agent is expected to be capable of implementing the entire extension in one cycle. The educational emphasis is therefore not on artificially decomposing the coding task. It is on specifying the economics clearly, embedding one decisive verification test in advance, and turning the completed code into something the learner can explain.

## What the Baseline Provides

The one-industry PDCU cycle is complete: implementation and numerical checks passed, human review and the learner explanation and understanding quiz were completed, and PR #2 was accepted and merged into `main`.

| PDCU phase | Existing one-industry artifact |
|---|---|
| Plan | A closed model specification, equilibrium conditions, normalization, solver contract, and pre-specified verification test |
| Do | A full-solution solver and a separate exact-hat solver |
| Check | A test comparing changes between two full-solution equilibria with the exact-hat result |
| Understand | An interactive code viewer and a code-understanding quiz |
| Latest State | The accepted revision of `docs/latest.md` describes the one-industry baseline merged into `main` |
| Lab Journal | The baseline Issue and linked implementation Pull Request record the plan, checks, and review |

The learner can inspect this completed baseline cycle before starting the extension, including its numerical evidence, completed human review, and understanding exercise.

The following sections describe the multi-industry exercise implemented on this branch. The existing production modules are generalized in place, with explicit industry axes even for J=1; no parallel historical solver is retained. Only after human review, learner understanding and an authorized merge will the candidate become the accepted Latest State on `main`. Until then, `docs/latest.md` explicitly distinguishes this candidate from the accepted one-industry starting state.

## Learning Objectives

By the end of the tutorial, the learner should be able to:

1. define one coherent and appropriately bounded research cycle;
2. turn that goal into a Plan with explicit scope, assumptions, and a pre-specified acceptance test;
3. use a coding agent to carry out the Do phase in an isolated branch while protecting the trusted Latest State;
4. take human responsibility for Check by evaluating the planned evidence rather than trusting plausible output;
5. use a code viewer and quiz during Understand to test their own grasp of what the agent produced;
6. distill the resulting insight into the Pull Request and rewrite the Single Source of Truth; and
7. use Issues, Pull Requests, Git history, and the updated Latest State to begin the next PDCU cycle.

## The Multi-Industry PDCU Cycle

The multi-industry extension is one PDCU cycle and one principal Pull Request.

### Plan

Open one GitHub Issue that fixes the target model before implementation begins. The Issue should state:

- the economic environment and equilibrium closure;
- the new industry-indexed primitives and endogenous variables;
- array shapes and the exporter/importer/industry index convention;
- the numeraire used by both solvers;
- the counterfactual shock; and
- the outputs to be compared; 

The coding agent should be asked to identify any missing economic assumption before editing code.

### Do

In one feature branch or isolated worktree, ask the coding agent to:

1. extend the existing full-solution model from one industry to multiple industries;
2. extend the exact-hat system to the same multi-industry economy;
3. generalize the existing production implementation in place rather than maintain separate one-industry and multi-industry solvers;
4. implement the pre-specified equivalence test; and
5. report numerical convergence diagnostics.

The implementation can be developed as a whole. There is no requirement to divide it into separate PDCU cycles for data structures, the levels solver, and the hat solver. The Git diff records how the trusted one-industry implementation became the multi-industry implementation.

### Check

The principal test is:

> Solve the model twice in levels, once at the baseline primitives and once at the counterfactual primitives. Starting from the first equilibrium, solve the same counterfactual with exact-hat algebra. Compare the realized changes from the two full solutions with the exact-hat result.

Formally:

1. Solve the baseline full equilibrium $E^0$.
2. Apply a nontrivial exogenous shock.
3. Solve the counterfactual full equilibrium $E^1$.
4. Compute realized changes $E^1/E^0$.
5. Use the baseline equilibrium objects from $E^0$ and the same exogenous shock to solve the exact-hat system.
6. Compare the two sets of changes under the same normalization.

The comparison should cover the endogenous objects returned by both routes, including:

- wage changes;
- industry price-index changes;
- aggregate cost-of-living changes;
- bilateral industry expenditure shares; and
- real-wage or welfare changes.

The test must also require both solvers to have converged. Subject to that condition, this equivalence test is the acceptance test for the multi-industry development; the tutorial does not require a long catalog of additional tests.

### Understand

After the test passes, the coding agent creates two educational artifacts:

1. an interactive multi-industry code viewer; and
2. a quiz that tests whether the learner understands the model and code.

The learner should use the viewer, close it, explain the solution process in their own words, and then take the quiz. This phase tests the learner's understanding, not the coding agent's ability to generate an explanation.

## Target Multi-Industry Model

Version 1 should remain small enough that the two solution routes are transparent.

- Countries: zero-based importer $n$ and exporter $i$, $N\ge2$
- Industries: zero-based $j$, $J\ge1$
- One factor: labor
- One wage per country
- Industry-specific technology $T_i^j$
- Industry-specific trade elasticity $\theta_j$
- Industry- and pair-specific iceberg trade costs $d_{ni}^j$
- Cobb-Douglas final demand with fixed industry shares $\alpha_n^j$
- Balanced trade
- No intermediate inputs, tariffs, tariff revenue, or exogenous trade deficits in Version 1

Country $n$'s expenditure share on industry-$j$ goods from country $i$ is

$$
\pi_{ni}^j =
\frac{T_i^j (w_i d_{ni}^j)^{-\theta_j}}
{\sum_k T_k^j (w_k d_{nk}^j)^{-\theta_j}}.
$$

The industry price index is

$$
P_n^j = \gamma_j
\left[\sum_i T_i^j (w_i d_{ni}^j)^{-\theta_j}\right]^{-1/\theta_j}.
$$

The approved normalized Cobb–Douglas utility and unit-expenditure index are

$$
U_n=\prod_j(c_n^j/\alpha_n^j)^{\alpha_n^j},
\qquad C_n=\prod_j(P_n^j)^{\alpha_n^j},
\qquad \text{real wage}_n=w_n/C_n.
$$

Labor is fully employed and mobile between domestic industries, but immobile internationally. Strictly positive expenditure shares sum to one by country. Balanced trade applies to aggregate country income. Fixed industry multipliers `gamma[j]` affect price levels and cancel from counterfactual ratios.

Industry expenditure is

$$
X_n^j = \alpha_n^j w_n L_n,
\qquad
\sum_j \alpha_n^j=1,
$$

and goods-market clearing determines relative wages:

$$
w_iL_i=\sum_n\sum_j\pi_{ni}^jX_n^j.
$$

For a trade-cost counterfactual with technology held fixed, exact-hat trade shares satisfy

$$
\pi_{ni}^{j\prime}=
\frac{\pi_{ni}^j
(\widehat w_i\widehat d_{ni}^j)^{-\theta_j}}
{\sum_k\pi_{nk}^j
(\widehat w_k\widehat d_{nk}^j)^{-\theta_j}},
$$

where juxtaposition denotes multiplication. The industry price-index change is

$$
\widehat P_n^j=
\left[
\sum_i\pi_{ni}^j
(\widehat w_i\widehat d_{ni}^j)^{-\theta_j}
\right]^{-1/\theta_j}.
$$

The full-solution and exact-hat implementations must use the same economic closure and numeraire. These model details support the PDCU exercise; mastering them is not the tutorial's primary learning objective.

Fixed preferences imply $\widehat C_n=\prod_j(\widehat P_n^j)^{\alpha_n^j}$ and $\widehat{\text{real wage}}_n=\widehat w_n/\widehat C_n$. Counterfactual expenditure is $X_n^{j\prime}=\alpha_n^j Y_n^0\widehat w_n$, and aggregate sales clear that country income. [Issue #5](https://github.com/AkariOno/pdcu-ek-multisector/issues/5) is the definitive Plan, including the heterogeneous three-country/two-industry fixture and 10% bilateral cost cut in industry 0 only.

## The Code Viewer

The new viewer should retain the useful interaction of the supplied legacy example: selecting an algorithm step highlights the corresponding implementation. It should add enough structure to make the economic logic visible before the code.

### Top: what is being solved

Show:

- primitives, unknowns, and the numeraire;
- trade-share, price-index, and market-clearing equations;
- the full-solution and exact-hat inputs;
- which conditions are imposed exactly and which are convergence residuals; and
- a compact diagram of the two verification routes from $E^0$ to $E^1$.

### Middle: expandable code tree

Provide tabs or a switch for:

- **Full solution:** primitives → unit costs → trade shares → price indices → market-clearing residual → equilibrium solver;
- **Exact hat:** baseline statistics + shocks → counterfactual shares → price changes → hat-market-clearing residual → hat solver; and
- **Equivalence test:** solve $E^0$ → solve $E^1$ → solve hats from $E^0$ → normalize → compare.

Each algorithm card should reveal the corresponding function body with one click. Industry, exporter, and importer dimensions should have stable visual labels so that an indexing error is easy to see.

### Bottom: verification certificate

Show a generated comparison table with, for every tested object:

- maximum absolute error;
- maximum relative error;
- acceptance tolerance; and
- pass/fail status.

Also show the residual norm and convergence status of each solver. The viewer must not claim equivalence merely because the comparison code ran.

### Viewer deliverables

The repository should contain:

- a standalone HTML viewer;
- reader-oriented annotated code used by the viewer;
- a reproducible viewer builder;
- a manifest containing the production-source hash and generated files; and
- a clear note that the annotated code is an explanatory reconstruction unless it is literally the production source.

## The Understanding Quiz

The quiz should be embedded at the end of the viewer or linked directly from it. It should test economic and code understanding rather than Python syntax.

A recommended quiz has eight questions:

1. **Index meaning:** Given `pi[n, i, j]`, identify importer, exporter, and industry.
2. **Code-to-equation mapping:** Select the code block that implements the denominator of $\pi_{ni}^j$.
3. **Equilibrium closure:** Explain which equation determines relative wages.
4. **Numeraire:** Predict what changes and what does not when a different wage is normalized to one.
5. **Exact-hat inputs:** Identify which baseline equilibrium objects are required by the hat solver.
6. **Two-route logic:** Explain why $E^1/E^0$ should equal the exact-hat result.
7. **Failure diagnosis:** Given a comparison table, identify whether the likely problem is normalization, indexing, non-convergence, or a wrong equation.
8. **Interpretation:** Explain how an industry-specific trade-cost reduction propagates through trade shares, wages, prices, and real income.

Use a mixture of multiple choice, code highlighting, and short written answers. After submission, show the correct answer and a short explanation. A score alone is not enough: the learner should see which part of the model or code they misunderstood.

## Suggested Repository Structure

```text
.
|-- README.md
|-- pyproject.toml
|-- src/
|   `-- ek_model/
|       |-- model.py
|       |-- full_solution.py
|       `-- exact_hat.py
|-- tests/
|   `-- test_exact_hat_equivalence.py
|-- viewer/
|   |-- model_viewer.html
|   |-- model_annotated.py
|   |-- manifest.json
|   `-- quiz.json
|-- scripts/
|   `-- build_model_viewer.py
`-- docs/
    `-- latest.md
```

The filenames describe roles rather than historical versions. The earlier one-industry implementation is preserved by Git, not copied into parallel production modules.

## GitHub Representation

| Research object | GitHub object |
|---|---|
| Completed one-industry baseline cycle | Completed Issues #1 and #3 and accepted, merged PR #2 |
| Multi-industry Plan | Approved Issue #5, kept open |
| Multi-industry Do | One feature branch or isolated worktree |
| Multi-industry Check | The Pull Request and its exact-hat equivalence test |
| Multi-industry Understand | Generated viewer and quiz; human explanation and quiz completion pending |
| Latest State | Reviewed `main` branch and `docs/latest.md`, both rewritten by the accepted cycle |
| Lab Journal | Issue and Pull Request history |

## Acceptance Criteria

The multi-industry Pull Request is ready to merge when:

1. the model document and implementation use the same equations and index convention;
2. the full-solution and exact-hat solvers converge under the test fixture;
3. the two-full-solutions versus exact-hat comparison passes at the stated tolerance;
4. the viewer is reproducibly generated from the reviewed source;
5. the viewer maps the major equations and algorithm steps to code;
6. the quiz covers both implementation and economic interpretation; and
7. the learner can summarize what changed from one industry to many;
8. the production tree does not retain duplicate one-industry and multi-industry implementations; and
9. `docs/latest.md` accurately describes the candidate and its evidence before merge, then records formal acceptance after authorization.

## Deliberate Non-Goals

The first tutorial does not include:

- input-output linkages or intermediate goods;
- tariffs and tariff-revenue recycling;
- trade deficits or transfers;
- estimation from real data; or
- large-scale performance optimization.

These are possible later PDCU exercises, but they are not prerequisites for treating the multi-industry extension as one coherent cycle.

## Recommended Reading

- Eaton and Kortum (2002), “Technology, Geography, and Trade.”
- Dekle, Eaton, and Kortum (2008), “Global Rebalancing with Gravity: Measuring the Burden of Adjustment.”
- Caliendo and Parro (2015), “Estimates of the Trade and Welfare Effects of NAFTA.”

## Status

The one-industry cycle was accepted and merged through PR #2, with its acceptance documentation reconciled by PR #4; Issues #1 and #3 are completed. This branch is the implemented and automatically verified **multi-industry candidate** for Issue #5. Both routes retain fixed damping `0.2`, at most 10,000 updates, country-0 normalization and strict full residual target `1e-13`. Acceptance still requires converged solvers, full residuals below `1e-11`, and separate absolute and relative errors strictly below `1e-9` for all five objects, plus frozen J=1 regression evidence. Human review and learner explanation/quiz remain unchecked. Issue #5 remains open and the implementation PR remains a draft.

# Automated validation record: multi-industry candidate

This records Do, automated Check and artifact verification for [Issue #5](https://github.com/AkariOno/pdcu-ek-multisector/issues/5). Economic specification/model choices were approved by the user. Multi-industry human implementation review and learner explanation/quiz completion remain pending. The candidate is not accepted or merged; the linked draft PR records actual Windows/Linux workflow results.

## Clean local environment and automated checks

Run on Windows using a newly created virtual environment, Python 3.12.14, NumPy 2.5.3 and pytest 9.1.1:

- Installation from `requirements-repro.txt`, followed by `pip install --no-deps -e .`, succeeds.
- `importlib.util.find_spec('scipy') is None` passes. The installed package imports and runs fresh verification outside the repository working directory, including access to packaged frozen reference data.
- `python -m pytest -q`: **109 passed**. Tests include direct multi-industry equations, normalized utility/unit-expenditure identity, preference validation without renormalization, importer/exporter/industry dimensions, industry gamma, share sums, expenditure budgets, country aggregate sales, numeraire, no shock, all-country residuals including country 0, independent exact hats, the direct simultaneous update, invalid controls/shocks, iteration limits, tiny-step stagnation and nonfinite/nonpositive states. Failed evaluation outputs preserve industry dimensions.
- Damping `0.1`, `0.2` and `0.3` yields equivalent fixture equilibria and passes the two-route comparison. Tests do not require monotonic residual decline.
- `python scripts/build_model_viewer.py`: actual verification and J=1 regression generate PASS evidence.
- `python scripts/build_model_viewer.py --check`: production/input/artifact hashes, verbatim source mappings and fresh numerical verification pass.
- Two independently generated artifact directories are byte-identical within the same environment. Certificate serialization round trips preserve rendering.
- Failure-path tests produce FAIL evidence and builder exit status 1 for injected non-convergence, real nonfinite evaluated objects and a tampered J=1 reference. Tests reject stale sources and artifacts. These failures are expected tests, not unaddressed failures of the delivered candidate.
- `git diff --check` passes.

The unchanged CI matrix retains `ubuntu-latest` and `windows-latest`, Python 3.12, clean dependency/package installation, an explicit SciPy-absence assertion, the complete test suite and artifact freshness/fresh numerical verification. CI run links and conclusions are recorded in the draft PR after publication; local checks do not substitute for an unobserved remote run.

## Numerical certificate

The deterministic three-country/two-industry fixture, normalized Cobb–Douglas index and industry-0-only symmetric 10% bilateral cut follow Issue #5 exactly. All three solves start independently from unit wages or hats. Default fixed damping is `0.2`, maximum updates 10,000 and strict full residual target `1e-13`.

| Route | Updates | Evaluations | Full residual infinity norm |
|---|---:|---:|---:|
| E0 | 41 | 42 | 5.3974921964810676e-14 |
| E1 | 41 | 42 | 8.2655820161500400e-14 |
| Exact hats | 38 | 39 | 8.3148552323416900e-14 |

All five comparisons pass both separate strict `1e-9` bounds:

| Object | Maximum absolute error | Maximum relative error |
|---|---:|---:|
| Wages | 4.440892098500626e-16 | 4.449319957590492e-16 |
| Industry prices | 2.220446049250313e-16 | 2.2235377419328625e-16 |
| Aggregate cost of living | 2.220446049250313e-16 | 2.2224564774844085e-16 |
| Bilateral industry share ratios | 1.887379141862766e-15 | 1.9032017626744743e-15 |
| Real wages/welfare | 4.440892098500626e-16 | 4.445295114559421e-16 |

The relative denominator is the absolute levels-route ratio. All comparisons are finite, all solvers converged and full residual norms satisfy the preserved `1e-11` acceptance bound. [Generated certificate](../viewer/verification.json) is the source of exact results, histories, tolerances and versions.

## Immutable J=1 compatibility evidence

Before production edits, full E0/E1/exact-hat results were captured from accepted PR #4 revision `9eb5b52bcefc6a5e4055acc5dfb2a6f6a184596b`. Capturing checked production source bytes against that revision. The reference stores its source commit and source SHA-256 hashes; its data hash is `a0e1c68038dac764d51acefada95b1f60580ca1fc8fb930b1a9f1d562a163ae9`.

All **18 comparisons** against that frozen reference pass, with maximum absolute and relative errors **0** in the certificate environment. The generalized model uses J=1, `alpha=1` and the original fixture/shock; no duplicate one-industry production solver is retained. Cost of living equals the sole industry price exactly. J=1 E0/E1/hats converge in 41/43/40 updates, with full residual norms `8.198879477336918e-14`, `8.262687422905555e-14` and `7.699009911526001e-14`.

## Browser and offline artifact checks

The generated HTML was inspected in the in-app Chromium browser using a loopback server restricted to the viewer directory. All assets are embedded; the content security policy prohibits network connections. After stopping the server, route selection, source expansion and quiz submission continued to work in the already-loaded page. Direct-file browser navigation was blocked by the browser's URL policy in the baseline workflow; no direct-file navigation test is claimed here.

Verified interactions:

- All three route tabs and expand-all controls work, with 9 levels, 7 hats and 7 verification steps.
- Source excerpts highlight actual production lines for normalized geometric-mean aggregation, exporter sales across both axes, industry shares/prices, the wage loop, fixed damping, numeraire and J=1 comparisons.
- Keyboard Home/End navigation selects the first/last routes.
- Per-solve residual plots and expanded wage traces match the certificate: E0/E1/hats have 42/42/39 evaluated states, starting at unit wages and ending at the reported residuals and returned wages.
- Expanded J=1 evidence shows all 18 comparisons and immutable reference provenance.
- Blank quiz submission reveals all eight explanations and reports 0/5 objective answers and 0/3 written responses.
- Five correct choices yield 5/5; replacing one with an incorrect choice yields 4/5 and distinct feedback. Three test written responses are counted separately, with nine self-assessment checkboxes/rubric items.
- The code-selection explanation opens the levels share function and highlights its three relevant lines.
- Reset clears choices, written responses and feedback; reload clears answers. No local/session storage or network submission is used.
- No browser console warnings/errors or horizontal page overflow appeared at the inspected desktop viewport. The certificate and residual plots were visually inspected.

Browser test responses were reset. These checks verify software interactions, not the learner's understanding. **Human review and learner explanation/eight-question quiz completion remain unchecked.**

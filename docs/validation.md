# Baseline validation record

This records automated checks and agent-operated UI checks. It does not certify human acceptance or a learner's understanding.

## Numerical and build checks

- Python 3.12.14 and NumPy 2.5.3 in a newly created environment where SciPy is absent (`importlib.util.find_spec('scipy') is None`). Reference dependencies are pinned in `requirements-repro.txt`.
- `python -m pytest -q`: 71 tests pass. Tests include independent direct-equation evaluation, no-shock identity, fixed-gamma behavior, invalid data and controls, independent exact hats, direct multiplicative-update/normalization checks, separate strict error bounds, full residual diagnostics, tiny-step stagnation, iteration limits, nonfinite/nonpositive states, JSON-safe failure histories, stale artifacts, and identical repeated builds.
- Damping `0.1`, `0.2`, and `0.3` yields equivalent fixture equilibria and passes the two-route check; monotone residual decline is not required.
- `python scripts/build_model_viewer.py --check`: source hashes, all rendered artifacts, and fresh numerical acceptance pass.
- The documented editable package installation succeeds and the public package imports outside pytest.
- [verification.json](../viewer/verification.json) is the generated source of exact numerical results; all three solvers converge and all four comparisons satisfy both strict `1e-9` bounds.
- Default wage updates: E0 41, E1 43, hats 40. Evaluations including initial states: 42, 44, 41. Full residual norms are below `1e-13`; the maximum comparison error is below `1.3e-14`.

## Browser checks

The generated HTML was inspected in the in-app Chromium browser through a loopback server restricted to the viewer directory. Direct `file:` navigation was blocked by the browser's URL policy; no direct-file browser test is claimed. The page embeds all assets and prohibits network connections through its content security policy. After stopping the loopback server, route selection, source expansion, and quiz submission continued to work in the already-loaded page.

Verified interactions:

- All three routes select correctly; the levels route has eight steps, hats and verification each have six.
- Selecting a step reveals verbatim source and relevant highlighted lines, including the exact-hat share equations.
- Keyboard Home navigation selects the first route.
- The explicit loop and wage-update steps highlight the evaluated residual, stopping rule, simultaneous log update, and country-0 normalization.
- Each residual plot and expandable wage trace matches the certificate: E0 has 42 evaluated states, E1 44, and exact hats 41, beginning at unit wages and ending at the returned equilibrium. The plot and scrollable trace were visually inspected.
- Blank quiz submission reveals all eight explanations and reports zero answered questions.
- Mixed correct/incorrect choices show distinct feedback and produce 4/5; correcting the remaining answer produces 5/5.
- Three test written responses are reported separately from objective scores; nine self-assessment rubric items appear.
- The code-selection explanation opens the levels trade-share function with its denominator highlighted.
- Reset removes selections, written answers, and feedback. Reload clears answers as documented.
- No horizontal page overflow or browser console warnings/errors were observed at the inspected desktop viewport.

The test responses were reset and are not recorded as learner completion. Human review and the learner explanation/quiz remain unchecked in the PR.

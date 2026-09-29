# Baseline validation record

This records automated checks and agent-operated UI checks. It does not certify human acceptance or a learner's understanding.

## Numerical and build checks

- Python 3.12.14, NumPy 2.5.3, SciPy 1.18.1; the reference dependencies are pinned in `requirements-repro.txt`.
- `python -m pytest -q`: 39 tests pass. Tests include independent direct-equation evaluation, no-shock identity, fixed-gamma behavior, invalid data, independent exact hats, separate strict error bounds, full residual diagnostics, failed/nonfinite optimizer results, failed certificate generation, stale artifacts, and identical repeated builds.
- `python scripts/build_model_viewer.py --check`: source hashes, all rendered artifacts, and fresh numerical acceptance pass.
- The documented editable package installation succeeds and the public package imports outside pytest.
- [verification.json](../viewer/verification.json) is the generated source of exact numerical results; all three solvers converge and all four comparisons satisfy both strict `1e-9` bounds.

## Browser checks

The generated HTML was inspected in the in-app Chromium browser through a loopback server restricted to the viewer directory. Direct `file:` navigation was blocked by the browser's URL policy; no direct-file browser test is claimed. The page embeds all assets and prohibits network connections through its content security policy. After stopping the loopback server, route selection, source expansion, and quiz submission continued to work in the already-loaded page.

Verified interactions:

- All three routes select correctly; each route expands and collapses all six steps.
- Selecting a step reveals verbatim source and relevant highlighted lines, including the exact-hat share equations.
- Keyboard Home navigation selects the first route.
- Blank quiz submission reveals all eight explanations and reports zero answered questions.
- Mixed correct/incorrect choices show distinct feedback and produce 4/5; correcting the remaining answer produces 5/5.
- Three test written responses are reported separately from objective scores; nine self-assessment rubric items appear.
- The code-selection explanation opens the levels trade-share function with its denominator highlighted.
- Reset removes selections, written answers, and feedback. Reload clears answers as documented.
- No horizontal page overflow or browser console warnings/errors were observed at the inspected desktop viewport.

The test responses were reset and are not recorded as learner completion. Human review and the learner explanation/quiz remain unchecked in the PR.

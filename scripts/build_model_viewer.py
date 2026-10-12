"""Build a self-contained viewer from real source and freshly computed evidence.

Run normally to regenerate; --check verifies committed hashes and reruns the
numerical acceptance test without rewriting evidence. Float bytes may differ
across platforms, so cross-environment checks do not replace the certificate.
"""

import argparse
import ast
import hashlib
import html
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ek_model.verification import run_verification  # noqa: E402


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def input_hashes(root=ROOT):
    production = sorted((root / "src/ek_model").glob("*.py"))
    production += sorted((root / "src/ek_model/data").glob("*.json"))
    inputs = [root / path for path in (
        "scripts/build_model_viewer.py", "viewer/template.html", "viewer/steps.json",
        "viewer/quiz.json", "pyproject.toml", "requirements-repro.txt",
    )]
    hashed = lambda paths: {p.relative_to(root).as_posix(): digest(p.read_bytes()) for p in paths}
    return {"production_sources": hashed(production), "builder_inputs": hashed(inputs)}


def source_steps(root=ROOT):
    """Extract actual AST spans; refuse stale symbol or highlight references."""
    steps = json.loads((root / "viewer/steps.json").read_text(encoding="utf-8"))
    for step in steps:
        path = root / "src/ek_model" / step["file"]
        source = path.read_text(encoding="utf-8")
        node = next(n for n in ast.parse(source).body if getattr(n, "name", None) == step["symbol"])
        start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        lines = source.splitlines()[start - 1:node.end_lineno]
        for marker in step["highlight"]:
            if not any(marker in line for line in lines):
                raise ValueError(f"Stale highlight in {step['id']}: {marker}")
        step.update(source="\n".join(lines), start_line=start, end_line=node.end_lineno,
                    path=path.relative_to(root).as_posix())
    return steps


def render_steps(steps):
    routes = {"levels": [], "hats": [], "check": []}
    for step in steps:
        code = []
        for number, line in enumerate(step["source"].splitlines(), step["start_line"]):
            selected = any(marker in line for marker in step["highlight"])
            css = "code-line highlighted" if selected else "code-line"
            code.append(f'<span class="{css}"><span class="line-number" aria-hidden="true">'
                        f'{number}</span>{html.escape(line) or " "}</span>')
        routes[step["route"]].append(
            f'<details class="step" id="{step["id"]}"><summary>{html.escape(step["title"])}</summary>'
            f'<div class="step-body"><p>{html.escape(step["explanation"])}</p>'
            f'<p class="equation">{html.escape(step["equation"])}</p>'
            f'<p class="source-location">{step["path"]}:{step["start_line"]}–{step["end_line"]}'
            ' · verbatim production excerpt · relevant lines highlighted</p>'
            f'<pre><code>{"".join(code)}</code></pre></div></details>'
        )
    return "\n".join(
        f'<section class="route" id="route-{route}" aria-labelledby="tab-{route}"'
        f'{" hidden" if route != "levels" else ""}>{"".join(cards)}</section>'
        for route, cards in routes.items()
    )


def render_certificate(certificate):
    escape = lambda value: html.escape(str(value))
    number = lambda value: "nonfinite / unavailable" if value is None else f"{value:.3e}"
    comparisons = "".join(
        f'<tr><th scope="row">{escape(name)}</th><td>{number(row["max_absolute_error"])}</td>'
        f'<td>{number(row["max_relative_error"])}</td><td>&lt; {number(row["tolerance"])}</td>'
        f'<td class="{"pass" if row["passed"] else "fail"}">{"PASS" if row["passed"] else "FAIL"}</td></tr>'
        for name, row in sorted(certificate["comparisons"].items())
    )
    solvers = "".join(
        f'<tr><th scope="row">{escape(name)}</th><td>{escape(row["converged"])}</td>'
        f'<td>{row["niter"]}</td><td>{row["nfev"]}</td><td>{row["damping"]}</td>'
        f'<td>{number(row["residual_norm"])}</td><td>{row["status"]}: {escape(row["message"])}</td></tr>'
        for name, row in sorted(certificate["solvers"].items())
    )
    failure = f'<p class="fail">{escape(certificate["failure"])}</p>' if certificate.get("failure") else ""
    return (
        f'<div class="certificate-status {"pass" if certificate["passed"] else "fail"}" id="certificate-status">'
        f'{"PASS" if certificate["passed"] else "FAIL"} · generated numerical evidence</div>{failure}'
        '<div class="table-scroll"><table><caption>Changes: levels ratios versus exact hats</caption>'
        '<thead><tr><th>Object</th><th>Max absolute error</th><th>Max relative error</th>'
        f'<th>Each error must be</th><th>Result</th></tr></thead><tbody>{comparisons}</tbody></table></div>'
        '<div class="table-scroll"><table><caption>All-country market-clearing diagnostics</caption>'
        '<thead><tr><th>Route</th><th>Converged</th><th>Wage updates</th><th>Evaluations</th><th>Damping</th>'
        f'<th>Full residual norm</th><th>Status / message</th></tr></thead><tbody>{solvers}</tbody></table></div>'
        '<p class="muted">Evaluations include the initial unit-wage state and each evaluated update. '
        'The residual is maxᵢ |(salesᵢ − incomeᵢ) / incomeᵢ|, including country 0.</p>'
        + render_histories(certificate) + render_regression(certificate) +
        '<details class="raw-evidence"><summary>Inspect fixture, shock, versions, contract & source hashes</summary>'
        f'<pre>{html.escape(json_bytes(certificate).decode())}</pre></details>'
    )


def render_regression(certificate):
    """Show all immutable-reference comparisons without claiming human acceptance."""
    regression = certificate.get("j1_regression")
    if not regression:
        return '<p class="fail">J=1 regression evidence unavailable on this failed run.</p>'
    rows = "".join(
        f'<tr><th scope="row">{html.escape(name)}</th><td>{row["max_absolute_error"]}</td>'
        f'<td>{row["max_relative_error"]}</td><td>{"PASS" if row["passed"] else "FAIL"}</td></tr>'
        for name, row in sorted(regression["comparisons"].items())
    )
    return (
        '<details id="j1-regression"><summary>J=1 compatibility with the accepted one-industry baseline · '
        f'{"PASS" if regression["passed"] else "FAIL"}</summary>'
        f'<p>Source commit: <code>{regression["source_commit"]}</code>. '
        f'Immutable data SHA-256: <code>{regression["reference_sha256"]}</code>. '
        'Both errors must separately be below 1e-9; all three regression solves must converge.</p>'
        '<div class="table-scroll"><table><thead><tr><th>Object</th><th>Absolute error</th>'
        f'<th>Relative error</th><th>Result</th></tr></thead><tbody>{rows}</tbody></table></div></details>'
    )


def render_histories(certificate):
    """Plot saved all-country residuals; expose every wage and residual snapshot."""
    panels = []
    target = certificate["contract"]["iteration_tolerance"]
    for name, solver in sorted(certificate["solvers"].items()):
        history = solver["history"]
        finite = [row for row in history if row["residual_norm"] is not None]
        if finite:
            # This floor is for plotting only; acceptance always uses the true norm.
            low = math.floor(math.log10(max(min(target, min(row["residual_norm"] for row in finite)), 1e-16)))
            high = max(low + 1, math.ceil(math.log10(max(target, max(row["residual_norm"] for row in finite), 1e-16))))
            max_iteration = max(1, history[-1]["iteration"])
            xpos = lambda i: 70 + 660 * i / max_iteration
            ypos = lambda norm: 24 + 140 * (high - math.log10(max(norm, 1e-16))) / (high - low)
            points = " ".join(f'{xpos(row["iteration"]):.3f},{ypos(row["residual_norm"]):.3f}' for row in finite)
            ticks = "".join(
                f'<text x="60" y="{ypos(10. ** exponent):.3f}" text-anchor="end">1e{exponent}</text>'
                for exponent in range(low, high + 1, max(1, (high - low) // 4))
            )
            plot = (
                f'<svg class="residual-plot" viewBox="0 0 760 210" role="img" aria-label="{name} full residual history">'
                f'<title>{name}: full market-clearing residual by wage update</title>'
                '<path d="M70 20 V164 H730" fill="none" stroke="#90a3b0"/>'
                f'<line x1="70" x2="730" y1="{ypos(target):.3f}" y2="{ypos(target):.3f}" stroke="#c28312" stroke-dasharray="5 4"/>'
                f'<polyline points="{points}" fill="none" stroke="#174bb4" stroke-width="2.5"/>{ticks}'
                f'<text x="70" y="185">0</text><text x="730" y="185" text-anchor="end">{history[-1]["iteration"]}</text>'
                '<text x="400" y="204" text-anchor="middle">Wage updates</text></svg>'
            )
        else:
            plot = '<p class="fail">No finite residual history is available.</p>'
        value = lambda x: "nonfinite" if x is None else f"{x:.12g}"
        rows = "".join(
            f'<tr><td>{row["iteration"]}</td><td>{html.escape(", ".join(value(w) for w in row["wages"]))}</td>'
            f'<td>{html.escape(", ".join(value(r) for r in row["residual"]))}</td>'
            f'<td>{value(row["residual_norm"])}</td></tr>' for row in history
        )
        panels.append(
            f'<section class="history-panel" id="history-{name}"><h3>{name} · iteration history</h3>{plot}'
            f'<p class="muted">Fixed damping {solver["damping"]}; dashed line = iteration target {target:g}. '
            'Vertical axis is logarithmic; zero norms use a display floor of 1e-16.</p>'
            f'<details><summary>Inspect {name} wages and full residual vectors ({len(history)} evaluated states)</summary>'
            '<div class="table-scroll"><table><thead><tr><th>Update</th><th>Wages / wage hats, country order</th>'
            f'<th>Residual vector, country order</th><th>Full norm</th></tr></thead><tbody>{rows}</tbody></table></div></details></section>'
        )
    return '<div class="iteration-histories">' + "".join(panels) + '</div>'


def render_quiz(quiz):
    cards = []
    for q in quiz:
        qid = html.escape(q["id"])
        if q["kind"] == "written":
            controls = f'<textarea id="answer-{qid}" aria-label="{html.escape(q["prompt"])}" rows="4" placeholder="Explain in your own words…"></textarea>'
        else:
            controls = "".join(
                f'<label class="option"><input type="radio" name="{qid}" value="{i}">'
                f'<{"code" if q["kind"] == "code" else "span"}>{html.escape(option)}'
                f'</{"code" if q["kind"] == "code" else "span"}></label>'
                for i, option in enumerate(q["options"])
            )
        rubric = ""
        if q.get("rubric"):
            rubric = '<p>Self-assessment rubric — check each point your answer covers:</p>' + "".join(
                f'<label class="rubric"><input type="checkbox">{html.escape(item)}</label>' for item in q["rubric"]
            )
        link = f'<button type="button" class="text-button" data-step="{q["step"]}">Show the matching production code →</button>' if q.get("step") else ""
        cards.append(
            f'<fieldset class="question" id="question-{qid}"><legend>{html.escape(q["title"])}</legend>'
            f'<p>{html.escape(q["prompt"])}</p>{controls}<div class="feedback" id="feedback-{qid}" hidden>'
            f'<p class="verdict"></p><p>{html.escape(q["explanation"])}</p>{rubric}{link}</div></fieldset>'
        )
    return "\n".join(cards)


def artifacts(certificate, root=ROOT):
    provenance = input_hashes(root)
    certificate = {**certificate, "provenance": provenance}
    steps = source_steps(root)
    quiz = json.loads((root / "viewer/quiz.json").read_text(encoding="utf-8"))
    annotated = [
        "# GENERATED READING COMPANION — not a standalone solver.",
        "# Each excerpt is verbatim production source; comments and ordering are explanatory.",
        "# Imports and other context may be omitted. Repeated functions illustrate different steps.",
        "# SHA-256 source provenance is recorded in manifest.json and verification.json.", "",
    ]
    for step in steps:
        annotated.extend([f"# {step['route'].upper()} / {step['title']}",
                          f"# {step['path']}:{step['start_line']}-{step['end_line']}",
                          f"# {step['explanation']}", step["source"], ""])
    template = (root / "viewer/template.html").read_text(encoding="utf-8")
    replacements = {
        "@@STEPS@@": render_steps(steps), "@@CERTIFICATE@@": render_certificate(certificate),
        "@@QUIZ@@": render_quiz(quiz),
        "@@QUIZ_DATA@@": json.dumps(quiz, ensure_ascii=False).replace("<", "\\u003c"),
    }
    for marker, content in replacements.items():
        if template.count(marker) != 1:
            raise ValueError(f"Expected one template marker: {marker}")
        template = template.replace(marker, content)
    generated = {
        "model_viewer.html": template.encode("utf-8"),
        "model_annotated.py": ("\n".join(annotated).rstrip() + "\n").encode("utf-8"),
        "verification.json": json_bytes(certificate),
    }
    manifest = {
        "schema_version": 1, **provenance,
        "generated_files": {f"viewer/{name}": digest(data) for name, data in generated.items()},
        "manifest_note": "The manifest does not hash itself. All paths are repository-relative.",
        "source_excerpts": [{key: s[key] for key in ("id", "path", "symbol", "start_line", "end_line")} for s in steps],
    }
    generated["manifest.json"] = json_bytes(manifest)
    return generated


def check_artifacts(root=ROOT):
    """Check source freshness and reproduce rendering from recorded evidence."""
    directory = root / "viewer"
    certificate = json.loads((directory / "verification.json").read_text(encoding="utf-8"))
    if certificate.get("provenance") != input_hashes(root):
        raise ValueError("Stale source/build-input hashes; regenerate the viewer")
    if not certificate.get("passed"):
        raise ValueError("The recorded certificate does not pass")
    for name, expected in artifacts(certificate, root).items():
        if (directory / name).read_bytes() != expected:
            raise ValueError(f"Stale or modified generated artifact: {name}")
    fresh = run_verification()
    if not fresh["passed"]:
        raise ValueError(f"Fresh numerical verification failed: {fresh['failure']}")
    for field in ("fixture", "contract"):
        if fresh[field] != certificate[field]:
            raise ValueError(f"Recorded {field} differs from the current experiment")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check freshness and rerun verification without writing")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "viewer")
    args = parser.parse_args(argv)
    if args.check:
        try:
            check_artifacts()
        except (ValueError, OSError, KeyError) as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            return 1
        print("PASS: source hashes, generated artifacts, and fresh numerical verification")
        return 0
    certificate = run_verification()
    # Write numerical evidence first so a failed run can never leave a stale PASS certificate.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "verification.json").write_bytes(json_bytes({**certificate, "provenance": input_hashes()}))
    for name, data in artifacts(certificate).items():
        (args.output_dir / name).write_bytes(data)
    print(f"{'PASS' if certificate['passed'] else 'FAIL'}: viewer and certificate generated in {args.output_dir}")
    return 0 if certificate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

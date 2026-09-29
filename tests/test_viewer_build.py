import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("builder", ROOT / "scripts/build_model_viewer.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def test_rebuilds_are_byte_identical_in_same_environment(tmp_path):
    snapshots = []
    for name in ("first", "second"):
        output = tmp_path / name
        subprocess.run([sys.executable, str(ROOT / "scripts/build_model_viewer.py"),
                        "--output-dir", str(output)], check=True, capture_output=True)
        snapshots.append({p.name: p.read_bytes() for p in output.iterdir()})
    assert snapshots[0] == snapshots[1]


def test_certificate_round_trip_does_not_change_rendering():
    certificate = builder.run_verification()
    loaded = json.loads(builder.json_bytes(certificate))
    assert builder.artifacts(certificate) == builder.artifacts(loaded)


def test_checked_in_artifacts_are_fresh():
    builder.check_artifacts()


def test_failed_build_writes_failure_evidence_and_exits_nonzero(tmp_path, monkeypatch):
    failed = builder.run_verification()
    failed.update(passed=False, failure="Injected non-convergence")
    failed["solvers"]["exact_hat"]["converged"] = False
    monkeypatch.setattr(builder, "run_verification", lambda: failed)
    assert builder.main(["--output-dir", str(tmp_path)]) == 1
    certificate = json.loads((tmp_path / "verification.json").read_text(encoding="utf-8"))
    assert not certificate["passed"]
    page = (tmp_path / "model_viewer.html").read_text(encoding="utf-8")
    assert 'id="certificate-status">FAIL' in page
    assert "Injected non-convergence" in page


@pytest.mark.parametrize("tamper", ["src/ek_model/full_solution.py", "viewer/model_viewer.html"])
def test_freshness_rejects_stale_source_or_artifact(tmp_path, tamper):
    # Copy only the small build inputs and outputs, not the virtual environment.
    for directory in ("src", "scripts", "viewer"):
        shutil.copytree(ROOT / directory, tmp_path / directory, ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("pyproject.toml", "requirements-repro.txt"):
        shutil.copyfile(ROOT / name, tmp_path / name)
    path = tmp_path / tamper
    path.write_bytes(path.read_bytes() + b"\n# modified\n")
    with pytest.raises(ValueError, match="Stale"):
        builder.check_artifacts(tmp_path)


def test_excerpts_are_literal_and_highlights_exist():
    for step in builder.source_steps():
        lines = (ROOT / step["path"]).read_text(encoding="utf-8").splitlines()
        assert step["source"] == "\n".join(lines[step["start_line"] - 1:step["end_line"]])
        assert all(marker in step["source"] for marker in step["highlight"])


def test_viewer_is_self_contained_and_quiz_has_eight_questions():
    page = (ROOT / "viewer/model_viewer.html").read_text(encoding="utf-8")
    assert "connect-src 'none'" in page
    for external in ('<script src=', '<link ', 'fetch(', 'localStorage', 'sessionStorage'):
        assert external not in page
    quiz = json.loads((ROOT / "viewer/quiz.json").read_text(encoding="utf-8"))
    assert len(quiz) == len({q["id"] for q in quiz}) == 8
    assert {q["kind"] for q in quiz} == {"choice", "code", "written"}
    for question in quiz:
        assert question["explanation"]
        if question["kind"] == "written":
            assert question["rubric"]
        else:
            assert 0 <= question["answer"] < len(question["options"])

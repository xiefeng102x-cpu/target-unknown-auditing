"""The public checkout contains code/evidence summaries, not paper sources/PDFs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import verify_srac_release_20261019 as verifier  # noqa: E402


def write_manifest(root: Path) -> None:
    rows = []
    for path in root.rglob("*"):
        if path.is_file() and path.name != verifier.MANIFEST:
            content = path.read_bytes()
            rows.append({"path": path.relative_to(root).as_posix(), "size": len(content),
                         "sha256": hashlib.sha256(content).hexdigest()})
    (root / verifier.MANIFEST).write_text(json.dumps({
        "schema": verifier.SCHEMA, "file_count": len(rows), "files": rows,
    }), encoding="utf-8")


def test_checkout_has_no_paper_payload() -> None:
    # Also cover the repository's legacy area when this package is in a clone;
    # an independently extracted package must not inspect unrelated siblings.
    scope = ROOT.parent if (ROOT.parent / ".git").exists() else ROOT
    files, links, _ = verifier.inventory(scope)
    assert not links
    assert not [name for name in files if Path(name).suffix.lower() in {".tex", ".pdf"}]


@pytest.mark.parametrize("suffix", (".tex", ".pdf", ".TEX", ".PDF"))
def test_declared_paper_file_is_rejected(tmp_path: Path, suffix: str) -> None:
    name = "manuscript" + suffix
    (tmp_path / name).write_text("not a distributed paper", encoding="utf-8")
    write_manifest(tmp_path)
    report = verifier.integrity(tmp_path)
    assert report["status"] == "FAIL"
    assert name in report["forbidden"]
    assert not report["hash_errors"]  # rejection is policy, not accidental hash failure


def test_legacy_publication_requirements_are_filtered_narrowly() -> None:
    paths = ("src/run_audit.py", "tests/test_exact.py", "protocol.json",
             "manuscript/main.tex", "manuscript/main.PDF",
             "manuscript/main_cert_zh_audit_receipt.json", "manuscript/MANUSCRIPT_SHA256.txt")
    assert verifier.code_only_static_required(paths) == paths[:3]


def test_code_only_manifest_still_checks_hashes(tmp_path: Path) -> None:
    code = tmp_path / "audit.py"
    code.write_text("print('audit')\n", encoding="utf-8")
    write_manifest(tmp_path)
    assert verifier.integrity(tmp_path)["status"] == "PASS_INTEGRITY_ONLY"
    code.write_text("print('tampered')\n", encoding="utf-8")
    report = verifier.integrity(tmp_path)
    assert report["status"] == "FAIL"
    assert report["hash_errors"] == ["audit.py"]


def test_local_virtual_environment_is_not_a_release_payload(tmp_path: Path) -> None:
    (tmp_path / "audit.py").write_text("print('audit')\n", encoding="utf-8")
    write_manifest(tmp_path)
    env = tmp_path / ".venv" / "lib"
    env.mkdir(parents=True)
    (env / "local_dependency.py").write_text("# installed locally\n", encoding="utf-8")
    assert verifier.integrity(tmp_path)["status"] == "PASS_INTEGRITY_ONLY"

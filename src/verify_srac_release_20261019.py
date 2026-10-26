#!/usr/bin/env python3
"""Verify the code-only public 2026-10-19 release, without private inputs.

Default: manifest/hygiene checks, data-independent statistical regression tests,
UI v1.1's rational reference suites (first and second), public-checkout endpoint smoke
tests, robust/directional self-tests, synthetic count fixtures, and the mechanism
Fisher self-test. ``--integrity-only`` never
reports a full validation PASS. Source hashes in the manifest are provenance
metadata; only distributed bytes can be independently checked in a checkout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "RELEASE_MANIFEST.json"
SCHEMA = "target-unknown-auditing-code-20261019"
PAPER_SUFFIXES = {".tex", ".pdf"}
IGNORED_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
OS_JUNK = {".DS_Store", "__MACOSX", "Thumbs.db", "desktop.ini"}
MODEL_SUFFIXES = {
    ".pt", ".pth", ".ckpt", ".onnx", ".h5", ".hdf5", ".safetensors",
    ".pkl", ".pickle", ".joblib", ".npy", ".npz", ".bin",
    ".zip", ".tar", ".gz", ".tgz", ".7z", ".rar", ".bz2", ".xz",
}
PRIVATE_NAMES = {
    "flags", "decisions", "scores", "raw_scores", "raw_score", "labels",
    "poison_ids", "poison_indices", "private_labels", "private_keys",
    "sealed_decisions", "sealed_per_decision", "credentials", "id_rsa", "id_ed25519",
}
SECRETS = {
    "llm_api_key": re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_-]{20,}\b"),
    "github_token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "private_key_header": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----"),
}
HEX256 = re.compile(r"[0-9a-f]{64}\Z")
HISTORICAL_DRAW_TABLES = set()
METHOD_TESTS = (
    "auditability_certificate_v1/tests/test_uniform_adjudication_certificate.py",
    "auditability_certificate_v1/tests/test_uniform_adjudication_exact.py",
    "auditability_certificate_v1/tests/test_protected_group_family_srac.py",
    "auditability_certificate_v1/tests/test_robust_interval_hull.py",
    "auditability_certificate_v1/tests/test_public_endpoint_compat.py",
    "tests/test_code_only_release.py",
    "tests/test_synthetic_certificate_records.py",
    "tests/test_verifier_selftest_contracts.py",
)
MIN_REGRESSIONS_PASSED = 43


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def redact(value):
    """Never echo a matched credential, including one embedded in a filename."""
    if isinstance(value, str):
        for pattern in SECRETS.values():
            value = pattern.sub("[REDACTED]", value)
        return value
    if isinstance(value, list):
        return [redact(x) for x in value]
    if isinstance(value, dict):
        return {redact(str(k)): redact(v) for k, v in value.items()}
    return value


def safe_relative(name) -> bool:
    if not isinstance(name, str) or not name or "\\" in name or ":" in name or "\x00" in name:
        return False
    p = PurePosixPath(name)
    return not p.is_absolute() and all(x not in {"", ".", ".."} for x in name.split("/"))


def prohibited(name: str) -> bool:
    p = PurePosixPath(name)
    if any(x.lower().startswith("manuscript") for x in p.parts):
        return True
    if any(x.lower() in {"data", "datasets", "models", "checkpoints", "audits"}
           or x.lower().startswith(("results", "human_pilot")) for x in p.parts):
        return True
    if any(x in OS_JUNK or x.startswith("._") for x in p.parts):
        return True
    if p.suffix.lower() in MODEL_SUFFIXES | PAPER_SUFFIXES or p.suffix.lower() in {".pem", ".key", ".p12", ".pfx"}:
        return True
    stem = p.stem.lower()
    if stem in PRIVATE_NAMES or stem.startswith(("flags_", "decisions_", "raw_scores_", "sealed_", "private_key")):
        return True
    if any(re.search(r"private.*keys?", x, re.I) for x in p.parts):
        return True
    return p.name == ".env" or p.name.startswith(".env.")


def code_only_static_required(paths):
    """Adapt only the historical publication surface, never statistical tests."""
    obsolete = {"main_cert_zh_audit_receipt.json", "MANUSCRIPT_SHA256.txt"}
    return tuple(name for name in paths
                 if PurePosixPath(name).suffix.lower() not in PAPER_SUFFIXES
                 and PurePosixPath(name).name not in obsolete)


def inventory(root: Path):
    files, links, junk_dirs = set(), [], []
    for base, dirs, names in os.walk(root, followlinks=False):
        parent = Path(base)
        kept = []
        for name in dirs:
            p = parent / name
            rel = p.relative_to(root).as_posix()
            if name in IGNORED_DIRS:
                continue
            if p.is_symlink():
                links.append(rel)
            elif name in OS_JUNK or name.startswith("._"):
                junk_dirs.append(rel)
            else:
                kept.append(name)
        dirs[:] = kept
        for name in names:
            p = parent / name
            rel = p.relative_to(root).as_posix()
            if rel == ".git":  # a worktree's .git pointer is not a distributed artifact
                continue
            if p.is_symlink():
                links.append(rel)
            elif rel != MANIFEST:
                files.add(rel)
    return files, links, junk_dirs


def integrity(root: Path) -> dict:
    errors = []
    mp = root / MANIFEST
    if mp.is_symlink() or not mp.is_file():
        return {"status": "FAIL", "errors": ["manifest missing or symlinked"]}
    try:
        m = json.loads(mp.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError):
        return {"status": "FAIL", "errors": ["manifest unreadable or invalid JSON"]}
    if not isinstance(m, dict) or m.get("schema") != SCHEMA:
        return {"status": "FAIL", "errors": ["unexpected manifest schema"]}
    rows = m.get("files")
    if not isinstance(rows, list) or not rows:
        return {"status": "FAIL", "errors": ["manifest files must be a nonempty list"]}
    expected = set()
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or not safe_relative(row.get("path")):
            errors.append(f"invalid manifest row/path at index {i}")
            continue
        name = row["path"]
        if name == MANIFEST or any(x in IGNORED_DIRS for x in PurePosixPath(name).parts):
            errors.append(f"manifest declares excluded path: {name}")
        if name in expected:
            errors.append(f"duplicate manifest path: {name}")
        expected.add(name)
        if not isinstance(row.get("size"), int) or isinstance(row.get("size"), bool) or row["size"] < 0:
            errors.append(f"invalid distributed size: {name}")
        if not isinstance(row.get("sha256"), str) or not HEX256.fullmatch(row["sha256"]):
            errors.append(f"invalid distributed hash: {name}")
        if "source_sha256" in row and (not isinstance(row["source_sha256"], str) or not HEX256.fullmatch(row["source_sha256"])):
            errors.append(f"invalid source hash metadata: {name}")
        if "source_size" in row and (not isinstance(row["source_size"], int) or isinstance(row["source_size"], bool) or row["source_size"] < 0):
            errors.append(f"invalid source size metadata: {name}")
        if row.get("frozen_bytes_preserved") is True and (row.get("source_sha256") != row.get("sha256") or row.get("source_size") != row.get("size")):
            errors.append(f"frozen source/distribution binding differs: {name}")
    if m.get("file_count") != len(rows):
        errors.append("manifest file_count inconsistent")
    observed, links, junk_dirs = inventory(root)
    missing, unexpected = sorted(expected - observed), sorted(observed - expected)
    hash_errors, size_errors, forbidden, secret_hits = [], [], list(junk_dirs), []
    by_name = {row["path"]: row for row in rows if isinstance(row, dict) and safe_relative(row.get("path"))}
    for name in sorted(observed):
        path = root / name
        if prohibited(name):
            forbidden.append(name)
        row = by_name.get(name)
        if row:
            if path.stat().st_size != row.get("size"):
                size_errors.append(name)
            if digest(path) != row.get("sha256"):
                hash_errors.append(name)
        # Decode all byte streams, including PDFs: credentials must not be hidden
        # merely by changing an extension. Never emit the matching byte string.
        text = path.read_bytes().decode("utf-8", errors="ignore")
        if path.suffix.lower() == ".csv":
            try:
                columns = {x.strip().lower() for x in next(csv.reader(text.splitlines()))}
            except (StopIteration, csv.Error):
                columns = set()
            row_identifiers = {"sample_id", "candidate_id", "candidate", "is_poison", "poison_label"}
            if name not in HISTORICAL_DRAW_TABLES:
                row_identifiers.add("draw")
            if columns & row_identifiers:
                forbidden.append(name)
        for label, pattern in SECRETS.items():
            if pattern.search(text):
                secret_hits.append({"path": name, "pattern": label})
    # Manifest fields may themselves contain accidentally copied credentials.
    mt = mp.read_text(encoding="utf-8")
    for label, pattern in SECRETS.items():
        if pattern.search(mt):
            secret_hits.append({"path": MANIFEST, "pattern": label})
    failed = any((errors, missing, unexpected, links, forbidden, hash_errors, size_errors, secret_hits))
    return dict(status="FAIL" if failed else "PASS_INTEGRITY_ONLY", file_count=len(observed),
                errors=errors, missing=missing, undeclared=unexpected, symlinks=sorted(links),
                forbidden=sorted(set(forbidden)), hash_errors=hash_errors, size_errors=size_errors,
                secret_hits=secret_hits, source_metadata="syntax checked; original source bytes not available")


def selftest_contract_error(name: str, parsed) -> str | None:
    """Require each new self-test's actual schema, not just a success exit code."""
    if not isinstance(parsed, dict) or parsed.get("passed") is not True:
        return "self-test did not report passed=true"
    if name in {"ui_robust", "ui_directional"}:
        pairs = parsed.get("pairs_checked")
        if not isinstance(pairs, int) or isinstance(pairs, bool) or pairs <= 0:
            return "self-test did not check any count pairs"
        if type(parsed.get("violations")) is not int or parsed["violations"] != 0:
            return "robustness self-test reports violations"
        reduction = "reduces_to_engine_at_R0" if name == "ui_robust" else "never_more_than_engine_at_B0"
        if parsed.get(reduction) is not True:
            return "self-test reduction or dominance check is missing or failed"
    if name == "synthetic_certificate_records":
        count = parsed.get("records_checked")
        if parsed.get("synthetic") is not True or not isinstance(count, int) or isinstance(count, bool) or count < 6:
            return "synthetic fixture schema or count is incomplete"
        if type(parsed.get("records_failed")) is not int or parsed["records_failed"] != 0:
            return "synthetic fixture decisions differ from expectations"
        decisions = parsed.get("decisions")
        if not isinstance(decisions, list) or len(decisions) != count:
            return "synthetic fixture decision list is incomplete"
        if any(not isinstance(row, dict) or row.get("actual") != row.get("expected")
               or row.get("actual") not in {"GO", "NO_GO", "ABSTAIN"} for row in decisions):
            return "synthetic fixture decision content is invalid"
    return None


def run_step(root: Path, name: str, arguments: list[str], timeout: int) -> dict:
    command = [sys.executable, "-B", *arguments]
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTEST_ADDOPTS="-p no:cacheprovider",
               PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", SRAC_PUBLIC_CLONE_VERIFY="1")
    env.pop("PYTHONPATH", None)
    started = time.monotonic()
    try:
        p = subprocess.run(command, cwd=root, env=env, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return dict(name=name, status="FAIL", reason="timeout", timeout_seconds=timeout)
    # Do not echo child output: a failed test or exception could print a secret.
    result = dict(name=name, status="PASS" if p.returncode == 0 else "FAIL", returncode=p.returncode,
                  duration_seconds=round(time.monotonic() - started, 3),
                  stdout_sha256=hashlib.sha256(p.stdout).hexdigest(),
                  stderr_sha256=hashlib.sha256(p.stderr).hexdigest())
    try:
        parsed = json.loads(p.stdout)
    except (ValueError, UnicodeError):
        parsed = None
    if name == "method_and_code_only_regressions":
        counts = re.findall(r"\b(\d+) passed\b", p.stdout.decode("utf-8",errors="replace"))
        if p.returncode != 0 or not counts or int(counts[-1]) < MIN_REGRESSIONS_PASSED:
            result.update(status="FAIL", reason="required statistical and code-only regression suite did not pass")
        else:
            result["tests_passed"] = int(counts[-1])
    else:
        contract_error = selftest_contract_error(name, parsed)
        if contract_error is not None:
            result.update(status="FAIL", reason=contract_error)
        for key in ("T1_comparisons", "T1_zero_observation_outcomes", "T6_mutant_failures_detected", "T3_intervals", "T4_census_populations", "n_failures", "truth_points", "max_error_over_alpha", "T2_max_error_over_alpha", "pairs_checked", "violations", "records_checked", "records_failed"):
            if isinstance(parsed, dict) and isinstance(parsed.get(key), (int, float)):
                result[key] = parsed[key]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--integrity-only", action="store_true")
    parser.add_argument("--test-timeout", type=int, default=120, help="seconds for each new self-test")
    parser.add_argument("--regression-timeout", type=int, default=300)
    args = parser.parse_args()
    report = {"status": "FAIL", "full_validation": False, "steps": []}
    try:
        root = args.root.resolve(strict=True)
        if not root.is_dir() or min(args.test_timeout, args.regression_timeout) <= 0:
            raise ValueError("invalid root or timeout")
        report["integrity"] = integrity(root)
        if report["integrity"]["status"] != "PASS_INTEGRITY_ONLY":
            return emit(report, 1)
        if args.integrity_only:
            report["status"] = "PASS_INTEGRITY_ONLY"
            report["not_verified"] = ["method, endpoint-smoke and code-only regression suite", "UI v1.1 rational-reference suites", "mechanism Fisher self-test", "robust and directional self-tests", "synthetic count-to-decision fixtures"]
            return emit(report, 0)
        missing_tests = [name for name in METHOD_TESTS if not (root/name).is_file()]
        if missing_tests:
            raise FileNotFoundError("required method tests are missing")
        steps = [
            ("method_and_code_only_regressions", ["-m", "pytest", *METHOD_TESTS, "-q", "-p", "no:cacheprovider"], args.regression_timeout),
            ("ui_v1_1", ["auditability_certificate_v1/analysis/test_srac_ui_engine_v1_1.py"], args.test_timeout),
            ("ui_v1_1_suite2", ["auditability_certificate_v1/analysis/test_srac_ui_engine_v1_1_suite2.py"], args.test_timeout),
            ("mechanism_fisher", ["auditability_certificate_v1/analysis/run_ui_mechanism_ablation_v1.py", "selftest"], args.test_timeout),
            ("ui_robust", ["auditability_certificate_v1/analysis/srac_ui_robust_v1.py"], args.test_timeout),
            ("ui_directional", ["auditability_certificate_v1/analysis/srac_ui_directional_v1.py"], args.test_timeout),
            ("synthetic_certificate_records", ["auditability_certificate_v1/analysis/replay_synthetic_certificate_records_v1.py"], args.test_timeout),
        ]
        for name, arguments, timeout in steps:
            result = run_step(root, name, arguments, timeout)
            report["steps"].append(result)
            if result["status"] != "PASS":
                return emit(report, 1)
        report["post_test_integrity"] = integrity(root)
        if report["post_test_integrity"]["status"] == "PASS_INTEGRITY_ONLY":
            report.update(status="PASS_FULL_VALIDATION", full_validation=True)
        return emit(report, 0 if report["full_validation"] else 1)
    except (OSError, ValueError, TypeError, KeyError) as err:
        report["error"] = type(err).__name__
        return emit(report, 1)


def emit(report: dict, code: int) -> int:
    print(json.dumps(redact(report), indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())

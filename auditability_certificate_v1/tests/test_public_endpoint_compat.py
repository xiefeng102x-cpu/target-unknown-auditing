"""Public-checkout smoke tests for frozen endpoint runners (no private inputs).

* Without the wrapper, runners that re-check the frozen hash of a path-normalized file stop at
  that check; through analysis/public_compat_runner.py they pass it and stop only at the
  missing controlled input.
* A normalized file whose bytes differ from the declared public hash is still rejected.
* The block-16 and block-12 endpoints run end to end on synthetic flags with a reduced copy of
  their frozen protocol (fewer budgets and draws, same code, same hash checks).
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
ANALYSIS = REPO / "auditability_certificate_v1/analysis"
WRAPPER = ANALYSIS / "public_compat_runner.py"
HASH_BLOCKED = ("run_gtsrb_srac_ui_confirmation.py", "run_cifar10_scan_lda_secondary_escalation.py",
                "run_cifar10_adapblend_secondary.py")
INPUT_ONLY = ("run_cifar10_scan_lda_confirmation.py", "run_cifar10_adapblend_scan_lda_confirmation.py")


def _run(args, cwd=REPO):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONPATH", None)
    return subprocess.run([sys.executable, "-B", *map(str, args)], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=600)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _missing_input(stderr: str) -> bool:
    last = stderr.strip().splitlines()[-1]
    return last.startswith("FileNotFoundError") and "manifest.json" in last


@pytest.mark.parametrize("entry", HASH_BLOCKED)
def test_hash_gate_blocks_without_wrapper(entry):
    p = _run([ANALYSIS / entry])
    assert p.returncode != 0 and "hash mismatch" in p.stderr.strip().splitlines()[-1]


@pytest.mark.parametrize("entry", HASH_BLOCKED + INPUT_ONLY)
def test_wrapper_passes_hash_gate_and_stops_at_missing_input(entry):
    p = _run([WRAPPER, ANALYSIS / entry])
    assert p.returncode != 0 and "hash mismatch" not in p.stderr
    assert _missing_input(p.stderr), p.stderr[-400:]
    assert ("[public-compat]" in p.stderr) == (entry in HASH_BLOCKED)


def test_wrapper_rejects_modified_normalized_file(tmp_path):
    copy_root = tmp_path / "repo"
    shutil.copytree(REPO, copy_root, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"))
    target = copy_root / "auditability_certificate_v1/analysis/run_posthoc_power_study_v1.py"
    target.write_bytes(target.read_bytes() + b"\n# modified\n")
    p = _run([copy_root / "auditability_certificate_v1/analysis/public_compat_runner.py",
              copy_root / "auditability_certificate_v1/analysis/run_gtsrb_srac_ui_confirmation.py"], cwd=copy_root)
    assert p.returncode != 0 and "endpoint-stage hash mismatch" in p.stderr.strip().splitlines()[-1]
    assert "[public-compat]" not in p.stderr


def _synthetic_flags(n_classes, target, n, victims, rng):
    rows = []
    for v, (seed, poison_recall) in enumerate(victims):
        label = rng.integers(0, n_classes, n)
        poison = np.zeros(n, dtype=bool)
        poison[rng.choice(np.flatnonzero(label != target), size=n // 20, replace=False)] = True
        label[poison] = target
        flag = np.where(poison, rng.random(n) < poison_recall, rng.random(n) < 0.004)
        rows.append(pd.DataFrame(dict(victim_id=f"synthetic_seed{seed}", seed=seed, sample_id=np.arange(n),
                                      assigned_group=label, flag=flag.astype(int), is_poison=poison.astype(int),
                                      victim_eligible=1)))
    return pd.concat(rows, ignore_index=True)


def _write_protocol(src: Path, dst: Path, budgets, draws):
    proto = copy.deepcopy(json.loads(src.read_text(encoding="utf-8")))
    proto["certificate"]["budgets"] = list(budgets)
    proto["certificate"]["protected_draws_per_seed_budget"] = draws
    dst.write_text(json.dumps(proto, indent=2), encoding="utf-8")
    (dst.parent / (dst.name + ".sha256")).write_text(f"{_sha(dst)}  {dst.name}\n", encoding="utf-8")
    return proto


def _write_scores(directory: Path, flags: pd.DataFrame, protocol: Path, status=None):
    directory.mkdir(parents=True)
    flags.to_csv(directory / "flags.csv", index=False)
    man = dict(protocol_sha256=_sha(protocol), flags_sha256=_sha(directory / "flags.csv"))
    if status:
        man["status"] = status
    (directory / "manifest.json").write_text(json.dumps(man), encoding="utf-8")


def test_block16_endpoint_end_to_end_on_synthetic_flags(tmp_path, monkeypatch):
    sys.path.insert(0, str(ANALYSIS))
    import public_compat_runner as W
    runner, log = W.load_entry(ANALYSIS / "run_gtsrb_srac_ui_confirmation.py")
    protocol = tmp_path / "protocol.json"
    proto = _write_protocol(runner.PROTOCOL, protocol, budgets=(2000,), draws=2)
    flags = _synthetic_flags(43, 38, 6000, ((1, 0.95), (2, 0.60)), np.random.default_rng(1))
    _write_scores(tmp_path / "scores", flags, protocol)
    monkeypatch.setattr(runner, "PROTOCOL", protocol)
    monkeypatch.setattr(runner, "SCORES", tmp_path / "scores")
    monkeypatch.setattr(runner, "OUTPUT", tmp_path / "out")
    monkeypatch.setattr(sys, "argv", ["run_gtsrb_srac_ui_confirmation.py", "--procs", "1"])
    runner.main()
    assert log == ["auditability_certificate_v1/analysis/run_posthoc_power_study_v1.py"]
    rep = json.loads((tmp_path / "out/report.json").read_text())
    dec = pd.read_csv(tmp_path / "out/decisions.csv")
    assert len(dec) == 2 * 3 * 1 * 2 and set(dec.engine) == {"srac_ui", "srac", "rla_bet"}
    assert set(dec.decision) <= {"GO", "NO_GO", "ABSTAIN"}
    assert rep["passing_victims"] == 1 and set(rep["endpoints"]) == {"V_validity", "P1_srac_ui_vs_srac_k2000", "P2_srac_ui_vs_rla_k2000"}
    assert rep["endpoints"]["V_validity"] is True
    assert proto["hashes"] == json.loads(runner.PROTOCOL.read_text())["hashes"]


def test_block12_endpoint_end_to_end_on_synthetic_flags(tmp_path):
    frozen = REPO / "auditability_certificate_v1/PROTOCOL_CIFAR10_SCAN_LDA_CONFIRMATION_V1.json"
    protocol = tmp_path / "protocol.json"
    _write_protocol(frozen, protocol, budgets=(2000, 8000), draws=1)
    flags = _synthetic_flags(10, 8, 12000, ((1, 0.95), (2, 0.60)), np.random.default_rng(2))
    _write_scores(tmp_path / "scores", flags, protocol, status="CIFAR10_SCAN_LDA_HANDOFF_COMPLETE")
    p = _run([WRAPPER, ANALYSIS / "run_cifar10_scan_lda_confirmation.py", "--protocol", protocol,
              "--scores", tmp_path / "scores", "--output", tmp_path / "out"])
    assert p.returncode == 0, p.stderr[-800:]
    rep = json.loads((tmp_path / "out/report.json").read_text())
    dec = pd.read_csv(tmp_path / "out/decisions.csv")
    assert len(dec) == 2 * 2 * 1 and rep["true_policy_pass_victims"] == 1
    assert rep["support_checks"]["zero_false_go"] and rep["support_checks"]["zero_false_nogo"]

from __future__ import annotations

from pathlib import Path
import importlib.util
import json

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "analysis/finalize_cifar10_vgg16_blended_srac_confirmation.py"
SPEC = importlib.util.spec_from_file_location("blended_finalizer", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    protocol = tmp_path / "protocol.json"
    results = tmp_path / "results"
    endpoint = tmp_path / "endpoint"
    write_json(protocol, {
        "formal_seeds": [1], "crossfit": {"folds": 1, "expected_rows": 10,
        "expected_clean_rows": 8, "expected_poison_rows": 2}, "detectors": ["d"],
        "srac_evaluation": {"audit_budgets": [2], "protected_draws_per_seed_fold_budget": 2,
        "audit_universe": "whole population"},
        "attack": {"implementation_name": "variant", "fidelity_boundary": "bounded"},
    })
    write_json(results / "manifest.json", {
        "status": "SCORE_HANDOFF_COMPLETE", "completed_seeds": [1], "all_victims_eligible": True,
    })
    write_json(results / "srac_handoff_verification.json", {"status": "PASS", "verified_seeds": [1]})
    write_json(results / "checkpoint_score_semantic_replay_verification.json", {
        "status": "PASS", "verified_seeds": [1], "per_seed": [{
            "score_replay": {"raw_sha256": {"byte_exact": False},
            "score_max_absolute_difference": {"score": 1e-12},
            "exact_field_mismatches": {"conventional_flag": 0}}
        }],
    })
    write_json(results / "seed_receipts.json", [{
        "seed": 1, "victim_metrics": {"balanced_accuracy": 0.9, "attack_success_rate": 0.95}
    }])
    write_json(endpoint / "report.json", {
        "status": "COMPLETE_FROZEN_ENDPOINT", "duplicate_decision_keys": 0,
        "decision_rows": 2, "by_budget": {"2": {"rows": 2}},
    })
    rows = pd.DataFrame([
        {"detector": "d", "actual_recall": 0.9, "actual_clean_flag_rate": 0.05,
         "decision": "GO", "false_go": 0, "false_nogo": 0},
        {"detector": "d", "actual_recall": 0.9, "actual_clean_flag_rate": 0.05,
         "decision": "GO", "false_go": 0, "false_nogo": 0},
    ])
    endpoint.mkdir(parents=True, exist_ok=True)
    rows.to_csv(endpoint / "decisions.csv", index=False)
    rows.iloc[:1].to_csv(endpoint / "summary.csv", index=False)
    rows.iloc[:1].to_csv(endpoint / "actual_ranges.csv", index=False)
    return protocol, results, endpoint


def test_finalizer_reports_semantic_not_byte_exact_pass(tmp_path: Path) -> None:
    protocol, results, endpoint = fixture(tmp_path)
    report = MODULE.finalize(protocol, results, endpoint)
    assert report["status"] == "PASS"
    assert report["verification"]["prospective_semantic_replay"] == "PASS"
    assert report["verification"]["raw_score_tables_byte_exact"] is False
    assert report["verification"]["exact_field_mismatch_count"] == 0


def test_finalizer_fails_closed_on_replay_failure(tmp_path: Path) -> None:
    protocol, results, endpoint = fixture(tmp_path)
    replay = results / "checkpoint_score_semantic_replay_verification.json"
    payload = json.loads(replay.read_text())
    payload["status"] = "FAIL"
    write_json(replay, payload)
    with pytest.raises(RuntimeError, match="expected 'PASS'"):
        MODULE.finalize(protocol, results, endpoint)


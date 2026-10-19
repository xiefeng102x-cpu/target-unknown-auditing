from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "auditability_certificate_v1/analysis/verify_cifar10_semantic_overlap_artifacts.py"
SPEC = importlib.util.spec_from_file_location("verify_cifar_semantic_artifacts", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def fixture_tree(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    protocol_path = tmp_path / "protocol.json"
    victims = tmp_path / "victims"
    scores = tmp_path / "scores"
    endpoint = tmp_path / "endpoint"
    victims.mkdir(); scores.mkdir(); endpoint.mkdir()
    protocol = {
        "formal_seeds": [1],
        "protected_groups": [{"name": "all", "class_ids": [0, 1]}],
        "certificate": {
            "budgets": [2000, 8000],
            "protected_draws_per_seed_budget": 2,
            "recall_target": 0.8,
            "max_group_clean_flag_rate_cap": 0.1,
        },
        "support_gate": {
            "minimum_eligible_victims": 1,
            "minimum_true_policy_pass_victims": 1,
            "zero_false_go": True,
            "zero_false_nogo": True,
            "minimum_victims_with_go_at_k_8000": 1,
        },
    }
    write_json(protocol_path, protocol)
    protocol_hash = digest(protocol_path)
    protocol_path.with_suffix(".json.sha256").write_text(protocol_hash + "  protocol.json\n")
    receipts = [{"seed": 1}]
    write_json(victims / "seed_receipts.json", receipts)
    victim_manifest = {
        "status": "SCORE_HANDOFF_COMPLETE",
        "freeze_manifest_sha256": protocol_hash,
        "completed_seeds": [1],
        "seed_receipts_sha256": digest(victims / "seed_receipts.json"),
    }
    write_json(victims / "manifest.json", victim_manifest)
    flags = pd.DataFrame(
        {
            "victim_id": ["v1"] * 4,
            "seed": [1] * 4,
            "sample_id": [10, 11, 12, 13],
            "assigned_group": [0, 0, 1, 1],
            "flag": [1, 1, 0, 0],
            "is_poison": [1, 1, 0, 0],
            "victim_eligible": [1] * 4,
        }
    )
    flags.to_csv(scores / "flags.csv", index=False)
    score_manifest = {
        "status": "CIFAR10_SEMANTIC_OVERLAP_SCAN_HANDOFF_COMPLETE",
        "protocol_sha256": protocol_hash,
        "victim_manifest_sha256": digest(victims / "manifest.json"),
        "flags_sha256": digest(scores / "flags.csv"),
    }
    write_json(scores / "manifest.json", score_manifest)
    truth = pd.DataFrame(
        [{"victim_id": "v1", "actual_recall": 1.0, "actual_max_group_clean_rate": 0.0, "true_policy_pass": 1}]
    )
    truth.to_csv(endpoint / "truth_by_victim.csv", index=False)
    decision_rows = []
    for k in [2000, 8000]:
        for draw in [0, 1]:
            decision_rows.append(
                {"victim_id": "v1", "k": k, "draw": draw, "decision": "GO", "false_go": 0, "false_nogo": 0}
            )
    decisions = pd.DataFrame(decision_rows)
    decisions.to_csv(endpoint / "decisions.csv", index=False)
    summary = pd.DataFrame(
        [
            {"k": k, "decisions": 2, "victims": 1, "go_rate": 1.0, "nogo_rate": 0.0,
             "abstain_rate": 0.0, "victims_with_go": 1, "false_go_count": 0, "false_nogo_count": 0}
            for k in [2000, 8000]
        ]
    )
    summary.to_csv(endpoint / "summary.csv", index=False)
    checks = {
        "minimum_eligible_victims": True,
        "minimum_true_policy_pass_victims": True,
        "zero_false_go": True,
        "zero_false_nogo": True,
        "minimum_victims_with_go_at_k_8000": True,
    }
    report = {
        "status": "PASS_CIFAR10_SEMANTIC_OVERLAP_CONFIRMATION",
        "protocol_sha256": protocol_hash,
        "score_manifest_sha256": digest(scores / "manifest.json"),
        "flags_sha256": digest(scores / "flags.csv"),
        "eligible_victims": 1,
        "true_policy_pass_victims": 1,
        "support_checks": checks,
        "outputs": {
            "decisions_sha256": digest(endpoint / "decisions.csv"),
            "truth_sha256": digest(endpoint / "truth_by_victim.csv"),
            "summary_sha256": digest(endpoint / "summary.csv"),
        },
    }
    write_json(endpoint / "report.json", report)
    return protocol_path, victims, scores, endpoint


def test_verifier_accepts_a_consistent_chain(tmp_path: Path) -> None:
    paths = fixture_tree(tmp_path)
    result = MODULE.verify(*paths)
    assert result["status"] == "VERIFIED"
    assert result["terminal_status"] == "PASS_CIFAR10_SEMANTIC_OVERLAP_CONFIRMATION"
    assert result["decision_rows"] == 4


def test_verifier_accepts_scan_lda_variant_and_k2000_gate(tmp_path: Path) -> None:
    protocol_path, victims, scores, endpoint = fixture_tree(tmp_path)
    protocol = json.loads(protocol_path.read_text())
    protocol["schema"] = "cifar10-scan-lda-semantic-overlap-confirmation-v1"
    protocol["support_gate"]["minimum_victims_with_go_at_k_2000"] = 1
    write_json(protocol_path, protocol)
    protocol_hash = digest(protocol_path)
    protocol_path.with_suffix(".json.sha256").write_text(protocol_hash + "  protocol.json\n")

    victim_manifest = json.loads((victims / "manifest.json").read_text())
    victim_manifest["freeze_manifest_sha256"] = protocol_hash
    write_json(victims / "manifest.json", victim_manifest)

    score_manifest = json.loads((scores / "manifest.json").read_text())
    score_manifest["status"] = "CIFAR10_SCAN_LDA_HANDOFF_COMPLETE"
    score_manifest["protocol_sha256"] = protocol_hash
    score_manifest["victim_manifest_sha256"] = digest(victims / "manifest.json")
    write_json(scores / "manifest.json", score_manifest)

    report = json.loads((endpoint / "report.json").read_text())
    report["status"] = "PASS_CIFAR10_SCAN_LDA_CONFIRMATION"
    report["protocol_sha256"] = protocol_hash
    report["score_manifest_sha256"] = digest(scores / "manifest.json")
    report["support_checks"]["minimum_victims_with_go_at_k_2000"] = True
    write_json(endpoint / "report.json", report)

    result = MODULE.verify(protocol_path, victims, scores, endpoint)
    assert result["terminal_status"] == "PASS_CIFAR10_SCAN_LDA_CONFIRMATION"
    assert result["support_checks"]["minimum_victims_with_go_at_k_2000"] is True


def test_verifier_rejects_cross_stage_hash_mismatch(tmp_path: Path) -> None:
    protocol, victims, scores, endpoint = fixture_tree(tmp_path)
    manifest = json.loads((scores / "manifest.json").read_text())
    manifest["victim_manifest_sha256"] = "0" * 64
    write_json(scores / "manifest.json", manifest)
    report = json.loads((endpoint / "report.json").read_text())
    report["score_manifest_sha256"] = digest(scores / "manifest.json")
    write_json(endpoint / "report.json", report)
    with pytest.raises(RuntimeError, match="score-to-victim hash mismatch"):
        MODULE.verify(protocol, victims, scores, endpoint)


def test_verifier_rejects_false_go_relabeling_even_when_rehashed(tmp_path: Path) -> None:
    protocol, victims, scores, endpoint = fixture_tree(tmp_path)
    decisions = pd.read_csv(endpoint / "decisions.csv")
    decisions.loc[0, "false_go"] = 1
    decisions.to_csv(endpoint / "decisions.csv", index=False)
    report = json.loads((endpoint / "report.json").read_text())
    report["outputs"]["decisions_sha256"] = digest(endpoint / "decisions.csv")
    write_json(endpoint / "report.json", report)
    with pytest.raises(RuntimeError, match="false-GO labels mismatch"):
        MODULE.verify(protocol, victims, scores, endpoint)

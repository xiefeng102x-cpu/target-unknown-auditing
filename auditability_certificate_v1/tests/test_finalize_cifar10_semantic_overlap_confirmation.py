from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "analysis/finalize_cifar10_semantic_overlap_confirmation.py"
SPEC = importlib.util.spec_from_file_location("finalize_cifar_semantic", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def test_finalizer_preserves_failed_outcome_and_recomputes_summary(tmp_path: Path) -> None:
    protocol_path = tmp_path / "protocol.json"
    protocol = {
        "study_id": "synthetic",
        "formal_seeds": [1],
        "certificate": {"budgets": [2, 4], "protected_draws_per_seed_budget": 2},
    }
    write_json(protocol_path, protocol)
    digest = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    protocol_path.with_suffix(".json.sha256").write_text(digest + "  protocol.json\n")

    victims = tmp_path / "victims"
    scores = tmp_path / "scores"
    endpoint = tmp_path / "endpoint"
    write_json(
        victims / "manifest.json",
        {
            "status": "SCORE_HANDOFF_COMPLETE",
            "freeze_manifest_sha256": digest,
            "completed_seeds": [1],
        },
    )
    write_json(
        victims / "seed_receipts.json",
        [
            {
                "seed": 1,
                "eligible": True,
                "victim_metrics": {"balanced_accuracy": 0.8, "attack_success_rate": 0.9},
            }
        ],
    )
    write_json(
        scores / "manifest.json",
        {
            "status": "CIFAR10_SEMANTIC_OVERLAP_SCAN_HANDOFF_COMPLETE",
            "protocol_sha256": digest,
        },
    )
    (scores / "flags.csv").write_text("flag\n1\n", encoding="utf-8")

    decisions = pd.DataFrame(
        [
            {"victim_id": "v1", "k": k, "draw": draw, "decision": decision, "false_go": 0, "false_nogo": 0}
            for k, decision in ((2, "ABSTAIN"), (4, "GO"))
            for draw in range(2)
        ]
    )
    endpoint.mkdir(parents=True)
    decisions.to_csv(endpoint / "decisions.csv", index=False)
    pd.DataFrame(
        [
            {"k": 2, "decisions": 2, "victims": 1, "go_rate": 0.0, "nogo_rate": 0.0, "abstain_rate": 1.0, "victims_with_go": 0, "false_go_count": 0, "false_nogo_count": 0},
            {"k": 4, "decisions": 2, "victims": 1, "go_rate": 1.0, "nogo_rate": 0.0, "abstain_rate": 0.0, "victims_with_go": 1, "false_go_count": 0, "false_nogo_count": 0},
        ]
    ).to_csv(endpoint / "summary.csv", index=False)
    pd.DataFrame(
        [{"victim_id": "v1", "actual_recall": 0.9, "actual_max_group_clean_rate": 0.05, "true_policy_pass": 1}]
    ).to_csv(endpoint / "truth_by_victim.csv", index=False)
    write_json(
        endpoint / "report.json",
        {
            "status": "FAIL_CIFAR10_SEMANTIC_OVERLAP_CONFIRMATION",
            "protocol_sha256": digest,
            "eligible_victims": 1,
            "support_checks": {"minimum_eligible_victims": False},
        },
    )

    report = MODULE.finalize(protocol_path, victims, scores, endpoint)
    assert report["status"] == "FAIL_CIFAR10_SEMANTIC_OVERLAP_CONFIRMATION"
    assert report["outcome_neutral_finalizer"] is True
    assert report["endpoint"]["decision_rows"] == 4
    assert report["endpoint"]["by_budget"][1]["go_rate"] == 1.0

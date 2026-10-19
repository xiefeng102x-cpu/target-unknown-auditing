#!/usr/bin/env python3
"""Evaluate the frozen GTSRB semantic overlapping-group confirmation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CERT = ROOT / "auditability_certificate_v1"
PROTOCOL_DEFAULT = CERT / "PROTOCOL_GTSRB_SEMANTIC_OVERLAP_CONFIRMATION_V1.json"
SCORES_DEFAULT = ROOT / "results/gtsrb_semantic_overlap_scan_scores_confirmation_v1"
OUTPUT_DEFAULT = CERT / "results_gtsrb_semantic_overlap_confirmation_v1"
CORE = CERT / "analysis/run_protected_group_family_srac.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_core():
    spec = importlib.util.spec_from_file_location("semantic_overlap_srac_core", CORE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def membership_matrix(labels: np.ndarray, groups: list[dict]) -> tuple[np.ndarray, list[str]]:
    columns = []
    names = []
    for row in groups:
        classes = np.asarray(row["class_ids"], dtype=int)
        if len(classes) == 0 or len(np.unique(classes)) != len(classes):
            raise ValueError(f"invalid class list for {row['name']}")
        if np.any((classes < 0) | (classes > 42)):
            raise ValueError(f"out-of-range class id for {row['name']}")
        mask = np.isin(labels, classes)
        if not np.any(mask):
            raise ValueError(f"empty protected group in population: {row['name']}")
        columns.append(mask)
        names.append(str(row["name"]))
    if len(set(names)) != len(names):
        raise ValueError("protected-group names must be unique")
    return np.column_stack(columns), names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_DEFAULT)
    parser.add_argument("--scores", type=Path, default=SCORES_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()

    sidecar = args.protocol.with_suffix(args.protocol.suffix + ".sha256")
    protocol_hash = sha256(args.protocol)
    if protocol_hash != sidecar.read_text(encoding="utf-8").split()[0]:
        raise RuntimeError("protocol sidecar mismatch")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["endpoint_runner_sha256"] != sha256(Path(__file__)):
        raise RuntimeError("endpoint runner differs from freeze")
    if protocol["endpoint_core_sha256"] != sha256(CORE):
        raise RuntimeError("endpoint core differs from freeze")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    score_manifest_path = args.scores / "manifest.json"
    score_manifest = json.loads(score_manifest_path.read_text(encoding="utf-8"))
    if score_manifest.get("status") != "SEMANTIC_OVERLAP_SCAN_HANDOFF_COMPLETE":
        raise RuntimeError("score handoff incomplete")
    if score_manifest.get("protocol_sha256") != protocol_hash:
        raise RuntimeError("score handoff protocol mismatch")
    flags_path = args.scores / "flags.csv"
    if score_manifest.get("flags_sha256") != sha256(flags_path):
        raise RuntimeError("score table hash mismatch")

    core = load_core()
    flags = pd.read_csv(flags_path)
    eligible = flags[flags.victim_eligible.eq(1)].copy()
    outputs = []
    truth_rows = []
    for victim_id, victim in eligible.groupby("victim_id", sort=True):
        victim = victim.reset_index(drop=True)
        memberships, names = membership_matrix(
            victim.assigned_group.to_numpy(dtype=int), protocol["protected_groups"]
        )
        poison = victim.is_poison.to_numpy(dtype=bool)
        frozen_flags = victim.flag.to_numpy(dtype=np.int8)
        clean_rates = []
        for index in range(memberships.shape[1]):
            clean = (~poison) & memberships[:, index]
            if np.any(clean):
                clean_rates.append(float(frozen_flags[clean].mean()))
        actual_recall = float(frozen_flags[poison].mean())
        actual_max_clean = max(clean_rates, default=0.0)
        true_pass = bool(
            actual_recall >= float(protocol["certificate"]["recall_target"])
            and actual_max_clean
            <= float(protocol["certificate"]["max_group_clean_flag_rate_cap"])
        )
        truth_rows.append(
            {
                "victim_id": victim_id,
                "actual_recall": actual_recall,
                "actual_max_group_clean_rate": actual_max_clean,
                "true_policy_pass": int(true_pass),
            }
        )
        decisions = core.audit_fixed_population_family(
            flags=frozen_flags,
            poison=poison,
            memberships=memberships,
            group_names=names,
            budgets=protocol["certificate"]["budgets"],
            draws=int(protocol["certificate"]["protected_draws_per_seed_budget"]),
            delta=float(protocol["certificate"]["delta_per_decision"]),
            tau=float(protocol["certificate"]["recall_target"]),
            alpha_cap=float(protocol["certificate"]["max_group_clean_flag_rate_cap"]),
            namespace=f"gtsrb-semantic-overlap-confirm-v1|{victim_id}",
            flagged_share=float(protocol["certificate"]["active_stratum_budget_share"]),
        )
        decisions.insert(0, "victim_id", victim_id)
        outputs.append(decisions)

    decisions = pd.concat(outputs, ignore_index=True)
    truth = pd.DataFrame(truth_rows)
    summary = (
        decisions.groupby("k", as_index=False)
        .agg(
            decisions=("decision", "size"),
            victims=("victim_id", "nunique"),
            go_rate=("decision", lambda x: float((x == "GO").mean())),
            nogo_rate=("decision", lambda x: float((x == "NO_GO").mean())),
            abstain_rate=("decision", lambda x: float((x == "ABSTAIN").mean())),
            victims_with_go=("victim_id", lambda x: int(decisions.loc[x.index].query("decision == 'GO'").victim_id.nunique())),
            false_go_count=("false_go", "sum"),
            false_nogo_count=("false_nogo", "sum"),
        )
    )
    args.output.mkdir(parents=True)
    decisions_path = args.output / "decisions.csv"
    truth_path = args.output / "truth_by_victim.csv"
    summary_path = args.output / "summary.csv"
    decisions.to_csv(decisions_path, index=False)
    truth.to_csv(truth_path, index=False)
    summary.to_csv(summary_path, index=False)

    gate = protocol["support_gate"]
    eligible_count = int(truth.victim_id.nunique())
    true_pass_count = int(truth.true_policy_pass.sum())
    by_k = summary.set_index("k")
    checks = {
        "minimum_eligible_victims": eligible_count >= int(gate["minimum_eligible_victims"]),
        "minimum_true_policy_pass_victims": true_pass_count >= int(gate["minimum_true_policy_pass_victims"]),
        "zero_false_go": int(decisions.false_go.sum()) == 0,
        "zero_false_nogo": int(decisions.false_nogo.sum()) == 0,
        "minimum_victims_with_go_at_k_2000": int(by_k.loc[2000, "victims_with_go"]) >= int(gate["minimum_victims_with_go_at_k_2000"]),
        "minimum_victims_with_go_at_k_8000": int(by_k.loc[8000, "victims_with_go"]) >= int(gate["minimum_victims_with_go_at_k_8000"]),
    }
    status = "PASS_GTSRB_SEMANTIC_OVERLAP_CONFIRMATION" if all(checks.values()) else "FAIL_GTSRB_SEMANTIC_OVERLAP_CONFIRMATION"
    report = {
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_id": protocol["study_id"],
        "protocol_sha256": protocol_hash,
        "score_manifest_sha256": sha256(score_manifest_path),
        "flags_sha256": sha256(flags_path),
        "endpoint_core_sha256": sha256(CORE),
        "endpoint_runner_sha256": sha256(Path(__file__)),
        "eligible_victims": eligible_count,
        "true_policy_pass_victims": true_pass_count,
        "protected_group_count": len(protocol["protected_groups"]),
        "support_checks": checks,
        "outputs": {
            "decisions_sha256": sha256(decisions_path),
            "truth_sha256": sha256(truth_path),
            "summary_sha256": sha256(summary_path),
        },
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

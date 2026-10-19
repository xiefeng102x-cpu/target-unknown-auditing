#!/usr/bin/env python3
"""Frozen exact group-safe SRAC endpoint for target-unknown SCAn confirmation."""

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


ROOT = Path(__file__).resolve().parents[1]
PAPER14 = ROOT.parent
PROTOCOL_DEFAULT = ROOT / "PROTOCOL_GTSRB_SCAN_GROUP_SAFE_CONFIRMATION_V1.json"
INPUT_DEFAULT = PAPER14 / "results/gtsrb_allclass_scan_scores_confirmation_v1"
OUTPUT_DEFAULT = ROOT / "results_gtsrb_scan_group_safe_confirmation_v1"
CORE = ROOT / "analysis/run_stratified_group_safe_srac.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_DEFAULT)
    parser.add_argument("--input", type=Path, default=INPUT_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    sidecar = args.protocol.with_suffix(args.protocol.suffix + ".sha256")
    protocol_hash = sha256(args.protocol)
    if protocol_hash != sidecar.read_text(encoding="utf-8").split()[0]:
        raise RuntimeError("protocol sidecar mismatch")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["endpoint_runner_sha256"] != sha256(Path(__file__)):
        raise RuntimeError("endpoint runner differs from frozen protocol")
    if protocol["endpoint_core_sha256"] != sha256(CORE):
        raise RuntimeError("endpoint core differs from frozen protocol")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    manifest_path = args.input / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "ALLCLASS_SCAN_SCORE_HANDOFF_COMPLETE":
        raise RuntimeError("SCAn handoff incomplete")
    if manifest.get("protocol_sha256") != protocol_hash:
        raise RuntimeError("SCAn handoff protocol mismatch")
    flags_path = args.input / "flags.csv"
    if sha256(flags_path) != manifest["flags_sha256"]:
        raise RuntimeError("SCAn flags hash mismatch")
    table = pd.read_csv(flags_path)
    core = load_module(CORE, "gtsrb_scan_group_safe_confirmation_core")
    certificate = protocol["certificate"]
    frames: list[pd.DataFrame] = []
    for victim_id, group in table.groupby("victim_id", sort=True):
        if group["sample_id"].duplicated().any():
            raise RuntimeError(f"duplicate sample IDs for {victim_id}")
        if not bool(group["victim_eligible"].iloc[0]):
            continue
        rows = core.audit_fixed_population(
            flags=group["flag"].to_numpy(dtype=np.int8),
            poison=group["is_poison"].to_numpy(dtype=np.int8).astype(bool),
            groups=group["assigned_group"].to_numpy(dtype=np.int64),
            budgets=certificate["budgets"],
            draws=int(certificate["protected_draws_per_seed_budget"]),
            delta=float(certificate["delta_per_decision"]),
            tau=float(certificate["recall_target"]),
            alpha_cap=float(certificate["max_group_clean_flag_rate_cap"]),
            namespace=f"{protocol['study_id']}|{victim_id}",
            active_share=float(certificate["active_stratum_budget_share"]),
        )
        rows.insert(0, "detector", "backdoorbench_scan")
        rows.insert(0, "victim_id", victim_id)
        frames.append(rows)
        print(f"endpoint {victim_id} complete", flush=True)
    if not frames:
        raise RuntimeError("no eligible confirmation victims")
    decisions = pd.concat(frames, ignore_index=True)
    summary = (
        decisions.groupby("k", sort=True)
        .agg(
            decisions=("decision", "size"),
            victims=("victim_id", "nunique"),
            go_rate=("decision", lambda values: float((values == "GO").mean())),
            nogo_rate=("decision", lambda values: float((values == "NO_GO").mean())),
            abstain_rate=("decision", lambda values: float((values == "ABSTAIN").mean())),
            false_go_count=("false_go", "sum"),
            false_nogo_count=("false_nogo", "sum"),
        )
        .reset_index()
    )
    actual = decisions.groupby("victim_id", sort=True).first().reset_index()
    go_victims_k2000 = int(
        decisions.loc[(decisions["k"].eq(2000)) & decisions["decision"].eq("GO"), "victim_id"].nunique()
    )
    gates = {
        "minimum_eligible_victims": int(manifest["eligible_seed_count"]) >= int(protocol["support_gate"]["minimum_eligible_victims"]),
        "minimum_true_policy_pass_victims": int(manifest["true_policy_pass_seed_count"]) >= int(protocol["support_gate"]["minimum_true_policy_pass_victims"]),
        "zero_false_go": int(decisions["false_go"].sum()) == 0,
        "zero_false_nogo": int(decisions["false_nogo"].sum()) == 0,
        "minimum_victims_with_go_at_k_2000": go_victims_k2000 >= int(protocol["support_gate"]["minimum_victims_with_go_at_k_2000"]),
    }
    passed = all(gates.values())
    args.output.mkdir(parents=True)
    decisions_path = args.output / "decisions.csv"
    summary_path = args.output / "summary.csv"
    decisions.to_csv(decisions_path, index=False)
    summary.to_csv(summary_path, index=False)
    report = {
        "status": "PASS_GTSRB_SCAN_GROUP_SAFE_CONFIRMATION" if passed else "FAIL_GTSRB_SCAN_GROUP_SAFE_CONFIRMATION",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_id": protocol["study_id"],
        "protocol_sha256": protocol_hash,
        "score_manifest_sha256": sha256(manifest_path),
        "decision_rows": len(decisions),
        "eligible_victims": int(decisions["victim_id"].nunique()),
        "true_policy_pass_victims": int(manifest["true_policy_pass_seed_count"]),
        "victims_with_go_at_k_2000": go_victims_k2000,
        "false_go_count": int(decisions["false_go"].sum()),
        "false_nogo_count": int(decisions["false_nogo"].sum()),
        "gates": gates,
        "outputs": {
            "decisions_sha256": sha256(decisions_path),
            "summary_sha256": sha256(summary_path),
        },
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

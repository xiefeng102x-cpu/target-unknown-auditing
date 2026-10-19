#!/usr/bin/env python3
"""Evaluate the frozen target-unknown GTSRB group-safe SRAC endpoint.

The detector-score handoff is immutable input.  This runner applies the exact
flag/group-stratified certificate to each detector and to every whole-group
flagging negative control.  Poison identities are used only by the protected
adjudication simulation and by retrospective error accounting.
"""

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
PROTOCOL_DEFAULT = ROOT / "PROTOCOL_GTSRB_ALLCLASS_GROUP_SAFE_V1.json"
SCORES_DEFAULT = PAPER14 / "results/gtsrb_allclass_detector_scores_v1"
OUTPUT_DEFAULT = ROOT / "results_gtsrb_allclass_group_safe_v1"
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


def summarize(decisions: pd.DataFrame) -> pd.DataFrame:
    return (
        decisions.groupby(["endpoint_family", "detector", "k"], sort=True)
        .agg(
            decisions=("decision", "size"),
            victims=("victim_id", "nunique"),
            actual_recall_mean=("actual_recall", "mean"),
            actual_max_group_clean_rate_mean=("actual_max_group_clean_rate", "mean"),
            go_rate=("decision", lambda values: float((values == "GO").mean())),
            nogo_rate=("decision", lambda values: float((values == "NO_GO").mean())),
            abstain_rate=("decision", lambda values: float((values == "ABSTAIN").mean())),
            false_go_count=("false_go", "sum"),
            false_nogo_count=("false_nogo", "sum"),
        )
        .reset_index()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_DEFAULT)
    parser.add_argument("--scores", type=Path, default=SCORES_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()

    protocol_sidecar = args.protocol.with_suffix(args.protocol.suffix + ".sha256")
    expected_protocol_hash = protocol_sidecar.read_text(encoding="utf-8").split()[0]
    protocol_hash = sha256(args.protocol)
    if protocol_hash != expected_protocol_hash:
        raise RuntimeError("protocol sidecar mismatch")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("endpoint_runner_sha256") != sha256(Path(__file__)):
        raise RuntimeError("endpoint runner hash differs from frozen protocol")
    if protocol.get("endpoint_core_sha256") != sha256(CORE):
        raise RuntimeError("endpoint core hash differs from frozen protocol")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    score_manifest_path = args.scores / "manifest.json"
    score_manifest = json.loads(score_manifest_path.read_text(encoding="utf-8"))
    if score_manifest.get("status") != "ALLCLASS_SCORE_HANDOFF_COMPLETE":
        raise RuntimeError("score handoff is incomplete")
    if score_manifest.get("protocol_sha256") != protocol_hash:
        raise RuntimeError("score handoff was produced under a different protocol")
    if int(score_manifest.get("seed_count", -1)) != len(protocol["seeds"]):
        raise RuntimeError("score manifest seed count mismatch")

    core = load_module(CORE, "frozen_stratified_group_safe_endpoint")
    certificate = protocol["certificate"]
    frames: list[pd.DataFrame] = []
    input_receipts: list[dict[str, object]] = []

    by_seed = {int(row["seed"]): row for row in score_manifest["seeds"]}
    for seed in map(int, protocol["seeds"]):
        receipt = by_seed.get(seed)
        if receipt is None:
            raise RuntimeError(f"missing score receipt for seed {seed}")
        score_path = args.scores / "per_seed" / f"seed_{seed}_allclass_scores.csv"
        if sha256(score_path) != receipt["score_sha256"]:
            raise RuntimeError(f"score hash mismatch for seed {seed}")
        table = pd.read_csv(score_path)
        required = {"victim_id", "sample_id", "assigned_group", "detector", "flag", "is_poison"}
        if not required.issubset(table.columns):
            raise RuntimeError(f"score schema mismatch for seed {seed}")
        input_receipts.append({"seed": seed, "path": str(score_path), "sha256": sha256(score_path)})

        population_views: dict[str, pd.DataFrame] = {}
        for detector in protocol["detectors"]:
            view = table.loc[table["detector"].eq(detector)].copy()
            if view.empty or view["sample_id"].duplicated().any():
                raise RuntimeError(f"invalid detector population: seed={seed}, detector={detector}")
            view = view.sort_values("sample_id", kind="stable").reset_index(drop=True)
            population_views[detector] = view
        anchor = population_views[protocol["detectors"][0]]
        anchor_identity = anchor[["sample_id", "assigned_group", "is_poison"]].reset_index(drop=True)
        for detector, view in population_views.items():
            if not anchor_identity.equals(
                view[["sample_id", "assigned_group", "is_poison"]].reset_index(drop=True)
            ):
                raise RuntimeError(f"detector populations differ for seed {seed}: {detector}")

        for detector, view in population_views.items():
            rows = core.audit_fixed_population(
                flags=view["flag"].to_numpy(dtype=np.int8),
                poison=view["is_poison"].to_numpy(dtype=np.int8).astype(bool),
                groups=view["assigned_group"].to_numpy(dtype=np.int64),
                budgets=certificate["budgets"],
                draws=int(certificate["protected_draws_per_seed_budget"]),
                delta=float(certificate["delta_per_detector_decision"]),
                tau=float(certificate["recall_target"]),
                alpha_cap=float(certificate["max_group_clean_flag_rate_cap"]),
                namespace=f"{protocol['study_id']}|seed={seed}|detector={detector}",
                active_share=float(certificate["active_stratum_budget_share"]),
            )
            rows.insert(0, "negative_control_group", np.nan)
            rows.insert(0, "endpoint_family", "real_detector")
            rows.insert(0, "detector", detector)
            rows.insert(0, "victim_id", anchor["victim_id"].iloc[0])
            frames.append(rows)

        groups = anchor["assigned_group"].to_numpy(dtype=np.int64)
        poison = anchor["is_poison"].to_numpy(dtype=np.int8).astype(bool)
        for group in sorted(np.unique(groups).tolist()):
            rows = core.audit_fixed_population(
                flags=(groups == group).astype(np.int8),
                poison=poison,
                groups=groups,
                budgets=certificate["budgets"],
                draws=int(certificate["protected_draws_per_seed_budget"]),
                delta=float(certificate["delta_per_detector_decision"]),
                tau=float(certificate["recall_target"]),
                alpha_cap=float(certificate["max_group_clean_flag_rate_cap"]),
                namespace=f"{protocol['study_id']}|seed={seed}|whole_group={group}",
                active_share=float(certificate["active_stratum_budget_share"]),
            )
            rows.insert(0, "negative_control_group", int(group))
            rows.insert(0, "endpoint_family", "whole_group_negative_control")
            rows.insert(0, "detector", "flag_whole_group")
            rows.insert(0, "victim_id", anchor["victim_id"].iloc[0])
            frames.append(rows)
        print(f"endpoint seed={seed} complete", flush=True)

    decisions = pd.concat(frames, ignore_index=True)
    summary = summarize(decisions)
    real = decisions.loc[decisions["endpoint_family"].eq("real_detector")]
    controls = decisions.loc[decisions["endpoint_family"].eq("whole_group_negative_control")]
    gates = {
        "zero_false_go": bool(int(decisions["false_go"].sum()) == 0),
        "zero_false_nogo": bool(int(decisions["false_nogo"].sum()) == 0),
        "zero_go_for_every_whole_group_negative_control": bool(not controls["decision"].eq("GO").any()),
        "nonzero_go_for_at_least_one_nontrivial_detector_at_k_2000": bool(
            real.loc[real["k"].eq(2000), "decision"].eq("GO").any()
        ),
    }
    expected_gates = protocol["support_gate"]
    if set(gates) != set(expected_gates):
        raise RuntimeError("implemented support gates differ from protocol")
    passed = all(gates[name] == bool(expected_gates[name]) for name in gates)

    args.output.mkdir(parents=True)
    decisions_path = args.output / "decisions.csv"
    summary_path = args.output / "summary.csv"
    decisions.to_csv(decisions_path, index=False)
    summary.to_csv(summary_path, index=False)
    report = {
        "status": "PASS_ALLCLASS_GROUP_SAFE_SUPPORT_GATE" if passed else "FAIL_ALLCLASS_GROUP_SAFE_SUPPORT_GATE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_id": protocol["study_id"],
        "protocol_sha256": protocol_hash,
        "endpoint_runner_sha256": sha256(Path(__file__)),
        "endpoint_core_sha256": sha256(CORE),
        "score_manifest_sha256": sha256(score_manifest_path),
        "victims": int(decisions["victim_id"].nunique()),
        "decision_rows": int(len(decisions)),
        "real_detector_decisions": int(len(real)),
        "negative_control_decisions": int(len(controls)),
        "false_go_count": int(decisions["false_go"].sum()),
        "false_nogo_count": int(decisions["false_nogo"].sum()),
        "real_go_count": int(real["decision"].eq("GO").sum()),
        "whole_group_control_go_count": int(controls["decision"].eq("GO").sum()),
        "gates": gates,
        "input_receipts": input_receipts,
        "outputs": {
            "decisions_sha256": sha256(decisions_path),
            "summary_sha256": sha256(summary_path),
        },
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(summary.loc[summary["endpoint_family"].eq("real_detector")].to_string(index=False))


if __name__ == "__main__":
    main()

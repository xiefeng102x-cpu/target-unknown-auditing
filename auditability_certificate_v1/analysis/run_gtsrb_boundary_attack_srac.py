#!/usr/bin/env python3
"""Run the frozen SRAC endpoint for the pre-draw GTSRB boundary attack."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER14 = ROOT.parent
CORE_PATH = ROOT / "analysis/run_uniform_adjudication_certificate.py"
DEFAULT_PROTOCOL = PAPER14 / "configs/srac_gtsrb_boundary_attack_confirmation_v1.json"
DEFAULT_INPUT = ROOT / "derived_inputs/gtsrb_boundary_attack_confirmation_v1"
DEFAULT_OUTPUT = ROOT / "results_gtsrb_boundary_attack_srac_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_core():
    spec = importlib.util.spec_from_file_location("srac_uniform_core", CORE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("study_id") != "srac_gtsrb_boundary_attack_confirmation_v1":
        raise RuntimeError("unexpected boundary-attack confirmation protocol")
    endpoint = protocol["srac_evaluation"]
    expected = {
        "recall_target": 0.8,
        "clean_flag_rate_cap": 0.1,
        "delta_per_decision": 0.005,
        "audit_budgets": [2000, 8000, 16000],
        "protected_draws_per_seed_budget": 200,
        "no_threshold_or_policy_retuning_after_freeze": True,
    }
    for key, value in expected.items():
        if endpoint.get(key) != value:
            raise RuntimeError(f"frozen endpoint mismatch for {key}")

    core = load_core()
    table, source_hashes = core.load_block(args.input.resolve())
    observed_seeds = sorted(
        int(value.rsplit("seed", 1)[1]) for value in table["victim_id"].unique()
    )
    frozen_seeds = [int(seed) for seed in protocol["seeds"]]
    if observed_seeds != frozen_seeds:
        raise RuntimeError("derived input does not contain the frozen seed block")
    rows = core.audit_block(
        "gtsrb_boundary_attack_confirmation",
        table,
        tuple(endpoint["audit_budgets"]),
        int(endpoint["protected_draws_per_seed_budget"]),
        float(endpoint["family_error_upper_bound"]),
        float(endpoint["recall_target"]),
        float(endpoint["clean_flag_rate_cap"]),
    )
    if set(rows["delta_cell"].unique()) != {float(endpoint["delta_per_decision"])}:
        raise RuntimeError("implementation changed the frozen per-decision delta")
    summary = core.summarize(rows)
    ranges = core.summarize_actual_ranges(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    decisions_path = args.output / "decisions.csv"
    summary_path = args.output / "summary.csv"
    ranges_path = args.output / "actual_ranges.csv"
    rows.to_csv(decisions_path, index=False)
    summary.to_csv(summary_path, index=False)
    ranges.to_csv(ranges_path, index=False)
    target = str(protocol["adaptive_attack_endpoint"]["target_detector"])
    target_rows = rows.loc[rows["detector"].eq(target)]
    by_budget: dict[str, object] = {}
    for budget, group in rows.groupby("k", sort=True):
        target_group = target_rows.loc[target_rows["k"].eq(budget)]
        by_budget[str(int(budget))] = {
            "rows": len(group),
            "go_rate": float(group["decision"].eq("GO").mean()),
            "nogo_rate": float(group["decision"].eq("NO_GO").mean()),
            "abstain_rate": float(group["decision"].eq("ABSTAIN").mean()),
            "false_go_count": int(group["false_go"].sum()),
            "false_nogo_count": int(group["false_nogo"].sum()),
            "target_detector_rows": len(target_group),
            "target_detector_go_rate": float(target_group["decision"].eq("GO").mean()),
            "target_detector_nogo_rate": float(target_group["decision"].eq("NO_GO").mean()),
            "target_detector_abstain_rate": float(
                target_group["decision"].eq("ABSTAIN").mean()
            ),
            "target_detector_false_go_count": int(target_group["false_go"].sum()),
        }
    report = {
        "status": "COMPLETE_FROZEN_ENDPOINT",
        "study_id": protocol["study_id"],
        "target_detector": target,
        "protocol_sha256": sha256(args.protocol),
        "core_sha256": sha256(CORE_PATH),
        "endpoint_runner_sha256": sha256(Path(__file__)),
        "input_hashes": source_hashes,
        "decision_rows": len(rows),
        "duplicate_decision_keys": int(
            rows.duplicated(["block", "victim_id", "fold", "detector", "k", "draw"]).sum()
        ),
        "by_budget": by_budget,
        "output_hashes": {
            "decisions": sha256(decisions_path),
            "summary": sha256(summary_path),
            "actual_ranges": sha256(ranges_path),
        },
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

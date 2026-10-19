#!/usr/bin/env python3
"""Run the frozen SRAC endpoint for CIFAR/VGG Blended-variant victims."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER14 = ROOT.parent
CORE_PATH = ROOT / "analysis/run_uniform_adjudication_certificate.py"
DEFAULT_PROTOCOL = PAPER14 / "configs/cifar10_vgg16_bn_blended_srac_confirmation_v1.json"
DEFAULT_INPUT = PAPER14 / "results/cifar10_vgg16_bn_blended_srac_confirmation_v1"
DEFAULT_OUTPUT = ROOT / "results_cifar10_vgg16_blended_srac_confirmation_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_core():
    spec = importlib.util.spec_from_file_location("srac_uniform_core_blended", CORE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run(protocol_path: Path, input_path: Path, output: Path) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != "cifar10_vgg16_bn_blended_srac_confirmation_v1":
        raise RuntimeError("unexpected CIFAR/VGG Blended SRAC protocol")
    endpoint = protocol["srac_evaluation"]
    expected = {
        "recall_target": 0.8,
        "clean_flag_rate_cap": 0.1,
        "delta_per_decision": 0.005,
        "audit_budgets": [120, 500, 2000],
        "protected_draws_per_seed_fold_budget": 200,
        "no_threshold_or_policy_retuning_after_freeze": True,
    }
    for key, value in expected.items():
        if endpoint.get(key) != value:
            raise RuntimeError(f"frozen endpoint mismatch for {key}")

    core = load_core()
    table, source_hashes = core.load_block(input_path.resolve())
    observed_seeds = sorted(int(value.rsplit("seed", 1)[1]) for value in table["victim_id"].unique())
    if observed_seeds != [int(seed) for seed in protocol["formal_seeds"]]:
        raise RuntimeError("score handoff does not contain the frozen seed block")
    rows = core.audit_block(
        "cifar10_vgg16_blended_srac_confirmation", table,
        tuple(endpoint["audit_budgets"]),
        int(endpoint["protected_draws_per_seed_fold_budget"]),
        float(endpoint["family_error_upper_bound"]),
        float(endpoint["recall_target"]),
        float(endpoint["clean_flag_rate_cap"]),
    )
    if set(rows["delta_cell"].unique()) != {float(endpoint["delta_per_decision"])}:
        raise RuntimeError("implementation changed the frozen per-decision delta")
    summary = core.summarize(rows)
    ranges = core.summarize_actual_ranges(rows)
    output.mkdir(parents=True, exist_ok=True)
    decisions_path = output / "decisions.csv"
    summary_path = output / "summary.csv"
    ranges_path = output / "actual_ranges.csv"
    rows.to_csv(decisions_path, index=False)
    summary.to_csv(summary_path, index=False)
    ranges.to_csv(ranges_path, index=False)
    by_budget = {}
    for budget, group in rows.groupby("k", sort=True):
        by_budget[str(int(budget))] = {
            "rows": len(group),
            "go_rate": float(group["decision"].eq("GO").mean()),
            "nogo_rate": float(group["decision"].eq("NO_GO").mean()),
            "abstain_rate": float(group["decision"].eq("ABSTAIN").mean()),
            "false_go_count": int(group["false_go"].sum()),
            "false_nogo_count": int(group["false_nogo"].sum()),
        }
    report = {
        "status": "COMPLETE_FROZEN_ENDPOINT",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(protocol_path),
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
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run(args.protocol, args.input, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

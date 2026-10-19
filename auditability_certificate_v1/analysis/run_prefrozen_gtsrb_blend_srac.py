#!/usr/bin/env python3
"""Execute the pre-frozen SRAC endpoint for the GTSRB global-blend study."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER14 = ROOT.parent
DEFAULT_PROTOCOL = PAPER14 / "configs" / "srac_gtsrb_global_blend_confirmation_v1.json"
DEFAULT_INPUT = ROOT / "derived_inputs" / "gtsrb_global_blend_prefrozen_v1"
DEFAULT_OUTPUT = ROOT / "results_gtsrb_global_blend_srac_v1"
CORE_PATH = ROOT / "analysis" / "run_uniform_adjudication_certificate.py"


def load_core():
    spec = importlib.util.spec_from_file_location("uniform_adjudication_core", CORE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("study_id") != "srac_gtsrb_global_blend_confirmation_v1":
        raise RuntimeError("unexpected confirmation protocol")
    endpoint = protocol["srac_evaluation"]
    if endpoint.get("no_threshold_or_policy_retuning_after_freeze") is not True:
        raise RuntimeError("SRAC endpoint is not frozen")
    expected = {
        "recall_target": 0.8,
        "clean_flag_rate_cap": 0.1,
        "delta_per_decision": 0.005,
        "audit_budgets": [2000, 8000, 16000],
        "protected_draws_per_seed_budget": 200,
    }
    for key, value in expected.items():
        if endpoint.get(key) != value:
            raise RuntimeError(f"frozen endpoint mismatch for {key}")

    core = load_core()
    table, source_hashes = core.load_block(args.input)
    observed_seeds = sorted(
        int(value.rsplit("seed", 1)[1]) for value in table["victim_id"].unique()
    )
    if observed_seeds != [int(seed) for seed in protocol["seeds"]]:
        raise RuntimeError("derived input does not contain every frozen seed")
    rows = core.audit_block(
        "gtsrb_global_blend_prefrozen",
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
    rows_path = args.output / "decisions.csv"
    summary_path = args.output / "summary.csv"
    ranges_path = args.output / "actual_ranges.csv"
    rows.to_csv(rows_path, index=False)
    summary.to_csv(summary_path, index=False)
    ranges.to_csv(ranges_path, index=False)
    by_budget: dict[str, object] = {}
    for budget, group in rows.groupby("k", sort=True):
        by_budget[str(int(budget))] = {
            "rows": len(group),
            "mean_sampled_poison": float(group["sampled_poison"].mean()),
            "go_rate": float(group["decision"].eq("GO").mean()),
            "nogo_rate": float(group["decision"].eq("NO_GO").mean()),
            "abstain_rate": float(group["decision"].eq("ABSTAIN").mean()),
            "false_go_count": int(group["false_go"].sum()),
            "false_nogo_count": int(group["false_nogo"].sum()),
            "sampling_fraction": float(group["k"].iloc[0] / group["n_total"].iloc[0]),
        }
    report = {
        "status": "COMPLETE_PREFROZEN_ENDPOINT",
        "study_id": protocol["study_id"],
        "protocol_sha256": sha256(args.protocol),
        "core_sha256": sha256(CORE_PATH),
        "input_hashes": source_hashes,
        "decision_rows": len(rows),
        "duplicate_decision_keys": int(
            rows.duplicated(["block", "victim_id", "fold", "detector", "k", "draw"]).sum()
        ),
        "by_budget": by_budget,
        "output_hashes": {
            "decisions": sha256(rows_path),
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

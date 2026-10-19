#!/usr/bin/env python3
"""Run a certificate-aware sampled-label corruption red team."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "PROTOCOL_SRAC_ADJUDICATION_REDTEAM_V1.json"
OUTPUT = ROOT / "results_srac_adjudication_redteam_v1"
CORE_PATH = ROOT / "analysis/run_uniform_adjudication_certificate.py"


def load_core():
    spec = importlib.util.spec_from_file_location("srac_core", CORE_PATH)
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


def best_corruption(row: object, budget: int, core: object) -> dict[str, object]:
    m1 = int(row.sampled_poison)
    c1 = int(row.sampled_caught_poison)
    m0 = int(row.sampled_clean)
    c0 = int(row.sampled_flagged_clean)
    delta = float(row.delta_cell)
    tau = 0.8
    alpha_cap = 0.1
    uncaught = m1 - c1
    best: tuple[tuple[object, ...], dict[str, object]] | None = None
    for poison_to_clean in range(min(budget, uncaught) + 1):
        remaining = budget - poison_to_clean
        clean_to_poison = min(remaining, c0)
        changed_m1 = m1 - poison_to_clean + clean_to_poison
        changed_c1 = c1 + clean_to_poison
        changed_m0 = m0 + poison_to_clean - clean_to_poison
        changed_c0 = c0 - clean_to_poison
        recall_lo, recall_hi = core.interval(changed_c1, changed_m1, delta)
        _, clean_hi = core.interval(changed_c0, changed_m0, delta)
        decision = core.decide(recall_lo, recall_hi, clean_hi, tau, alpha_cap)
        used = poison_to_clean + clean_to_poison
        candidate = {
            "poison_to_clean": poison_to_clean,
            "clean_to_poison": clean_to_poison,
            "flips_used": used,
            "corrupt_sampled_poison": changed_m1,
            "corrupt_sampled_caught_poison": changed_c1,
            "corrupt_sampled_clean": changed_m0,
            "corrupt_sampled_flagged_clean": changed_c0,
            "naive_recall_lo": recall_lo,
            "naive_recall_hi": recall_hi,
            "naive_clean_hi": clean_hi,
            "naive_decision": decision,
        }
        key = (
            int(decision == "GO"),
            recall_lo - tau,
            alpha_cap - clean_hi,
            -used,
        )
        if best is None or key > best[0]:
            best = (key, candidate)
    assert best is not None
    return best[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != "srac_adjudication_redteam_v1":
        raise RuntimeError("unexpected red-team protocol")
    core = load_core()
    input_frames: list[pd.DataFrame] = []
    input_hashes: dict[str, str] = {}
    for source, spec in protocol["inputs"].items():
        if source == "certificate_core":
            continue
        path = ROOT / spec["path"]
        actual = sha256(path)
        if actual != spec["sha256"]:
            raise RuntimeError(f"input hash mismatch: {source}")
        frame = pd.read_csv(path)
        frame.insert(0, "source_study", source)
        input_frames.append(frame)
        input_hashes[source] = actual
    if sha256(CORE_PATH) != protocol["inputs"]["certificate_core"]["sha256"]:
        raise RuntimeError("certificate core hash mismatch")
    source_rows = pd.concat(input_frames, ignore_index=True)
    output_rows: list[dict[str, object]] = []
    for row in source_rows.itertuples(index=False):
        original_failure = bool(
            float(row.actual_recall) < 0.8 or float(row.actual_clean_flag_rate) > 0.1
        )
        for corruption_rate in protocol["corruption_rates"]:
            budget = int(math.ceil(float(corruption_rate) * int(row.k)))
            attack = best_corruption(row, budget, core)
            robust_lo, robust_hi, robust_clean_hi = core.robust_bounds(
                int(attack["corrupt_sampled_poison"]),
                int(attack["corrupt_sampled_caught_poison"]),
                int(attack["corrupt_sampled_clean"]),
                int(attack["corrupt_sampled_flagged_clean"]),
                budget,
                float(row.delta_cell),
            )
            robust_decision = core.decide(
                robust_lo, robust_hi, robust_clean_hi, 0.8, 0.1
            )
            output_rows.append(
                {
                    "source_study": row.source_study,
                    "block": row.block,
                    "victim_id": row.victim_id,
                    "fold": int(row.fold),
                    "detector": row.detector,
                    "k": int(row.k),
                    "draw": int(row.draw),
                    "corruption_rate": float(corruption_rate),
                    "budget_B": budget,
                    "original_decision": row.decision,
                    "actual_recall": float(row.actual_recall),
                    "actual_clean_flag_rate": float(row.actual_clean_flag_rate),
                    "truth_failure": int(original_failure),
                    **attack,
                    "naive_false_go": int(
                        attack["naive_decision"] == "GO" and original_failure
                    ),
                    "robust_recall_lo": robust_lo,
                    "robust_recall_hi": robust_hi,
                    "robust_clean_hi": robust_clean_hi,
                    "robust_decision": robust_decision,
                    "robust_false_go": int(robust_decision == "GO" and original_failure),
                    "delta_cell": float(row.delta_cell),
                }
            )
    rows = pd.DataFrame(output_rows)
    summary = (
        rows.groupby(
            ["source_study", "block", "detector", "k", "corruption_rate", "budget_B"],
            sort=True,
        )
        .agg(
            rows=("draw", "size"),
            truth_failures=("truth_failure", "sum"),
            original_go_rate=("original_decision", lambda x: float((x == "GO").mean())),
            naive_go_rate=("naive_decision", lambda x: float((x == "GO").mean())),
            robust_go_rate=("robust_decision", lambda x: float((x == "GO").mean())),
            robust_nogo_rate=("robust_decision", lambda x: float((x == "NO_GO").mean())),
            robust_abstain_rate=("robust_decision", lambda x: float((x == "ABSTAIN").mean())),
            naive_false_go_count=("naive_false_go", "sum"),
            robust_false_go_count=("robust_false_go", "sum"),
            mean_flips_used=("flips_used", "mean"),
        )
        .reset_index()
    )
    summary["naive_false_go_rate_all"] = summary["naive_false_go_count"] / summary["rows"]
    summary["robust_false_go_rate_all"] = summary["robust_false_go_count"] / summary["rows"]
    summary["naive_false_go_rate_among_failures"] = summary["naive_false_go_count"].div(
        summary["truth_failures"].where(summary["truth_failures"] > 0)
    )
    summary["robust_false_go_rate_among_failures"] = summary["robust_false_go_count"].div(
        summary["truth_failures"].where(summary["truth_failures"] > 0)
    )
    args.output.mkdir(parents=True, exist_ok=True)
    decisions_path = args.output / "redteam_decisions.csv.gz"
    summary_path = args.output / "summary.csv"
    witnesses_path = args.output / "naive_false_go_witnesses.csv"
    rows.to_csv(
        decisions_path,
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )
    summary.to_csv(summary_path, index=False)
    witnesses = rows.loc[rows["naive_false_go"].eq(1)].copy()
    witnesses.to_csv(witnesses_path, index=False)
    report = {
        "status": "PASS" if int(rows["robust_false_go"].sum()) == 0 else "FAIL",
        "protocol_sha256": sha256(args.protocol),
        "core_sha256": sha256(CORE_PATH),
        "source_rows": len(source_rows),
        "attacked_rows": len(rows),
        "naive_false_go_count": int(rows["naive_false_go"].sum()),
        "robust_false_go_count": int(rows["robust_false_go"].sum()),
        "naive_false_go_witness_rows": len(witnesses),
        "input_hashes": input_hashes,
        "output_hashes": {
            "redteam_decisions": sha256(decisions_path),
            "summary": sha256(summary_path),
            "naive_false_go_witnesses": sha256(witnesses_path),
        },
    }
    report_path = args.output / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

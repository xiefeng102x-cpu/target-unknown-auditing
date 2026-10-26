#!/usr/bin/env python3
"""Replay hand-constructed count fixtures; no experimental data or labels are read.

The expectations follow census arithmetic or the documented empty-count rule.
These examples validate the public count-to-decision interface; they do not
reproduce any scientific experiment or establish statistical validity on their own.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import srac_ui_engine_v1_1 as engine


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE = ROOT / "fixtures/synthetic_certificate_counts_v1.json"
SCHEMA = "srac-ui-synthetic-count-records-v1"


def check_counts(size: int, sampled: int, observed: int) -> None:
    if any(not isinstance(v, int) or isinstance(v, bool) for v in (size, sampled, observed)):
        raise ValueError("counts must be integers")
    if not 0 <= observed <= sampled <= size:
        raise ValueError("infeasible counts")


def replay_record(record: dict) -> str:
    for side in ("flagged", "unflagged"):
        check_counts(record[side + "_size"], record[side + "_sampled"], record[side + "_sampled_poisons"])
    if not all(0 < record[key] < 1 for key in ("delta", "tau", "cap")):
        raise ValueError("policy parameters must lie in (0,1)")
    groups = record["groups"]
    for group in groups:
        for side in ("flagged", "unflagged"):
            check_counts(group[side + "_size"], group[side + "_sampled"], group[side + "_sampled_clean"])
            if group[side + "_size"] > record[side + "_size"] or group[side + "_sampled"] > record[side + "_sampled"]:
                raise ValueError("group is larger than its stratum")
    assertions = 1 + sum(group["flagged_size"] > 0 for group in groups)
    alpha = record["delta"] / (2 * assertions)
    go, no_go = engine.recall_tests(
        record["flagged_size"], record["flagged_sampled"], record["flagged_sampled_poisons"],
        record["unflagged_size"], record["unflagged_sampled"], record["unflagged_sampled_poisons"],
        record["tau"], alpha,
    )
    for group in groups:
        if group["flagged_size"] == 0:
            continue
        group_go, group_no_go = engine.group_tests(
            group["flagged_size"], group["flagged_sampled"], group["flagged_sampled_clean"],
            group["unflagged_size"], group["unflagged_sampled"], group["unflagged_sampled_clean"],
            record["cap"], alpha,
        )
        go = go and group_go
        no_go = no_go or group_no_go
    return "GO" if go else ("NO_GO" if no_go else "ABSTAIN")


def replay_fixture(path: Path = DEFAULT_FIXTURE) -> dict:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("schema") != SCHEMA or document.get("synthetic") is not True:
        raise ValueError("only the declared synthetic fixture schema is supported")
    records = document.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("nonempty fixture records are required")
    decisions = []
    for record in records:
        actual = replay_record(record)
        decisions.append({"name": record["name"], "expected": record["expected_decision"], "actual": actual})
    failed = sum(row["actual"] != row["expected"] for row in decisions)
    return {"passed": failed == 0, "synthetic": True, "records_checked": len(decisions),
            "records_failed": failed, "decisions": decisions}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    args = parser.parse_args()
    result = replay_fixture(args.fixture)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

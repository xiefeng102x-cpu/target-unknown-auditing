import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER14 = ROOT.parent
RESULT = ROOT / "results_gtsrb_global_blend_srac_v1"
DERIVED = ROOT / "derived_inputs/gtsrb_global_blend_prefrozen_v1"
PROTOCOL = PAPER14 / "configs/srac_gtsrb_global_blend_confirmation_v1.json"
VICTIM_RESULT = PAPER14 / "results/srac_gtsrb_global_blend_confirmation_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_prefrozen_blend_endpoint_reconciles():
    report = json.loads((RESULT / "report.json").read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    verification = json.loads(
        (VICTIM_RESULT / "verification_receipt.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "COMPLETE_PREFROZEN_ENDPOINT"
    assert report["protocol_sha256"] == sha256(PROTOCOL)
    assert report["protocol_sha256"] == (
        "4c2425cc967730c4dfbda2b7c5ac6802e4dbe6679ba844ff65cb076c8219fde5"
    )
    assert protocol["srac_evaluation"]["audit_budgets"] == [2000, 8000, 16000]
    assert verification["status"] == "PASS"
    assert verification["eligible_seed_count"] == 10
    assert verification["primary_gate_pass"] is False

    truth = read_csv(DERIVED / "private_truth.csv")
    truth_rows = Counter(row["victim_id"] for row in truth)
    poison_rows = Counter(
        row["victim_id"] for row in truth if int(row["is_poison"]) == 1
    )
    assert len(truth_rows) == 10
    assert set(truth_rows.values()) == {25108}
    assert set(poison_rows.values()) == {1132}

    decisions = read_csv(RESULT / "decisions.csv")
    keys = {
        (
            row["block"],
            row["victim_id"],
            row["fold"],
            row["detector"],
            row["k"],
            row["draw"],
        )
        for row in decisions
    }
    assert len(decisions) == report["decision_rows"] == 18000
    assert len(keys) == len(decisions)
    assert {float(row["delta_cell"]) for row in decisions} == {0.005}
    assert sum(int(row["false_go"]) for row in decisions) == 0
    assert sum(int(row["false_nogo"]) for row in decisions) == 0

    expected = {
        "2000": (1259, 1800, 2941),
        "8000": (4000, 1800, 200),
        "16000": (4000, 1800, 200),
    }
    for budget, counts in expected.items():
        block = [row for row in decisions if row["k"] == budget]
        observed = Counter(row["decision"] for row in block)
        assert len(block) == 6000
        assert (observed["GO"], observed["NO_GO"], observed["ABSTAIN"]) == counts

    for name in ("decisions", "summary", "actual_ranges"):
        assert report["output_hashes"][name] == sha256(RESULT / f"{name}.csv")

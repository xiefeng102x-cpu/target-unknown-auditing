import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results_srac_adjudication_redteam_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_adjudication_redteam_exposes_naive_failure_and_robust_control():
    report = json.loads((RESULT / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert report["source_rows"] == 186000
    assert report["attacked_rows"] == 744000
    assert report["naive_false_go_count"] == 245
    assert report["robust_false_go_count"] == 0
    for name, filename in (
        ("redteam_decisions", "redteam_decisions.csv.gz"),
        ("summary", "summary.csv"),
        ("naive_false_go_witnesses", "naive_false_go_witnesses.csv"),
    ):
        assert report["output_hashes"][name] == sha256(RESULT / filename)

    summary = pd.read_csv(RESULT / "summary.csv")
    assert int(summary["naive_false_go_count"].sum()) == 245
    assert int(summary["robust_false_go_count"].sum()) == 0
    nonzero = summary.loc[summary["naive_false_go_count"].gt(0)]
    assert len(nonzero) == 3
    assert set(nonzero["block"]) == {"gtsrb_tinycnn_prefrozen"}
    assert set(nonzero["k"]) == {2000}

    witnesses = pd.read_csv(RESULT / "naive_false_go_witnesses.csv")
    assert len(witnesses) == 245
    assert witnesses["naive_false_go"].eq(1).all()
    assert witnesses["robust_false_go"].eq(0).all()
    assert witnesses["truth_failure"].eq(1).all()
    assert witnesses["naive_decision"].eq("GO").all()
    assert ~witnesses["robust_decision"].eq("GO").any()

"""Recompute the reference SRAC-UI certificate records from their counts alone."""
import json
from pathlib import Path
import srac_ui_engine_v1_1 as E

RECORDS = Path(__file__).resolve().parent.parent / "certificate_records_v1/srac_ui_reference_records_v1.json"


def test_reference_records():
    doc = json.loads(RECORDS.read_text())
    assert doc["records"]
    for r in doc["records"]:
        go, ng = E.recall_tests(r["flagged_size"], r["flagged_sampled"], r["flagged_sampled_poisons"], r["unflagged_size"],
                                r["unflagged_sampled"], r["unflagged_sampled_poisons"], r["tau"], r["alpha_per_test"])
        assert (go, ng) == (r["expected_recall_go_test"], r["expected_recall_nogo_test"])
        for g in r["groups"]:
            g_go, g_ng = E.group_tests(g["flagged_size"], g["flagged_sampled"], g["flagged_sampled_clean"], g["unflagged_size"],
                                       g["unflagged_sampled"], g["unflagged_sampled_clean"], r["cap"], r["alpha_per_test"])
            assert (g_go, g_ng) == (g["expected_go_test"], g["expected_nogo_test"])
            go &= g_go; ng |= g_ng
        assert ("GO" if go else ("NO_GO" if ng else "ABSTAIN")) == r["expected_decision"]
        assert r["alpha_per_test"] == r["delta"] / (2 * r["assertions"])

"""New release self-tests must provide their full success schemas."""
from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import verify_srac_release_20261019 as verifier


@pytest.mark.parametrize("name,reduction", (
    ("ui_robust", "reduces_to_engine_at_R0"),
    ("ui_directional", "never_more_than_engine_at_B0"),
))
def test_full_robustness_schema_required(name, reduction):
    correct = {"passed": True, "pairs_checked": 12, "violations": 0, reduction: True}
    assert verifier.selftest_contract_error(name, correct) is None
    for field in correct:
        incomplete = dict(correct)
        incomplete.pop(field)
        assert verifier.selftest_contract_error(name, incomplete) is not None
    assert verifier.selftest_contract_error(name, dict(correct, violations=1)) is not None


def test_synthetic_fixture_schema_and_decisions_required():
    correct = {"passed": True, "synthetic": True, "records_checked": 6, "records_failed": 0,
               "decisions": [{"actual": "GO", "expected": "GO"}] * 6}
    assert verifier.selftest_contract_error("synthetic_certificate_records", correct) is None
    assert verifier.selftest_contract_error("synthetic_certificate_records", dict(correct, records_failed=1)) is not None
    assert verifier.selftest_contract_error("synthetic_certificate_records", dict(correct, decisions=[])) is not None
    assert verifier.selftest_contract_error("synthetic_certificate_records", dict(correct, synthetic=False)) is not None


@pytest.mark.parametrize("path", (
    "results_trial/report.json", "data/records.json", "models/model.txt",
    "audits/private.md", "human_pilot_v2/answers.json",
))
def test_private_directory_classes_are_not_payload(path):
    assert verifier.prohibited(path)

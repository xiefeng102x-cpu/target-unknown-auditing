"""Data-independent public count-to-decision regression fixtures."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "auditability_certificate_v1/analysis"))
import replay_synthetic_certificate_records_v1 as replay


def records():
    return json.loads(replay.DEFAULT_FIXTURE.read_text())["records"]


def test_synthetic_reference_records():
    result = replay.replay_fixture()
    assert result["synthetic"] is True
    assert result["passed"] is True
    assert result["records_checked"] == 6
    assert result["records_failed"] == 0
    assert {row["actual"] for row in result["decisions"]} == {"GO", "NO_GO", "ABSTAIN"}


@pytest.mark.parametrize("name", ("no_sample", "no_poisons_and_no_removals", "empty_universe"))
def test_empty_information_abstains(name):
    record = next(row for row in records() if row["name"] == name)
    assert replay.replay_record(record) == "ABSTAIN"


def test_impossible_counts_rejected():
    record = copy.deepcopy(records()[0])
    record["flagged_sampled_poisons"] = record["flagged_sampled"] + 1
    with pytest.raises(ValueError, match="infeasible counts"):
        replay.replay_record(record)


def test_fixture_cannot_claim_real_experiment(tmp_path):
    document = json.loads(replay.DEFAULT_FIXTURE.read_text())
    document["synthetic"] = False
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="synthetic fixture"):
        replay.replay_fixture(path)

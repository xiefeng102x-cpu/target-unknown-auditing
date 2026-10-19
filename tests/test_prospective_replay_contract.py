from pathlib import Path
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prospective_replay_contract import ReplayPolicy, compare_csv, compare_frames  # noqa: E402


POLICY = ReplayPolicy(
    key_fields=("victim_id", "detector", "fold", "sample_id", "role"),
    exact_fields=("conventional_flag",),
    score_fields=("score",),
    absolute_tolerance=1e-10,
)


def rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"victim_id": 1, "detector": "ac", "fold": 0, "sample_id": 9,
             "role": "train", "conventional_flag": 1, "score": 0.25},
            {"victim_id": 1, "detector": "ac", "fold": 0, "sample_id": 10,
             "role": "train", "conventional_flag": 0, "score": -0.75},
        ]
    )


def test_sub_tolerance_drift_passes_but_raw_hashes_remain_visible(tmp_path: Path) -> None:
    expected = rows()
    replayed = rows()
    replayed.loc[0, "score"] += 1e-12
    expected_path = tmp_path / "expected.csv"
    replayed_path = tmp_path / "replayed.csv"
    expected.to_csv(expected_path, index=False)
    replayed.to_csv(replayed_path, index=False)

    report = compare_csv(expected_path, replayed_path, POLICY)

    assert report["status"] == "PASS"
    assert report["raw_sha256"]["byte_exact"] is False
    assert report["score_max_absolute_difference"]["score"] <= 1e-10


@pytest.mark.parametrize("mutation", ["key", "flag", "score"])
def test_endpoint_relevant_or_material_changes_fail(mutation: str) -> None:
    expected = rows()
    replayed = rows()
    if mutation == "key":
        replayed.loc[0, "sample_id"] = 999
    elif mutation == "flag":
        replayed.loc[0, "conventional_flag"] = 0
    else:
        replayed.loc[0, "score"] += 1e-4

    assert compare_frames(expected, replayed, POLICY)["status"] == "FAIL"


def test_duplicate_keys_are_rejected() -> None:
    duplicated = pd.concat([rows(), rows().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate replay keys"):
        compare_frames(duplicated, rows(), POLICY)


def test_missing_exact_values_match_only_when_both_are_missing() -> None:
    expected = rows()
    replayed = rows()
    expected.loc[0, "conventional_flag"] = None
    replayed.loc[0, "conventional_flag"] = None
    assert compare_frames(expected, replayed, POLICY)["status"] == "PASS"
    replayed.loc[0, "conventional_flag"] = 1
    assert compare_frames(expected, replayed, POLICY)["status"] == "FAIL"


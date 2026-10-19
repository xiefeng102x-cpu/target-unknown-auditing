from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replay_verify_cifar10_vgg16_blended_srac import policy_from_protocol  # noqa: E402


def protocol() -> dict:
    return {
        "prospective_replay_contract": {
            "contract_version": "prospective-semantic-replay-v1",
            "key_fields": ["victim_id", "detector", "fold", "sample_id", "role"],
            "exact_fields": ["conventional_flag"],
            "score_fields": ["score"],
            "absolute_tolerance": 1e-10,
            "relative_tolerance": 0.0,
        }
    }


def test_policy_is_loaded_exactly() -> None:
    value = policy_from_protocol(protocol())
    assert value.absolute_tolerance == 1e-10
    assert value.exact_fields == ("conventional_flag",)


def test_nonzero_relative_tolerance_is_rejected() -> None:
    value = protocol()
    value["prospective_replay_contract"]["relative_tolerance"] = 1e-8
    with pytest.raises(RuntimeError, match="relative tolerance"):
        policy_from_protocol(value)


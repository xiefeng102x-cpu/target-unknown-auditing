from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CERT = ROOT / "auditability_certificate_v1"
PROTOCOL = CERT / "PROTOCOL_CIFAR10_SEMANTIC_OVERLAP_CONFIRMATION_V1.json"
EXPECTED_PROTOCOL_SHA256 = "5a53c5798b728d40357d7e8f271cfb4a038d32f4597a98f581432407d3fc40bd"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def test_protocol_and_every_bound_file_are_immutable() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert sha256(PROTOCOL) == EXPECTED_PROTOCOL_SHA256
    assert (
        PROTOCOL.with_suffix(PROTOCOL.suffix + ".sha256")
        .read_text(encoding="utf-8")
        .split()[0]
        == EXPECTED_PROTOCOL_SHA256
    )
    stage_paths = {
        "victim_stage": {
            "overlay_sha256": "configs/cifar10_vgg16_bn_blended_semantic_overlap_confirmation_v1.json",
            "victim_runner_sha256": "src/run_cifar10_vgg16_bn_blended_semantic_overlap_confirmation.py",
            "base_config_sha256": "configs/cifar10_vgg16_bn_blended_srac_confirmation_v1.json",
            "base_engine_sha256": "src/run_cifar10_vgg16_bn_srac_confirmation.py",
            "architecture_sha256": "src/cifar10_vgg16_bn.py",
            "development_engine_sha256": "src/run_cifar10_vgg16_bn_development.py",
            "blended_wrapper_sha256": "src/run_cifar10_vgg16_bn_blended_development.py",
            "score_families_sha256": "src/blind_score_families.py",
        },
        "score_stage": {
            "score_runner_sha256": "src/run_cifar10_semantic_overlap_scan_confirmation.py",
            "adapter_sha256": "src/backdoorbench_feature_adapter.py",
        },
        "endpoint_stage": {
            "endpoint_core_sha256": "auditability_certificate_v1/analysis/run_protected_group_family_srac.py",
            "endpoint_runner_sha256": "auditability_certificate_v1/analysis/run_cifar10_semantic_overlap_confirmation.py",
        },
    }
    for stage, bindings in stage_paths.items():
        for hash_key, path in bindings.items():
            assert sha256(ROOT / path) == protocol["hashes"][stage][hash_key]


def test_groups_are_semantic_nested_crossing_and_cover_all_classes() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    groups = protocol["protected_groups"]
    names = [row["name"] for row in groups]
    sets = [set(row["class_ids"]) for row in groups]
    assert len(groups) == len(set(names)) == 8
    assert all(values and min(values) >= 0 and max(values) <= 9 for values in sets)
    assert set().union(*sets) == set(range(10))
    assert any(left < right for left in sets for right in sets)
    assert any(
        left & right and not (left <= right or right <= left)
        for left in sets
        for right in sets
    )
    assert protocol["formal_seeds"] == [4411, 4412, 4413, 4414, 4415]


def test_support_gate_and_stopping_rule_are_locked() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["support_gate"] == {
        "minimum_eligible_victims": 4,
        "minimum_true_policy_pass_victims": 2,
        "zero_false_go": True,
        "zero_false_nogo": True,
        "minimum_victims_with_go_at_k_8000": 1,
    }
    assert "no_seed_replacement" not in protocol["support_gate"]
    assert "without replacement" in protocol["stopping_rule"]


def test_overlay_changes_only_declared_execution_fields() -> None:
    overlay_path = ROOT / "configs/cifar10_vgg16_bn_blended_semantic_overlap_confirmation_v1.json"
    overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
    base_path = ROOT / overlay["base_config_path"]
    base = json.loads(base_path.read_text(encoding="utf-8"))
    assert sha256(base_path) == overlay["base_config_sha256"]
    assert overlay["formal_seeds"] == [4411, 4412, 4413, 4414, 4415]
    assert overlay["formal_ack"] == "OPEN_CIFAR10_VGG16_BLENDED_SEMANTIC_OVERLAP_SEEDS_4411_4415"
    for field in (
        "dataset",
        "victim",
        "development_provenance",
        "attack",
        "eligibility",
        "reference",
        "crossfit",
        "detectors",
    ):
        assert field not in overlay
        assert field in base

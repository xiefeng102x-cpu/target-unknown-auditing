from pathlib import Path
import json
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import run_cifar10_vgg16_bn_blended_srac_confirmation as adapter  # noqa: E402


def test_expected_hash_surface_names_all_reused_engines(monkeypatch) -> None:
    monkeypatch.setattr(adapter.engine, "sha256", lambda path: Path(path).name)
    hashes = adapter.expected_hashes()
    assert set(hashes) == {
        "config_sha256", "adapter_sha256", "base_confirmation_engine_sha256",
        "architecture_sha256", "development_engine_sha256", "blended_wrapper_sha256",
        "score_families_sha256", "semantic_replay_contract_sha256",
        "development_freeze_sha256", "development_report_sha256",
        "selected_development_checkpoint_sha256", "implementation_audit_sha256",
        "handoff_verifier_sha256", "semantic_replay_verifier_sha256",
        "endpoint_runner_sha256", "srac_core_sha256",
    }


def test_write_scores_relabels_all_handoffs_without_changing_receipts(tmp_path: Path, monkeypatch) -> None:
    seed = 4401
    old = f"cifar10_badnets_vgg16bn_seed{seed}"
    paths = []
    for name in ("score.csv", "truth.csv", "trusted.csv"):
        path = tmp_path / name
        path.write_text(f"victim_id,value\n{old},1\n", encoding="utf-8")
        paths.append(path)
    receipt = {"detector": "unchanged"}
    monkeypatch.setattr(adapter, "_BASE_WRITE_SCORES", lambda *a, **k: (*paths, receipt))

    returned = adapter.write_scores(tmp_path, {}, seed, np.array([]), np.empty((0, 1)),
                                    np.empty((0, 1)), np.empty((0, 1)), np.array([]),
                                    np.empty((0, 1)), np.array([]), np.array([]))

    assert returned[3] is receipt
    assert all("badnets" not in path.read_text() for path in paths)
    assert all(f"cifar10_blended_vgg16bn_seed{seed}" in path.read_text() for path in paths)


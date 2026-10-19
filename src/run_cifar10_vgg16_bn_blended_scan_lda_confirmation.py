#!/usr/bin/env python3
"""Train fresh CIFAR-10/VGG victims for the SCAn+LDA semantic-overlap confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

import run_cifar10_vgg16_bn_blended_development as blended
import run_cifar10_vgg16_bn_srac_confirmation as engine


ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "configs/cifar10_vgg16_bn_blended_scan_lda_confirmation_v1.json"
PROTOCOL = ROOT / "auditability_certificate_v1/PROTOCOL_CIFAR10_SCAN_LDA_CONFIRMATION_V1.json"
DEFAULT_OUTPUT = ROOT / "results/cifar10_vgg16_bn_blended_scan_lda_confirmation_v1"
ACK = "OPEN_CIFAR10_VGG16_BLENDED_SCAN_LDA_SEEDS_4416_4425"
SEEDS = (4416, 4417, 4418, 4419, 4420, 4421, 4422, 4423, 4424, 4425)
_BASE_WRITE_SCORES = engine.write_scores


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def merged_config() -> dict[str, Any]:
    overlay = json.loads(OVERLAY.read_text(encoding="utf-8"))
    base = ROOT / str(overlay["base_config_path"])
    if sha256(base) != overlay["base_config_sha256"]:
        raise RuntimeError("base configuration hash mismatch")
    config = json.loads(base.read_text(encoding="utf-8"))
    config["protocol_id"] = overlay["protocol_id"]
    config["status"] = "FROZEN_BEFORE_FORMAL_SEED_EXECUTION"
    config["formal_seeds"] = list(map(int, overlay["formal_seeds"]))
    config["output"] = overlay["victim_output"]
    return config


def expected_hashes() -> dict[str, str]:
    return {
        "overlay_sha256": sha256(OVERLAY),
        "victim_runner_sha256": sha256(Path(__file__)),
        "base_config_sha256": sha256(
            ROOT / "configs/cifar10_vgg16_bn_blended_srac_confirmation_v1.json"
        ),
        "base_engine_sha256": sha256(ROOT / "src/run_cifar10_vgg16_bn_srac_confirmation.py"),
        "architecture_sha256": sha256(ROOT / "src/cifar10_vgg16_bn.py"),
        "development_engine_sha256": sha256(ROOT / "src/run_cifar10_vgg16_bn_development.py"),
        "blended_wrapper_sha256": sha256(ROOT / "src/run_cifar10_vgg16_bn_blended_development.py"),
        "score_families_sha256": sha256(ROOT / "src/blind_score_families.py"),
    }


def load_and_verify() -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    config = merged_config()
    if protocol.get("status") != "FROZEN_BEFORE_FORMAL_SEED_EXECUTION_AND_PROTECTED_DRAWS":
        raise RuntimeError("formal freeze status mismatch")
    if protocol.get("hashes", {}).get("victim_stage") != expected_hashes():
        raise RuntimeError("victim-stage frozen hashes mismatch")
    if tuple(config["formal_seeds"]) != SEEDS or tuple(protocol["formal_seeds"]) != SEEDS:
        raise RuntimeError("unexpected confirmation seeds")
    if protocol.get("formal_ack") != ACK:
        raise RuntimeError("formal acknowledgement mismatch")
    return config, protocol


def write_scores(*args: Any, **kwargs: Any):
    score_path, truth_path, trusted_path, receipts = _BASE_WRITE_SCORES(*args, **kwargs)
    seed = int(args[2] if len(args) > 2 else kwargs["seed"])
    old = f"cifar10_badnets_vgg16bn_seed{seed}"
    new = f"cifar10_blended_scan_lda_vgg16bn_seed{seed}"
    for path in (score_path, truth_path, trusted_path):
        text = path.read_text(encoding="utf-8")
        if old not in text:
            raise RuntimeError(f"expected base victim identifier absent from {path}")
        path.write_text(text.replace(old, new), encoding="utf-8")
    return score_path, truth_path, trusted_path, receipts


def install() -> None:
    config = merged_config()
    blend, _ = blended.build_blender(config)
    engine.development.patch = blend
    engine.development.CONFIG = OVERLAY
    engine.CONFIG = OVERLAY
    engine.FREEZE = PROTOCOL
    engine.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    engine.ACK = ACK
    engine.load_and_verify = load_and_verify
    engine.write_scores = write_scores


def selftest() -> None:
    config = merged_config()
    assert tuple(config["formal_seeds"]) == SEEDS
    assert config["attack"]["implementation_name"] == "BackdoorBench-parameterized Blended variant"
    blend, trigger = blended.build_blender(config)
    observed = blend(torch.zeros(1, 3, 32, 32))
    expected = float(config["attack"]["train_blended_alpha"]) * trigger.unsqueeze(0)
    if not torch.allclose(observed, expected, atol=1e-7):
        raise RuntimeError("Blended transform selftest failed")
    print("SCAn+LDA victim wrapper selftest PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ack")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    install()
    if args.selftest:
        selftest()
    elif args.formal:
        engine.run_formal(args.output, torch.device("mps"), resume=args.resume, ack=args.ack)
    else:
        parser.error("choose --selftest or --formal")


if __name__ == "__main__":
    main()

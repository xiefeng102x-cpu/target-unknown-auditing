#!/usr/bin/env python3
"""Formal CIFAR-10/VGG16-BN Blended-variant handoff for SRAC.

This thin adapter deliberately reuses the audited CIFAR/VGG training and score
engine without modifying the already frozen BadNets implementation.  It becomes
executable only after a separate confirmation config and freeze manifest exist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from blind_score_families import METHODS
import run_cifar10_vgg16_bn_blended_development as blended
import run_cifar10_vgg16_bn_srac_confirmation as engine


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/cifar10_vgg16_bn_blended_srac_confirmation_v1.json"
FREEZE = ROOT / "SRAC_CIFAR10_VGG16_BLENDED_CONFIRMATION_FREEZE_V1.json"
DEFAULT_OUTPUT = ROOT / "results/cifar10_vgg16_bn_blended_srac_confirmation_v1"
ACK = "OPEN_CIFAR10_VGG16_BLENDED_SRAC_SEEDS_4401_4405"
_BASE_WRITE_SCORES = engine.write_scores


def expected_hashes() -> dict[str, str]:
    return {
        "config_sha256": engine.sha256(CONFIG),
        "adapter_sha256": engine.sha256(Path(__file__)),
        "base_confirmation_engine_sha256": engine.sha256(
            ROOT / "src/run_cifar10_vgg16_bn_srac_confirmation.py"
        ),
        "architecture_sha256": engine.sha256(ROOT / "src/cifar10_vgg16_bn.py"),
        "development_engine_sha256": engine.sha256(
            ROOT / "src/run_cifar10_vgg16_bn_development.py"
        ),
        "blended_wrapper_sha256": engine.sha256(
            ROOT / "src/run_cifar10_vgg16_bn_blended_development.py"
        ),
        "score_families_sha256": engine.sha256(ROOT / "src/blind_score_families.py"),
        "semantic_replay_contract_sha256": engine.sha256(
            ROOT / "PROSPECTIVE_SEMANTIC_REPLAY_CONTRACT_FREEZE_V1.json"
        ),
        "development_freeze_sha256": engine.sha256(
            ROOT / "CIFAR10_VGG16_BLENDED_DEVELOPMENT_FREEZE_V1.json"
        ),
        "development_report_sha256": engine.sha256(
            ROOT / "results/cifar10_vgg16_bn_blended_development_v1/development_report.json"
        ),
        "selected_development_checkpoint_sha256": engine.sha256(
            ROOT / "results/cifar10_vgg16_bn_blended_development_v1/development_seed4400_epoch15.pt"
        ),
        "implementation_audit_sha256": engine.sha256(
            ROOT / "auditability_certificate_v1/BLENDED_DEVELOPMENT_IMPLEMENTATION_AUDIT_2026-10-14.md"
        ),
        "handoff_verifier_sha256": engine.sha256(
            ROOT / "src/verify_cifar10_vgg16_blended_srac_handoff.py"
        ),
        "semantic_replay_verifier_sha256": engine.sha256(
            ROOT / "src/replay_verify_cifar10_vgg16_blended_srac.py"
        ),
        "endpoint_runner_sha256": engine.sha256(
            ROOT / "auditability_certificate_v1/analysis/run_cifar10_vgg16_blended_srac_confirmation.py"
        ),
        "srac_core_sha256": engine.sha256(
            ROOT / "auditability_certificate_v1/analysis/run_uniform_adjudication_certificate.py"
        ),
    }


def load_and_verify() -> tuple[dict[str, Any], dict[str, Any]]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    if frozen.get("status") != "FROZEN_BEFORE_FORMAL_SEED_EXECUTION_AND_PROTECTED_DRAWS":
        raise RuntimeError("formal freeze status mismatch")
    if frozen.get("hashes") != expected_hashes():
        raise RuntimeError("formal freeze manifest mismatch")
    if tuple(config.get("formal_seeds", ())) != (4401, 4402, 4403, 4404, 4405):
        raise RuntimeError("unexpected formal seeds")
    if tuple(config.get("detectors", ())) != METHODS:
        raise RuntimeError("unexpected detector set")
    replay = config.get("prospective_replay_contract", {})
    if replay.get("contract_version") != "prospective-semantic-replay-v1":
        raise RuntimeError("prospective semantic replay contract is absent")
    if float(replay.get("absolute_tolerance", -1.0)) != 1e-10:
        raise RuntimeError("unexpected replay tolerance")
    if config.get("attack", {}).get("implementation_name") != "BackdoorBench-parameterized Blended variant":
        raise RuntimeError("attack naming boundary is not frozen")
    return config, frozen


def write_scores(*args: Any, **kwargs: Any):
    """Reuse the score engine and replace only the human-readable victim tag."""

    score_path, truth_path, trusted_path, receipts = _BASE_WRITE_SCORES(*args, **kwargs)
    seed = int(args[2] if len(args) > 2 else kwargs["seed"])
    old = f"cifar10_badnets_vgg16bn_seed{seed}"
    new = f"cifar10_blended_vgg16bn_seed{seed}"
    for path in (score_path, truth_path, trusted_path):
        text = path.read_text(encoding="utf-8")
        if old not in text:
            raise RuntimeError(f"expected base victim identifier absent from {path}")
        path.write_text(text.replace(old, new), encoding="utf-8")
    return score_path, truth_path, trusted_path, receipts


def install() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8")) if CONFIG.is_file() else None
    if config is not None:
        blend, _ = blended.build_blender(config)
        engine.development.patch = blend
        engine.development.CONFIG = CONFIG
    engine.CONFIG = CONFIG
    engine.FREEZE = FREEZE
    engine.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    engine.ACK = ACK
    engine.load_and_verify = load_and_verify
    engine.write_scores = write_scores


def selftest() -> None:
    # The adapter can be tested before the config/freeze exist. The blend itself
    # is already covered by the frozen development wrapper's self-test.
    engine.selftest()
    if CONFIG.is_file():
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        blend, trigger = blended.build_blender(config)
        observed = blend(torch.zeros(1, 3, 32, 32))
        expected = float(config["attack"]["train_blended_alpha"]) * trigger.unsqueeze(0)
        if not torch.allclose(observed, expected, atol=1e-7):
            raise RuntimeError("formal Blended transform selftest failed")
    print("blended SRAC confirmation adapter selftest PASS")


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

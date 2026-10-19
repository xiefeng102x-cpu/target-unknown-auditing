#!/usr/bin/env python3
"""Victim-only development for a native CIFAR-10/VGG16 Blended block.

The reused development engine exposes only BCA and ASR.  This wrapper replaces
its BadNets patch with the canonical BackdoorBench blend and redirects it to a
separate development configuration.  Detector and SRAC outcomes remain
unavailable by construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

import run_cifar10_vgg16_bn_development as engine


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/cifar10_vgg16_bn_blended_development_v1.json"
DEFAULT_OUTPUT = ROOT / "results/cifar10_vgg16_bn_blended_development_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_blender(config: dict):
    attack = config["attack"]
    asset = ROOT / attack["trigger_asset_relative_to_paper14"]
    if sha256(asset) != attack["trigger_asset_sha256"]:
        raise RuntimeError("Blended trigger asset hash mismatch")
    with Image.open(asset) as image:
        resized = TF.resize(
            image.convert("RGB"),
            [32, 32],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        # Match BackdoorBench's Resize-then-ToTensor order exactly.
        trigger = TF.pil_to_tensor(resized).to(torch.float32).div_(255.0)
    alpha_train = float(attack["train_blended_alpha"])
    alpha_test = float(attack["test_blended_alpha"])
    if alpha_train != alpha_test:
        raise RuntimeError("the reused engine requires equal train/test blend alpha")

    def blend(values: torch.Tensor, size: int = 4) -> torch.Tensor:
        del size
        pattern = trigger.to(device=values.device, dtype=values.dtype).unsqueeze(0)
        return ((1.0 - alpha_train) * values + alpha_train * pattern).clamp_(0.0, 1.0)

    return blend, trigger


def install() -> tuple[dict, torch.Tensor]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config.get("status") != "NONFORMAL_VICTIM_ONLY_DEVELOPMENT":
        raise RuntimeError("development wrapper cannot open a formal protocol")
    if not config["development_gate"].get("detector_or_certificate_outcomes_forbidden"):
        raise RuntimeError("development information boundary is not locked")
    blend, trigger = build_blender(config)
    engine.CONFIG = CONFIG
    engine.patch = blend
    return config, trigger


def selftest() -> None:
    config, trigger = install()
    zeros = torch.zeros(2, 3, 32, 32)
    observed = engine.patch(zeros)
    expected = float(config["attack"]["train_blended_alpha"]) * trigger.unsqueeze(0)
    if observed.shape != zeros.shape or not torch.allclose(observed[:1], expected, atol=1e-7):
        raise RuntimeError("blend transform selftest failed")
    if float(observed.min()) < 0.0 or float(observed.max()) > 1.0:
        raise RuntimeError("blend transform leaves image range")
    engine.selftest()
    print("blended development wrapper selftest PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--develop", action="store_true")
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    install()
    if args.selftest:
        selftest()
    elif args.develop:
        report = engine.train(args.output, torch.device(args.device))
        report_path = args.output / "development_report.json"
        report["wrapper_sha256"] = sha256(Path(__file__))
        report["trigger_asset_sha256"] = json.loads(CONFIG.read_text(encoding="utf-8"))["attack"]["trigger_asset_sha256"]
        engine.atomic_json(report_path, report)
    else:
        parser.error("choose --selftest or --develop")


if __name__ == "__main__":
    main()

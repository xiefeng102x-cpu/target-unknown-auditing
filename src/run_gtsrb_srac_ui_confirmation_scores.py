#!/usr/bin/env python3
"""Block 16 launcher: frozen SCAn scoring of the new GTSRB victims (seeds 255-264).

Reuses the unchanged block-8 score runner (pinned BackdoorBench SCAn, deterministic half of
each class's official test records as trusted references) with block-16 paths.
"""
from __future__ import annotations
import hashlib, importlib.util, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_RUNNER = ROOT / "src/run_gtsrb_semantic_overlap_scan_confirmation.py"
VICTIM_PROTOCOL = ROOT / "configs/srac_gtsrb_srac_ui_confirmation_v1.json"
VICTIM_RESULTS = ROOT / "results/srac_gtsrb_srac_ui_confirmation_v1"
PROTOCOL = ROOT / "auditability_certificate_v1/PROTOCOL_GTSRB_SRAC_UI_CONFIRMATION_V1.json"
OUTPUT = ROOT / "results/gtsrb_srac_ui_scan_scores_confirmation_v1"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_VICTIM_TRAINING":
        raise RuntimeError("block-16 protocol is not frozen")
    if protocol["runner_launcher_sha256"] != sha256(Path(__file__)):
        raise RuntimeError("launcher differs from frozen protocol")
    spec = importlib.util.spec_from_file_location("srac_ui_score_base", BASE_RUNNER)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
    module.VICTIM_PROTOCOL = VICTIM_PROTOCOL; module.VICTIM_RESULTS = VICTIM_RESULTS
    module.PROTOCOL_DEFAULT = PROTOCOL; module.OUTPUT_DEFAULT = OUTPUT
    module.main()


if __name__ == "__main__":
    main()

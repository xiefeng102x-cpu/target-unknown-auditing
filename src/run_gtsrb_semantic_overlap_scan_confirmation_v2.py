#!/usr/bin/env python3
"""Versioned launcher for the corrected semantic-overlap SCAn handoff."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_RUNNER = ROOT / "src/run_gtsrb_semantic_overlap_scan_confirmation.py"
VICTIM_PROTOCOL = ROOT / "configs/srac_gtsrb_semantic_overlap_confirmation_v2.json"
VICTIM_RESULTS = ROOT / "results/srac_gtsrb_semantic_overlap_confirmation_v2"
DEFAULT_PROTOCOL = ROOT / "auditability_certificate_v1/PROTOCOL_GTSRB_SEMANTIC_OVERLAP_CONFIRMATION_V2.json"
DEFAULT_OUTPUT = ROOT / "results/gtsrb_semantic_overlap_scan_scores_confirmation_v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    probe.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    known, _ = probe.parse_known_args()
    protocol = json.loads(known.protocol.read_text(encoding="utf-8"))
    sidecar = known.protocol.with_suffix(known.protocol.suffix + ".sha256")
    if sha256(known.protocol) != sidecar.read_text(encoding="utf-8").split()[0]:
        raise RuntimeError("protocol sidecar mismatch")
    if protocol["runner_launcher_sha256"] != sha256(Path(__file__)):
        raise RuntimeError("launcher differs from frozen protocol")
    if protocol["runner_sha256"] != sha256(BASE_RUNNER):
        raise RuntimeError("base score runner differs from frozen protocol")

    spec = importlib.util.spec_from_file_location("semantic_overlap_score_base_v2", BASE_RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.VICTIM_PROTOCOL = VICTIM_PROTOCOL
    module.VICTIM_RESULTS = VICTIM_RESULTS
    module.PROTOCOL_DEFAULT = DEFAULT_PROTOCOL
    module.OUTPUT_DEFAULT = DEFAULT_OUTPUT
    module.main()


if __name__ == "__main__":
    main()

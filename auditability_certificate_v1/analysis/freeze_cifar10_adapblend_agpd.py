#!/usr/bin/env python3
"""Freeze block 14 (AGPD on the Adap-Blend victims) before any detector output exists.

Writes the code and input hashes into the protocol, sets its status, writes the .sha256
sidecar and a freeze receipt.  Refuses if the protocol is already frozen or if any score or
endpoint output exists.
"""
from __future__ import annotations
import hashlib, json, platform, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CERT = ROOT / "auditability_certificate_v1"
PROTOCOL = CERT / "PROTOCOL_CIFAR10_ADAPBLEND_AGPD_V1.json"
RECEIPT = CERT / "CIFAR10_ADAPBLEND_AGPD_FREEZE_RECEIPT_V1.json"
SCORE_STAGE = ["src/run_cifar10_adapblend_agpd_scores.py", "src/agpd_feature_adapter.py",
               "src/run_cifar10_vgg16_bn_adapblend_scan_lda_confirmation.py", "src/cifar10_vgg16_bn.py",
               "src/run_cifar10_vgg16_bn_development.py", "configs/cifar10_vgg16_bn_adapblend_scan_lda_confirmation_v1.json"]
ENDPOINT_STAGE = ["auditability_certificate_v1/analysis/run_cifar10_adapblend_agpd_endpoint.py",
                  "auditability_certificate_v1/analysis/srac_family_core_scan_lda_v1.py"]
VICTIMS = ROOT / "results/cifar10_vgg16_bn_adapblend_scan_lda_confirmation_v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("status") != "DRAFT_NOT_FROZEN":
        raise SystemExit(f"protocol status is {protocol.get('status')!r}; refusing to re-freeze")
    for out in (protocol["score_output_directory"], protocol["endpoint_output_directory"]):
        if (ROOT / out).exists():
            raise SystemExit(f"output already exists: {out}; refusing to freeze after outputs")
    seed_receipts = json.loads((VICTIMS / "seed_receipts.json").read_text(encoding="utf-8"))
    protocol["hashes"] = {"score_stage": {p: sha256(ROOT / p) for p in SCORE_STAGE},
                          "endpoint_stage": {p: sha256(ROOT / p) for p in ENDPOINT_STAGE}}
    protocol["inputs"] = {"victim_manifest_sha256": sha256(VICTIMS / "manifest.json"),
                          "seed_receipts_sha256": sha256(VICTIMS / "seed_receipts.json"),
                          "checkpoint_sha256": {str(r["seed"]): r["model_sha256"] for r in seed_receipts}}
    protocol["status"] = "FROZEN_BEFORE_DETECTOR_OUTPUTS"
    protocol["frozen_at"] = datetime.now(timezone.utc).isoformat()
    PROTOCOL.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    digest = sha256(PROTOCOL)
    PROTOCOL.with_suffix(".json.sha256").write_text(f"{digest}  {PROTOCOL.name}\n", encoding="utf-8")
    receipt = {"written_utc": protocol["frozen_at"], "status": "frozen before detector outputs",
               "protocol": PROTOCOL.name, "protocol_sha256": digest, "hashes": protocol["hashes"],
               "inputs": protocol["inputs"], "python": sys.version.split()[0], "platform": platform.platform()}
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()

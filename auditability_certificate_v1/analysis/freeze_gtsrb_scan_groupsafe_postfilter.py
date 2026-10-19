#!/usr/bin/env python3
"""Freeze block 15 (GO -> removal -> retraining) before any retraining output exists."""
from __future__ import annotations
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/srac_gtsrb_scan_groupsafe_postfilter_v1.json"
RECEIPT = ROOT / "auditability_certificate_v1/GTSRB_SCAN_GROUPSAFE_POSTFILTER_FREEZE_RECEIPT_V1.json"
OUTPUT = ROOT / "results/srac_gtsrb_scan_groupsafe_postfilter_v1"
HASHED = ["src/run_gtsrb_scan_groupsafe_postfilter.py", "src/run_tifs_native_gtsrb.py", "src/run_tifs_native_cifar10.py",
          "configs/srac_gtsrb_allclass_scan_confirmation_v1.json", "results/gtsrb_allclass_scan_scores_confirmation_v1/flags.csv",
          "auditability_certificate_v1/results_gtsrb_scan_group_safe_confirmation_v1/decisions.csv",
          "results/srac_gtsrb_allclass_scan_confirmation_v1/victim_metrics.csv"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    proto = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if proto.get("status") != "DRAFT_NOT_FROZEN":
        sys.exit(f"protocol status {proto.get('status')!r}; refusing to re-freeze")
    if OUTPUT.exists():
        sys.exit(f"{OUTPUT} exists; refusing to freeze after outputs")
    proto["hashes"] = {p: sha256(ROOT / p) for p in HASHED}
    proto["status"] = "FROZEN_BEFORE_RETRAINING"
    proto["frozen_at"] = datetime.now(timezone.utc).isoformat()
    PROTOCOL.write_text(json.dumps(proto, indent=2) + "\n", encoding="utf-8")
    digest = sha256(PROTOCOL)
    PROTOCOL.with_suffix(".json.sha256").write_text(f"{digest}  {PROTOCOL.name}\n", encoding="utf-8")
    RECEIPT.write_text(json.dumps({"written_utc": proto["frozen_at"], "protocol": str(PROTOCOL.relative_to(ROOT)),
                                   "protocol_sha256": digest, "hashes": proto["hashes"]}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"protocol_sha256": digest, "frozen_at": proto["frozen_at"]}, indent=2))


if __name__ == "__main__":
    main()

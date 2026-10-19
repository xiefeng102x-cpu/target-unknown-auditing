#!/usr/bin/env python3
"""Freeze block 16 before any seed 255-264 is trained: finalize the victim config, bind every
script and input by hash, write the protocol sidecar and a receipt."""
from __future__ import annotations
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]; CERT = ROOT / "auditability_certificate_v1"
PROTOCOL = CERT / "PROTOCOL_GTSRB_SRAC_UI_CONFIRMATION_V1.json"
CONFIG = ROOT / "configs/srac_gtsrb_srac_ui_confirmation_v1.json"
RECEIPT = CERT / "GTSRB_SRAC_UI_CONFIRMATION_FREEZE_RECEIPT_V1.json"
ENDPOINT = ["auditability_certificate_v1/analysis/run_gtsrb_srac_ui_confirmation.py",
            "auditability_certificate_v1/analysis/run_stratified_ui_audit_v1.py",
            "auditability_certificate_v1/analysis/run_audit_baselines_v1.py",
            "auditability_certificate_v1/analysis/run_posthoc_power_study_v1.py"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    proto = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if proto.get("status") != "DRAFT_NOT_FROZEN":
        sys.exit(f"status {proto.get('status')!r}; refusing to re-freeze")
    for out in (proto["victim_output_directory"], proto["score_output_directory"], proto["endpoint_output_directory"]):
        if (ROOT / out).exists():
            sys.exit(f"{out} exists; refusing to freeze after outputs")
    now = datetime.now(timezone.utc).isoformat()
    cfg = json.loads(CONFIG.read_text(encoding="utf-8")); cfg["frozen_at"] = now
    CONFIG.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    for key, rel in (("victim_protocol_sha256", proto["victim_protocol_path"]), ("victim_runner_sha256", proto["victim_runner_path"]),
                     ("adapter_sha256", proto["adapter_path"]), ("runner_sha256", proto["runner_path"]),
                     ("runner_launcher_sha256", proto["runner_launcher_path"])):
        proto[key] = sha256(ROOT / rel)
    proto["hashes"] = {"endpoint_stage": {rel: sha256(ROOT / rel) for rel in ENDPOINT}}
    proto["status"] = "FROZEN_BEFORE_VICTIM_TRAINING"; proto["frozen_at"] = now
    PROTOCOL.write_text(json.dumps(proto, indent=2) + "\n", encoding="utf-8")
    digest = sha256(PROTOCOL)
    PROTOCOL.with_suffix(".json.sha256").write_text(f"{digest}  {PROTOCOL.name}\n", encoding="utf-8")
    RECEIPT.write_text(json.dumps({"written_utc": now, "protocol_sha256": digest, "config_sha256": proto["victim_protocol_sha256"],
                                   "bindings": {k: proto[k] for k in ("victim_runner_sha256", "adapter_sha256", "runner_sha256", "runner_launcher_sha256")},
                                   "endpoint_stage": proto["hashes"]["endpoint_stage"]}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"protocol_sha256": digest, "frozen_at": now}, indent=2))


if __name__ == "__main__":
    main()

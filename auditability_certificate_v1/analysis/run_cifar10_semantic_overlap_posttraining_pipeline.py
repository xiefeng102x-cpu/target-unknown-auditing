#!/usr/bin/env python3
"""Fail-closed post-training pipeline for the frozen CIFAR confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CERT = ROOT / "auditability_certificate_v1"
PROTOCOL = CERT / "PROTOCOL_CIFAR10_SEMANTIC_OVERLAP_CONFIRMATION_V1.json"
VICTIMS = ROOT / "results/cifar10_vgg16_bn_blended_semantic_overlap_confirmation_v1"
SCORES = ROOT / "results/cifar10_semantic_overlap_scan_scores_confirmation_v1"
ENDPOINT = CERT / "results_cifar10_semantic_overlap_confirmation_v1"
SCORE_RUNNER = ROOT / "src/run_cifar10_semantic_overlap_scan_confirmation.py"
ENDPOINT_RUNNER = CERT / "analysis/run_cifar10_semantic_overlap_confirmation.py"
ACTIVE_ENDPOINT_CORE = CERT / "analysis/run_protected_group_family_srac.py"
FROZEN_ENDPOINT_CORE = (
    CERT
    / "analysis/frozen/cifar10_semantic_overlap_v1/run_protected_group_family_srac.py"
)
FINALIZER = CERT / "analysis/finalize_cifar10_semantic_overlap_confirmation.py"
FINAL_RECEIPT = ENDPOINT / "audit_receipt_2026-10-15.json"
ACK = "OPEN_CIFAR10_SEMANTIC_OVERLAP_POSTTRAINING_PIPELINE_V1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def classify_stage(
    *,
    common_gates_ok: bool,
    training_complete: bool,
    score_state: str,
    endpoint_state: str,
    final_receipt_present: bool,
) -> str:
    """Classify the only safe next action without rerunning a completed stage."""
    if not common_gates_ok:
        return "BLOCKED_COMMON_GATE"
    if not training_complete:
        return "TRAINING_INCOMPLETE"
    if score_state == "invalid" or endpoint_state == "invalid":
        return "BLOCKED_PARTIAL_OR_CORRUPT_OUTPUT"
    if score_state == "absent" and endpoint_state != "absent":
        return "BLOCKED_INCONSISTENT_STAGE_ORDER"
    if endpoint_state == "complete" and score_state != "complete":
        return "BLOCKED_INCONSISTENT_STAGE_ORDER"
    if final_receipt_present:
        return "TERMINAL_COMPLETE" if endpoint_state == "complete" else "BLOCKED_INCONSISTENT_STAGE_ORDER"
    if score_state == "absent" and endpoint_state == "absent":
        return "READY_SCORE"
    if score_state == "complete" and endpoint_state == "absent":
        return "READY_ENDPOINT"
    if score_state == "complete" and endpoint_state == "complete":
        return "READY_FINALIZER"
    return "BLOCKED_INCONSISTENT_STAGE_ORDER"


def score_state(protocol_hash: str) -> str:
    if not SCORES.exists():
        return "absent"
    manifest_path = SCORES / "manifest.json"
    flags_path = SCORES / "flags.csv"
    if not (manifest_path.is_file() and flags_path.is_file()):
        return "invalid"
    manifest = load_json(manifest_path)
    valid = bool(
        manifest.get("status") == "CIFAR10_SEMANTIC_OVERLAP_SCAN_HANDOFF_COMPLETE"
        and manifest.get("protocol_sha256") == protocol_hash
        and manifest.get("flags_sha256") == sha256(flags_path)
    )
    return "complete" if valid else "invalid"


def endpoint_state(protocol_hash: str) -> str:
    if not ENDPOINT.exists():
        return "absent"
    report_path = ENDPOINT / "report.json"
    decisions = ENDPOINT / "decisions.csv"
    summary = ENDPOINT / "summary.csv"
    truth = ENDPOINT / "truth_by_victim.csv"
    if not all(path.is_file() for path in (report_path, decisions, summary, truth)):
        return "invalid"
    report = load_json(report_path)
    valid_status = report.get("status") in {
        "PASS_CIFAR10_SEMANTIC_OVERLAP_CONFIRMATION",
        "FAIL_CIFAR10_SEMANTIC_OVERLAP_CONFIRMATION",
    }
    output_hashes = report.get("outputs", {})
    valid = bool(
        valid_status
        and report.get("protocol_sha256") == protocol_hash
        and output_hashes.get("decisions_sha256") == sha256(decisions)
        and output_hashes.get("summary_sha256") == sha256(summary)
        and output_hashes.get("truth_sha256") == sha256(truth)
    )
    return "complete" if valid else "invalid"


def readiness() -> dict[str, object]:
    protocol_hash = sha256(PROTOCOL)
    sidecar_ok = (
        PROTOCOL.with_suffix(PROTOCOL.suffix + ".sha256")
        .read_text(encoding="utf-8")
        .split()[0]
        == protocol_hash
    )
    protocol = load_json(PROTOCOL)
    victim_manifest_path = VICTIMS / "manifest.json"
    manifest = load_json(victim_manifest_path) if victim_manifest_path.is_file() else {}
    training_complete = bool(
        manifest.get("status") == "SCORE_HANDOFF_COMPLETE"
        and manifest.get("completed_seeds") == protocol["formal_seeds"]
        and manifest.get("freeze_manifest_sha256") == protocol_hash
    )
    frozen_stage_hashes_ok = bool(
        protocol["hashes"]["score_stage"]["score_runner_sha256"] == sha256(SCORE_RUNNER)
        and protocol["hashes"]["endpoint_stage"]["endpoint_runner_sha256"]
        == sha256(ENDPOINT_RUNNER)
        and protocol["hashes"]["endpoint_stage"]["endpoint_core_sha256"]
        == sha256(FROZEN_ENDPOINT_CORE)
    )
    active_endpoint_core_matches_freeze = bool(
        protocol["hashes"]["endpoint_stage"]["endpoint_core_sha256"]
        == sha256(ACTIVE_ENDPOINT_CORE)
    )
    observed_score_state = score_state(protocol_hash)
    observed_endpoint_state = endpoint_state(protocol_hash)
    terminal_outputs_present = bool(
        FINAL_RECEIPT.is_file()
        and observed_score_state == "complete"
        and observed_endpoint_state == "complete"
    )
    execution_hashes_ok = bool(
        frozen_stage_hashes_ok
        and (terminal_outputs_present or active_endpoint_core_matches_freeze)
    )
    stage = classify_stage(
        common_gates_ok=bool(sidecar_ok and execution_hashes_ok),
        training_complete=training_complete,
        score_state=observed_score_state,
        endpoint_state=observed_endpoint_state,
        final_receipt_present=FINAL_RECEIPT.is_file(),
    )
    return {
        "protocol_sha256": protocol_hash,
        "protocol_sidecar_ok": sidecar_ok,
        "stage_hashes_ok": frozen_stage_hashes_ok,
        "active_endpoint_core_matches_freeze": active_endpoint_core_matches_freeze,
        "postfreeze_core_update_detected": not active_endpoint_core_matches_freeze,
        "training_complete": training_complete,
        "completed_seeds": manifest.get("completed_seeds", []),
        "score_state": observed_score_state,
        "endpoint_state": observed_endpoint_state,
        "final_receipt_present": FINAL_RECEIPT.is_file(),
        "pipeline_stage": stage,
        "ready": stage.startswith("READY_"),
    }


def commands() -> list[list[str]]:
    return [
        [
            sys.executable,
            str(SCORE_RUNNER),
            "--protocol",
            str(PROTOCOL),
            "--victims",
            str(VICTIMS),
            "--output",
            str(SCORES),
            "--device",
            "mps",
        ],
        [
            sys.executable,
            str(ENDPOINT_RUNNER),
            "--protocol",
            str(PROTOCOL),
            "--scores",
            str(SCORES),
            "--output",
            str(ENDPOINT),
        ],
        [
            sys.executable,
            str(FINALIZER),
            "--protocol",
            str(PROTOCOL),
            "--victims",
            str(VICTIMS),
            "--scores",
            str(SCORES),
            "--endpoint",
            str(ENDPOINT),
        ],
    ]


def pending_commands(stage: str) -> list[list[str]]:
    start = {
        "READY_SCORE": 0,
        "READY_ENDPOINT": 1,
        "READY_FINALIZER": 2,
    }.get(stage)
    return commands()[start:] if start is not None else []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--ack")
    args = parser.parse_args()
    if args.check == args.execute:
        parser.error("choose exactly one of --check or --execute")
    state = readiness()
    print(json.dumps(state, indent=2, sort_keys=True), flush=True)
    if args.check:
        return
    if args.ack != ACK:
        raise RuntimeError(f"formal post-training execution requires --ack {ACK}")
    if not state["ready"]:
        raise RuntimeError("post-training pipeline is not ready; no command was run")
    for command in pending_commands(str(state["pipeline_stage"])):
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()

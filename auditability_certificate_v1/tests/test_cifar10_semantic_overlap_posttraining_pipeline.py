from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "analysis/run_cifar10_semantic_overlap_posttraining_pipeline.py"
SPEC = importlib.util.spec_from_file_location("cifar_semantic_pipeline", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_pipeline_order_is_score_then_endpoint_then_outcome_neutral_finalizer() -> None:
    commands = MODULE.commands()
    assert len(commands) == 3
    assert commands[0][1].endswith("run_cifar10_semantic_overlap_scan_confirmation.py")
    assert commands[1][1].endswith("run_cifar10_semantic_overlap_confirmation.py")
    assert commands[2][1].endswith("finalize_cifar10_semantic_overlap_confirmation.py")
    assert commands[0][-1] == "mps"


def test_live_readiness_is_the_conjunction_of_all_fail_closed_gates() -> None:
    state = MODULE.readiness()
    assert state["protocol_sidecar_ok"] is True
    assert state["stage_hashes_ok"] is True
    assert MODULE.sha256(MODULE.FROZEN_ENDPOINT_CORE) == (
        "9171814bd8be311461cb6c3a493be1b8faba58c06dbd76f497d51c47bd9d5422"
    )
    assert state["active_endpoint_core_matches_freeze"] is False
    assert state["postfreeze_core_update_detected"] is True
    if (MODULE.ENDPOINT / "decisions.csv").is_file():
        assert state["pipeline_stage"] == "TERMINAL_COMPLETE"
    else:
        # The review-sized public profile deliberately omits private per-decision
        # tables. It must not claim terminal replay readiness without them.
        assert state["endpoint_state"] == "invalid"
        assert state["pipeline_stage"] == "BLOCKED_COMMON_GATE"
    assert state["ready"] is state["pipeline_stage"].startswith("READY_")


def test_completed_stages_are_resumed_without_rerunning_them() -> None:
    assert MODULE.classify_stage(
        common_gates_ok=True, training_complete=True, score_state="absent",
        endpoint_state="absent", final_receipt_present=False,
    ) == "READY_SCORE"
    assert MODULE.classify_stage(
        common_gates_ok=True, training_complete=True, score_state="complete",
        endpoint_state="absent", final_receipt_present=False,
    ) == "READY_ENDPOINT"
    assert MODULE.classify_stage(
        common_gates_ok=True, training_complete=True, score_state="complete",
        endpoint_state="complete", final_receipt_present=False,
    ) == "READY_FINALIZER"
    assert [command[1] for command in MODULE.pending_commands("READY_ENDPOINT")] == [
        str(MODULE.ENDPOINT_RUNNER), str(MODULE.FINALIZER)
    ]
    assert [command[1] for command in MODULE.pending_commands("READY_FINALIZER")] == [
        str(MODULE.FINALIZER)
    ]


def test_corrupt_partial_outputs_and_terminal_receipt_close_execution() -> None:
    assert MODULE.classify_stage(
        common_gates_ok=True, training_complete=True, score_state="invalid",
        endpoint_state="absent", final_receipt_present=False,
    ) == "BLOCKED_PARTIAL_OR_CORRUPT_OUTPUT"
    assert MODULE.classify_stage(
        common_gates_ok=True, training_complete=True, score_state="complete",
        endpoint_state="invalid", final_receipt_present=False,
    ) == "BLOCKED_PARTIAL_OR_CORRUPT_OUTPUT"
    assert MODULE.classify_stage(
        common_gates_ok=True, training_complete=True, score_state="complete",
        endpoint_state="complete", final_receipt_present=True,
    ) == "TERMINAL_COMPLETE"
    assert MODULE.pending_commands("TERMINAL_COMPLETE") == []


def test_common_gate_or_training_failure_closes_execution() -> None:
    assert MODULE.classify_stage(
        common_gates_ok=False, training_complete=True, score_state="absent",
        endpoint_state="absent", final_receipt_present=False,
    ) == "BLOCKED_COMMON_GATE"
    assert MODULE.classify_stage(
        common_gates_ok=True, training_complete=False, score_state="absent",
        endpoint_state="absent", final_receipt_present=False,
    ) == "TRAINING_INCOMPLETE"

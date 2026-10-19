#!/usr/bin/env python3
"""Prospective semantic replay contract for detector-score handoffs.

This module is for protocols frozen *after* its introduction.  It does not
reinterpret an earlier byte-exact replay.  Identity-bearing fields and endpoint
decisions are exact; continuous diagnostic scores may differ only within a
predeclared numerical tolerance.  Raw-file hashes remain evidence and are never
discarded merely because semantic replay passes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


CONTRACT_VERSION = "prospective-semantic-replay-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ReplayPolicy:
    """Fields and tolerances that must be frozen before endpoint generation."""

    key_fields: tuple[str, ...]
    exact_fields: tuple[str, ...]
    score_fields: tuple[str, ...]
    absolute_tolerance: float

    def __post_init__(self) -> None:
        if not self.key_fields:
            raise ValueError("at least one replay key is required")
        if self.absolute_tolerance < 0.0 or not np.isfinite(self.absolute_tolerance):
            raise ValueError("absolute_tolerance must be finite and nonnegative")
        overlap = set(self.exact_fields) & set(self.score_fields)
        if overlap:
            raise ValueError(f"fields cannot be both exact and tolerant: {sorted(overlap)}")

    def as_dict(self) -> dict:
        return {
            "contract_version": CONTRACT_VERSION,
            "key_fields": list(self.key_fields),
            "exact_fields": list(self.exact_fields),
            "score_fields": list(self.score_fields),
            "absolute_tolerance": self.absolute_tolerance,
            "relative_tolerance": 0.0,
            "raw_sha256_disposition": "retained_as_diagnostic_even_when_semantic_replay_passes",
        }


def _require_columns(frame: pd.DataFrame, fields: Iterable[str], label: str) -> None:
    missing = sorted(set(fields) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def compare_frames(
    expected: pd.DataFrame,
    replayed: pd.DataFrame,
    policy: ReplayPolicy,
) -> dict:
    """Compare one-to-one replay rows under a prospectively frozen policy."""

    fields = policy.key_fields + policy.exact_fields + policy.score_fields
    _require_columns(expected, fields, "expected frame")
    _require_columns(replayed, fields, "replayed frame")
    if expected.duplicated(list(policy.key_fields)).any():
        raise ValueError("expected frame has duplicate replay keys")
    if replayed.duplicated(list(policy.key_fields)).any():
        raise ValueError("replayed frame has duplicate replay keys")

    joined = expected[list(fields)].merge(
        replayed[list(fields)],
        on=list(policy.key_fields),
        how="outer",
        suffixes=("_expected", "_replayed"),
        indicator=True,
        validate="one_to_one",
    )
    missing_expected = int(joined["_merge"].eq("right_only").sum())
    missing_replayed = int(joined["_merge"].eq("left_only").sum())
    exact_mismatches: dict[str, int] = {}
    score_max_abs: dict[str, float] = {}
    score_nonfinite: dict[str, int] = {}
    score_exceedances: dict[str, int] = {}

    paired = joined["_merge"].eq("both")
    for field in policy.exact_fields:
        left = joined.loc[paired, f"{field}_expected"]
        right = joined.loc[paired, f"{field}_replayed"]
        equal = left.eq(right) | (left.isna() & right.isna())
        exact_mismatches[field] = int((~equal).sum())
    for field in policy.score_fields:
        left = pd.to_numeric(joined.loc[paired, f"{field}_expected"], errors="coerce").to_numpy(float)
        right = pd.to_numeric(joined.loc[paired, f"{field}_replayed"], errors="coerce").to_numpy(float)
        finite = np.isfinite(left) & np.isfinite(right)
        score_nonfinite[field] = int((~finite).sum())
        differences = np.abs(left[finite] - right[finite])
        score_max_abs[field] = float(differences.max(initial=0.0))
        score_exceedances[field] = int(np.sum(differences > policy.absolute_tolerance))

    passed = (
        missing_expected == 0
        and missing_replayed == 0
        and not any(exact_mismatches.values())
        and not any(score_nonfinite.values())
        and not any(score_exceedances.values())
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "contract": policy.as_dict(),
        "expected_rows": int(len(expected)),
        "replayed_rows": int(len(replayed)),
        "missing_expected_keys": missing_expected,
        "missing_replayed_keys": missing_replayed,
        "exact_field_mismatches": exact_mismatches,
        "score_max_absolute_difference": score_max_abs,
        "score_nonfinite_pairs": score_nonfinite,
        "score_tolerance_exceedances": score_exceedances,
    }


def compare_csv(
    expected_path: Path,
    replayed_path: Path,
    policy: ReplayPolicy,
) -> dict:
    """Compare CSV semantics while retaining both raw byte hashes."""

    report = compare_frames(pd.read_csv(expected_path), pd.read_csv(replayed_path), policy)
    report["raw_sha256"] = {
        "expected": sha256(expected_path),
        "replayed": sha256(replayed_path),
        "byte_exact": sha256(expected_path) == sha256(replayed_path),
    }
    return report


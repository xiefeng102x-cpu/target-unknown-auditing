#!/usr/bin/env python3
"""Build a score-blind adjudication packet with duplicates and gold controls.

The public packet never contains detector names, scores, flags, or gold poison
identities.  The private key is required for evaluation and must not be given
to adjudicators.  This tool prepares a study; it does not create evidence that
an adjudicator is reliable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED = {"victim_id", "sample_id", "role", "is_poison"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def token(rng: np.random.Generator, prefix: str) -> str:
    return prefix + "_" + bytes(rng.integers(0, 256, size=16, dtype=np.uint8)).hex()


def build(
    truth: pd.DataFrame,
    *,
    uniform_rows: int,
    challenge_rows_per_class: int,
    duplicate_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if not REQUIRED.issubset(truth.columns):
        raise RuntimeError(f"truth table lacks columns: {sorted(REQUIRED - set(truth.columns))}")
    base = truth.loc[truth["role"].eq("train"), list(REQUIRED)].drop_duplicates(
        ["victim_id", "sample_id"]
    )
    if len(base) != int(truth["role"].eq("train").sum()):
        raise RuntimeError("truth table has duplicate train identities")
    if set(base["is_poison"].astype(int).unique()) - {0, 1}:
        raise RuntimeError("gold labels are not binary")
    if not 0.0 <= duplicate_fraction <= 1.0:
        raise RuntimeError("duplicate_fraction must lie in [0,1]")
    rng = np.random.default_rng(seed)

    if uniform_rows > len(base):
        raise RuntimeError("uniform_rows exceeds the available population")
    uniform_idx = rng.choice(base.index.to_numpy(), size=uniform_rows, replace=False)
    selected: list[dict] = []
    for index in uniform_idx:
        selected.append({**base.loc[index].to_dict(), "study_stratum": "uniform"})

    # Gold challenges measure correctness on both classes. They are analyzed
    # separately and never used as an estimate of field prevalence.
    remaining = base.drop(index=uniform_idx)
    for label, name in ((0, "clean_challenge"), (1, "poison_challenge")):
        available = remaining.loc[remaining["is_poison"].astype(int).eq(label)]
        if challenge_rows_per_class > len(available):
            raise RuntimeError(f"insufficient rows for {name}")
        chosen = rng.choice(available.index.to_numpy(), size=challenge_rows_per_class, replace=False)
        for index in chosen:
            selected.append({**available.loc[index].to_dict(), "study_stratum": name})
        remaining = remaining.drop(index=chosen)

    units = pd.DataFrame(selected).reset_index(drop=True)
    units["unit_token"] = [token(rng, "unit") for _ in range(len(units))]
    duplicate_count = int(round(duplicate_fraction * len(units)))
    duplicate_units = set(
        rng.choice(units["unit_token"].to_numpy(), size=duplicate_count, replace=False)
    ) if duplicate_count else set()

    presentations: list[dict] = []
    private: list[dict] = []
    for row in units.to_dict("records"):
        repeats = 2 if row["unit_token"] in duplicate_units else 1
        for repeat in range(repeats):
            presentation_id = token(rng, "case")
            presentations.append({
                "presentation_id": presentation_id,
                # A custodian or retrieval service resolves this one-time handle.
                # Never expose the underlying sample ID: repeated presentations
                # would otherwise reveal that they are duplicates.
                "record_handle": presentation_id,
                "adjudicated_is_poison": "",
                "adjudicator_id": "",
                "adjudicated_at": "",
                "notes": "",
            })
            private.append({
                "presentation_id": presentation_id,
                "unit_token": row["unit_token"],
                "victim_id": row["victim_id"],
                "sample_id": int(row["sample_id"]),
                "study_stratum": row["study_stratum"],
                "repeat_index": repeat,
                "gold_is_poison": int(row["is_poison"]),
            })
    public = pd.DataFrame(presentations).sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)
    key = pd.DataFrame(private)
    if public["presentation_id"].duplicated().any() or key["presentation_id"].duplicated().any():
        raise RuntimeError("presentation token collision")
    if set(public["presentation_id"]) != set(key["presentation_id"]):
        raise RuntimeError("public/private presentation mismatch")
    receipt = {
        "status": "PACKET_CREATED_NOT_ADJUDICATED",
        "population_rows": int(len(base)),
        "unique_study_units": int(len(units)),
        "uniform_units": int((units["study_stratum"] == "uniform").sum()),
        "clean_challenge_units": int((units["study_stratum"] == "clean_challenge").sum()),
        "poison_challenge_units": int((units["study_stratum"] == "poison_challenge").sum()),
        "duplicate_units": duplicate_count,
        "presentations": int(len(public)),
        "seed": seed,
        "warning": "Packet creation is not operational adjudicator evidence. Keep the private key hidden until every adjudication is locked.",
        "retrieval_requirement": "A blinded custodian must resolve each record_handle to the private victim/sample mapping without revealing that mapping or duplicate linkage to the adjudicator.",
    }
    return public, key, receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--uniform-rows", type=int, default=500)
    parser.add_argument("--challenge-rows-per-class", type=int, default=100)
    parser.add_argument("--duplicate-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=731001)
    args = parser.parse_args()
    truth = pd.read_csv(args.truth)
    public, key, receipt = build(
        truth,
        uniform_rows=args.uniform_rows,
        challenge_rows_per_class=args.challenge_rows_per_class,
        duplicate_fraction=args.duplicate_fraction,
        seed=args.seed,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    public_path = args.output / "PUBLIC_adjudication_packet.csv"
    key_path = args.output / "PRIVATE_adjudication_key.csv"
    public.to_csv(public_path, index=False)
    key.to_csv(key_path, index=False)
    receipt.update({
        "source_truth_sha256": sha256(args.truth),
        "public_packet_sha256": sha256(public_path),
        "private_key_sha256": sha256(key_path),
        "nonce": secrets.token_hex(16),
    })
    (args.output / "packet_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

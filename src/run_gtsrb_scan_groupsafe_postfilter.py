#!/usr/bin/env python3
"""Block 15: does a group-safe GO screen remove the backdoor?  (GO -> removal -> retraining)

Source: the ten frozen block-6 GTSRB victims (seeds 235-244) and their frozen SCAn flags,
which meet the all-class group-safe policy and receive GO in 22.8%, 90.0%, and 100% of
audits at k = 2000, 8000, 16000.  For every victim the training set is retrained from
scratch, with the unchanged victim protocol and the victim seed, after five removals:

  original_poisoned           no removal (the frozen suspect checkpoint; no retraining)
  detector_scan               remove every SCAn-flagged record
  random_matched_global       remove the same number of records uniformly from the whole set
  random_matched_flagged      remove the same number uniformly from the classes SCAn flagged
  oracle_all_poison           remove every poison (clean-training reference)

Primary contrast: paired detector_scan minus random_matched_flagged ASR (the stronger
control, since SCAn flags only inside the flagged classes); secondary: minus
random_matched_global.  Utility: paired BCA differences.  Support rule per contrast: 95%
bootstrap upper bound of the paired ASR difference below zero and mean paired BCA
difference at least -0.05.  The frozen block-6 GO rates are attached per victim.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import run_tifs_native_cifar10 as common
import run_tifs_native_gtsrb as gtsrb


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_DEFAULT = ROOT / "configs/srac_gtsrb_scan_groupsafe_postfilter_v1.json"
OUTPUT_DEFAULT = ROOT / "results/srac_gtsrb_scan_groupsafe_postfilter_v1"
CONDITIONS = ("original_poisoned", "detector_scan", "random_matched_global", "random_matched_flagged", "oracle_all_poison")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def sample_ids(data) -> list[str]:
    clean = [f"train_clean_{int(index)}" for index in data.train_indices]
    poison = [f"train_poison_source{int(s)}_base{int(i)}" for s, i in zip(data.poison_sources, data.poison_base_indices)]
    return clean + poison


def removal_masks(data, flags: pd.DataFrame, seed: int, rng_base: int) -> dict[str, np.ndarray]:
    ids = sample_ids(data); n = len(ids); n_clean = len(data.train_indices)
    frame = flags.set_index("sample_id")
    if len(frame) != n or set(frame.index) != set(ids):
        raise RuntimeError(f"seed {seed}: flag table does not cover the training population")
    frame = frame.loc[ids]
    truth = np.r_[np.zeros(n_clean, dtype=bool), np.ones(n - n_clean, dtype=bool)]
    if not np.array_equal(frame.is_poison.to_numpy(dtype=bool), truth):
        raise RuntimeError(f"seed {seed}: poison identities do not match the population order")
    if not np.array_equal(frame.assigned_group.to_numpy(dtype=int), np.asarray(data.train_labels, dtype=int)):
        raise RuntimeError(f"seed {seed}: assigned labels do not match the population order")
    detector = frame.flag.to_numpy(dtype=bool)
    count = int(detector.sum())
    flagged_classes = np.unique(np.asarray(data.train_labels)[detector])
    in_flagged = np.isin(np.asarray(data.train_labels), flagged_classes)
    rng = np.random.default_rng(rng_base + 101 * seed)
    glob = np.zeros(n, dtype=bool); glob[rng.permutation(n)[:count]] = True
    rng = np.random.default_rng(rng_base + 101 * seed + 1009)
    cls = np.zeros(n, dtype=bool); pool = np.flatnonzero(in_flagged); cls[rng.permutation(pool)[:count]] = True
    oracle = truth.copy()
    return {"detector_scan": detector, "random_matched_global": glob, "random_matched_flagged": cls,
            "oracle_all_poison": oracle, "_flagged_classes": flagged_classes}


def sanitized(data, remove: np.ndarray):
    n_clean = len(data.train_indices)
    return dataclasses.replace(
        data,
        train_indices=data.train_indices[~remove[:n_clean]],
        poison_base_indices=data.poison_base_indices[~remove[n_clean:]],
        poison_sources=data.poison_sources[~remove[n_clean:]],
        train_images=data.train_images[~remove],
        train_labels=data.train_labels[~remove])


def bootstrap(values: np.ndarray, seed: int, draws: int) -> list[float]:
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def summarize(rows: pd.DataFrame, protocol: dict, go_rates: dict) -> dict:
    out = {"conditions": {}, "contrasts": {}, "go_rates_by_seed": go_rates}
    for c in CONDITIONS:
        b = rows[rows.condition == c]
        if len(b):
            out["conditions"][c] = {"seeds": int(len(b)), "mean_asr": float(b.attack_success_rate.mean()),
                                    "mean_bca": float(b.balanced_clean_accuracy.mean()),
                                    "mean_clean_removed": float(b.clean_removed.mean()), "mean_poison_removed": float(b.poison_removed.mean())}
    a = protocol["analysis"]
    for ctrl in ("random_matched_flagged", "random_matched_global"):
        d = rows[rows.condition == "detector_scan"].set_index("seed"); r = rows[rows.condition == ctrl].set_index("seed")
        common_seeds = sorted(set(d.index) & set(r.index))
        if not common_seeds:
            continue
        asr = (d.loc[common_seeds].attack_success_rate - r.loc[common_seeds].attack_success_rate).to_numpy()
        bca = (d.loc[common_seeds].balanced_clean_accuracy - r.loc[common_seeds].balanced_clean_accuracy).to_numpy()
        ci = bootstrap(asr, int(a["bootstrap_seed"]), int(a["bootstrap_draws"]))
        out["contrasts"][f"detector_scan_minus_{ctrl}"] = {
            "seeds": len(common_seeds), "mean_asr_difference": float(asr.mean()), "asr_ci95": ci,
            "mean_bca_difference": float(bca.mean()), "per_seed_asr_difference": asr.tolist(),
            "supported": bool(ci[1] < 0 and bca.mean() >= -0.05) if len(common_seeds) == 10 else None}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", type=Path, default=PROTOCOL_DEFAULT)
    ap.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    ap.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--eval-batch-size", type=int, default=512)
    ap.add_argument("--check-masks", action="store_true", help="build and verify all removal masks; no training")
    args = ap.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if not args.check_masks:
        if protocol.get("status") != "FROZEN_BEFORE_RETRAINING":
            raise RuntimeError("protocol is not frozen")
        if args.protocol.with_suffix(".json.sha256").read_text().split()[0] != sha256(args.protocol):
            raise RuntimeError("protocol sidecar mismatch")
        for rel, digest in protocol["hashes"].items():
            if sha256(ROOT / rel) != digest:
                raise RuntimeError(f"frozen hash mismatch: {rel}")
    victim_protocol = json.loads((ROOT / protocol["victim_protocol"]).read_text(encoding="utf-8"))
    victims = ROOT / protocol["source_victims"]
    flags_all = pd.read_csv(ROOT / protocol["source_flags"])
    decisions = pd.read_csv(ROOT / protocol["source_decisions"])
    seeds = [int(s) for s in protocol["seeds"]]
    data_root, _ = gtsrb.validate_dataset(victim_protocol)
    out = args.output
    log = gtsrb.RunLog((out if not args.check_masks else Path("/tmp/b15_check")) / "run.log")
    cache, _ = gtsrb.build_resized_cache(data_root, log)
    vm = pd.read_csv(victims / "victim_metrics.csv").set_index("seed")

    if args.check_masks:
        for seed in seeds:
            data = gtsrb.build_seed_data(cache["train_images"], cache["train_labels"], victim_protocol, seed)
            gtsrb.validate_seed_data(data, cache["train_labels"], victim_protocol)
            f = flags_all[flags_all.seed == seed]
            m = removal_masks(data, f, seed, int(protocol["random_removal_rng_base"]))
            sizes = {k: int(v.sum()) for k, v in m.items() if not k.startswith("_")}
            if not (sizes["detector_scan"] == sizes["random_matched_global"] == sizes["random_matched_flagged"]):
                raise RuntimeError("matched sizes differ")
            print(f"seed={seed} masks ok: removal size {sizes['detector_scan']}, flagged classes {m['_flagged_classes'].tolist()}")
        print("mask check PASS"); return

    device = gtsrb.resolve_device(args.device, protocol)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "per_seed_metrics.csv"
    rows = pd.read_csv(path) if path.exists() else pd.DataFrame()
    done = set(zip(rows.seed, rows.condition)) if len(rows) else set()
    go_rates = {str(s): decisions[decisions.victim_id.str.endswith(f"seed{s}")].groupby("k").decision.apply(lambda x: float((x == "GO").mean())).to_dict()
                for s in seeds}
    for seed in seeds:
        data = gtsrb.build_seed_data(cache["train_images"], cache["train_labels"], victim_protocol, seed)
        gtsrb.validate_seed_data(data, cache["train_labels"], victim_protocol)
        if sha256(victims / f"models/seed_{seed}_suspect.pt") != vm.loc[seed, "checkpoint_sha256"]:
            raise RuntimeError(f"seed {seed}: source checkpoint hash mismatch")
        masks = removal_masks(data, flags_all[flags_all.seed == seed], seed, int(protocol["random_removal_rng_base"]))
        n_clean = len(data.train_indices)
        if (seed, "original_poisoned") not in done:
            orig = dict(seed=seed, condition="original_poisoned", rows_removed=0, clean_removed=0, poison_removed=0,
                            balanced_clean_accuracy=float(vm.loc[seed, "balanced_clean_accuracy"]),
                            attack_success_rate=float(vm.loc[seed, "attack_success_rate"]), best_epoch=int(vm.loc[seed, "best_epoch"]),
                            remove_mask_sha256="", checkpoint_sha256=vm.loc[seed, "checkpoint_sha256"], training_seconds=0.0)
            rows = pd.concat([rows, pd.DataFrame([orig])], ignore_index=True); rows.to_csv(path, index=False); done.add((seed, "original_poisoned"))
        for cond in CONDITIONS[1:]:
            if (seed, cond) in done:
                log(f"seed={seed} condition={cond} skip existing"); continue
            remove = masks[cond]; started = time.time()
            log(f"seed={seed} condition={cond} start removed={int(remove[:n_clean].sum())}/{int(remove[n_clean:].sum())}")
            model, training = gtsrb.train_model(sanitized(data, remove), victim_protocol, seed, args.threads, device, log)
            metrics = gtsrb.victim_metrics(model, cache, victim_protocol, args.eval_batch_size, device)
            ck = out / "models" / cond / f"seed_{seed}.pt"; ck.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"seed": seed, "condition": cond, "remove_mask_sha256": sha256_array(remove.astype(np.uint8)),
                        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, ck)
            row = (dict(seed=seed, condition=cond, rows_removed=int(remove.sum()), clean_removed=int(remove[:n_clean].sum()),
                            poison_removed=int(remove[n_clean:].sum()), balanced_clean_accuracy=metrics["balanced_clean_accuracy"],
                            attack_success_rate=metrics["attack_success_rate"], best_epoch=training["best_epoch"],
                            remove_mask_sha256=sha256_array(remove.astype(np.uint8)), checkpoint_sha256=sha256(ck),
                            training_seconds=time.time() - started))
            log(f"seed={seed} condition={cond} BCA={metrics['balanced_clean_accuracy']:.4f} ASR={metrics['attack_success_rate']:.4f}")
            rows = pd.concat([rows, pd.DataFrame([row])], ignore_index=True); rows.to_csv(path, index=False); done.add((seed, cond))
    rows = rows.sort_values(["seed", "condition"]).reset_index(drop=True); rows.to_csv(path, index=False)
    summary = summarize(rows, protocol, go_rates)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    complete = len(rows) == len(seeds) * len(CONDITIONS)
    (out / "manifest.json").write_text(json.dumps({"status": "complete" if complete else "partial", "protocol_sha256": sha256(args.protocol),
                                                  "runner_sha256": sha256(Path(__file__)), "device": str(device),
                                                  "per_seed_metrics_sha256": sha256(path)}, indent=2) + "\n")
    print(json.dumps(summary["contrasts"], indent=2)); print(json.dumps(summary["conditions"], indent=2))


if __name__ == "__main__":
    main()

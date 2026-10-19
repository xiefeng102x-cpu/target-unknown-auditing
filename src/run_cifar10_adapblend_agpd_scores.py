#!/usr/bin/env python3
"""Frozen AGPD (ICLR 2025) full-population handoff on the Adap-Blend victims of block 13.

Block 14.  The victims, attack plan, protected family, budgets, and certificate are those
of block 13, unchanged; only the screen differs.  The screen is the repository AGPD
adapter (src/agpd_feature_adapter.py, BackdoorBench upstream f02e3534) with the
parameters of the earlier BloodMNIST AGPD protocol: activation-gradient features of the
assigned-class logit at pre-declared convolutional layers, one-sided MAD class anomaly
with threshold e^2 (fallback rho 0.3), progressive filtering with tau 0.5 and window 10.
It is target-unknown: target classes come from the class-anomaly step.

Every training member is scored in the form in which it was presented during training
(payload and cover rows carry their partial trigger).  The clean reference set holds
12 correctly classified held-out validation images per class, chosen by a fixed seeded
order.  Payload identities are carried privately for post-draw adjudication only.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import inspect
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch

import agpd_feature_adapter as agpd
import run_cifar10_vgg16_bn_adapblend_scan_lda_confirmation as victim
from cifar10_vgg16_bn import CIFAR10VGG16BN


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_DEFAULT = ROOT / "auditability_certificate_v1/PROTOCOL_CIFAR10_ADAPBLEND_AGPD_V1.json"
VICTIM_RESULTS = ROOT / "results/cifar10_vgg16_bn_adapblend_scan_lda_confirmation_v1"
OUTPUT_DEFAULT = ROOT / "results/cifar10_adapblend_agpd_scores_v1"
STATUS = "CIFAR10_ADAPBLEND_AGPD_HANDOFF_COMPLETE"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def call_agpd(model, candidates, assigned, references, reference_labels, parameters, device, batch_size):
    """Call the repository adapter by keyword so that any signature drift fails loudly."""
    kwargs = dict(model=model, candidate_images=candidates, assigned_labels=assigned,
                  reference_images=references, reference_labels=reference_labels,
                  layer_names=list(parameters["layer_names"]), num_classes=10,
                  tau=float(parameters["tau"]), anomaly_threshold=float(parameters["anomaly_threshold"]),
                  fallback_rho_threshold=float(parameters["fallback_rho_threshold"]),
                  window_size=int(parameters["window_size"]), batch_size=int(batch_size), device=device)
    accepted = inspect.signature(agpd.run_agpd_adapter).parameters
    missing = [name for name in kwargs if name not in accepted]
    if missing:
        raise RuntimeError(f"AGPD adapter signature drift; unknown keywords {missing}; accepted {list(accepted)}")
    return agpd.run_agpd_adapter(**kwargs)


def _jsonable(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def result_receipt(result) -> dict:
    if dataclasses.is_dataclass(result):
        raw = dataclasses.asdict(result)
    elif hasattr(result, "_asdict"):
        raw = result._asdict()
    else:
        raw = dict(vars(result))
    out = {}
    for key, value in raw.items():
        if key == "flags":
            continue
        if isinstance(value, (np.ndarray, torch.Tensor)):
            arr = np.asarray(value.detach().cpu() if isinstance(value, torch.Tensor) else value)
            out[key] = arr.tolist() if arr.size <= 64 else {"shape": list(arr.shape), "sha256": array_sha256(arr)}
        elif isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
        elif isinstance(value, (list, tuple)) and len(value) <= 64:
            out[key] = [v.item() if hasattr(v, "item") else v for v in value]
        elif isinstance(value, dict):
            out[key] = json.loads(json.dumps({str(k): v for k, v in value.items()}, default=_jsonable))
        else:
            out[key] = str(value)[:500]
    return out


def flags_of(result, n: int) -> np.ndarray:
    flags = np.asarray(result.flags.detach().cpu() if isinstance(result.flags, torch.Tensor) else result.flags).astype(bool).reshape(-1)
    if flags.shape != (n,):
        raise RuntimeError(f"AGPD returned {flags.shape} flags for {n} candidates")
    return flags


def presented_images(images, indices, masks, trigger, alpha, batch_size=1024) -> torch.Tensor:
    """Float [0,1] NCHW CPU tensor of the training members as presented in training."""
    out = []
    cpu = torch.device("cpu")
    for start in range(0, len(indices), batch_size):
        local = indices[start:start + batch_size]
        values = victim.development.uint8_batch(images, local, cpu)
        rows = [i for i, sid in enumerate(local) if int(sid) in masks]
        if rows:
            stacked = torch.from_numpy(np.stack([masks[int(local[i])] for i in rows]))
            selector = torch.tensor(rows)
            values[selector] = victim.apply_trigger(values[selector], stacked, trigger, alpha)
        out.append(values.to(torch.float32).cpu())
    return torch.cat(out)


def predict(model, values: torch.Tensor, device, batch_size=512) -> np.ndarray:
    preds = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            preds.append(model(values[start:start + batch_size].to(device)).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def reference_set(model, images, labels, validation_indices, *, per_class, seed_base, device):
    """First `per_class` correctly classified validation images per class, in a fixed seeded order."""
    chosen = []
    for label in range(10):
        order = np.random.default_rng(seed_base + 1009 * label).permutation(validation_indices[labels[validation_indices] == label])
        values = victim.development.uint8_batch(images, order, torch.device("cpu")).to(torch.float32)
        correct = order[predict(model, values, device) == label]
        if len(correct) < per_class:
            raise RuntimeError(f"fewer than {per_class} correctly classified class-{label} validation images")
        chosen.append(correct[:per_class])
    ids = np.concatenate(chosen).astype(np.int64)
    return ids


def selftest() -> None:
    """Synthetic API check on CPU: untrained VGG, random images.  Reveals no study outcome."""
    torch.manual_seed(0)
    model = CIFAR10VGG16BN().eval()
    rng = np.random.default_rng(0)
    candidates = torch.from_numpy(rng.random((300, 3, 32, 32), dtype=np.float32))
    assigned = rng.integers(0, 10, 300)
    references = torch.from_numpy(rng.random((120, 3, 32, 32), dtype=np.float32))
    ref_labels = np.repeat(np.arange(10), 12)
    params = json.loads(PROTOCOL_DEFAULT.read_text(encoding="utf-8"))["detector_parameters"]
    names = dict(model.named_modules())
    for layer in params["layer_names"]:
        if layer not in names or not isinstance(names[layer], torch.nn.Conv2d):
            raise RuntimeError(f"layer {layer} is not a Conv2d of CIFAR10VGG16BN")
    result = call_agpd(model, candidates, assigned, references, ref_labels, params, torch.device("cpu"), 64)
    flags = flags_of(result, 300)
    receipt = result_receipt(result)
    print(f"AGPD selftest PASS: {int(flags.sum())} synthetic flags; receipt keys {sorted(receipt)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_DEFAULT)
    parser.add_argument("--victims", type=Path, default=VICTIM_RESULTS)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest(); return

    sidecar = args.protocol.with_suffix(args.protocol.suffix + ".sha256")
    protocol_hash = sha256(args.protocol)
    if protocol_hash != sidecar.read_text(encoding="utf-8").split()[0]:
        raise RuntimeError("protocol sidecar mismatch: run the freeze step first")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_DETECTOR_OUTPUTS":
        raise RuntimeError("protocol is not frozen")
    for name, digest in protocol["hashes"]["score_stage"].items():
        if sha256(ROOT / name) != digest:
            raise RuntimeError(f"score-stage hash mismatch: {name}")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available; the frozen protocol scores on MPS")
    if args.device != protocol["execution"]["device"]:
        raise RuntimeError("device differs from the frozen protocol")
    device = torch.device(args.device)

    config = victim.load_config()
    trigger = victim.load_trigger(config)
    alpha = float(config["attack"]["train_alpha"])
    manifest_path = args.victims / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "SCORE_HANDOFF_COMPLETE" or manifest.get("completed_seeds") != protocol["formal_seeds"]:
        raise RuntimeError("block-13 victim stage incomplete")
    if sha256(manifest_path) != protocol["inputs"]["victim_manifest_sha256"]:
        raise RuntimeError("victim manifest differs from the frozen input")
    development = victim.development
    train_images, train_labels, _, _, dataset_receipt = development.resolve_dataset(config)
    train_indices, validation_indices = development.fixed_split(
        train_labels, rows_per_class=int(config["dataset"]["validation_rows_per_class"]),
        seed=int(config["dataset"]["validation_split_seed"]))
    receipts_by_seed = {int(r["seed"]): r for r in json.loads((args.victims / "seed_receipts.json").read_text(encoding="utf-8"))}
    target = int(config["attack"]["target_class"])
    parameters = protocol["detector_parameters"]
    frames, receipts = [], []
    for seed in map(int, protocol["formal_seeds"]):
        seed_receipt = receipts_by_seed[seed]
        checkpoint = args.victims / f"models/seed_{seed}_suspect.pt"
        if sha256(checkpoint) != seed_receipt["model_sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch for seed {seed}")
        model = CIFAR10VGG16BN()
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False), strict=True)
        model.to(device).eval()
        payload, cover = victim.attack_plan(train_labels, train_indices, config, seed)
        if array_sha256(payload) != seed_receipt["poison_indices_sha256"] or array_sha256(cover) != seed_receipt["cover_indices_sha256"]:
            raise RuntimeError(f"attack-plan mismatch for seed {seed}")
        is_payload = np.isin(train_indices, payload, assume_unique=True)
        is_cover = np.isin(train_indices, cover, assume_unique=True)
        assigned = train_labels[train_indices].copy(); assigned[is_payload] = target
        masks = victim.mask_table(np.concatenate([payload, cover]), seed, config)
        candidates = presented_images(train_images, train_indices, masks, trigger, alpha)
        refs = reference_set(model, train_images, train_labels, validation_indices,
                             per_class=int(parameters["reference_rows_per_class"]),
                             seed_base=int(parameters["reference_seed_base"]) + seed, device=device)
        references = development.uint8_batch(train_images, refs, torch.device("cpu")).to(torch.float32)
        torch.manual_seed(int(parameters["torch_seed_base"]) + seed)
        result = call_agpd(model, candidates, assigned, references, train_labels[refs], parameters, device,
                           int(protocol["execution"]["batch_size"]))
        flags = flags_of(result, len(train_indices))
        clean = ~is_payload
        class_rates = {str(c): float(flags[clean & (assigned == c)].mean()) for c in range(10)}
        eligible = bool(seed_receipt["eligible"])
        frames.append(pd.DataFrame({
            "victim_id": f"cifar10_adapblend_agpd_vgg16bn_seed{seed}", "seed": seed,
            "sample_id": train_indices.astype(np.int64), "assigned_group": assigned.astype(np.int64),
            "flag": flags.astype(np.int8), "is_poison": is_payload.astype(np.int8),
            "is_cover": is_cover.astype(np.int8), "victim_eligible": int(eligible)}))
        receipts.append({
            "seed": seed, "checkpoint_sha256": sha256(checkpoint), "eligible": eligible,
            "recall": float(flags[is_payload].mean()), "cover_flag_rate": float(flags[is_cover].mean()),
            "recall_including_cover": float(flags[is_payload | is_cover].mean()),
            "all_clean_flag_rate": float(flags[clean].mean()), "clean_flag_rate_by_class": class_rates,
            "flagged_rows": int(flags.sum()), "reference_ids_sha256": array_sha256(refs),
            "agpd_receipt": result_receipt(result)})
        print(f"seed={seed} eligible={eligible} recall={receipts[-1]['recall']:.6f} "
              f"max_class_clean={max(class_rates.values()):.6f} flagged={int(flags.sum())}", flush=True)
        del candidates
        if device.type == "mps":
            torch.mps.empty_cache()

    args.output.mkdir(parents=True)
    flags_path = args.output / "flags.csv"
    pd.concat(frames, ignore_index=True).to_csv(flags_path, index=False)
    out = {"status": STATUS, "created_utc": datetime.now(timezone.utc).isoformat(), "study_id": protocol["study_id"],
           "protocol_sha256": protocol_hash, "victim_manifest_sha256": sha256(manifest_path), "dataset": dataset_receipt,
           "device": device.type, "flags_sha256": sha256(flags_path), "seed_count": len(receipts), "seeds": receipts}
    (args.output / "manifest.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "seeds"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

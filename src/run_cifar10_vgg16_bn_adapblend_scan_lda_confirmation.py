#!/usr/bin/env python3
"""Train fresh CIFAR-10/VGG-16-BN victims under an Adap-Blend attack (block 13).

The block tests, prospectively, whether the frozen SCAn+LDA screen of block 12 keeps
positive coverage under an attack designed against latent separability.  This stage
only trains victims and records clean accuracy and attack success; detector and
certificate outcomes are computed by later, separately hash-bound stages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import balanced_accuracy_score
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from cifar10_vgg16_bn import CIFAR10VGG16BN, parameter_count
import run_cifar10_vgg16_bn_development as development


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/cifar10_vgg16_bn_adapblend_scan_lda_confirmation_v1.json"
PROTOCOL = ROOT / "auditability_certificate_v1/PROTOCOL_CIFAR10_ADAPBLEND_SCAN_LDA_CONFIRMATION_V1.json"
DEFAULT_OUTPUT = ROOT / "results/cifar10_vgg16_bn_adapblend_scan_lda_confirmation_v1"
ACK = "OPEN_CIFAR10_VGG16_ADAPBLEND_SCAN_LDA_SEEDS_4426_4435"
SEEDS = tuple(range(4426, 4436))
PIECE = 8  # 32x32 image, 4x4 grid of 8x8 pieces


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def expected_hashes() -> dict[str, str]:
    config = load_config()
    return {
        "config_sha256": sha256(CONFIG),
        "victim_runner_sha256": sha256(Path(__file__)),
        "architecture_sha256": sha256(ROOT / "src/cifar10_vgg16_bn.py"),
        "development_engine_sha256": sha256(ROOT / "src/run_cifar10_vgg16_bn_development.py"),
        "trigger_asset_sha256": sha256(ROOT / config["attack"]["trigger_asset_relative_to_paper14"]),
    }


def load_and_verify() -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    config = load_config()
    sidecar = PROTOCOL.with_suffix(PROTOCOL.suffix + ".sha256")
    if sidecar.read_text(encoding="utf-8").split()[0] != sha256(PROTOCOL):
        raise RuntimeError("protocol sidecar mismatch")
    if protocol.get("status") != "FROZEN_BEFORE_FORMAL_SEED_EXECUTION_AND_PROTECTED_DRAWS":
        raise RuntimeError("formal freeze status mismatch")
    if protocol.get("hashes", {}).get("victim_stage") != expected_hashes():
        raise RuntimeError("victim-stage frozen hashes mismatch")
    if tuple(config["formal_seeds"]) != SEEDS or tuple(protocol["formal_seeds"]) != SEEDS:
        raise RuntimeError("unexpected confirmation seeds")
    if protocol.get("formal_ack") != ACK or config.get("formal_ack") != ACK:
        raise RuntimeError("formal acknowledgement mismatch")
    return config, protocol


def load_trigger(config: dict[str, Any]) -> torch.Tensor:
    attack = config["attack"]
    asset = ROOT / attack["trigger_asset_relative_to_paper14"]
    if sha256(asset) != attack["trigger_asset_sha256"]:
        raise RuntimeError("trigger asset hash mismatch")
    with Image.open(asset) as image:
        resized = TF.resize(image.convert("RGB"), [32, 32], interpolation=InterpolationMode.BILINEAR, antialias=True)
        return TF.pil_to_tensor(resized).to(torch.float32).div_(255.0)


def attack_plan(labels: np.ndarray, train_indices: np.ndarray, config: dict[str, Any], seed: int) -> tuple[np.ndarray, np.ndarray]:
    attack = config["attack"]
    target = int(attack["target_class"])
    payload = development.poison_plan(labels, train_indices, target=target, count=int(attack["payload_count"]), seed=seed)
    eligible = train_indices[labels[train_indices] != target]
    eligible = np.setdiff1d(eligible, payload, assume_unique=True)
    cover = np.random.default_rng(4_243_100 + seed).choice(eligible, size=int(attack["cover_count"]), replace=False)
    return np.sort(payload).astype(np.int64), np.sort(cover).astype(np.int64)


def piece_mask(seed: int, sample_id: int, pieces: int = 16, active: int = 8) -> np.ndarray:
    chosen = np.random.default_rng(4_244_100 + 1009 * seed + int(sample_id)).choice(pieces, size=active, replace=False)
    mask = np.zeros((1, 32, 32), dtype=np.float32)
    for piece in chosen:
        row, col = divmod(int(piece), 4)
        mask[:, row * PIECE:(row + 1) * PIECE, col * PIECE:(col + 1) * PIECE] = 1.0
    return mask


def mask_table(sample_ids: np.ndarray, seed: int, config: dict[str, Any]) -> dict[int, np.ndarray]:
    attack = config["attack"]
    return {
        int(sid): piece_mask(seed, int(sid), int(attack["pieces"]), int(attack["active_pieces_per_training_row"]))
        for sid in sample_ids
    }


def apply_trigger(values: torch.Tensor, masks: torch.Tensor, trigger: torch.Tensor, alpha: float) -> torch.Tensor:
    pattern = trigger.to(device=values.device, dtype=values.dtype).unsqueeze(0)
    weight = alpha * masks.to(device=values.device, dtype=values.dtype)
    return (values + weight * (pattern - values)).clamp_(0.0, 1.0)


def evaluate(model: CIFAR10VGG16BN, images: np.ndarray, labels: np.ndarray, device: torch.device,
             target: int, batch_size: int, trigger: torch.Tensor, alpha: float) -> dict[str, float]:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            values = development.uint8_batch(images, np.arange(start, min(start + batch_size, len(images))), device)
            predictions.append(model(values).argmax(1).cpu().numpy())
    predicted = np.concatenate(predictions)
    non_target = np.flatnonzero(labels != target)
    triggered: list[np.ndarray] = []
    full = torch.ones(1, 1, 32, 32)
    with torch.no_grad():
        for start in range(0, len(non_target), batch_size):
            values = development.uint8_batch(images, non_target[start:start + batch_size], device)
            values = apply_trigger(values, full.expand(len(values), 1, 32, 32), trigger, alpha)
            triggered.append(model(values).argmax(1).cpu().numpy())
    return {
        "clean_accuracy": float(np.mean(predicted == labels)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "attack_success_rate": float(np.mean(np.concatenate(triggered) == target)),
    }


def train_seed(config: dict[str, Any], seed: int, device: torch.device, train_images: np.ndarray,
               train_labels: np.ndarray, test_images: np.ndarray, test_labels: np.ndarray,
               train_indices: np.ndarray, payload: np.ndarray, cover: np.ndarray, trigger: torch.Tensor,
               output: Path, resume: bool, max_batches: int | None = None) -> tuple[CIFAR10VGG16BN, dict[str, Any]]:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    victim = config["victim"]; attack = config["attack"]
    target = int(attack["target_class"])
    is_payload = np.isin(train_indices, payload, assume_unique=True)
    is_modified = is_payload | np.isin(train_indices, cover, assume_unique=True)
    assigned = train_labels[train_indices].copy(); assigned[is_payload] = target
    masks = mask_table(train_indices[is_modified], seed, config)
    modified_positions = np.flatnonzero(is_modified)
    mask_bank = np.stack([masks[int(train_indices[p])] for p in modified_positions]).astype(np.float32)
    mask_slot = np.full(len(train_indices), -1, dtype=np.int64)
    mask_slot[modified_positions] = np.arange(len(modified_positions))
    alpha = float(attack["train_alpha"])
    model = CIFAR10VGG16BN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(victim["learning_rate"]), weight_decay=float(victim["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(victim["schedule_horizon_epochs"]), eta_min=0.0)
    resume_path = output / "resume" / f"seed_{seed}_latest.pt"
    start_epoch, history = 1, []
    if resume and resume_path.is_file():
        payload_state = torch.load(resume_path, map_location="cpu", weights_only=False)
        if payload_state.get("seed") != seed or payload_state.get("config_sha256") != sha256(CONFIG) \
                or payload_state.get("protocol_sha256") != sha256(PROTOCOL):
            raise RuntimeError(f"resume contract mismatch for seed {seed}")
        model.load_state_dict(payload_state["model"], strict=True)
        optimizer.load_state_dict(payload_state["optimizer"]); scheduler.load_state_dict(payload_state["scheduler"])
        start_epoch = int(payload_state["epoch"]) + 1; history = list(payload_state.get("history", []))
    batch_size = int(victim["batch_size"]); started = time.time()
    for epoch in range(start_epoch, int(victim["training_epochs"]) + 1):
        model.train()
        order = np.random.default_rng(4_242_200 + 1009 * seed + epoch).permutation(len(train_indices))
        augmentation_rng = torch.Generator().manual_seed(4_242_300 + 1009 * seed + epoch)
        loss_sum = 0.0; correct = 0; seen = 0
        for batch_number, begin in enumerate(range(0, len(order), batch_size)):
            if max_batches is not None and batch_number >= max_batches:
                break
            positions = order[begin:begin + batch_size]
            values = development.augment(development.uint8_batch(train_images, train_indices[positions], torch.device("cpu")), augmentation_rng)
            modified = torch.from_numpy(is_modified[positions])
            if bool(modified.any()):
                values[modified] = apply_trigger(values[modified], torch.from_numpy(mask_bank[mask_slot[positions][is_modified[positions]]]), trigger, alpha)
            values = values.to(device); labels = torch.from_numpy(assigned[positions]).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(values); loss = F.cross_entropy(logits, labels)
            loss.backward(); optimizer.step()
            loss_sum += float(loss.detach()) * len(positions)
            correct += int((logits.argmax(1) == labels).sum().detach()); seen += len(positions)
        if max_batches is not None:
            return model, {"smoke_loss": loss_sum / max(seen, 1)}
        scheduler.step()
        row = {"epoch": epoch, "train_loss": loss_sum / seen, "train_accuracy": correct / seen,
               "learning_rate": optimizer.param_groups[0]["lr"], "elapsed_seconds": time.time() - started}
        history.append(row)
        print(f"adapblend seed={seed} epoch={epoch} loss={row['train_loss']:.4f} acc={row['train_accuracy']:.4f}", flush=True)
        resume_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = resume_path.with_name(resume_path.name + ".tmp")
        torch.save({"seed": seed, "epoch": epoch, "config_sha256": sha256(CONFIG), "protocol_sha256": sha256(PROTOCOL),
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "history": history}, temporary)
        os.replace(temporary, resume_path)
    metrics = evaluate(model, test_images, test_labels, device, target, batch_size, trigger, float(attack["test_alpha"]))
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    digest = hashlib.sha256()
    for name, value in state.items():
        array = value.contiguous().numpy(); digest.update(name.encode()); digest.update(str(array.shape).encode()); digest.update(array.tobytes())
    return model, {"history": history, "victim_metrics": metrics, "model_state_sha256": digest.hexdigest()}


def run_formal(output: Path, device: torch.device, *, resume: bool, ack: str | None) -> dict[str, Any]:
    if ack != ACK:
        raise RuntimeError(f"formal execution requires --ack {ACK}")
    config, protocol = load_and_verify()
    if device.type != "mps" or not torch.backends.mps.is_available():
        raise RuntimeError("the frozen protocol requires MPS")
    if output.exists() and any(output.iterdir()) and not resume:
        raise RuntimeError("formal output is nonempty; use --resume for exact continuation")
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("status") == "SCORE_HANDOFF_COMPLETE":
            raise RuntimeError("victim stage is already complete")
        if existing.get("freeze_manifest_sha256") != sha256(PROTOCOL):
            raise RuntimeError("resume output was created under a different protocol")
    trigger = load_trigger(config)
    train_images, train_labels, test_images, test_labels, dataset_receipt = development.resolve_dataset(config)
    train_indices, validation_indices = development.fixed_split(
        train_labels, rows_per_class=int(config["dataset"]["validation_rows_per_class"]),
        seed=int(config["dataset"]["validation_split_seed"]))
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "experiment": config["protocol_id"], "status": "RUNNING_BLINDED", "formal": True,
        "freeze_manifest_sha256": sha256(PROTOCOL), "frozen_hashes": protocol["hashes"]["victim_stage"],
        "dataset": dataset_receipt, "seeds": list(SEEDS), "device": str(device),
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "train_indices_sha256": array_hash(train_indices), "validation_indices_sha256": array_hash(validation_indices),
        "created_unix": time.time(), "command_line": sys.argv, "completed_seeds": [],
    }
    partial = output / "seed_receipts.partial.json"
    receipts: list[dict[str, Any]] = json.loads(partial.read_text(encoding="utf-8")) if (resume and partial.is_file()) else []
    done = {int(row["seed"]) for row in receipts}
    manifest["completed_seeds"] = sorted(done); atomic_json(manifest_path, manifest)
    for seed in SEEDS:
        if seed in done:
            continue
        start = time.time(); print(f"adapblend seed {seed}: training fresh victim", flush=True)
        payload, cover = attack_plan(train_labels, train_indices, config, seed)
        model, training = train_seed(config, seed, device, train_images, train_labels, test_images, test_labels,
                                     train_indices, payload, cover, trigger, output, resume)
        model_path = output / "models" / f"seed_{seed}_suspect.pt"; model_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = model_path.with_name(model_path.name + ".tmp")
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, temporary); os.replace(temporary, model_path)
        metrics = training["victim_metrics"]
        eligible = metrics["balanced_accuracy"] >= float(config["eligibility"]["balanced_accuracy_min"]) \
            and metrics["attack_success_rate"] >= float(config["eligibility"]["attack_success_rate_min"])
        receipts.append({
            "seed": seed, "eligible": bool(eligible), "victim_metrics": metrics,
            "poison_count": int(len(payload)), "poison_indices_sha256": array_hash(payload),
            "cover_count": int(len(cover)), "cover_indices_sha256": array_hash(cover),
            "model_file": str(model_path), "model_sha256": sha256(model_path),
            "model_state_sha256": training["model_state_sha256"], "training_history": training["history"],
            "elapsed_seconds": time.time() - start,
        })
        atomic_json(partial, receipts)
        manifest["completed_seeds"] = [int(r["seed"]) for r in receipts]; atomic_json(manifest_path, manifest)
        print(f"adapblend seed {seed}: BCA={metrics['balanced_accuracy']:.4f} ASR={metrics['attack_success_rate']:.4f} eligible={eligible}", flush=True)
    receipts.sort(key=lambda r: int(r["seed"]))
    atomic_json(output / "seed_receipts.json", receipts)
    manifest.update({"status": "SCORE_HANDOFF_COMPLETE", "completed_seeds": list(SEEDS),
                     "all_victims_eligible": all(r["eligible"] for r in receipts),
                     "seed_receipts_sha256": sha256(output / "seed_receipts.json"), "completed_unix": time.time()})
    atomic_json(manifest_path, manifest)
    print(json.dumps({"status": manifest["status"], "eligible": sum(r["eligible"] for r in receipts)}), flush=True)
    return manifest


def selftest(smoke_cpu: bool = False) -> None:
    config = load_config()
    if tuple(config["formal_seeds"]) != SEEDS:
        raise RuntimeError("config seeds mismatch")
    trigger = load_trigger(config)
    if trigger.shape != (3, 32, 32):
        raise RuntimeError("trigger shape")
    m1, m2 = piece_mask(4426, 17), piece_mask(4426, 17)
    if not np.array_equal(m1, m2) or int(m1.sum()) != 8 * PIECE * PIECE:
        raise RuntimeError("piece mask is not deterministic or has the wrong area")
    if np.array_equal(piece_mask(4426, 17), piece_mask(4426, 18)) and np.array_equal(piece_mask(4426, 17), piece_mask(4426, 19)):
        raise RuntimeError("piece masks do not vary across rows")
    zeros = torch.zeros(2, 3, 32, 32)
    mask = torch.from_numpy(np.stack([m1, m1]))
    out = apply_trigger(zeros.clone(), mask, trigger, 0.15)
    if not torch.allclose(out, 0.15 * mask * trigger.unsqueeze(0), atol=1e-7):
        raise RuntimeError("masked blend selftest failed")
    full = apply_trigger(zeros.clone(), torch.ones(2, 1, 32, 32), trigger, 0.2)
    if not torch.allclose(full, 0.2 * trigger.unsqueeze(0).expand(2, 3, 32, 32), atol=1e-7):
        raise RuntimeError("full-trigger selftest failed")
    train_images, train_labels, test_images, test_labels, _ = development.resolve_dataset(config)
    train_indices, _ = development.fixed_split(train_labels, rows_per_class=int(config["dataset"]["validation_rows_per_class"]),
                                               seed=int(config["dataset"]["validation_split_seed"]))
    if len(train_indices) != 45_000:
        raise RuntimeError("training split size")
    for seed in SEEDS:
        payload, cover = attack_plan(train_labels, train_indices, config, seed)
        if len(payload) != 2250 or len(cover) != 2250 or len(np.intersect1d(payload, cover)):
            raise RuntimeError("payload/cover plan")
        if np.any(train_labels[payload] == 8) or np.any(train_labels[cover] == 8):
            raise RuntimeError("payload or cover drawn from the target class")
        if not (np.isin(payload, train_indices).all() and np.isin(cover, train_indices).all()):
            raise RuntimeError("plan leaves the training split")
    if parameter_count() < 14_000_000:
        raise RuntimeError("architecture")
    if smoke_cpu:
        payload, cover = attack_plan(train_labels, train_indices, config, SEEDS[0])
        _, info = train_seed(config, SEEDS[0], torch.device("cpu"), train_images, train_labels, test_images, test_labels,
                             train_indices, payload, cover, trigger, Path("/tmp/adapblend_smoke"), False, max_batches=2)
        if not np.isfinite(info["smoke_loss"]):
            raise RuntimeError("smoke training loss is not finite")
        print(f"cpu smoke: two batches, loss {info['smoke_loss']:.4f}")
    print("Adap-Blend victim runner selftest PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--smoke-cpu", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ack")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.selftest:
        selftest(smoke_cpu=args.smoke_cpu)
    elif args.formal:
        run_formal(args.output, torch.device("mps"), resume=args.resume, ack=args.ack)
    else:
        parser.error("choose --selftest or --formal")


if __name__ == "__main__":
    main()

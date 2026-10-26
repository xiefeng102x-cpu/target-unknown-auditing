#!/usr/bin/env python3
"""Automated adjudication channel without master labels (post hoc pilot; plan
REFERENCE_CHANNEL_PLAN_V1.md), part 1: train the reference ensemble on the trusted official
GTSRB test split and label every record of the ten SRAC-UI confirmation populations.

The channel function `adjudicate(images, assigned_labels)` sees only images and assigned
labels.  Record identifiers are used to rebuild the images with the frozen victim protocol;
the master poison labels are read only by `score()`, after all channel labels exist.

usage: run_reference_channel_v1.py --output DIR [--threads 4]
"""
from __future__ import annotations
import argparse, hashlib, json, re, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn.functional as F

HERE = Path(__file__).resolve().parent; CERT = HERE.parent; ROOT = CERT.parent
sys.path.insert(0, str(ROOT / "src"))
import run_tifs_native_gtsrb as G  # noqa: E402

PROTOCOL = ROOT / "configs/srac_gtsrb_srac_ui_confirmation_v1.json"
FLAGS = ROOT / "results/gtsrb_srac_ui_scan_scores_confirmation_v1/flags.csv"
CACHE = ROOT / "data/gtsrb_official/gtsrb_32x32_uint8_v1.npz"
SPLIT_SEED, MODEL_SEEDS = 2026102010, (2026102011, 2026102012, 2026102013)
EPOCHS, BATCH, LR, WD = 30, 256, 2e-3, 1e-4


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------ reference ensemble
def augment(x, rng):
    """x: uint8 (n, 32, 32, 3) -> float tensor with random 2-pixel shifts and +-10% brightness."""
    n = len(x); out = np.empty_like(x)
    pad = np.pad(x, ((0, 0), (2, 2), (2, 2), (0, 0)), mode="edge")
    dx = rng.integers(0, 5, n); dy = rng.integers(0, 5, n)
    for i in range(n):
        out[i] = pad[i, dy[i]:dy[i] + 32, dx[i]:dx[i] + 32]
    t = G.to_tensor(out)
    return (t * torch.from_numpy(rng.uniform(0.9, 1.1, (n, 1, 1, 1)).astype(np.float32))).clamp(0, 1)


def train_one(images, labels, seed, log):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = G.GTSRBTinyCNN(); opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    counts = np.bincount(labels, minlength=43).astype(np.float64)
    w = 1.0 / np.sqrt(np.maximum(counts, 1.0)); w /= w.mean(); cw = torch.from_numpy(w.astype(np.float32))
    for epoch in range(1, EPOCHS + 1):
        model.train(); order = rng.permutation(len(images)); tot = 0.0
        for s in range(0, len(order), BATCH):
            idx = order[s:s + BATCH]
            loss = F.cross_entropy(model(augment(images[idx], rng)), torch.from_numpy(labels[idx]).long(), weight=cw)
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(idx)
        log(f"model seed={seed} epoch={epoch}/{EPOCHS} loss={tot / len(order):.4f}")
    model.eval()
    return model


def softmax_mean(models, images):
    out = np.zeros((len(images), 43))
    with torch.no_grad():
        for m in models:
            for s in range(0, len(images), 1024):
                out[s:s + 1024] += F.softmax(m(G.to_tensor(images[s:s + 1024])), 1).numpy()
    return out / len(models)


def adjudicate(models, images, assigned):
    """The channel: poison iff the ensemble's argmax differs from the assigned label."""
    p = softmax_mean(models, images)
    return p.argmax(1) != assigned, p[np.arange(len(assigned)), assigned], p.argmax(1)


# ------------------------------------------------------------------ records
def rebuild(cache, protocol, seed, ids):
    """Images and assigned labels of the given record ids for one victim seed."""
    data = G.build_seed_data(cache["train_images"], cache["train_labels"], protocol, seed)
    poison_pos = {(int(s), int(b)): i for i, (s, b) in enumerate(zip(data.poison_sources, data.poison_base_indices))}
    clean_pos = {int(t): i for i, t in enumerate(data.train_indices)}
    n_clean = len(data.train_indices); pos = np.empty(len(ids), dtype=np.int64)
    for j, rid in enumerate(ids):
        m = re.fullmatch(r"train_clean_(\d+)", rid)
        if m:
            pos[j] = clean_pos[int(m.group(1))]; continue
        m = re.fullmatch(r"train_poison_source(\d+)_base(\d+)", rid)
        pos[j] = n_clean + poison_pos[(int(m.group(1)), int(m.group(2)))]
    return data.train_images[pos], data.train_labels[pos]


def cp(x, n, conf=0.95):
    from scipy.stats import beta
    a = (1 - conf) / 2
    lo = 0.0 if x == 0 else float(beta.ppf(a, x, n - x + 1)); hi = 1.0 if x == n else float(beta.ppf(1 - a, x + 1, n - x))
    return [lo, hi]


def score(lab):
    """Uses the master labels; called only after every channel label exists."""
    rows = []
    for key, g in [("pooled", lab)] + list(lab.groupby("victim_id")):
        clean, poison = g[g.is_poison == 0], g[g.is_poison == 1]
        fp, fn = int(clean.channel_poison.sum()), int((~poison.channel_poison.astype(bool)).sum())
        r = dict(population=key, records=len(g), clean=len(clean), poisons=len(poison),
                 clean_labelled_poison=fp, clean_error=fp / len(clean), clean_error_ci95=cp(fp, len(clean)),
                 poison_labelled_clean=fn, poison_error=fn / len(poison), poison_error_ci95=cp(fn, len(poison)),
                 overall_error=(fp + fn) / len(g), overall_error_ci95=cp(fp + fn, len(g)))
        for f in (1, 0):
            s = g[g.flag == f]; w = int((s.channel_poison.astype(bool) != s.is_poison.astype(bool)).sum())
            r[f"stratum{f}_records"] = len(s); r[f"stratum{f}_wrong"] = w
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--output", type=Path, required=True); ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.mkdir(parents=True); torch.set_num_threads(a.threads)
    logf = open(a.output / "run.log", "w")
    log = lambda m: (logf.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {m}\n"), logf.flush())
    protocol = json.loads(PROTOCOL.read_text()); cache = dict(np.load(CACHE))
    ti, tl = cache["test_images"], cache["test_labels"]
    rng = np.random.default_rng(SPLIT_SEED); tr, ho = [], []
    for c in range(43):
        idx = rng.permutation(np.flatnonzero(tl == c)); k = int(round(0.1 * len(idx))); ho += list(idx[:k]); tr += list(idx[k:])
    tr, ho = np.array(sorted(tr)), np.array(sorted(ho))
    models = [train_one(ti[tr], tl[tr], s, log) for s in MODEL_SEEDS]
    ho_pred = softmax_mean(models, ti[ho]).argmax(1); ho_err = int((ho_pred != tl[ho]).sum())
    log(f"held-out clean error {ho_err}/{len(ho)}")
    torch.save([m.state_dict() for m in models], a.output / "reference_ensemble.pt")
    flags = pd.read_csv(FLAGS); flags = flags[flags.victim_eligible.eq(1)]
    out = []
    for v, g in flags.groupby("victim_id", sort=True):
        seed = int(g.seed.iloc[0]); imgs, assigned = rebuild(cache, protocol, seed, g.sample_id.tolist())
        if not np.array_equal(assigned, g.assigned_group.to_numpy()):
            raise RuntimeError(f"assigned labels do not match the flags file for {v}")
        ch, p_assigned, pred = adjudicate(models, imgs, assigned)
        out.append(pd.DataFrame(dict(victim_id=v, sample_id=g.sample_id.to_numpy(), assigned_label=assigned, ref_pred=pred,
                                     ref_prob_assigned=np.round(p_assigned, 6), channel_poison=ch.astype(int))))
        log(f"labelled {v}: {int(ch.sum())} records labelled poison")
    lab = pd.concat(out, ignore_index=True)
    lab.to_csv(a.output / "channel_labels.csv", index=False)       # written before scoring
    lab = lab.merge(flags[["victim_id", "sample_id", "flag", "is_poison"]], on=["victim_id", "sample_id"], how="left")
    rep = dict(status="POST_HOC_PILOT", trusted_train=int(len(tr)), trusted_heldout=int(len(ho)),
               heldout_clean_errors=ho_err, heldout_clean_error=ho_err / len(ho), heldout_clean_error_ci95=cp(ho_err, len(ho)),
               scoring=score(lab), script_sha256=sha(__file__), cache_sha256=sha(CACHE), flags_sha256=sha(FLAGS),
               protocol_sha256=sha(PROTOCOL), channel_labels_sha256=sha(a.output / "channel_labels.csv"))
    (a.output / "report.json").write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps({k: rep[k] for k in ("heldout_clean_error", "heldout_clean_error_ci95")}), json.dumps(rep["scoring"][0], indent=1))


if __name__ == "__main__":
    main()

"""Post-hoc sensitivity of the frozen group-safe certificate to imperfect adjudication.

Developmental evidence only.  Frozen flags, frozen protected families, frozen design
(two-sided exact intervals, 50/50 flag-stratum allocation, 200 draws, budgets
2000/8000/16000, delta=0.005).  Sampled labels are flipped independently with
probability eps.  The robust certificate splits delta equally between the count
intervals and the adjudication-error event, takes B as the exact Binomial(k, eps)
upper (1 - delta/2) quantile, and widens every cell interval to the hull
[L(x-B), U(x+B)] (exact because both endpoints are nondecreasing in x).  The naive
certificate ignores errors (B=0, full delta).  eps=0 reproduces the frozen endpoint.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[2]; CERT = ROOT / "auditability_certificate_v1"
spec = importlib.util.spec_from_file_location("ph", CERT / "analysis/run_posthoc_power_study_v1.py")
H = importlib.util.module_from_spec(spec); sys.modules["ph"] = H; spec.loader.exec_module(H)
DATASETS = {
    "gtsrb_semantic_v2": dict(protocol=CERT / "PROTOCOL_GTSRB_SEMANTIC_OVERLAP_CONFIRMATION_V2.json",
        flags=ROOT / "results/gtsrb_semantic_overlap_scan_scores_confirmation_v2/flags.csv",
        frozen=CERT / "results_gtsrb_semantic_overlap_confirmation_v2/decisions.csv", ns="gtsrb-semantic-overlap-confirm-v1|{v}"),
    "cifar10_scan_lda_v1": dict(protocol=CERT / "PROTOCOL_CIFAR10_SCAN_LDA_CONFIRMATION_V1.json",
        flags=ROOT / "results/cifar10_scan_lda_scores_confirmation_v1/flags.csv",
        frozen=CERT / "results_cifar10_scan_lda_confirmation_v1/decisions.csv", ns="cifar10-scan-lda-confirm-v1|{v}"),
}
EPS = (0.0, 0.001, 0.005, 0.01, 0.02)


def decide(pop, chosen, obs, eta, B, tau, cap):
    """obs: observed (possibly corrupted) poison labels for all records (only sampled ones are read)."""
    M = pop["memberships"]
    def iv(n_pop, idx):
        x = int(obs[idx].sum()); n = len(idx)
        lo = H.interval(int(n_pop), n, max(0, x - B), float(eta)).lower
        hi = H.interval(int(n_pop), n, min(n, x + B), float(eta)).upper
        return lo, hi
    c_lo, c_hi = iv(pop["N"][1], chosen[1]); m_lo, m_hi = iv(pop["N"][0], chosen[0])
    if c_lo + m_lo == 0:
        rec_lo, rec_hi = 0.0, 1.0
    else:
        rec_lo = H.ratio(c_lo, m_hi, 0.0); rec_hi = H.ratio(c_hi, m_lo, 1.0)
    go_g, ng_g = True, False
    for g in range(M.shape[1]):
        fs, us = pop["gsize"][g]
        if fs == 0:
            continue
        f_lo, f_hi = iv(fs, chosen[1][M[chosen[1], g]])
        u_lo, u_hi = iv(us, chosen[0][M[chosen[0], g]]) if us else (0, 0)
        hi = H.ratio(fs - f_lo, us - u_hi, 1.0); lo = H.ratio(fs - f_hi, us - u_lo, 0.0)
        go_g &= hi <= cap; ng_g |= lo > cap
    if rec_lo >= tau and go_g:
        return "GO"
    if rec_hi < tau or ng_g:
        return "NO_GO"
    return "ABSTAIN"


def run_victim(args):
    ds, victim, frame, cfg = args
    proto = json.loads(Path(cfg["protocol"]).read_text()); c = proto["certificate"]
    delta, tau, cap = c["delta_per_decision"], c["recall_target"], c["max_group_clean_flag_rate_cap"]
    pop = H.build_population(frame, proto["protected_groups"]); q = pop["q"]
    true_pass = pop["recall"] >= tau and pop["max_clean"] <= cap
    ns = cfg["ns"].format(v=victim); poison = pop["poison"]; rows = []
    for k in c["budgets"]:
        n1, n0 = H.allocate(pop["N"][1], pop["N"][0], k, "0.5")
        for draw in range(c["protected_draws_per_seed_budget"]):
            chosen = {}
            for f, n in ((1, n1), (0, n0)):
                rng = np.random.default_rng(H.stable_seed(ns, k, draw, f))
                chosen[f] = rng.choice(pop["idx"][f], size=n, replace=False)
            sampled = np.concatenate([chosen[1], chosen[0]])
            for eps in EPS:
                obs = poison.copy()
                if eps > 0:
                    flip = np.random.default_rng(H.stable_seed("adj-error-v1", ns, k, draw, eps)).random(len(sampled)) < eps
                    obs[sampled[flip]] = ~obs[sampled[flip]]
                    nerr = int(flip.sum())
                else:
                    nerr = 0
                variants = [("naive", 0, delta)] if eps == 0 else [("naive", 0, delta), ("robust", int(binom.ppf(1 - delta / 2, k, eps)), delta / 2)]
                for name, B, dcert in variants:
                    d = decide(pop, chosen, obs, dcert / (2 * q), B, tau, cap)
                    rows.append(dict(dataset=ds, victim_id=victim, k=k, draw=draw, eps=eps, variant=name, B=B, errors=nerr,
                                     decision=d, true_pass=int(true_pass)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--output", type=Path, required=True); ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    jobs = []
    for ds, cfg in DATASETS.items():
        f = pd.read_csv(cfg["flags"]); f = f[f.victim_eligible.eq(1)]
        for v, fr in f.groupby("victim_id", sort=True):
            jobs.append((ds, v, fr.reset_index(drop=True)[["victim_id","seed","sample_id","assigned_group","flag","is_poison","victim_eligible"]], {k: str(x) for k, x in cfg.items()}))
    with Pool(a.procs) as p:
        out = pd.concat(p.map(run_victim, jobs), ignore_index=True)
    out["false_go"] = ((out.decision == "GO") & (out.true_pass == 0)).astype(int)
    out["false_nogo"] = ((out.decision == "NO_GO") & (out.true_pass == 1)).astype(int)
    val = {}
    for ds, cfg in DATASETS.items():
        fz = pd.read_csv(cfg["frozen"])[["victim_id", "k", "draw", "decision"]]
        mine = out[(out.dataset == ds) & (out.eps == 0)][["victim_id", "k", "draw", "decision"]]
        m = fz.merge(mine, on=["victim_id", "k", "draw"], suffixes=("_f", "_m"))
        val[ds] = dict(rows=len(fz), matched=len(m), identical=bool(len(m) == len(fz) and (m.decision_f == m.decision_m).all()))
    a.output.mkdir(parents=True); out.to_csv(a.output / "decisions.csv", index=False)
    s = out.groupby(["dataset", "eps", "variant", "k", "true_pass"]).agg(n=("decision", "size"), B=("B", "max"),
        go=("decision", lambda x: (x == "GO").mean()), nogo=("decision", lambda x: (x == "NO_GO").mean()),
        false_go=("false_go", "sum"), false_nogo=("false_nogo", "sum"), mean_errors=("errors", "mean")).reset_index()
    s.to_csv(a.output / "summary.csv", index=False)
    h = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    rep = dict(validation=val, rows=len(out), script_sha256=h(__file__), decisions_sha256=h(a.output / "decisions.csv"), summary_sha256=h(a.output / "summary.csv"))
    (a.output / "report.json").write_text(json.dumps(rep, indent=2)); print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()

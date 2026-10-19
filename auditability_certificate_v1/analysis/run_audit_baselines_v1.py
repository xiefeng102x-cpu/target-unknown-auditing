"""Post-hoc same-budget comparison of audit designs for the group-safe object (developmental).

Every design certifies the same object as SRAC, on the same frozen flags, families,
budgets k in {2000, 8000, 16000}, 200 draws, tau = 0.80, cap = 0.10, and delta = 0.005
per decision split over both decision directions:

  srac       frozen SRAC: flag-stratified 50/50 sample, exact hypergeometric intervals,
             eta = delta/(2Q).  Must reproduce every frozen decision (validation gate).
  uni_exact  simple random sample of k from the whole universe (no stratification), exact
             hypergeometric intervals for the same joint counts, intersected with the
             known flag-stratum sizes, eta = delta/(2Q)  (TAR-style uniform audit).
  rla_bet    comparison risk-limiting audit (SHANGRLA-style assertions with the flags as
             reported outcomes): uniform sample without replacement in random order;
             recall and every group cap become means of non-negative discrepancies whose
             reported part is known exactly; each assertion is tested sequentially with a
             without-replacement betting test martingale (Waudby-Smith and Ramdas) using
             predictable aGRAPA bets; level delta/(2A) per assertion and direction (A =
             number of assertions); anytime-valid, so the audit stops at the first
             decision and records the labels used.
  ppi_clt    prediction-powered / difference estimator with the flags as predictions on a
             uniform sample, finite-population-corrected CLT intervals with a Bonferroni
             normal quantile at delta/(2A).  Asymptotic: no finite-sample guarantee.

Blocks: 8, 11, 12, 13, 14 (all frozen group-safe blocks with semantic families).
Reported: GO/NO-GO/ABSTAIN rates, false GO and false NO-GO counts, mean labels used.
Nothing becomes a frozen endpoint.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]; CERT = ROOT / "auditability_certificate_v1"
spec = importlib.util.spec_from_file_location("ph", CERT / "analysis/run_posthoc_power_study_v1.py")
H = importlib.util.module_from_spec(spec); sys.modules["ph"] = H; spec.loader.exec_module(H)
BLOCKS = {
    "8": dict(protocol=CERT / "PROTOCOL_GTSRB_SEMANTIC_OVERLAP_CONFIRMATION_V2.json",
              flags=ROOT / "results/gtsrb_semantic_overlap_scan_scores_confirmation_v2/flags.csv",
              frozen=CERT / "results_gtsrb_semantic_overlap_confirmation_v2/decisions.csv", ns="gtsrb-semantic-overlap-confirm-v1|{v}"),
    "11": dict(protocol=CERT / "PROTOCOL_CIFAR10_SEMANTIC_OVERLAP_CONFIRMATION_V1.json",
               flags=ROOT / "results/cifar10_semantic_overlap_scan_scores_confirmation_v1/flags.csv",
               frozen=CERT / "results_cifar10_semantic_overlap_confirmation_v1/decisions.csv", ns="cifar10-semantic-overlap-confirm-v1|{v}"),
    "12": dict(protocol=CERT / "PROTOCOL_CIFAR10_SCAN_LDA_CONFIRMATION_V1.json",
               flags=ROOT / "results/cifar10_scan_lda_scores_confirmation_v1/flags.csv",
               frozen=CERT / "results_cifar10_scan_lda_confirmation_v1/decisions.csv", ns="cifar10-scan-lda-confirm-v1|{v}"),
    "13": dict(protocol=CERT / "PROTOCOL_CIFAR10_ADAPBLEND_SCAN_LDA_CONFIRMATION_V1.json",
               flags=ROOT / "results/cifar10_adapblend_scan_lda_scores_confirmation_v1/flags.csv",
               frozen=CERT / "results_cifar10_adapblend_scan_lda_confirmation_v1/decisions.csv", ns="cifar10-adapblend-scan-lda-confirm-v1|{v}"),
    "14": dict(protocol=CERT / "PROTOCOL_CIFAR10_ADAPBLEND_AGPD_V1.json",
               flags=ROOT / "results/cifar10_adapblend_agpd_scores_v1/flags.csv",
               frozen=CERT / "results_cifar10_adapblend_agpd_v1/decisions.csv", ns="cifar10-adapblend-agpd-v1|{v}"),
}
DESIGNS = ("srac", "uni_exact", "rla_bet", "ppi_clt")


# ----------------------------------------------------------------- exact uniform audit
def uni_exact(pop, sample, eta, tau, cap):
    N = len(pop["poison"]); k = len(sample); M = pop["memberships"]; Z = pop["poison"]; A = pop["flags"].astype(bool)
    def iv(mask, lo_cap, hi_cap):
        x = int(mask[sample].sum()); r = H.interval(N, k, x, float(eta))
        return max(r.lower, lo_cap), min(r.upper, hi_cap)
    c_lo, c_hi = iv(Z & A, 0, pop["N"][1]); m_lo, m_hi = iv(Z & ~A, 0, pop["N"][0])
    if c_lo + m_lo == 0:
        rec_lo, rec_hi = 0.0, 1.0
    else:
        rec_lo = H.ratio(c_lo, m_hi, 0.0); rec_hi = H.ratio(c_hi, m_lo, 1.0)
    go, ng = True, False
    for g in range(M.shape[1]):
        fs, us = pop["gsize"][g]
        if fs == 0:
            continue
        f_lo, f_hi = iv(Z & A & M[:, g], 0, fs)
        u_lo, u_hi = iv(Z & ~A & M[:, g], 0, us) if us else (0, 0)
        hi = H.ratio(fs - f_lo, us - u_hi, 1.0); lo = H.ratio(fs - f_hi, us - u_lo, 0.0)
        go &= hi <= cap; ng |= lo > cap
    return ("GO" if rec_lo >= tau and go else "NO_GO" if rec_hi < tau or ng else "ABSTAIN"), k


# ----------------------------------------------------------------- comparison RLA (betting)
def assertions(pop, tau, cap):
    """Discrepancy vectors d (>= 0, scaled to [0,1]) and thresholds m (scaled) such that
    the GO claim is mean(d) < m and the NO-GO claim is mean(d) > m."""
    N = len(pop["poison"]); Z = pop["poison"]; A = pop["flags"].astype(bool); M = pop["memberships"]
    out = []
    # recall >= tau  <=>  sum Z(A - tau) >= 0; reported value A(1 - tau); discrepancy in {0, tau, 1 - tau}
    b = max(tau, 1 - tau)
    d = np.where(A & ~Z, 1 - tau, 0.0) + np.where(~A & Z, tau, 0.0)
    out.append(("recall", d / b, (1 - tau) * A.sum() / N / b))
    for g in range(M.shape[1]):
        fs, us = pop["gsize"][g]
        if fs == 0:
            continue
        inH = M[:, g]; b = max(cap, 1 - cap)
        # F_H <= cap  <=>  sum_H (1-Z)(A - cap) <= 0; reported value -cap(1-A); discrepancy in {0, cap, 1 - cap}
        e = np.where(inH & A & ~Z, 1 - cap, 0.0) + np.where(inH & ~A & Z, cap, 0.0)
        out.append((f"g{g}", e / b, cap * (inH & ~A).sum() / N / b))
    return out


def first_rejection(x, N, m, alpha, side):
    """First 1-based time at which the WoR betting test rejects; inf if never within len(x).
    side='below': H0 mean >= m (reject => mean < m).  side='above': H0 mean <= m."""
    t = np.arange(1, len(x) + 1, dtype=float)
    S_prev = np.concatenate([[0.0], np.cumsum(x)[:-1]])
    m_t = (N * m - S_prev) / (N - t + 1)
    mu = (S_prev + 0.5 * m) / t
    ss = np.concatenate([[0.0], np.cumsum(x ** 2)[:-1]])
    var = np.maximum((ss - 2 * mu * S_prev + (t - 1) * mu ** 2 + 0.25 * m) / t, 1e-8)
    if side == "below":
        diff = m_t - mu
        lam = np.clip(diff / (var + diff ** 2), 0, None)
        lam = np.minimum(lam, 0.75 / np.maximum(1 - m_t, 1e-12))
        inc = 1 + lam * (m_t - x)
        impossible = m_t < 0          # observed total already exceeds N*m: H0 certainly true
        inc = np.where(impossible, 0.0, inc)
    else:
        diff = mu - m_t
        lam = np.clip(diff / (var + diff ** 2), 0, None)
        lam = np.minimum(lam, 0.75 / np.maximum(m_t, 1e-12))
        inc = 1 + lam * (x - m_t)
        certain = m_t < 0             # observed total exceeds N*m: H0 certainly false
        inc = np.where(certain, np.inf, inc)
    if side == "below":
        inc = np.where(m_t > 1, np.inf, inc)   # remaining mean would have to exceed the bound 1: H0 false
    with np.errstate(divide="ignore"):
        logk = np.cumsum(np.log(np.maximum(inc, 0.0)))
    hit = np.flatnonzero(logk >= np.log(1 / alpha))
    return float(hit[0] + 1) if hit.size else np.inf


def rla_bet(asserts, sample, N, delta):
    alpha = delta / (2 * len(asserts)); k = len(sample)
    go_t = max(first_rejection(d[sample], N, m, alpha, "below") for _, d, m in asserts)
    ng_t = min(first_rejection(d[sample], N, m, alpha, "above") for _, d, m in asserts)
    if ng_t <= k and ng_t < go_t:
        return "NO_GO", int(ng_t)
    if go_t <= k:
        return "GO", int(go_t)
    return "ABSTAIN", k


# ----------------------------------------------------------------- PPI / difference estimator
def ppi_clt(pop, sample, delta, tau, cap):
    N = len(pop["poison"]); k = len(sample); Z = pop["poison"][sample].astype(float); A = pop["flags"][sample].astype(float)
    M = pop["memberships"]; fpc = 1 - k / N; G = [g for g in range(M.shape[1]) if pop["gsize"][g][0] > 0]
    z = norm.ppf(1 - delta / (2 * (1 + len(G))))
    def ratio_ci(t1_known, y1, t2_known, y2):
        T1 = t1_known + N * y1.mean(); T2 = t2_known + N * y2.mean()
        if T2 <= 0:
            return 0.0, 1.0
        R = T1 / T2; g = (y1 - R * y2) / T2
        se = N * np.sqrt(fpc * g.var(ddof=1) / k) if k > 1 else np.inf
        return R - z * se, R + z * se
    N1 = float(pop["flags"].sum())
    rl, rh = ratio_ci(N1, -A * (1 - Z), N1, Z - A)
    go, ng = rl >= tau, rh < tau
    for g in G:
        inH = M[sample, g].astype(float); N0H = float(((pop["flags"] == 0) & M[:, g]).sum())
        fl, fh = ratio_ci(0.0, inH * (1 - Z) * A, N0H, inH * (A - Z))
        go &= fh <= cap; ng |= fl > cap
    return ("GO" if go else "NO_GO" if ng else "ABSTAIN"), k


def run_victim(args):
    block, victim, frame, cfg = args
    proto = json.loads(Path(cfg["protocol"]).read_text()); c = proto["certificate"]
    delta, tau, cap = c["delta_per_decision"], c["recall_target"], c["max_group_clean_flag_rate_cap"]
    pop = H.build_population(frame, proto["protected_groups"]); q = pop["q"]; eta = delta / (2 * q)
    N = len(pop["poison"]); asserts = assertions(pop, tau, cap)
    true_pass = int(pop["recall"] >= tau and pop["max_clean"] <= cap)
    ns = cfg["ns"].format(v=victim); rows = []
    for k in c["budgets"]:
        n1, n0 = H.allocate(pop["N"][1], pop["N"][0], k, "0.5")
        for draw in range(c["protected_draws_per_seed_budget"]):
            chosen = {}
            for f, n in ((1, n1), (0, n0)):
                rng = np.random.default_rng(H.stable_seed(ns, k, draw, f))
                chosen[f] = rng.choice(pop["idx"][f], size=n, replace=False)
            d_srac, *_ = H.decide(pop, chosen, dict(R=eta, G=eta), tau, cap)
            uni = np.random.default_rng(H.stable_seed("audit-baselines-v1", ns, k, draw)).choice(N, size=k, replace=False)
            d_u, _ = uni_exact(pop, uni, eta, tau, cap)
            d_r, used = rla_bet(asserts, uni, N, delta)
            d_p, _ = ppi_clt(pop, uni, delta, tau, cap)
            for design, dec, lab in (("srac", d_srac, k), ("uni_exact", d_u, k), ("rla_bet", d_r, used), ("ppi_clt", d_p, k)):
                rows.append(dict(block=block, victim_id=victim, k=k, draw=draw, design=design, decision=dec, labels=lab,
                                 true_pass=true_pass, recall=pop["recall"], max_clean=pop["max_clean"], Q=q, A=len(asserts)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--output", type=Path, required=True); ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    jobs = []
    for block, cfg in BLOCKS.items():
        f = pd.read_csv(cfg["flags"], usecols=["victim_id", "seed", "sample_id", "assigned_group", "flag", "is_poison", "victim_eligible"])
        f = f[f.victim_eligible.eq(1)]
        for v, fr in f.groupby("victim_id", sort=True):
            jobs.append((block, v, fr.reset_index(drop=True), {k: str(x) for k, x in cfg.items()}))
    with Pool(a.procs) as p:
        out = pd.concat(p.imap_unordered(run_victim, jobs), ignore_index=True)
    out = out.sort_values(["block", "victim_id", "k", "draw", "design"]).reset_index(drop=True)
    out["false_go"] = ((out.decision == "GO") & (out.true_pass == 0)).astype(int)
    out["false_nogo"] = ((out.decision == "NO_GO") & (out.true_pass == 1)).astype(int)
    val = {}
    for block, cfg in BLOCKS.items():
        fz = pd.read_csv(cfg["frozen"])[["victim_id", "k", "draw", "decision"]]
        mine = out[(out.block == block) & (out.design == "srac")][["victim_id", "k", "draw", "decision"]]
        m = fz.merge(mine, on=["victim_id", "k", "draw"], suffixes=("_f", "_m"))
        val[block] = dict(frozen_rows=len(fz), matched=len(m), identical=bool(len(m) == len(fz) and (m.decision_f == m.decision_m).all()))
    a.output.mkdir(parents=True)
    out.to_csv(a.output / "decisions.csv", index=False)
    s = out.groupby(["design", "block", "true_pass", "k"]).agg(
        n=("decision", "size"), go=("decision", lambda x: float((x == "GO").mean())),
        nogo=("decision", lambda x: float((x == "NO_GO").mean())), false_go=("false_go", "sum"),
        false_nogo=("false_nogo", "sum"), mean_labels=("labels", "mean")).reset_index()
    s.to_csv(a.output / "summary.csv", index=False)
    h = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    tot = out.groupby("design")[["false_go", "false_nogo"]].sum().to_dict("index")
    rep = dict(validation=val, rows=len(out), totals=tot, script_sha256=h(__file__),
               decisions_sha256=h(a.output / "decisions.csv"), summary_sha256=h(a.output / "summary.csv"))
    (a.output / "report.json").write_text(json.dumps(rep, indent=2)); print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()

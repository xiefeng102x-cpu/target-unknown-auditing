"""Post-hoc development of a flag-stratified union-intersection audit (SRAC-UI), V1.

Developmental: designed after the equal-budget comparison (results_audit_baselines_v1)
showed that a uniform betting comparison audit beats the frozen SRAC engine at k = 2000.
Same object, policy, delta, frozen flags, families, budgets, and the frozen SRAC
flag-stratified draws (50/50 allocation, same draw namespace).

Each assertion is linear in two hidden stratum counts:
  recall >= tau        <=>  (1 - tau) c - tau m >= 0        c: flagged poisons, m: missed poisons
  F_H <= cap           <=>  (1 - cap) CF - cap CU <= 0       CF/CU: clean flagged/unflagged in H
The two counts are observed through independent without-replacement samples from the two
flag strata (for a group: conditionally on the number of sampled members of H in each
stratum).  The test statistic is the unbiased estimate of the linear form,
  T = (1 - tau) N1 x1 / n1 - tau N0 x0 / n0     (recall; analogous for each group),
whose exact null law is a convolution of two hypergeometric laws.  The p-value is the
supremum of P(T as or more extreme | counts) over all null counts consistent with the
sample (union-intersection), restricted to an exact 1-gamma confidence set for the
nuisance count (Berger-Boos, gamma = 0.05 alpha added to the p-value); monotonicity of the
law in each count gives valid block bounds, so the supremum is decided exactly by branch
and bound.  Each assertion and
direction gets alpha = delta / (2 A), A = number of assertions (recall + every group with
a flagged record), so false GO <= delta/2 and false NO-GO <= delta/2 per decision.

Designs reported:
  srac          frozen endpoint (validation: reproduce every frozen decision)
  srac_ui       fixed budget; flagged-stratum shares 0.5 (the frozen SRAC draws), 0.25, 0.1
                (fresh draws from namespace "srac-ui-v1"); every share is reported
  srac_ui_esc   escalation 2000 -> 8000 -> 16000, nested flag-stratified samples with share
                0.25 (the share of the earlier pre-declared escalations), alpha/3 per look,
                stop at the first non-ABSTAIN
Comparisons with rla_bet, uni_exact, and ppi_clt use the stored results_audit_baselines_v1.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
from scipy.stats import hypergeom
from scipy.special import gammaln

ROOT = Path(__file__).resolve().parents[2]; CERT = ROOT / "auditability_certificate_v1"
spec = importlib.util.spec_from_file_location("bl", CERT / "analysis/run_audit_baselines_v1.py")
BL = importlib.util.module_from_spec(spec); sys.modules["bl"] = BL; spec.loader.exec_module(BL)
H = BL.H; BLOCKS = BL.BLOCKS
BUDGETS = (2000, 8000, 16000)


def hg_pmf(x, N, K, n):
    """Hypergeometric pmf by log-gamma differences (vectorized, broadcasting; zero outside support).
    Agrees with scipy.stats.hypergeom.pmf to ~1e-13 and is two orders of magnitude faster."""
    x = np.asarray(x, dtype=np.float64); K = np.asarray(K, dtype=np.float64)
    ok = (x >= 0) & (x <= K) & (n - x >= 0) & (n - x <= N - K)
    xs = np.where(ok, x, 0.0); Ks = np.where(ok, K, n)
    lg = (gammaln(Ks + 1) - gammaln(xs + 1) - gammaln(Ks - xs + 1) + gammaln(N - Ks + 1) - gammaln(n - xs + 1)
          - gammaln(N - Ks - n + xs + 1) - (gammaln(N + 1) - gammaln(n + 1) - gammaln(N - n + 1)))
    return np.where(ok, np.exp(lg), 0.0)


def linear_p(Na, na, Ka, wa, Nb, nb, Kb, wb, t, side):
    """Vectorized over parameter pairs (Ka[i], Kb[i]):
    P(wa*Na*Xa/na - wb*Nb*Xb/nb >= t)  (side 'ge')  or  <= t  (side 'le'),
    Xa ~ HG(Na, Ka, na), Xb ~ HG(Nb, Kb, nb) independent; a stratum with n = 0 contributes 0.
    Exact: full pmfs, tails by cumulative sums."""
    Ka = np.atleast_1d(np.asarray(Ka, dtype=np.int64)); Kb = np.atleast_1d(np.asarray(Kb, dtype=np.int64))
    P = max(len(Ka), len(Kb)); Ka = np.broadcast_to(Ka, (P,)); Kb = np.broadcast_to(Kb, (P,))
    if nb > 0:
        lo = max(0, nb - (Nb - int(Kb.min()))); hi = min(nb, int(Kb.max()))
        xb = np.arange(lo, hi + 1)
        pb = hg_pmf(xb[None, :], Nb, Kb[:, None], nb)
        vb = wb * Nb * xb / nb
    else:
        pb = np.ones((P, 1)); vb = np.array([0.0])
    if na > 0:
        a = wa * Na / na
        pa = hg_pmf(np.arange(na + 1)[None, :], Na, Ka[:, None], na)          # (P, na+1)
        if side == "ge":
            tail = np.concatenate([np.cumsum(pa[:, ::-1], axis=1)[:, ::-1], np.zeros((P, 1))], axis=1)  # P(Xa >= x)
            thr = np.ceil((t + vb) / a - 1e-9).astype(np.int64)
            q = np.where(thr[None, :] <= 0, 1.0, tail[:, np.clip(thr, 0, na + 1)])
        else:
            cdf = np.cumsum(pa, axis=1)                                          # P(Xa <= x)
            thr = np.floor((t + vb) / a + 1e-9).astype(np.int64)
            q = np.where(thr[None, :] < 0, 0.0, np.where(thr[None, :] >= na, 1.0, cdf[:, np.clip(thr, 0, na)]))
    else:
        q = np.broadcast_to((((-vb) >= t) if side == "ge" else ((-vb) <= t)).astype(float)[None, :], pb.shape)
    return np.minimum((pb * q).sum(axis=1), 1.0)


GAMMA_SHARE = 0.05   # Berger-Boos: 5% of each test's alpha restricts the nuisance count


def sup_at_most(lo, hi, bound, thr, fan=24):
    """True iff max over integers u in [lo, hi] of p(u) <= thr, where bound(a, b) returns, for
    arrays of sub-ranges [a, b], valid upper bounds of max p over each sub-range that are exact
    when a == b (branch and bound over monotone bounds)."""
    if lo > hi:
        return True
    blocks = [(lo, hi)]
    while blocks:
        A, Bb = [], []
        for a, b in blocks:
            edges = np.unique(np.linspace(a, b + 1, min(fan, b - a + 1) + 1).astype(np.int64))
            A.extend(edges[:-1]); Bb.extend(edges[1:] - 1)
        A = np.asarray(A); Bb = np.asarray(Bb)
        val = bound(A, Bb)
        bad = ~(val <= thr)                                   # NaN counts as a violation
        if np.any(bad & (A == Bb)):
            return False
        blocks = list(zip(A[bad], Bb[bad]))
    return True


def recall_tests(N1, n1, x1, N0, n0, x0, tau, alpha, gamma=None):
    """UI tests of the linear assertion (1-tau) c - tau m >= 0 with the statistic
    T = (1-tau) N1 x1/n1 - tau N0 x0/n0 over all null counts consistent with the sample.
    GO (H0: recall < tau): p(c) = P(T >= t | c, m_go(c)) is increasing in c and decreasing in
    m, and m_go is increasing, so p over [a, b] is at most P(T >= t | b, m_go(a)).
    NO-GO (H0: recall >= tau): p(c) = P(T <= t | c, m_ng(c)) is at most P(T <= t | a, m_ng(b))."""
    if x1 == 0 and x0 == 0:
        return False, False                                   # the empty population is in both nulls
    t = (1 - tau) * N1 * x1 / n1 - tau * N0 * x0 / n0
    c_lo, c_hi = x1, N1 - (n1 - x1); m_max = N0 - (n0 - x0)
    gamma = GAMMA_SHARE * alpha if gamma is None else gamma
    if n1 > 0:                                                 # Berger-Boos restriction to a 1-gamma set for c
        r = H.interval(int(N1), int(n1), int(x1), float(gamma / 2)); c_lo, c_hi = max(c_lo, r.lower), min(c_hi, r.upper)
    thr = alpha - gamma
    k = (1 - tau) / tau
    m_go = lambda c: np.maximum(np.floor(k * c).astype(np.int64) + 1, x0)
    m_ng = lambda c: np.minimum(np.floor(k * c).astype(np.int64), m_max)
    feas_hi = min(c_hi, int(np.ceil(m_max / k)) - 1)          # GO-null points need m_go(c) <= m_max
    while feas_hi >= c_lo and m_go(np.array([feas_hi]))[0] > m_max:
        feas_hi -= 1
    go = sup_at_most(c_lo, min(c_hi, feas_hi), lambda a, b: linear_p(N1, n1, b, 1 - tau, N0, n0, m_go(a), tau, t, "ge"), thr)
    ng_lo = max(c_lo, 1, int(np.ceil(x0 / k)) if k > 0 else c_lo)
    while ng_lo <= c_hi and m_ng(np.array([ng_lo]))[0] < x0:
        ng_lo += 1
    ng = sup_at_most(ng_lo, c_hi, lambda a, b: linear_p(N1, n1, a, 1 - tau, N0, n0, m_ng(b), tau, t, "le"), thr)
    return go, ng


def group_tests(fs, n1, y1, us, n0, y0, cap, alpha, gamma=None):
    """UI tests of (1-cap) CF - cap CU <= 0 with T = (1-cap) fs y1/n1 - cap us y0/n0 over all
    null counts consistent with the sample (y1: clean among the n1 sampled flagged members
    of H; y0: clean among the n0 sampled unflagged members).
    GO (H0: F > cap): p(CU) = P(T <= t | CF_go(CU), CU), at most P(T <= t | CF_go(a), b).
    NO-GO (H0: F <= cap): p(CU) = P(T >= t | CF_ng(CU), CU), at most P(T >= t | CF_ng(b), a)."""
    t = (1 - cap) * fs * (y1 / n1 if n1 else 0.0) - cap * us * (y0 / n0 if n0 else 0.0)
    cu_lo, cu_hi = (y0, us - (n0 - y0)) if n0 > 0 else (0, us)
    gamma = GAMMA_SHARE * alpha if gamma is None else gamma
    if n0 > 0 and us > 0:                                      # Berger-Boos restriction to a 1-gamma set for CU
        r = H.interval(int(us), int(n0), int(y0), float(gamma / 2)); cu_lo, cu_hi = max(cu_lo, r.lower), min(cu_hi, r.upper)
    thr = alpha - gamma
    cf_min, cf_max = (y1, fs - (n1 - y1)) if n1 > 0 else (0, fs)
    k = cap / (1 - cap)
    cf_go = lambda cu: np.maximum(np.floor(k * cu).astype(np.int64) + 1, cf_min)
    cf_ng = lambda cu: np.minimum(np.floor(k * cu).astype(np.int64), cf_max)
    # GO-null points need cf_go(CU) <= cf_max  <=>  CU < (cf_max) / k  (and >= cf_min handled by max)
    go_hi = min(cu_hi, int(np.ceil(cf_max / k)) - 1)
    while go_hi >= cu_lo and cf_go(np.array([go_hi]))[0] > cf_max:
        go_hi -= 1
    go = sup_at_most(cu_lo, go_hi, lambda a, b: linear_p(fs, n1, cf_go(a), 1 - cap, us, n0, b, cap, t, "le"), thr)
    if cu_lo == 0 and cf_min == 0:
        go = False                                            # a group without clean records is in H0
    ng_lo = max(cu_lo, int(np.ceil(cf_min / k)) if k > 0 else cu_lo)
    while ng_lo <= cu_hi and cf_ng(np.array([ng_lo]))[0] < cf_min:
        ng_lo += 1
    ng = sup_at_most(ng_lo, cu_hi, lambda a, b: linear_p(fs, n1, cf_ng(b), 1 - cap, us, n0, a, cap, t, "ge"), thr)
    return go, ng


def decide_ui(pop, chosen, alpha, tau, cap):
    Z = pop["poison"]; M = pop["memberships"]; s1, s0 = chosen[1], chosen[0]
    go, ng = recall_tests(pop["N"][1], len(s1), int(Z[s1].sum()), pop["N"][0], len(s0), int(Z[s0].sum()), tau, alpha)
    for g in range(M.shape[1]):
        fs, us = pop["gsize"][g]
        if fs == 0:
            continue
        a1 = s1[M[s1, g]]; a0 = s0[M[s0, g]]
        g_go, g_ng = group_tests(fs, len(a1), int((~Z[a1]).sum()), us, len(a0), int((~Z[a0]).sum()), cap, alpha)
        go &= g_go; ng |= g_ng
    return "GO" if go else ("NO_GO" if ng else "ABSTAIN")


def n_assertions(pop):
    return 1 + sum(1 for fs, _ in pop["gsize"] if fs > 0)


SHARES = ("0.5", "0.25", "0.1")
ESC_SHARE = "0.25"


def draw_strata(pop, ns, k, draw, share):
    n1, n0 = H.allocate(pop["N"][1], pop["N"][0], k, share)
    chosen = {}
    for f, n in ((1, n1), (0, n0)):
        key = (ns, k, draw, f) if share == "0.5" else ("srac-ui-v1", share, ns, k, draw, f)
        rng = np.random.default_rng(H.stable_seed(*key))
        chosen[f] = rng.choice(pop["idx"][f], size=n, replace=False)
    return chosen


def run_victim(args):
    block, victim, frame, cfg = args
    proto = json.loads(Path(cfg["protocol"]).read_text()); c = proto["certificate"]
    delta, tau, cap = c["delta_per_decision"], c["recall_target"], c["max_group_clean_flag_rate_cap"]
    pop = H.build_population(frame, proto["protected_groups"]); q = pop["q"]; eta = delta / (2 * q)
    A = n_assertions(pop); alpha = delta / (2 * A)
    true_pass = int(pop["recall"] >= tau and pop["max_clean"] <= cap)
    ns = cfg["ns"].format(v=victim); rows = []
    for k in c["budgets"]:
        for draw in range(c["protected_draws_per_seed_budget"]):
            for share in SHARES:
                chosen = draw_strata(pop, ns, k, draw, share)   # share 0.5 = the frozen SRAC draw
                if share == "0.5":
                    d_srac, *_ = H.decide(pop, chosen, dict(R=eta, G=eta), tau, cap)
                    rows.append(dict(design="srac", share=share, k=k, draw=draw, decision=d_srac, labels=k))
                rows.append(dict(design="srac_ui", share=share, k=k, draw=draw, decision=decide_ui(pop, chosen, alpha, tau, cap), labels=k))
    for draw in range(c["protected_draws_per_seed_budget"]):
        perm = {f: np.random.default_rng(H.stable_seed("srac-ui-escalation-v1", ns, draw, f)).permutation(pop["idx"][f]) for f in (1, 0)}
        final, used = "ABSTAIN", BUDGETS[-1]
        for k in BUDGETS:
            n1, n0 = H.allocate(pop["N"][1], pop["N"][0], k, ESC_SHARE)
            d = decide_ui(pop, {1: perm[1][:n1], 0: perm[0][:n0]}, alpha / len(BUDGETS), tau, cap)
            if d != "ABSTAIN":
                final, used = d, k; break
        rows.append(dict(design="srac_ui_esc", share=ESC_SHARE, k=0, draw=draw, decision=final, labels=used))
    out = pd.DataFrame(rows)
    out.insert(0, "victim_id", victim); out.insert(0, "block", block)
    out["true_pass"] = true_pass; out["recall"] = pop["recall"]; out["max_clean"] = pop["max_clean"]; out["A"] = A; out["Q"] = q
    return out


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
    out = out.sort_values(["block", "victim_id", "design", "share", "k", "draw"]).reset_index(drop=True)
    out["false_go"] = ((out.decision == "GO") & (out.true_pass == 0)).astype(int)
    out["false_nogo"] = ((out.decision == "NO_GO") & (out.true_pass == 1)).astype(int)
    val = {}
    for block, cfg in BLOCKS.items():
        fz = pd.read_csv(cfg["frozen"])[["victim_id", "k", "draw", "decision"]]
        mine = out[(out.block == block) & (out.design == "srac")][["victim_id", "k", "draw", "decision"]]
        m = fz.merge(mine, on=["victim_id", "k", "draw"], suffixes=("_f", "_m"))
        val[block] = dict(frozen_rows=len(fz), identical=bool(len(m) == len(fz) and (m.decision_f == m.decision_m).all()))
    a.output.mkdir(parents=True)
    out.to_csv(a.output / "decisions.csv", index=False)
    s = out.groupby(["design", "share", "true_pass", "k"]).agg(n=("decision", "size"), go=("decision", lambda x: float((x == "GO").mean())),
                                                     nogo=("decision", lambda x: float((x == "NO_GO").mean())), false_go=("false_go", "sum"),
                                                     false_nogo=("false_nogo", "sum"), mean_labels=("labels", "mean")).reset_index()
    s.to_csv(a.output / "summary.csv", index=False)
    h = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    rep = dict(validation=val, rows=len(out), totals=out.groupby("design")[["false_go", "false_nogo"]].sum().to_dict("index"),
               script_sha256=h(__file__), decisions_sha256=h(a.output / "decisions.csv"), summary_sha256=h(a.output / "summary.csv"))
    (a.output / "report.json").write_text(json.dumps(rep, indent=2)); print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()

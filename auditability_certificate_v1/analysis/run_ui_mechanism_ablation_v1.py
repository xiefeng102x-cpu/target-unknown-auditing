#!/usr/bin/env python3
"""Post-hoc mechanism analysis of SRAC-UI (developmental; plan UI_MECHANISM_ABLATION_PLAN_V1.md).

On exactly the samples that SRAC-UI v1.1 used with flagged share 0.1 (post-hoc blocks 8, 11-14,
namespace "srac-ui-v1"; prospective block 16, namespace "gtsrb-srac-ui-confirm-v1"), every
sample is decided by the following engines, all with false GO <= delta/2 and false NO-GO <=
delta/2 per decision:

  interval_frozen  interval SRAC, frozen split eta = delta/(2Q)
  interval_used    interval SRAC, eta = delta/(2 Q_used) over decision-relevant cells
  interval_split   interval SRAC, one-sided, delta/2 per error direction over used cells
                   (recall cells delta/4 in total, group cells delta/4 in total)
  ui_split         SRAC-UI v1.1 test with the interval_split risk allocation:
                   recall test alpha = delta/4, each of the G group tests delta/(4G)
  ui_nobb          SRAC-UI v1.1 allocation (alpha = delta/(2A)) without the Berger-Boos
                   restriction (gamma = 0, full nuisance range); block 16, draws 0-19 only
  ui_full          SRAC-UI v1.1 as reported (stored decisions)
  suite            SUITE-style test (Ottoboni et al. 2018): for each assertion, one-sided
                   stratum p-values combined by Fisher's function (chi-square, 4 df),
                   maximized over the null boundary; alpha = delta/(2A); no Berger-Boos

Contrasts (victim-bootstrap over policy-passing screens for GO, failing screens for NO-GO):
  ui_split - interval_split   the linear test at the interval allocation
  ui_full  - ui_split         per-assertion allocation
  ui_full  - ui_nobb          Berger-Boos restriction
  ui_full  - suite            exact convolution vs Fisher combination (nearest neighbour)
  ui_full  - interval_split   the whole engine at matched risk

usage: run_ui_mechanism_ablation_v1.py selftest
       run_ui_mechanism_ablation_v1.py run --parts DIR [--procs 2] [--only 8,16]
       run_ui_mechanism_ablation_v1.py combine --parts DIR --output DIR
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, itertools, json, sys, types
from fractions import Fraction as Fr
from math import comb
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import chi2

HERE = Path(__file__).resolve().parent; CERT = HERE.parent; ROOT = CERT.parent
spec = importlib.util.spec_from_file_location("ui_v1", HERE / "run_stratified_ui_audit_v1.py")
V1 = importlib.util.module_from_spec(spec); sys.modules["ui_v1"] = V1; spec.loader.exec_module(V1)
H = V1.H
sys.path.insert(0, str(HERE))
import srac_ui_engine_v1_1 as E  # noqa: E402
spec2 = importlib.util.spec_from_file_location("srac_ui_engine_v1_1_nobb", HERE / "srac_ui_engine_v1_1.py")
ENB = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(ENB)
ENB._bb_interval = lambda N, n, x, g: types.SimpleNamespace(lower=0, upper=int(N))   # no restriction

B16 = dict(protocol=CERT / "PROTOCOL_GTSRB_SRAC_UI_CONFIRMATION_V1.json",
           flags=ROOT / "results/gtsrb_srac_ui_scan_scores_confirmation_v1/flags.csv",
           ns="gtsrb-srac-ui-confirm-v1|{v}",
           ui=CERT / "results_gtsrb_srac_ui_confirmation_v1_1_erratum/decisions.csv")
POSTHOC_UI = CERT / "results_stratified_ui_audit_v1_1/decisions.csv"
DATASET = {"8": "GTSRB", "16": "GTSRB", "11": "CIFAR-10", "12": "CIFAR-10", "13": "CIFAR-10", "14": "CIFAR-10"}
ENGINES = ("interval_frozen", "interval_used", "interval_split", "ui_split", "ui_nobb", "ui_full", "suite")
CONTRASTS = (("ui_split", "interval_split"), ("ui_full", "ui_split"), ("ui_full", "ui_nobb"),
             ("ui_full", "suite"), ("ui_full", "interval_split"))
BOOT_SEED, BOOT_DRAWS = 2026101816, 10000
MARGIN = 1e-6
NOBB_DRAWS = 20   # ui_nobb (full nuisance range) is ~100x slower: block 16, draws 0-19 only


# ---------------------------------------------------------------- SUITE-style tests
def _fisher(p1, p0):
    with np.errstate(divide="ignore"):
        stat = -2.0 * (np.log(p1) + np.log(p0))
    return chi2.sf(stat, 4)


def _tail(x, N, K, n, upper):
    """P(X >= x) (upper) or P(X <= x) for X ~ HG(N, K, n), vectorized over K; sums the shorter
    side directly and uses the complement otherwise (absolute error ~1e-15, far below the
    thresholds); log-gamma pmf as in the engine."""
    K = np.atleast_1d(np.asarray(K, dtype=np.int64))
    if n == 0:
        return np.ones(K.shape)
    lo, hi = (x, n) if upper else (0, x)
    lo, hi = max(lo, 0), min(hi, n)
    if lo > hi:
        return np.zeros(K.shape)
    inside = hi - lo + 1; outside = (n + 1) - inside
    if inside <= outside:
        xs, comp = np.arange(lo, hi + 1), False
    else:
        xs, comp = (np.arange(0, lo) if upper else np.arange(hi + 1, n + 1)), True
    out = np.empty(K.shape)
    step = max(1, 2_000_000 // max(len(xs), 1))
    for a in range(0, len(K), step):
        v = E.hg_pmf(xs[None, :], N, K[a:a + step, None], n).sum(axis=1) if len(xs) else np.zeros(len(K[a:a + step]))
        out[a:a + step] = 1.0 - v if comp else v
    return np.clip(out, 0.0, 1.0)


def _sf(x, N, K, n):      # P(X >= x)
    return _tail(x, N, K, n, True)


def _cdf(x, N, K, n):     # P(X <= x)
    return _tail(x, N, K, n, False)


def _tstar(thr):
    """Largest product t of two p-values with Fisher p-value t(1 - ln t) <= thr."""
    lo, hi = 1e-300, thr
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        if mid * (1 - np.log(mid)) <= thr:
            lo = mid
        else:
            hi = mid
    return lo


def _lim(N, n, x, eta):
    """(L, U): smallest K with P(X >= x | K) > eta and largest K with P(X <= x | K) > eta,
    by bisection over the feasible counts (both tails are monotone in K); the whole range
    when the stratum is unsampled."""
    if n == 0:
        return 0, N
    flo, fhi = x, N - (n - x)
    a, b = flo, fhi                                   # smallest K with sf > eta (sf increases in K)
    while a < b:
        m = (a + b) // 2
        if _sf(x, N, m, n)[0] > eta:
            b = m
        else:
            a = m + 1
    L = a
    a, b = flo, fhi                                   # largest K with cdf > eta (cdf decreases in K)
    while a < b:
        m = (a + b + 1) // 2
        if _cdf(x, N, m, n)[0] > eta:
            a = m
        else:
            b = m - 1
    return L, a


def _maxfisher(lo, hi, p1, p0):
    if lo > hi:
        return 0.0
    u = np.arange(lo, hi + 1)
    return float(np.max(_fisher(p1(u), p0(u))))


def suite_recall(N1, n1, x1, N0, n0, x0, tau, alpha):
    """Fisher combination of the two one-sided stratum p-values at every null-boundary point,
    maximized over the boundary.  A point where either p-value is <= t* (Fisher p-value of the
    product t* equals the threshold) cannot attain the maximum above the threshold, so the
    search is restricted, exactly, to the points where both exceed t*."""
    tau = E.rational(tau); w1, w0 = tau.denominator - tau.numerator, tau.numerator
    if x1 == 0 and x0 == 0:
        return False, False
    thr = alpha * (1 - MARGIN); ts = _tstar(thr)
    c_lo, c_hi = (x1, N1 - (n1 - x1)) if n1 > 0 else (0, N1)
    m_max = N0 - (n0 - x0) if n0 > 0 else N0
    L1, U1 = _lim(N1, n1, x1, ts); L0, U0 = _lim(N0, n0, x0, ts)
    # GO: m = m_go(c); p1 = P(X1 >= x1 | c) > t* iff c >= L1; p0 = P(X0 <= x0 | m) > t* iff m <= U0
    go_hi = min(c_hi, E._ceil_div(w0 * m_max, w1) - 1, E._ceil_div(w0 * U0, w1) - 1)
    m_go = lambda c: np.maximum((w1 * c) // w0 + 1, x0)
    go = _maxfisher(max(c_lo, L1), go_hi, lambda c: _sf(x1, N1, c, n1), lambda c: _cdf(x0, N0, m_go(c), n0)) <= thr
    # NO-GO: m = m_ng(c); p1 = P(X1 <= x1 | c) > t* iff c <= U1; p0 = P(X0 >= x0 | m) > t* iff m >= L0
    ng_lo = max(c_lo, 1, E._ceil_div(w0 * x0, w1), E._ceil_div(w0 * L0, w1))
    m_ng = lambda c: np.minimum((w1 * c) // w0, m_max)
    ng = _maxfisher(ng_lo, min(c_hi, U1), lambda c: _cdf(x1, N1, c, n1), lambda c: _sf(x0, N0, m_ng(c), n0)) <= thr
    return bool(go), bool(ng)


def suite_group(fs, n1, y1, us, n0, y0, cap, alpha):
    cap = E.rational(cap); a_, b_ = cap.numerator, cap.denominator; w1, w0 = b_ - a_, a_
    thr = alpha * (1 - MARGIN); ts = _tstar(thr)
    cu_lo, cu_hi = (y0, us - (n0 - y0)) if n0 > 0 else (0, us)
    cf_min, cf_max = (y1, fs - (n1 - y1)) if n1 > 0 else (0, fs)
    L1, U1 = _lim(fs, n1, y1, ts); L0, U0 = _lim(us, n0, y0, ts)
    # GO: CF = cf_go(CU); p1 = P(Y1 <= y1 | CF) > t* iff CF <= U1; p0 = P(Y0 >= y0 | CU) > t* iff CU >= L0
    go_hi = min(cu_hi, E._ceil_div(w1 * cf_max, w0) - 1, E._ceil_div(w1 * U1, w0) - 1)
    cf_go = lambda cu: np.maximum((w0 * cu) // w1 + 1, cf_min)
    go = _maxfisher(max(cu_lo, L0), go_hi, lambda cu: _cdf(y1, fs, cf_go(cu), n1), lambda cu: _sf(y0, us, cu, n0)) <= thr
    if cu_lo == 0 and cf_min == 0:
        go = False
    # NO-GO: CF = cf_ng(CU); p1 = P(Y1 >= y1 | CF) > t* iff CF >= L1; p0 = P(Y0 <= y0 | CU) > t* iff CU <= U0
    ng_lo = max(cu_lo, E._ceil_div(w1 * cf_min, w0), E._ceil_div(w1 * L1, w0))
    cf_ng = lambda cu: np.minimum((w0 * cu) // w1, cf_max)
    ng = _maxfisher(ng_lo, min(cu_hi, U0), lambda cu: _sf(y1, fs, cf_ng(cu), n1), lambda cu: _cdf(y0, us, cu, n0)) <= thr
    return bool(go), bool(ng)


# ---------------------------------------------------------------- generic decision
def decide(pop, chosen, tau, cap, rfun, gfun, a_r, a_g):
    Z = pop["poison"]; M = pop["memberships"]; s1, s0 = chosen[1], chosen[0]
    go, ng = rfun(pop["N"][1], len(s1), int(Z[s1].sum()), pop["N"][0], len(s0), int(Z[s0].sum()), tau, a_r)
    for g in range(M.shape[1]):
        fs, us = pop["gsize"][g]
        if fs == 0:
            continue
        b1 = s1[M[s1, g]]; b0 = s0[M[s0, g]]
        g_go, g_ng = gfun(fs, len(b1), int((~Z[b1]).sum()), us, len(b0), int((~Z[b0]).sum()), cap, a_g)
        go &= g_go; ng |= g_ng
    return "GO" if go else ("NO_GO" if ng else "ABSTAIN")


ui_r = lambda *a: E.recall_tests(*a)
ui_g = lambda *a: E.group_tests(*a)
nb_r = lambda N1, n1, x1, N0, n0, x0, t, al: ENB.recall_tests(N1, n1, x1, N0, n0, x0, t, al, gamma=0.0)
nb_g = lambda fs, n1, y1, us, n0, y0, c, al: ENB.group_tests(fs, n1, y1, us, n0, y0, c, al, gamma=0.0)


def decide_all(pop, chosen, delta, tau, cap, with_nobb=True):
    A = E.n_assertions(pop); G = A - 1; alpha = delta / (2 * A)
    q_used = 2 + sum(int(fs > 0) + int(fs > 0 and us > 0) for fs, us in pop["gsize"])
    out = {}
    out["interval_frozen"], *_ = H.decide(pop, chosen, dict(R=delta / (2 * pop["q"]), G=delta / (2 * pop["q"])), tau, cap)
    out["interval_used"], *_ = H.decide(pop, chosen, dict(R=delta / (2 * q_used), G=delta / (2 * q_used)), tau, cap)
    out["interval_split"], *_ = H.decide(pop, chosen, H.etas("onesided_split", delta / 2, q_used), tau, cap)
    a_r, a_g = (delta / 4, delta / (4 * G)) if G > 0 else (delta / 2, delta / 2)
    out["ui_split"] = decide(pop, chosen, tau, cap, ui_r, ui_g, a_r, a_g)
    out["ui_nobb"] = decide(pop, chosen, tau, cap, nb_r, nb_g, alpha, alpha) if with_nobb else ""
    out["suite"] = decide(pop, chosen, tau, cap, suite_recall, suite_group, alpha, alpha)
    return out


# ---------------------------------------------------------------- per victim
def run_victim(job):
    block, victim, frame, cfg = job
    proto = json.loads(Path(cfg["protocol"]).read_text()); c = proto["certificate"]
    delta, tau, cap = c["delta_per_decision"], c["recall_target"], c["max_group_clean_flag_rate_cap"]
    pop = H.build_population(frame, proto["protected_groups"])
    tp = int(pop["recall"] >= tau and pop["max_clean"] <= cap)
    ns = cfg["ns"].format(v=victim); rows = []
    for k in c["budgets"]:
        for draw in range(c["protected_draws_per_seed_budget"]):
            if block == "16":
                n1, n0 = H.allocate(pop["N"][1], pop["N"][0], k, "0.1")
                ch = {f: np.random.default_rng(H.stable_seed(ns, "srac_ui", k, draw, f)).choice(pop["idx"][f], size=n, replace=False)
                      for f, n in ((1, n1), (0, n0))}
            else:
                ch = V1.draw_strata(pop, ns, k, draw, "0.1")
            nobb = block == "16" and draw < NOBB_DRAWS
            rows.append(dict(block=block, victim_id=victim, k=k, draw=draw, true_pass=tp, **decide_all(pop, ch, delta, tau, cap, nobb)))
    return pd.DataFrame(rows)


def jobs(only):
    out = []
    for block, cfg in V1.BLOCKS.items():
        if only and block not in only:
            continue
        f = pd.read_csv(cfg["flags"], usecols=["victim_id", "seed", "sample_id", "assigned_group", "flag", "is_poison", "victim_eligible"])
        for v, fr in f[f.victim_eligible.eq(1)].groupby("victim_id", sort=True):
            out.append((block, v, fr.reset_index(drop=True), {k: str(x) for k, x in cfg.items()}))
    if not only or "16" in only:
        f = pd.read_csv(B16["flags"]); f = f[f.victim_eligible.eq(1)]
        for v, fr in f.groupby("victim_id", sort=True):
            out.append(("16", v, fr.reset_index(drop=True)[["victim_id", "seed", "sample_id", "assigned_group", "flag", "is_poison", "victim_eligible"]],
                        dict(protocol=str(B16["protocol"]), ns=B16["ns"])))
    return out


# ---------------------------------------------------------------- self-test (exact size of the SUITE-style test)
def selftest():
    sys.path.insert(0, str(HERE)); import test_srac_ui_engine_v1_1 as T  # noqa: E402
    worst = Fr(0); fails = []; n = 0
    for kind, thrs in (("recall", (Fr(4, 5), Fr(2, 3))), ("group", (Fr(1, 10), Fr(1, 4)))):
        fn = suite_recall if kind == "recall" else suite_group
        for (Na, na, Nb, nb), thr, alpha in itertools.product(T.CONFIGS, thrs, (0.05, 0.2)):
            ref = T.Ref(kind, Na, na, Nb, nb, thr); dec = {}
            for xa in range(na + 1):
                for xb in range(nb + 1):
                    dec[xa, xb] = fn(Na, na, xa, Nb, nb, xb, thr, alpha)
            tot = comb(Na, na) * comb(Nb, nb)
            for Ka in range(Na + 1):
                wa = T.hg_w(Na, Ka, na)
                for Kb in range(Nb + 1):
                    wb = T.hg_w(Nb, Kb, nb); h, vac = ref.holds(Ka, Kb); n += 1
                    pgo = Fr(sum(wa[i] * wb[j] for (i, j), d in dec.items() if d[0]), tot)
                    png = Fr(sum(wa[i] * wb[j] for (i, j), d in dec.items() if d[1]), tot)
                    for bad, p in (((not h) or vac, pgo), (h or vac, png)):
                        if bad:
                            worst = max(worst, p / Fr(alpha))
                            if p > Fr(alpha):
                                fails.append((kind, (Na, na, Nb, nb), str(thr), alpha, (Ka, Kb), float(p)))
    res = dict(truth_points=n, max_error_over_alpha=float(worst), failures=fails[:20], passed=not fails)
    print(json.dumps(res, indent=1)); return res


# ---------------------------------------------------------------- combine
def boot(d, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), size=(BOOT_DRAWS, len(d)))].mean(axis=1)
    return [float(np.quantile(m, .025)), float(np.quantile(m, .975))]


def combine(parts: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    d = pd.concat([pd.read_csv(p, dtype={"block": str}) for p in sorted(parts.glob("*.csv"))], ignore_index=True)
    ph = pd.read_csv(POSTHOC_UI, dtype={"block": str, "share": str})
    ph = ph[(ph.design == "srac_ui") & (ph.share == "0.1")][["block", "victim_id", "k", "draw", "decision"]]
    b16 = pd.read_csv(B16["ui"]); b16 = b16[b16.engine == "srac_ui"][["victim_id", "k", "draw", "decision"]].assign(block="16")
    ui = pd.concat([ph, b16], ignore_index=True).rename(columns={"decision": "ui_full"})
    d = d.merge(ui, on=["block", "victim_id", "k", "draw"], how="left")
    if d.ui_full.isna().any() or len(d) != len(ui):
        raise RuntimeError("UI decisions do not match the sample set")
    d["dataset"] = d.block.map(DATASET)
    d["set"] = np.where(d.block == "16", "block16_prospective_samples", "posthoc_" + d.dataset)
    output.mkdir(parents=True)
    d.to_csv(output / "decisions.csv", index=False)
    rows, cons = [], []
    for (st, k), g in d.groupby(["set", "k"]):
        p = g[g.true_pass == 1]; f = g[g.true_pass == 0]
        r = dict(set=st, k=int(k), passing_victims=int(p.victim_id.nunique()), failing_victims=int(f.victim_id.nunique()))
        for e in ENGINES:
            if e == "ui_nobb":
                p_, f_ = p[p.ui_nobb.isin(["GO", "NO_GO", "ABSTAIN"])], f[f.ui_nobb.isin(["GO", "NO_GO", "ABSTAIN"])]
                r["ui_nobb_go"] = float(p_.ui_nobb.eq("GO").mean()) if len(p_) else None
                r["ui_nobb_nogo"] = float(f_.ui_nobb.eq("NO_GO").mean()) if len(f_) else None
                r["ui_nobb_n"] = int(len(p_) + len(f_)); continue
            r[f"{e}_go"] = float(p[e].eq("GO").mean()) if len(p) else None
            r[f"{e}_nogo"] = float(f[e].eq("NO_GO").mean()) if len(f) else None
            r[f"{e}_false"] = int(((g[e] == "GO") & (g.true_pass == 0)).sum() + ((g[e] == "NO_GO") & (g.true_pass == 1)).sum())
        rows.append(r)
        for a, b in CONTRASTS:
            for direction, sub, lab in (("GO", p, "GO"), ("NO_GO", f, "NO_GO")):
                if sub.empty:
                    continue
                if "ui_nobb" in (a, b):
                    sub = sub[sub.ui_nobb.isin(["GO", "NO_GO", "ABSTAIN"])]
                    if sub.empty:
                        continue
                diff = (sub[a].eq(lab).astype(int) - sub[b].eq(lab).astype(int)).groupby(sub.victim_id).mean().to_numpy()
                cons.append(dict(set=st, k=int(k), direction=direction, contrast=f"{a} - {b}", victims=int(len(diff)),
                                 mean=float(diff.mean()), ci95=boot(diff)))
    pd.DataFrame(rows).to_csv(output / "summary.csv", index=False)
    rep = dict(status="POST_HOC_DEVELOPMENTAL", rows=len(d), summary=rows, contrasts=cons,
               script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               decisions_sha256=hashlib.sha256((output / "decisions.csv").read_bytes()).hexdigest())
    (output / "report.json").write_text(json.dumps(rep, indent=2) + "\n")
    print(pd.DataFrame(cons).to_string())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=("selftest", "run", "combine"))
    ap.add_argument("--parts", type=Path); ap.add_argument("--output", type=Path); ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--only", default=""); a = ap.parse_args()
    if a.cmd == "selftest":
        sys.exit(0 if selftest()["passed"] else 1)
    if a.cmd == "combine":
        return combine(a.parts, a.output)
    a.parts.mkdir(parents=True, exist_ok=True)
    only = [b for b in a.only.split(",") if b]
    todo = [j for j in jobs(only) if not (a.parts / f"{j[0]}__{j[1]}.csv").exists()]
    print(f"{len(todo)} victims to compute", flush=True)
    with Pool(a.procs) as p:
        for df in p.imap_unordered(run_victim, todo):
            df.to_csv(a.parts / f"{df.block.iloc[0]}__{df.victim_id.iloc[0]}.csv", index=False)
            print("part done", df.block.iloc[0], df.victim_id.iloc[0], flush=True)


if __name__ == "__main__":
    main()

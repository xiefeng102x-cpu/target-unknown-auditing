#!/usr/bin/env python3
"""Fixed-budget comparison of SRAC-UI with a stratified betting union-intersection test in the
style of SWEETER (Spertus and Stark 2022) (post hoc; plan SWEETER_COMPARISON_PLAN_V1.md).

For every linear assertion and direction, the null is a set of count pairs (stratum or cell
counts).  For each point on the null boundary, each stratum (or group cell) gets a
without-replacement betting test martingale (Waudby-Smith and Ramdas) for its share of the
null, with the predictable aGRAPA bets of the betting comparison audit; the two final values
are multiplied (independent strata, so the product has expectation at most one under that
point) and the test rejects iff the minimum of the product over the boundary is at least
1/alpha.  Every null point is dominated by a boundary point, so the size is at most alpha by
Markov's inequality.  The samples, their order, the strata, the risk allocation, and the GO /
NO-GO rule are those of SRAC-UI; only the test inside each assertion changes.

  sweeter        alpha = delta/(2A) per assertion and direction (SRAC-UI as reported)
  sweeter_dir    GO tests at delta/2, NO-GO tests at delta/(2A) (direction-specific)

usage: run_sweeter_comparison_v1.py selftest
       run_sweeter_comparison_v1.py run --parts DIR [--procs 2]
       run_sweeter_comparison_v1.py combine --parts DIR --output DIR
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys, time
from fractions import Fraction
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent; CERT = HERE.parent; ROOT = CERT.parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("ui_v1", HERE / "run_stratified_ui_audit_v1.py")
V1 = importlib.util.module_from_spec(spec); sys.modules["ui_v1"] = V1; spec.loader.exec_module(V1)
H = V1.H
import srac_ui_engine_v1_1 as E  # noqa: E402

B16 = dict(protocol=CERT / "PROTOCOL_GTSRB_SRAC_UI_CONFIRMATION_V1.json",
           flags=ROOT / "results/gtsrb_srac_ui_scan_scores_confirmation_v1/flags.csv",
           ns="gtsrb-srac-ui-confirm-v1|{v}")
GOALLOC = CERT / "results_go_allocation_v1/decisions.csv"
COLS = ["victim_id", "seed", "sample_id", "assigned_group", "flag", "is_poison", "victim_eligible"]
DRAWS = {"16": 200, "8": 20, "11": 20, "12": 20, "13": 20, "14": 20}
DATASET = {"8": "GTSRB", "16": "GTSRB", "11": "CIFAR-10", "12": "CIFAR-10", "13": "CIFAR-10", "14": "CIFAR-10"}
SEED, BOOT = 2026102101, 10000
CHUNK = 256


# ------------------------------------------------------------------ betting test martingales
from numba import njit, prange  # noqa: E402


@njit(cache=True, parallel=False)
def _log_tsm_nb(x, N, mus, above):
    n = x.shape[0]; P = mus.shape[0]; out = np.zeros(P)
    for p in range(P):
        m = mus[p]; S = 0.0; ss = 0.0; acc = 0.0
        for i in range(n):
            t = i + 1.0
            m_t = (N * m - S) / (N - t + 1)
            mu = (S + 0.5 * m) / t
            var = (ss - 2 * mu * S + (t - 1) * mu * mu + 0.25 * m) / t
            if var < 1e-8:
                var = 1e-8
            xi = x[i]
            if above:
                diff = mu - m_t
                lam = diff / (var + diff * diff)
                if lam < 0:
                    lam = 0.0
                cap = 0.75 / (m_t if m_t > 1e-12 else 1e-12)
                if lam > cap:
                    lam = cap
                inc = 1 + lam * (xi - m_t)
                if m_t < 0:
                    acc = np.inf
                    break
            else:
                diff = m_t - mu
                lam = diff / (var + diff * diff)
                if lam < 0:
                    lam = 0.0
                cap = 0.75 / ((1 - m_t) if (1 - m_t) > 1e-12 else 1e-12)
                if lam > cap:
                    lam = cap
                inc = 1 + lam * (m_t - xi)
                if m_t > 1:
                    acc = np.inf
                    break
                if m_t < 0:
                    inc = 0.0
            if inc <= 0:
                acc = -np.inf
                break
            acc += np.log(inc)
            S += xi; ss += xi * xi
        out[p] = acc
    return out


def log_tsm(x, N, mus, side):
    """Numba version of log_tsm_np (identical arithmetic and edge rules)."""
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float64)); mus = np.ascontiguousarray(np.atleast_1d(np.asarray(mus, dtype=np.float64)))
    if len(x) == 0:
        return np.zeros(len(mus))
    return _log_tsm_nb(x, float(N), mus, side == "above")


def log_tsm_np(x, N, mus, side):
    """Final log value of the WoR betting test martingale (aGRAPA bets, as in the betting
    comparison audit) for H0: mean <= mu (side 'above', rejects for large values) or
    H0: mean >= mu (side 'below'), vectorized over null means mus.  Empty sample: 0."""
    x = np.asarray(x, dtype=float); n = len(x); mus = np.atleast_1d(np.asarray(mus, dtype=float))
    if n == 0:
        return np.zeros(len(mus))
    t = np.arange(1, n + 1, dtype=float)
    S_prev = np.concatenate([[0.0], np.cumsum(x)[:-1]]); ss = np.concatenate([[0.0], np.cumsum(x ** 2)[:-1]])
    out = np.empty(len(mus))
    for a in range(0, len(mus), CHUNK):
        m = mus[a:a + CHUNK, None]
        m_t = (N * m - S_prev[None, :]) / (N - t[None, :] + 1)
        mu = (S_prev[None, :] + 0.5 * m) / t[None, :]
        var = np.maximum((ss[None, :] - 2 * mu * S_prev[None, :] + (t[None, :] - 1) * mu ** 2 + 0.25 * m) / t[None, :], 1e-8)
        if side == "above":
            diff = mu - m_t
            lam = np.minimum(np.clip(diff / (var + diff ** 2), 0, None), 0.75 / np.maximum(m_t, 1e-12))
            inc = 1 + lam * (x[None, :] - m_t)
            inc = np.where(m_t < 0, np.inf, inc)                  # observed total already exceeds N*mu
        else:
            diff = m_t - mu
            lam = np.minimum(np.clip(diff / (var + diff ** 2), 0, None), 0.75 / np.maximum(1 - m_t, 1e-12))
            inc = 1 + lam * (m_t - x[None, :])
            inc = np.where(m_t < 0, 0.0, inc)                     # H0 certainly true
            inc = np.where(m_t > 1, np.inf, inc)                  # remaining mean would exceed 1
        with np.errstate(divide="ignore"):
            out[a:a + CHUNK] = np.log(np.maximum(inc, 0.0)).sum(axis=1)
    return out


def min_log(a_vals, b_vals):
    s = a_vals + b_vals
    s = s[~np.isnan(s)]
    return float(s.min()) if len(s) else np.inf


def _recall_w(tau):
    tau = E.rational(tau)
    return tau.denominator - tau.numerator, tau.numerator


def recall_go(N1, x1seq, N0, x0seq, tau):
    """min over the GO-null boundary (recall < tau) of the log product statistic."""
    w1, w0 = _recall_w(tau)
    if np.sum(x1seq) == 0 and np.sum(x0seq) == 0:
        return -np.inf
    c = np.arange(0, N1 + 1)
    # boundary m = floor(w1 c / w0) + 1 <= N0; stratum 1 H0: C1 <= c ('above'); stratum 0 H0: M0 >= m ('below')
    m = (w1 * c) // w0 + 1; ok = m <= N0
    return min_log(log_tsm(x1seq, N1, c[ok] / N1, "above"), log_tsm(x0seq, N0, m[ok] / N0, "below")) if ok.any() else np.inf


def recall_ng(N1, x1seq, N0, x0seq, tau):
    """min over the NO-GO-null boundary (recall >= tau) of the log product statistic."""
    w1, w0 = _recall_w(tau)
    if np.sum(x1seq) == 0 and np.sum(x0seq) == 0:
        return -np.inf
    c = np.arange(1, N1 + 1); m = np.minimum((w1 * c) // w0, N0)
    # stratum 1 H0: C1 >= c ('below'); stratum 0 H0: M0 <= m ('above')
    return min_log(log_tsm(x1seq, N1, c / N1, "below"), log_tsm(x0seq, N0, m / N0, "above"))


def _cap_w(cap):
    cap = E.rational(cap)
    return cap.denominator - cap.numerator, cap.numerator


def group_go(fs, y1seq, us, y0seq, cap):
    """Clean counts CF (flagged cell, size fs) and CU (unflagged cell, size us); GO null F > cap."""
    w1, w0 = _cap_w(cap)
    if np.sum(y0seq) == 0 and np.sum(y1seq) == 0:
        return -np.inf      # as SRAC-UI: a group that may hold no clean record stays in the GO null
    u = np.arange(0, us + 1)
    # boundary CF = floor(w0 u / w1) + 1 <= fs; cell 1 H0: CF >= cf ('below'); cell 0 H0: CU <= u ('above')
    cf = (w0 * u) // w1 + 1; ok = cf <= fs
    if not ok.any():
        return np.inf
    return min_log(log_tsm(y1seq, fs, cf[ok] / fs, "below"), log_tsm(y0seq, us, u[ok] / us if us else u[ok] * 0.0, "above"))


def group_ng(fs, y1seq, us, y0seq, cap):
    """NO-GO null F <= cap: CF <= floor(w0 u / w1); cell 1 H0: CF <= cf ('above'); cell 0 H0: CU >= u ('below')."""
    w1, w0 = _cap_w(cap)
    u = np.arange(0, us + 1); cf = np.minimum((w0 * u) // w1, fs)
    return min_log(log_tsm(y1seq, fs, cf / fs, "above"), log_tsm(y0seq, us, u / us if us else u * 0.0, "below"))


def decide_both(pop, chosen, tau, cap, delta):
    """Decisions at alpha = delta/(2A) (sweeter) and with GO tests at delta/2 (sweeter_dir).
    GO needs every GO test, NO-GO any NO-GO test; evaluation stops as soon as the outcome is
    determined, which does not change any decision."""
    Z = pop["poison"]; M = pop["memberships"]; s1, s0 = chosen[1], chosen[0]
    A = E.n_assertions(pop); t_lo, t_hi, t_ng = np.log(2 / delta), np.log(2 * A / delta), np.log(2 * A / delta)
    cells = []
    for k in range(M.shape[1]):
        fs, us = pop["gsize"][k]
        if fs == 0:
            continue
        a1 = s1[M[s1, k]]; a0 = s0[M[s0, k]]
        cells.append((fs, (~Z[a1]).astype(float), us, (~Z[a0]).astype(float)))
    x1, x0 = Z[s1].astype(float), Z[s0].astype(float)
    gmin = recall_go(pop["N"][1], x1, pop["N"][0], x0, tau)
    for fs, y1, us, y0 in cells:
        if gmin < t_lo:
            break
        gmin = min(gmin, group_go(fs, y1, us, y0, cap))
    go_dir, go_full = gmin >= t_lo, gmin >= t_hi
    ng = False
    if not go_full:
        ng = recall_ng(pop["N"][1], x1, pop["N"][0], x0, tau) >= t_ng
        for fs, y1, us, y0 in cells:
            if ng:
                break
            ng = group_ng(fs, y1, us, y0, cap) >= t_ng
    dec = lambda go: "GO" if go else ("NO_GO" if ng else "ABSTAIN")
    return dec(go_full), dec(go_dir)


# ------------------------------------------------------------------ self-test (Monte Carlo size at null boundary points)
def selftest(reps=20000):
    rng = np.random.default_rng(7); worst = []; t0 = time.time()
    cfgs = [(30, 10, 60, 20), (20, 20, 50, 10), (40, 8, 200, 40)]
    for N1, n1, N0, n0 in cfgs:
        for tau in (Fraction(4, 5), Fraction(2, 3)):
            w1, w0 = tau.denominator - tau.numerator, tau.numerator
            for c in (max(1, N1 // 4), N1 // 2, N1 - 2):
                for kind, m in (("go", (w1 * c) // w0 + 1), ("ng", (w1 * c) // w0)):
                    if not (0 <= m <= N0):
                        continue
                    pop1 = np.zeros(N1); pop1[:c] = 1; pop0 = np.zeros(N0); pop0[:m] = 1
                    rej = {0.05: 0, 0.2: 0}
                    for _ in range(reps):
                        x1 = rng.permutation(pop1)[:n1]; x0 = rng.permutation(pop0)[:n0]
                        s = recall_go(N1, x1, N0, x0, tau) if kind == "go" else recall_ng(N1, x1, N0, x0, tau)
                        for al in rej:
                            rej[al] += s >= np.log(1 / al)
                    for al, r in rej.items():
                        worst.append(dict(cfg=(N1, n1, N0, n0), tau=str(tau), c=c, m=m, kind=kind, alpha=al, rate=r / reps))
    mx = max(w["rate"] / w["alpha"] for w in worst)
    ok = all(w["rate"] <= w["alpha"] + 3 * np.sqrt(w["alpha"] * (1 - w["alpha"]) / reps) for w in worst)
    res = dict(points=len(worst), reps=reps, max_rate_over_alpha=mx, passed=bool(ok), seconds=round(time.time() - t0, 1))
    print(json.dumps(res)); return res


# ------------------------------------------------------------------ per victim
def run_victim(job):
    block, victim, frame, cfg = job
    proto = json.loads(Path(cfg["protocol"]).read_text()); c = proto["certificate"]
    delta, tau, cap = c["delta_per_decision"], c["recall_target"], c["max_group_clean_flag_rate_cap"]
    pop = H.build_population(frame, proto["protected_groups"]); A = E.n_assertions(pop)
    tp = int(pop["recall"] >= tau and pop["max_clean"] <= cap); ns = cfg["ns"].format(v=victim); rows = []
    for k in c["budgets"]:
        for draw in range(DRAWS[block]):
            if block == "16":
                n1, n0 = H.allocate(pop["N"][1], pop["N"][0], k, "0.1")
                ch = {f: np.random.default_rng(H.stable_seed(ns, "srac_ui", k, draw, f)).choice(pop["idx"][f], size=n, replace=False)
                      for f, n in ((1, n1), (0, n0))}
            else:
                ch = V1.draw_strata(pop, ns, k, draw, "0.1")
            t0 = time.time(); d_full, d_dir = decide_both(pop, ch, tau, cap, delta); sec = time.time() - t0
            rows.append(dict(block=block, victim_id=victim, k=k, draw=draw, true_pass=tp, A=A,
                             sweeter=d_full, sweeter_dir=d_dir, seconds=round(sec, 3)))
    return pd.DataFrame(rows)


def jobs():
    out = []
    for block, cfg in V1.BLOCKS.items():
        f = pd.read_csv(cfg["flags"], usecols=COLS)
        for v, fr in f[f.victim_eligible.eq(1)].groupby("victim_id", sort=True):
            out.append((block, v, fr.reset_index(drop=True), {k: str(x) for k, x in cfg.items()}))
    f = pd.read_csv(B16["flags"], usecols=COLS); f = f[f.victim_eligible.eq(1)]
    for v, fr in f.groupby("victim_id", sort=True):
        out.append(("16", v, fr.reset_index(drop=True), dict(protocol=str(B16["protocol"]), ns=B16["ns"])))
    return out


# ------------------------------------------------------------------ combine
def cluster_boot(scr, cluster):
    keys = list(dict.fromkeys(cluster)); groups = [scr[cluster == k] for k in keys]
    rng = np.random.default_rng(SEED); out = np.empty(BOOT)
    for i in range(BOOT):
        out[i] = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))]).mean()
    return [float(np.quantile(out, .025)), float(np.quantile(out, .975))]


def combine(parts, output):
    if output.exists():
        raise FileExistsError(output)
    d = pd.concat([pd.read_csv(p, dtype={"block": str}) for p in sorted(parts.glob("*.csv"))], ignore_index=True)
    ga = pd.read_csv(GOALLOC, dtype={"block": str})[["block", "victim_id", "k", "draw", "ui_full_stored", "ui_dir", "rla_bet", "rla_dir"]]
    d = d.merge(ga, on=["block", "victim_id", "k", "draw"], how="left", validate="one_to_one")
    if d.ui_full_stored.isna().any():
        raise RuntimeError("missing SRAC-UI decisions")
    d["dataset"] = d.block.map(DATASET); d["set"] = np.where(d.block == "16", "confirmation", "posthoc_" + d.dataset)
    d["model"] = d.dataset + ":" + d.victim_id.str.extract(r"seed(\d+)")[0]
    output.mkdir(parents=True); d.to_csv(output / "decisions.csv", index=False)
    summ, cons = [], []
    for (st, k), g in d.groupby(["set", "k"]):
        p, f = g[g.true_pass == 1], g[g.true_pass == 0]
        r = dict(set=st, k=int(k), audits=int(len(g)), seconds_mean=float(g.seconds.mean()), seconds_max=float(g.seconds.max()))
        for e in ("ui_full_stored", "ui_dir", "sweeter", "sweeter_dir", "rla_bet", "rla_dir"):
            r[f"{e}_go"] = float(p[e].eq("GO").mean()); r[f"{e}_nogo"] = float(f[e].eq("NO_GO").mean()) if len(f) else None
            r[f"{e}_false"] = int(((g[e] == "GO") & (g.true_pass == 0)).sum() + ((g[e] == "NO_GO") & (g.true_pass == 1)).sum())
        summ.append(r)
        for a, b in (("ui_full_stored", "sweeter"), ("ui_dir", "sweeter_dir"), ("sweeter", "rla_bet"), ("sweeter_dir", "rla_dir")):
            for direction, lab, sub in (("GO", "GO", p), ("NO_GO", "NO_GO", f)):
                if sub.empty:
                    continue
                x = sub[a].eq(lab).astype(int) - sub[b].eq(lab).astype(int); scr = x.groupby(sub.victim_id).mean()
                cl = sub.groupby("victim_id").model.first().loc[scr.index].to_numpy()
                cons.append(dict(set=st, k=int(k), direction=direction, contrast=f"{a} - {b}", screens=int(len(scr)),
                                 models=int(len(set(cl))), mean=float(scr.mean()), ci95=cluster_boot(scr.to_numpy(), cl)))
    rep = dict(status="POST_HOC_DEVELOPMENTAL", rows=int(len(d)), summary=summ, contrasts=cons,
               script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               decisions_sha256=hashlib.sha256((output / "decisions.csv").read_bytes()).hexdigest())
    (output / "report.json").write_text(json.dumps(rep, indent=2) + "\n")
    print(pd.DataFrame(cons).query("k == 2000 and direction == 'GO'").to_string())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=("selftest", "run", "combine"))
    ap.add_argument("--parts", type=Path); ap.add_argument("--output", type=Path); ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    if a.cmd == "selftest":
        sys.exit(0 if selftest()["passed"] else 1)
    if a.cmd == "combine":
        return combine(a.parts, a.output)
    a.parts.mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs() if not (a.parts / f"{j[0]}__{j[1]}.csv").exists()]
    print(f"{len(todo)} victims to compute", flush=True)
    with Pool(a.procs) as p:
        for df in p.imap_unordered(run_victim, todo):
            df.to_csv(a.parts / f"{df.block.iloc[0]}__{df.victim_id.iloc[0]}.csv", index=False)
            print("part done", df.block.iloc[0], df.victim_id.iloc[0], flush=True)


if __name__ == "__main__":
    main()

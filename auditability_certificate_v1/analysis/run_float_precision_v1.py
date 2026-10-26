#!/usr/bin/env python3
"""Floating-point precision of SRAC-UI engine v1.1 at audited sizes (post hoc; plan
FLOAT_PRECISION_PLAN_V1.md).

1. Instrument the engine's tail function linear_p and recompute calls whose value lies in
   [1e-8, 1e-1] with 40-digit mpmath arithmetic (relative error of the double result).
2. Instrument scipy hypergeom sf/cdf inside the exact count-interval routine (Berger-Boos set)
   and recompute calls in [1e-9, 1e-1] the same way.
3. Re-decide draws 0-19 with MARGIN 0 and 1e-4 and count changes against MARGIN 1e-6.

usage: run_float_precision_v1.py --output DIR [--inst-draws 5] [--stab-draws 20]
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys, time
from pathlib import Path
import mpmath as mp
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent; CERT = HERE.parent; ROOT = CERT.parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("ui_v1", HERE / "run_stratified_ui_audit_v1.py")
V1 = importlib.util.module_from_spec(spec); sys.modules["ui_v1"] = V1; spec.loader.exec_module(V1)
H = V1.H
import srac_ui_engine_v1_1 as E  # noqa: E402

mp.mp.dps = 40
CAP_NEAR, CAP_OTHER = 300, 100
import os
if os.environ.get("FP_DRYRUN"): CAP_NEAR, CAP_OTHER = 5, 3
B16_PROTO = CERT / "PROTOCOL_GTSRB_SRAC_UI_CONFIRMATION_V1.json"
B16_FLAGS = ROOT / "results/gtsrb_srac_ui_scan_scores_confirmation_v1/flags.csv"
B16_STORED = CERT / "results_gtsrb_srac_ui_confirmation_v1_1_erratum/decisions.csv"
PH_STORED = CERT / "results_stratified_ui_audit_v1_1/decisions.csv"
SC_STORED = CERT / "results_ui_scaling_v1/decisions.csv"
COLS = ["victim_id", "seed", "sample_id", "assigned_group", "flag", "is_poison", "victim_eligible"]


# ------------------------------------------------------------------ 40-digit references
def mp_pmf_vec(N, K, n, lo, hi):
    """pmf of HG(N, K, n) at x = lo..hi (all in the support), 40 digits."""
    if lo > hi:
        return []
    lg = lambda z: mp.loggamma(z + 1)
    x = lo
    v = mp.exp(lg(K) - lg(x) - lg(K - x) + lg(N - K) - lg(n - x) - lg(N - K - n + x) - (lg(N) - lg(n) - lg(N - n)))
    out = [v]
    for x in range(lo, hi):
        v = v * (K - x) * (n - x) / ((x + 1) * (N - K - n + x + 1))
        out.append(v)
    return out


def support(N, K, n):
    return max(0, n - (N - K)), min(n, K)


def mp_linear_p(Na, na, Ka, A, Nb, nb, Kb, B, s, side):
    if nb > 0:
        lo, hi = support(Nb, Kb, nb); xb = list(range(lo, hi + 1)); pb = mp_pmf_vec(Nb, Kb, nb, lo, hi)
    else:
        xb, pb = [0], [mp.mpf(1)]
    if na > 0 and A > 0:
        la, ha = support(Na, Ka, na); pa = mp_pmf_vec(Na, Ka, na, la, ha)
        # tails over the full 0..na index
        full = [mp.mpf(0)] * (na + 1)
        for i, x in enumerate(range(la, ha + 1)):
            full[x] = pa[i]
        if side == "ge":
            tail = [mp.mpf(0)] * (na + 2)
            for x in range(na, -1, -1):
                tail[x] = tail[x + 1] + full[x]
        else:
            cdf = []; acc = mp.mpf(0)
            for x in range(na + 1):
                acc += full[x]; cdf.append(acc)
    tot = mp.mpf(0)
    for x, p in zip(xb, pb):
        rhs = s + B * x
        if na > 0 and A > 0:
            if side == "ge":
                t = E._ceil_div(rhs, A); q = mp.mpf(1) if t <= 0 else tail[min(max(t, 0), na + 1)]
            else:
                t = rhs // A; q = mp.mpf(0) if t < 0 else (mp.mpf(1) if t >= na else cdf[t])
        else:
            q = mp.mpf(1) if ((0 >= rhs) if side == "ge" else (0 <= rhs)) else mp.mpf(0)
        tot += p * q
    return min(tot, mp.mpf(1))


def mp_hyper(kind, x, N, K, n):
    lo, hi = support(N, K, n)
    if kind == "sf":     # P(X > x)
        a, b = max(lo, x + 1), hi
    else:                # P(X <= x)
        a, b = lo, min(hi, x)
    return mp.fsum(mp_pmf_vec(N, K, n, a, b)) if a <= b else mp.mpf(0)


# ------------------------------------------------------------------ instrumentation
class Recorder:
    def __init__(self):
        self.near, self.other = [], []
        self.level = None

    def add(self, rec, val):
        near = self.level is not None and self.level / 10 <= val <= self.level * 10
        lst, cap = (self.near, CAP_NEAR) if near else (self.other, CAP_OTHER)
        if len(lst) < cap:
            lst.append(rec)

    def full(self):
        return len(self.near) >= CAP_NEAR and len(self.other) >= CAP_OTHER


LP, HG = Recorder(), Recorder()
_orig_lp = E.linear_p


def lp_wrap(Na, na, Ka, A, Nb, nb, Kb, B, s, side):
    out = _orig_lp(Na, na, Ka, A, Nb, nb, Kb, B, s, side)
    if not LP.full():
        Ka_ = np.broadcast_to(np.atleast_1d(Ka), out.shape); Kb_ = np.broadcast_to(np.atleast_1d(Kb), out.shape)
        for i in np.flatnonzero((out >= 1e-8) & (out <= 1e-1))[:5]:
            LP.add((int(Na), int(na), int(Ka_[i]), int(A), int(Nb), int(nb), int(Kb_[i]), int(B), int(s), side, float(out[i])), float(out[i]))
    return out


class HGWrap:
    def __init__(self, base):
        self.base = base

    def sf(self, x, N, K, n):
        v = float(self.base.sf(x, N, K, n))
        if 1e-9 <= v <= 1e-1 and not HG.full():
            HG.add(("sf", int(x), int(N), int(K), int(n), v), v)
        return v

    def cdf(self, x, N, K, n):
        v = float(self.base.cdf(x, N, K, n))
        if 1e-9 <= v <= 1e-1 and not HG.full():
            HG.add(("cdf", int(x), int(N), int(K), int(n), v), v)
        return v


# ------------------------------------------------------------------ settings
def settings():
    out = []
    proto = json.loads(B16_PROTO.read_text())
    f = pd.read_csv(B16_FLAGS); f = f[f.victim_eligible.eq(1)]
    for v, fr in f.groupby("victim_id", sort=True):
        out.append(("confirmation", v, fr.reset_index(drop=True)[COLS], proto, "gtsrb-srac-ui-confirm-v1|{v}".format(v=v), "b16"))
    cfg = V1.BLOCKS["12"]; proto12 = json.loads(Path(cfg["protocol"]).read_text())
    f = pd.read_csv(cfg["flags"], usecols=COLS)
    for v, fr in f[f.victim_eligible.eq(1)].groupby("victim_id", sort=True):
        out.append(("cifar10_scan_lda", v, fr.reset_index(drop=True), proto12, cfg["ns"].format(v=v), "ph"))
    cfg = V1.BLOCKS["8"]; proto8 = json.loads(Path(cfg["protocol"]).read_text())
    f = pd.read_csv(cfg["flags"], usecols=COLS)
    for v, fr in f[f.victim_eligible.eq(1)].groupby("victim_id", sort=True):
        rep = pd.concat([fr.reset_index(drop=True)] * 10, ignore_index=True); rep["sample_id"] = np.arange(len(rep))
        out.append(("tenfold_semantic_family", v, rep, proto8, "ui-scaling-v1|{v}|r10".format(v=v), "sc"))
    return out


def draw(pop, kind, ns, k, d):
    if kind == "ph":
        return V1.draw_strata(pop, ns, k, d, "0.1")
    n1, n0 = H.allocate(pop["N"][1], pop["N"][0], k, "0.1")
    key = (lambda f: (ns, "srac_ui", k, d, f)) if kind == "b16" else (lambda f: (ns, k, d, f))
    return {f: np.random.default_rng(H.stable_seed(*key(f))).choice(pop["idx"][f], size=n, replace=False) for f, n in ((1, n1), (0, n0))}


def measure(recs, ref):
    rows = []
    for r in recs:
        exact = ref(r)
        rows.append(float(abs(mp.mpf(r[-1]) - exact) / exact) if exact > 0 else float("inf"))
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--inst-draws", type=int, default=5); ap.add_argument("--stab-draws", type=int, default=20)
    a = ap.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    core = E._core(); core.hypergeom = HGWrap(core.hypergeom)
    sets = settings(); res = dict(instrumented={}, stability={})
    # ---- 1-2 instrumented runs (first three victims of each setting; plus a stress level)
    for name in ("confirmation", "cifar10_scan_lda", "tenfold_semantic_family", "confirmation_stress"):
        base = "confirmation" if name == "confirmation_stress" else name
        LP.__init__(); HG.__init__(); E.linear_p = lp_wrap; t0 = time.time()
        for s_name, v, frame, proto, ns, kind in [x for x in sets if x[0] == base][:3]:
            c = proto["certificate"]; delta, tau, cap = c["delta_per_decision"], c["recall_target"], c["max_group_clean_flag_rate_cap"]
            pop = H.build_population(frame, proto["protected_groups"]); A = E.n_assertions(pop)
            alpha = delta / (2 * 44) if name.endswith("stress") else delta / (2 * A)
            LP.level = (alpha - 0.05 * alpha); HG.level = 0.05 * alpha / 2
            for k in c["budgets"]:
                for d in range(a.inst_draws):
                    E.decide_ui(pop, draw(pop, kind, ns, k, d), alpha, tau, cap)
        E.linear_p = _orig_lp
        lp_err = measure(LP.near + LP.other, lambda r: mp_linear_p(*r[:10]))
        hg_err = measure(HG.near + HG.other, lambda r: mp_hyper(r[0], *r[1:5]))
        nl = len(LP.near); nh = len(HG.near)
        res["instrumented"][name] = dict(
            linear_p_calls=len(lp_err), linear_p_near_threshold=nl,
            linear_p_max_rel_err=max(lp_err) if lp_err else None,
            linear_p_max_rel_err_near=max(lp_err[:nl]) if nl else None,
            hypergeom_calls=len(hg_err), hypergeom_near_level=nh,
            hypergeom_max_rel_err=max(hg_err) if hg_err else None,
            hypergeom_max_rel_err_near=max(hg_err[:nh]) if nh else None,
            seconds=round(time.time() - t0, 1))
        print(name, res["instrumented"][name], flush=True)
    core.hypergeom = core.hypergeom.base
    # ---- 3 stability
    st16 = pd.read_csv(B16_STORED); st16 = st16[st16.engine == "srac_ui"].set_index(["victim_id", "k", "draw"]).decision
    stph = pd.read_csv(PH_STORED, dtype={"block": str, "share": str})
    stph = stph[(stph.design == "srac_ui") & (stph.share == "0.1") & (stph.block == "12")].set_index(["victim_id", "k", "draw"]).decision
    stsc = pd.read_csv(SC_STORED); stsc = stsc[stsc.replication == 10].set_index(["victim_id", "k", "draw"]).decision
    stored = dict(b16=st16, ph=stph, sc=stsc)
    for s_name in ("confirmation", "cifar10_scan_lda", "tenfold_semantic_family"):
        cnt = dict(decisions=0, stored_mismatch=0, changed_margin_0=0, changed_margin_1e4=0); t0 = time.time()
        for _, v, frame, proto, ns, kind in [x for x in sets if x[0] == s_name]:
            c = proto["certificate"]; delta, tau, cap = c["delta_per_decision"], c["recall_target"], c["max_group_clean_flag_rate_cap"]
            pop = H.build_population(frame, proto["protected_groups"]); alpha = delta / (2 * E.n_assertions(pop))
            for k in c["budgets"]:
                for d in range(min(a.stab_draws, 5) if kind == "sc" else a.stab_draws):
                    ch = draw(pop, kind, ns, k, d)
                    E.MARGIN = 1e-6; base = E.decide_ui(pop, ch, alpha, tau, cap)
                    E.MARGIN = 0.0; d0 = E.decide_ui(pop, ch, alpha, tau, cap)
                    E.MARGIN = 1e-4; d4 = E.decide_ui(pop, ch, alpha, tau, cap)
                    E.MARGIN = 1e-6
                    cnt["decisions"] += 1; cnt["stored_mismatch"] += int(base != stored[kind][(v, k, d)])
                    cnt["changed_margin_0"] += int(d0 != base); cnt["changed_margin_1e4"] += int(d4 != base)
        cnt["seconds"] = round(time.time() - t0, 1); res["stability"][s_name] = cnt
        print(s_name, cnt, flush=True)
    a.output.mkdir(parents=True)
    res.update(status="POST_HOC_CHECK", mp_dps=mp.mp.dps,
               script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               engine_sha256=hashlib.sha256((HERE / "srac_ui_engine_v1_1.py").read_bytes()).hexdigest())
    (a.output / "report.json").write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()

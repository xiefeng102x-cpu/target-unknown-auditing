#!/usr/bin/env python3
"""Direction-specific risk allocation (post hoc; plan GO_ALLOCATION_PLAN_V1.md).

GO needs every GO-direction test to reject, so by the intersection-union principle each
GO-direction test may use level delta/2 (no division by the number of assertions); the NO-GO
direction keeps its union bound.  On the stored samples, decide:

  ui_dir              SRAC-UI v1.1, GO tests at delta/2, NO-GO tests at delta/(2A)
  srac_dir            interval SRAC (share 0.5), GO cells delta/4 per one-sided tail
                      (two cells per assertion), NO-GO cells delta/(2Q) as frozen
  interval_split_dir  interval SRAC on the UI samples, GO as srac_dir, NO-GO onesided_split
  rla_dir             betting audit, GO tests delta/2, NO-GO tests delta/(2A)

and recompute the stored engines (srac, interval_split, rla_bet) as validation gates.

usage: run_go_allocation_v1.py run --parts DIR [--procs 2] [--only 8,16] [--draws 200]
       run_go_allocation_v1.py combine --parts DIR --output DIR
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent; CERT = HERE.parent; ROOT = CERT.parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("ui_v1", HERE / "run_stratified_ui_audit_v1.py")
V1 = importlib.util.module_from_spec(spec); sys.modules["ui_v1"] = V1; spec.loader.exec_module(V1)
BL = V1.BL; H = V1.H
import srac_ui_engine_v1_1 as E  # noqa: E402

B16 = dict(protocol=CERT / "PROTOCOL_GTSRB_SRAC_UI_CONFIRMATION_V1.json",
           flags=ROOT / "results/gtsrb_srac_ui_scan_scores_confirmation_v1/flags.csv",
           ns="gtsrb-srac-ui-confirm-v1|{v}",
           stored=CERT / "results_gtsrb_srac_ui_confirmation_v1_1_erratum/decisions.csv")
MECH = CERT / "results_ui_mechanism_ablation_v1/decisions.csv"
BASE = CERT / "results_audit_baselines_v1/decisions.csv"
DATASET = {"8": "GTSRB", "16": "GTSRB", "11": "CIFAR-10", "12": "CIFAR-10", "13": "CIFAR-10", "14": "CIFAR-10"}
SEED, DRAWS_BOOT = 2026102001, 10000


# ------------------------------------------------------------------ interval SRAC, two etas
def ratio(a, b, empty):
    d = a + b
    return a / d if d else empty


def interval_dir(pop, chosen, eg, en, tau, cap):
    """H.decide with separate per-tail levels for the GO direction (eg) and NO-GO (en).
    eg, en: dict(R=..., G=...).  With eg == en it equals H.decide exactly."""
    poison, M = pop["poison"], pop["memberships"]
    def side(n_pop, idx, et, which):
        iv = H.interval(int(n_pop), int(len(idx)), int(poison[idx].sum()), float(et))
        return iv.lower if which == "L" else iv.upper
    N1, N0 = pop["N"][1], pop["N"][0]
    cL = side(N1, chosen[1], eg["R"], "L"); mU = side(N0, chosen[0], eg["R"], "U")
    cU = side(N1, chosen[1], en["R"], "U"); mL = side(N0, chosen[0], en["R"], "L")
    rec_lo = ratio(cL, mU, 0.0); rec_hi = ratio(cU, mL, 1.0)
    mLg = side(N0, chosen[0], eg["R"], "L") if eg != en else mL
    cLn = side(N1, chosen[1], en["R"], "L") if eg != en else cL
    if cL + mLg == 0:
        rec_lo = 0.0
    if cLn + mL == 0:
        rec_hi = 1.0
    go_groups, nogo_groups = True, False
    for g in range(M.shape[1]):
        f_size, u_size = pop["gsize"][g]
        if f_size == 0:
            continue
        in1 = chosen[1][M[chosen[1], g]]; in0 = chosen[0][M[chosen[0], g]]
        fL = side(f_size, in1, eg["G"], "L"); fU = side(f_size, in1, en["G"], "U")
        if u_size:
            uU = side(u_size, in0, eg["G"], "U"); uL = side(u_size, in0, en["G"], "L")
        else:
            uL = uU = 0
        go_groups &= ratio(f_size - fL, u_size - uU, 1.0) <= cap
        nogo_groups |= ratio(f_size - fU, u_size - uL, 0.0) > cap
    if rec_lo >= tau and go_groups:
        return "GO"
    if rec_hi < tau or nogo_groups:
        return "NO_GO"
    return "ABSTAIN"


# ------------------------------------------------------------------ SRAC-UI, GO direction only
def ui_go(pop, chosen, alpha, tau, cap):
    Z = pop["poison"]; M = pop["memberships"]; s1, s0 = chosen[1], chosen[0]
    go, _ = E.recall_tests(pop["N"][1], len(s1), int(Z[s1].sum()), pop["N"][0], len(s0), int(Z[s0].sum()), tau, alpha)
    if not go:
        return False
    for g in range(M.shape[1]):
        fs, us = pop["gsize"][g]
        if fs == 0:
            continue
        a1 = s1[M[s1, g]]; a0 = s0[M[s0, g]]
        g_go, _ = E.group_tests(fs, len(a1), int((~Z[a1]).sum()), us, len(a0), int((~Z[a0]).sum()), cap, alpha)
        if not g_go:
            return False
    return True


def ui_dir(pop, chosen, delta, tau, cap, stored):
    """GO at delta/2 per assertion; NO-GO direction at delta/(2A) is read from the stored
    decision (stored NO_GO <=> a NO-GO test rejected and not every GO test did; stored GO at
    delta/(2A) implies GO at delta/2, which is checked)."""
    if ui_go(pop, chosen, delta / 2, tau, cap):
        return "GO"
    if stored == "GO":   # cannot happen if the tests are monotone in alpha; decide NO-GO explicitly
        A = E.n_assertions(pop); d = E.decide_ui(pop, chosen, delta / (2 * A), tau, cap)
        return "CHECK_" + d
    return stored


# ------------------------------------------------------------------ betting audit
def rla(asserts, sample, N, a_go, a_ng):
    k = len(sample)
    go_t = max(BL.first_rejection(d[sample], N, m, a_go, "below") for _, d, m in asserts)
    ng_t = min(BL.first_rejection(d[sample], N, m, a_ng, "above") for _, d, m in asserts)
    if ng_t <= k and ng_t < go_t:
        return "NO_GO", int(ng_t)
    if go_t <= k:
        return "GO", int(go_t)
    return "ABSTAIN", k


def etas_go(delta):
    return dict(R=delta / 4, G=delta / 4)


# ------------------------------------------------------------------ per victim
def run_victim(job):
    block, victim, frame, cfg, draws, stored = job
    proto = json.loads(Path(cfg["protocol"]).read_text()); c = proto["certificate"]
    delta, tau, cap = c["delta_per_decision"], c["recall_target"], c["max_group_clean_flag_rate_cap"]
    pop = H.build_population(frame, proto["protected_groups"]); q = pop["q"]
    q_used = 2 + sum(int(fs > 0) + int(fs > 0 and us > 0) for fs, us in pop["gsize"])
    e_frozen = dict(R=delta / (2 * q), G=delta / (2 * q)); e_split = H.etas("onesided_split", delta / 2, q_used)
    N = len(pop["poison"]); asserts = BL.assertions(pop, tau, cap); A = len(asserts)
    tp = int(pop["recall"] >= tau and pop["max_clean"] <= cap)
    ns = cfg["ns"].format(v=victim); rows = []
    for k in c["budgets"]:
        for draw in range(draws):
            r = dict(block=block, victim_id=victim, k=k, draw=draw, true_pass=tp)
            if block == "16":
                n1, n0 = H.allocate(pop["N"][1], pop["N"][0], k, "0.1")
                ch_ui = {f: np.random.default_rng(H.stable_seed(ns, "srac_ui", k, draw, f)).choice(pop["idx"][f], size=n, replace=False)
                         for f, n in ((1, n1), (0, n0))}
                n1, n0 = H.allocate(pop["N"][1], pop["N"][0], k, "0.5")
                ch_s = {f: np.random.default_rng(H.stable_seed(ns, "srac", k, draw, f)).choice(pop["idx"][f], size=n, replace=False)
                        for f, n in ((1, n1), (0, n0))}
                uni = np.random.default_rng(H.stable_seed(ns, "rla_bet", k, draw)).choice(N, size=k, replace=False)
            else:
                ch_ui = V1.draw_strata(pop, ns, k, draw, "0.1")
                n1, n0 = H.allocate(pop["N"][1], pop["N"][0], k, "0.5")
                ch_s = {f: np.random.default_rng(H.stable_seed(ns, k, draw, f)).choice(pop["idx"][f], size=n, replace=False)
                        for f, n in ((1, n1), (0, n0))}
                uni = np.random.default_rng(H.stable_seed("audit-baselines-v1", ns, k, draw)).choice(N, size=k, replace=False)
                r["interval_split"] = interval_dir(pop, ch_ui, e_split, e_split, tau, cap)
                r["interval_split_dir"] = interval_dir(pop, ch_ui, etas_go(delta), e_split, tau, cap)
            st = stored[(k, draw)]
            r["ui_full_stored"] = st["ui"]
            r["ui_dir"] = ui_dir(pop, ch_ui, delta, tau, cap, st["ui"])
            r["srac"] = interval_dir(pop, ch_s, e_frozen, e_frozen, tau, cap)
            r["srac_dir"] = interval_dir(pop, ch_s, etas_go(delta), e_frozen, tau, cap)
            r["rla_bet"], r["rla_bet_labels"] = rla(asserts, uni, N, delta / (2 * A), delta / (2 * A))
            r["rla_dir"], r["rla_dir_labels"] = rla(asserts, uni, N, delta / 2, delta / (2 * A))
            r["srac_stored"], r["rla_bet_stored"] = st["srac"], st["rla_bet"]
            if block != "16":
                r["interval_split_stored"] = st["interval_split"]
            rows.append(r)
    return pd.DataFrame(rows)


def stored_maps():
    out = {}
    m = pd.read_csv(MECH, dtype={"block": str})
    b = pd.read_csv(BASE, dtype={"block": str})
    for (blk, v), g in m.groupby(["block", "victim_id"]):
        if blk == "16":
            continue
        bb = b[(b.block == blk) & (b.victim_id == v)]
        bs = bb[bb.design == "srac"].set_index(["k", "draw"]).decision
        br = bb[bb.design == "rla_bet"].set_index(["k", "draw"]).decision
        out[(blk, v)] = {(int(r.k), int(r.draw)): dict(ui=r.ui_full, interval_split=r.interval_split,
                                                     srac=bs[(r.k, r.draw)], rla_bet=br[(r.k, r.draw)])
                         for r in g.itertuples()}
    s = pd.read_csv(B16["stored"])
    for v, g in s.groupby("victim_id"):
        piv = g.pivot_table(index=["k", "draw"], columns="engine", values="decision", aggfunc="first")
        out[("16", v)] = {(int(k), int(dr)): dict(ui=row["srac_ui"], srac=row["srac"], rla_bet=row["rla_bet"])
                          for (k, dr), row in piv.iterrows()}
    return out


def jobs(only, draws):
    sm = stored_maps(); out = []
    for block, cfg in V1.BLOCKS.items():
        if only and block not in only:
            continue
        f = pd.read_csv(cfg["flags"], usecols=["victim_id", "seed", "sample_id", "assigned_group", "flag", "is_poison", "victim_eligible"])
        for v, fr in f[f.victim_eligible.eq(1)].groupby("victim_id", sort=True):
            out.append((block, v, fr.reset_index(drop=True), {k: str(x) for k, x in cfg.items()}, draws, sm[(block, v)]))
    if not only or "16" in only:
        f = pd.read_csv(B16["flags"]); f = f[f.victim_eligible.eq(1)]
        for v, fr in f.groupby("victim_id", sort=True):
            out.append(("16", v, fr.reset_index(drop=True)[["victim_id", "seed", "sample_id", "assigned_group", "flag", "is_poison", "victim_eligible"]],
                        dict(protocol=str(B16["protocol"]), ns=B16["ns"]), draws, sm[("16", v)]))
    return out


# ------------------------------------------------------------------ combine
def cluster_boot(scr, cluster, seed=SEED):
    keys = list(dict.fromkeys(cluster))
    groups = [scr[cluster == k] for k in keys]
    rng = np.random.default_rng(seed); out = np.empty(DRAWS_BOOT)
    for i in range(DRAWS_BOOT):
        pick = rng.integers(0, len(groups), len(groups))
        out[i] = np.concatenate([groups[j] for j in pick]).mean()
    return [float(np.quantile(out, .025)), float(np.quantile(out, .975))]


ENGINES = ("ui_full_stored", "ui_dir", "srac", "srac_dir", "interval_split", "interval_split_dir", "rla_bet", "rla_dir")
CONTRASTS = (("ui_dir", "ui_full_stored"), ("srac_dir", "srac"), ("interval_split_dir", "interval_split"),
             ("rla_dir", "rla_bet"), ("ui_dir", "srac_dir"), ("ui_dir", "rla_dir"), ("ui_full_stored", "srac"),
             ("ui_full_stored", "rla_bet"))


def combine(parts: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    d = pd.concat([pd.read_csv(p, dtype={"block": str}) for p in sorted(parts.glob("*.csv"))], ignore_index=True)
    gates = {}
    for e in ("srac", "rla_bet", "interval_split"):
        sub = d[d[f"{e}_stored"].notna()] if f"{e}_stored" in d else d.iloc[0:0]
        gates[e] = dict(compared=int(len(sub)), mismatches=int((sub[e] != sub[f"{e}_stored"]).sum()))
    gates["ui_dir_check"] = int(d.ui_dir.astype(str).str.startswith("CHECK_").sum())
    if any(v["mismatches"] for k, v in gates.items() if isinstance(v, dict)) or gates["ui_dir_check"]:
        raise RuntimeError(f"validation gate failed: {gates}")
    d["dataset"] = d.block.map(DATASET)
    d["set"] = np.where(d.block == "16", "confirmation", "posthoc_" + d.dataset)
    d["model"] = d.dataset + ":" + d.victim_id.str.extract(r"seed(\d+)")[0]
    output.mkdir(parents=True)
    d.to_csv(output / "decisions.csv", index=False)
    summ, cons = [], []
    for (st, k), g in d.groupby(["set", "k"]):
        p, f = g[g.true_pass == 1], g[g.true_pass == 0]
        row = dict(set=st, k=int(k), passing_screens=int(p.victim_id.nunique()), failing_screens=int(f.victim_id.nunique()))
        for e in ENGINES:
            if g[e].isna().all():
                continue
            row[f"{e}_go"] = float(p[e].eq("GO").mean()) if len(p) else None
            row[f"{e}_nogo"] = float(f[e].eq("NO_GO").mean()) if len(f) else None
            row[f"{e}_false"] = int(((g[e] == "GO") & (g.true_pass == 0)).sum() + ((g[e] == "NO_GO") & (g.true_pass == 1)).sum())
        row["rla_bet_labels_passing"] = float(p.rla_bet_labels.mean()) if len(p) else None
        row["rla_dir_labels_passing"] = float(p.rla_dir_labels.mean()) if len(p) else None
        summ.append(row)
        for a, b in CONTRASTS:
            if g[a].isna().all() or g[b].isna().all():
                continue
            for direction, lab, sub in (("GO", "GO", p), ("NO_GO", "NO_GO", f)):
                if sub.empty:
                    continue
                x = sub[a].eq(lab).astype(int) - sub[b].eq(lab).astype(int)
                scr = x.groupby(sub.victim_id).mean()
                cl = sub.groupby("victim_id").model.first().loc[scr.index].to_numpy()
                cons.append(dict(set=st, k=int(k), direction=direction, contrast=f"{a} - {b}", screens=int(len(scr)),
                                 models=int(len(set(cl))), mean=float(scr.mean()), ci95=cluster_boot(scr.to_numpy(), cl)))
    rep = dict(status="POST_HOC_DEVELOPMENTAL", gates=gates, rows=int(len(d)), summary=summ, contrasts=cons,
               engine_sha256=hashlib.sha256((HERE / "srac_ui_engine_v1_1.py").read_bytes()).hexdigest(),
               script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               decisions_sha256=hashlib.sha256((output / "decisions.csv").read_bytes()).hexdigest())
    (output / "report.json").write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps(gates)); print(pd.DataFrame(cons).query("k == 2000").to_string())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=("run", "combine"))
    ap.add_argument("--parts", type=Path, required=True); ap.add_argument("--output", type=Path)
    ap.add_argument("--procs", type=int, default=2); ap.add_argument("--only", default=""); ap.add_argument("--draws", type=int, default=200)
    a = ap.parse_args()
    if a.cmd == "combine":
        return combine(a.parts, a.output)
    a.parts.mkdir(parents=True, exist_ok=True)
    only = [b for b in a.only.split(",") if b]
    todo = [j for j in jobs(only, a.draws) if not (a.parts / f"{j[0]}__{j[1]}.csv").exists()]
    print(f"{len(todo)} victims to compute", flush=True)
    with Pool(a.procs) as p:
        for df in p.imap_unordered(run_victim, todo):
            df.to_csv(a.parts / f"{df.block.iloc[0]}__{df.victim_id.iloc[0]}.csv", index=False)
            print("part done", df.block.iloc[0], df.victim_id.iloc[0], flush=True)


if __name__ == "__main__":
    main()

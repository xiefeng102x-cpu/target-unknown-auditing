#!/usr/bin/env python3
"""Directional label-error budgets (post hoc; plan DIRECTIONAL_BUDGET_PLAN_V1.md).

usage: run_directional_budget_v1.py run --parts DIR [--procs 2]
       run_directional_budget_v1.py combine --parts DIR --output DIR
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import binom, hypergeom

HERE = Path(__file__).resolve().parent; CERT = HERE.parent; ROOT = CERT.parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("ui_v1", HERE / "run_stratified_ui_audit_v1.py")
V1 = importlib.util.module_from_spec(spec); sys.modules["ui_v1"] = V1; spec.loader.exec_module(V1)
H = V1.H
import srac_ui_directional_v1 as D  # noqa: E402

PROTOCOL = CERT / "PROTOCOL_GTSRB_SRAC_UI_CONFIRMATION_V1.json"
FLAGS = ROOT / "results/gtsrb_srac_ui_scan_scores_confirmation_v1/flags.csv"
CHANNEL = CERT / "results_reference_channel_v1/channel_labels.csv"
NS = "gtsrb-srac-ui-confirm-v1|{v}"
COLS = ["victim_id", "seed", "sample_id", "assigned_group", "flag", "is_poison", "victim_eligible"]
DRAWS, DELTA_ADJ = 200, 0.001
F0 = (0.0, 0.001, 0.003, 0.01, 0.03); H1 = (0.0, 0.01, 0.035); G0S = (0.0, 0.0003); P1MISS = 0.011


def q(n, p):
    return int(binom.ppf(1 - DELTA_ADJ / 4, n, p)) if p > 0 and n > 0 else 0


def run_victim(job):
    v, frame = job
    proto = json.loads(PROTOCOL.read_text()); c = proto["certificate"]
    delta, tau, cap = c["delta_per_decision"], c["recall_target"], c["max_group_clean_flag_rate_cap"]
    pop = H.build_population(frame, proto["protected_groups"]); Z = pop["poison"]; fl = pop["flags"].astype(bool)
    obs = frame.channel_poison.to_numpy().astype(bool)
    W1 = int((fl & ~Z & obs).sum()); W0 = int((~fl & Z & ~obs).sum())
    C1, M0 = int((fl & ~Z).sum()), int((~fl & Z).sum()); P1 = int((fl & Z).sum()); C0 = int((~fl & ~Z).sum())
    tp = int(pop["recall"] >= tau and pop["max_clean"] <= cap); a_go = delta / 2; rows = []
    for k in c["budgets"]:
        n1, n0 = H.allocate(pop["N"][1], pop["N"][0], k, "0.1")
        for draw in range(DRAWS):
            ch = {f: np.random.default_rng(H.stable_seed(NS.format(v=v), "srac_ui", k, draw, f)).choice(pop["idx"][f], size=n, replace=False)
                  for f, n in ((1, n1), (0, n0))}
            base = dict(victim_id=v, k=k, draw=draw, true_pass=tp)
            rows.append(dict(base, setting="A_master", f0=None, h1=None, g0=None, B1=0, B0=0, err1=0, err0=0,
                             decision=D.decide_go(pop, ch, Z, a_go, tau, cap, 0, 0)))
            b1 = int(hypergeom.ppf(1 - DELTA_ADJ / 4, pop["N"][1], W1, n1)); b0 = int(hypergeom.ppf(1 - DELTA_ADJ / 4, pop["N"][0], W0, n0))
            e1 = int((~Z[ch[1]] & obs[ch[1]]).sum()); e0 = int((Z[ch[0]] & ~obs[ch[0]]).sum())
            rows.append(dict(base, setting="B_channel_oracle", f0=None, h1=None, g0=None, B1=b1, B0=b0, err1=e1, err0=e0,
                             decision=D.decide_go(pop, ch, obs, a_go, tau, cap, b1, b0)))
            for f0, h1, G0 in ((a, b, g) for a in F0 for b in H1 for g in G0S):
                if True:
                    rng = np.random.default_rng(H.stable_seed("directional-v1", v, k, draw, f0, h1, G0))
                    lab = Z.copy()
                    s1, s0 = ch[1], ch[0]
                    u1 = rng.random(len(s1)); u0 = rng.random(len(s0))
                    p_c1 = min(1.0, h1 * pop["N"][1] / max(C1, 1)); p_p1 = min(1.0, P1MISS * pop["N"][1] / max(P1, 1))
                    p_p0 = min(1.0, G0 * pop["N"][0] / max(M0, 1)); p_c0 = min(1.0, f0 * pop["N"][0] / max(C0, 1))
                    flip1 = np.where(Z[s1], u1 < p_p1, u1 < p_c1); flip0 = np.where(Z[s0], u0 < p_p0, u0 < p_c0)
                    lab[s1] = np.where(flip1, ~Z[s1], Z[s1]); lab[s0] = np.where(flip0, ~Z[s0], Z[s0])
                    e1 = int((~Z[s1] & flip1).sum()); e0 = int((Z[s0] & flip0).sum())
                    B1, B0 = q(n1, h1), q(n0, G0)
                    rows.append(dict(base, setting="C_synthetic", f0=f0, h1=h1, g0=G0, B1=B1, B0=B0, err1=e1, err0=e0,
                                     decision=D.decide_go(pop, ch, lab, a_go, tau, cap, B1, B0)))
    return pd.DataFrame(rows)


def jobs():
    lab = pd.read_csv(CHANNEL)
    f = pd.read_csv(FLAGS, usecols=COLS); f = f[f.victim_eligible.eq(1)]
    f = f.merge(lab[["victim_id", "sample_id", "channel_poison"]], on=["victim_id", "sample_id"], how="left", validate="one_to_one")
    return [(v, g.reset_index(drop=True)) for v, g in f.groupby("victim_id", sort=True)]


def combine(parts, output):
    if output.exists():
        raise FileExistsError(output)
    d = pd.concat([pd.read_csv(p) for p in sorted(parts.glob("*.csv"))], ignore_index=True)
    output.mkdir(parents=True); d.to_csv(output / "decisions.csv", index=False)
    keys = ["setting", "f0", "h1", "g0", "k"]
    d[["f0", "h1", "g0"]] = d[["f0", "h1", "g0"]].fillna(-1)
    summ = []
    for key, g in d.groupby(keys):
        p, fl = g[g.true_pass == 1], g[g.true_pass == 0]
        summ.append(dict(zip(keys, [key[0]] + [None if x < 0 else x for x in key[1:4]] + [int(key[4])]),
                         audits=int(len(g)), go_passing=float(p.decision.eq("GO").mean()),
                         false_go=int(fl.decision.eq("GO").sum()), failing_audits=int(len(fl)),
                         B1_mean=float(g.B1.mean()), B0_mean=float(g.B0.mean()),
                         budget_covered=float(((g.err1 <= g.B1) & (g.err0 <= g.B0)).mean())))
    rep = dict(status="POST_HOC_DEVELOPMENTAL", summary=summ, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               directional_sha256=hashlib.sha256((HERE / "srac_ui_directional_v1.py").read_bytes()).hexdigest(),
               decisions_sha256=hashlib.sha256((output / "decisions.csv").read_bytes()).hexdigest())
    (output / "report.json").write_text(json.dumps(rep, indent=2) + "\n")
    print(pd.DataFrame(summ).to_string())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=("run", "combine"))
    ap.add_argument("--parts", type=Path, required=True); ap.add_argument("--output", type=Path); ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    if a.cmd == "combine":
        return combine(a.parts, a.output)
    a.parts.mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs() if not (a.parts / f"{j[0]}.csv").exists()]
    with Pool(a.procs) as p:
        for df in p.imap_unordered(run_victim, todo):
            df.to_csv(a.parts / f"{df.victim_id.iloc[0]}.csv", index=False); print("done", df.victim_id.iloc[0], flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Post-hoc analyses B and C of plan UI_REDTEAM_SCALING_PLAN_V1.md.

B (scaling).  Each block-8 victim's frozen population (flags, labels, classes) is replicated
r = 10 times, which keeps the poison prevalence, the recall, and every group's clean flag rate
exactly and multiplies N by 10 (about 2.5 x 10^5 records).  SRAC-UI v1.1 (flagged share 0.1,
alpha = delta/(2A)) decides 40 fresh protected draws per budget k in {2000, 8000, 16000}
(namespace "ui-scaling-v1").  The same 40 draws (indices) are decided on the original population
(r = 1) for a matched comparison.  Question: does the GO rate at fixed k stay roughly
constant when N grows 10-fold, i.e. does the adjudicated fraction k/N fall as 1/N?

C (exact paired tests, block 16).  For the nine policy-passing block-16 screens, the
per-screen GO-rate differences of SRAC-UI over interval SRAC and over the betting audit at
k = 2000 (corrected decisions) are tested with an exact one-sided sign test and an exact
one-sided paired permutation (sign-flip) test of the mean.

usage: run_ui_scaling_v1.py run --output DIR [--procs 2]
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, itertools, json, sys
from math import comb
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent; CERT = HERE.parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("ui_v1", HERE / "run_stratified_ui_audit_v1.py")
V1 = importlib.util.module_from_spec(spec); sys.modules["ui_v1"] = V1; spec.loader.exec_module(V1)
import srac_ui_engine_v1_1 as E  # noqa: E402
H = V1.H
REPLICATION, DRAWS, BUDGETS, NS = 10, 40, (2000, 8000, 16000), "ui-scaling-v1|{v}|r{r}"
B16 = CERT / "results_gtsrb_srac_ui_confirmation_v1_1_erratum/decisions.csv"


def replicate(frame, r):
    out = pd.concat([frame] * r, ignore_index=True)
    out["sample_id"] = np.arange(len(out))
    return out


def run_victim(args):
    victim, frame, cfg = args
    proto = json.loads(Path(cfg["protocol"]).read_text()); c = proto["certificate"]
    delta, tau, cap = c["delta_per_decision"], c["recall_target"], c["max_group_clean_flag_rate_cap"]
    rows = []
    for r in (1, REPLICATION):
        pop = H.build_population(replicate(frame, r), proto["protected_groups"])
        alpha = delta / (2 * E.n_assertions(pop)); tp = int(pop["recall"] >= tau and pop["max_clean"] <= cap)
        ns = NS.format(v=victim, r=r); N = len(pop["poison"])
        for k in BUDGETS:
            n1, n0 = H.allocate(pop["N"][1], pop["N"][0], k, "0.1")
            for draw in range(DRAWS):
                ch = {f: np.random.default_rng(H.stable_seed(ns, k, draw, f)).choice(pop["idx"][f], size=n, replace=False)
                      for f, n in ((1, n1), (0, n0))}
                rows.append(dict(victim_id=victim, replication=r, N=N, k=k, draw=draw, true_pass=tp,
                                 recall=pop["recall"], max_clean=pop["max_clean"], decision=E.decide_ui(pop, ch, alpha, tau, cap)))
    return pd.DataFrame(rows)


def exact_tests(diffs):
    d = np.asarray(diffs, dtype=float); n = len(d)
    pos, neg = int((d > 0).sum()), int((d < 0).sum()); m = pos + neg
    sign_p = sum(comb(m, j) for j in range(pos, m + 1)) / 2 ** m if m else 1.0
    obs = d.mean(); cnt = 0
    for s in itertools.product((1, -1), repeat=n):
        cnt += (np.dot(s, d) / n) >= obs - 1e-12
    return dict(screens=n, positive=pos, negative=neg, mean=float(obs), sign_test_p=float(sign_p), permutation_p=cnt / 2 ** n)


def part_c():
    d = pd.read_csv(B16); p = d[(d.true_pass == 1) & (d.k == 2000)]
    per = p.groupby(["victim_id", "engine"]).decision.apply(lambda x: float((x == "GO").mean())).unstack("engine")
    return {ref: exact_tests((per.srac_ui - per[ref]).to_numpy()) for ref in ("srac", "rla_bet")}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["run"]); ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--procs", type=int, default=2); a = ap.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    cfg = V1.BLOCKS["8"]
    f = pd.read_csv(cfg["flags"], usecols=["victim_id", "seed", "sample_id", "assigned_group", "flag", "is_poison", "victim_eligible"])
    jobs = [(v, fr.reset_index(drop=True), {k: str(x) for k, x in cfg.items()}) for v, fr in f[f.victim_eligible.eq(1)].groupby("victim_id", sort=True)]
    with Pool(a.procs) as p:
        d = pd.concat(p.imap_unordered(run_victim, jobs), ignore_index=True)
    d = d.sort_values(["victim_id", "replication", "k", "draw"]).reset_index(drop=True)
    pas, fail = d[d.true_pass == 1], d[d.true_pass == 0]
    summ = {}
    for r, g in d.groupby("replication"):
        gp, gf = g[g.true_pass == 1], g[g.true_pass == 0]
        summ[int(r)] = dict(mean_N=float(g.groupby("victim_id").N.first().mean()),
                            go_passing_by_k={int(k): float((gp[gp.k == k].decision == "GO").mean()) for k in BUDGETS},
                            nogo_failing_by_k={int(k): float((gf[gf.k == k].decision == "NO_GO").mean()) for k in BUDGETS},
                            adjudicated_fraction_by_k={int(k): float(k / g.groupby("victim_id").N.first().mean()) for k in BUDGETS},
                            false_go=int(((g.decision == "GO") & (g.true_pass == 0)).sum()),
                            false_nogo=int(((g.decision == "NO_GO") & (g.true_pass == 1)).sum()))
    a.output.mkdir(parents=True)
    d.to_csv(a.output / "decisions.csv", index=False)
    rep = dict(status="POST_HOC_DEVELOPMENTAL", scaling=summ, block16_exact_tests_k2000=part_c(),
               passing_screens=int(pas.victim_id.nunique()), failing_screens=int(fail.victim_id.nunique()),
               script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               decisions_sha256=hashlib.sha256((a.output / "decisions.csv").read_bytes()).hexdigest())
    (a.output / "report.json").write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()

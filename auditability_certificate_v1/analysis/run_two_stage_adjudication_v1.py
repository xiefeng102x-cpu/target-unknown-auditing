#!/usr/bin/env python3
"""Two-stage adjudication simulation (plan TWO_STAGE_ADJUDICATION_PLAN_V1.md).

usage: run_two_stage_adjudication_v1.py run --parts DIR [--procs 2]
       run_two_stage_adjudication_v1.py combine --parts DIR --output DIR
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import binom

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_directional_budget_v1 as R  # noqa: E402  (H, D, PROTOCOL, NS, jobs)

H, D = R.H, R.D
DRAWS, DELTA_ADJ = 200, 0.001
R_FC, R_FP, R_UP = 0.05, 0.01, 0.01
FS = (0.01, 0.02, 0.05); QS = (0.02, 0.05, 0.1, 0.25, 1.0); MS = (0.0, 0.02)


def q_hi(n, p):
    return int(binom.ppf(1 - DELTA_ADJ / 4, n, p)) if p > 0 and n > 0 else 0


def run_victim(job):
    v, frame = job
    proto = json.loads(R.PROTOCOL.read_text()); c = proto["certificate"]
    delta, tau, cap = c["delta_per_decision"], c["recall_target"], c["max_group_clean_flag_rate_cap"]
    pop = H.build_population(frame, proto["protected_groups"]); Z = pop["poison"]; fl = pop["flags"].astype(bool)
    N1, N0 = pop["N"][1], pop["N"][0]; C1, M0 = int((fl & ~Z).sum()), int((~fl & Z).sum())
    tp = int(pop["recall"] >= tau and pop["max_clean"] <= cap); rows = []
    for k in c["budgets"]:
        n1, n0 = H.allocate(N1, N0, k, "0.1")
        for draw in range(DRAWS):
            ch = {f: np.random.default_rng(H.stable_seed(R.NS.format(v=v), "srac_ui", k, draw, f)).choice(pop["idx"][f], size=n, replace=False)
                  for f, n in ((1, n1), (0, n0))}
            s1, s0 = ch[1], ch[0]; z1, z0 = Z[s1], Z[s0]
            for f in FS:
                for qq in QS:
                    for m in MS:
                        rng = np.random.default_rng(H.stable_seed("two-stage-v1", v, k, draw, f, qq, m))
                        u1, u0, u2 = rng.random(len(s1)), rng.random(len(s0)), rng.random(len(s0))
                        call1 = np.where(z1, u1 >= R_FP, u1 < R_FC)                    # True = poison call
                        stage1 = np.where(z0, u0 >= R_UP, u0 < f)
                        keep = np.where(z0, u2 >= m, u2 < qq)                           # stage 2 agrees
                        call0 = stage1 & keep
                        lab = Z.copy(); lab[s1] = call1; lab[s0] = call0
                        e1 = int((~z1 & call1).sum()); e0 = int((z0 & ~call0).sum())
                        B1 = q_hi(n1, R_FC * C1 / N1); B0 = q_hi(n0, (R_UP + (1 - R_UP) * m) * M0 / N0)
                        rows.append(dict(victim_id=v, k=k, draw=draw, true_pass=tp, f=f, q=qq, m=m, B1=B1, B0=B0,
                                         err1=e1, err0=e0, stage2=int(stage1.sum()), final_fa=int((~z0 & call0).sum()),
                                         decision=D.decide_go(pop, ch, lab, delta / 2, tau, cap, B1, B0)))
    return pd.DataFrame(rows)


def combine(parts, output):
    if output.exists():
        raise FileExistsError(output)
    d = pd.concat([pd.read_csv(p) for p in sorted(parts.glob("*.csv"))], ignore_index=True)
    output.mkdir(parents=True); d.to_csv(output / "decisions.csv", index=False)
    summ = []
    for (f, qq, m, k), g in d.groupby(["f", "q", "m", "k"]):
        p, fl = g[g.true_pass == 1], g[g.true_pass == 0]
        summ.append(dict(f=f, q=qq, m=m, k=int(k), go_passing=float(p.decision.eq("GO").mean()),
                         false_go=int(fl.decision.eq("GO").sum()), failing_audits=int(len(fl)),
                         B1_mean=float(g.B1.mean()), B0_mean=float(g.B0.mean()),
                         budget_covered=float(((g.err1 <= g.B1) & (g.err0 <= g.B0)).mean()),
                         stage2_share_of_k=float((g.stage2 / g.k).mean())))
    rep = dict(status="POST_HOC_SIMULATION", summary=summ, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
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
    todo = [j for j in R.jobs() if not (a.parts / f"{j[0]}.csv").exists()]
    with Pool(a.procs) as p:
        for df in p.imap_unordered(run_victim, todo):
            df.to_csv(a.parts / f"{df.victim_id.iloc[0]}.csv", index=False); print("done", df.victim_id.iloc[0], flush=True)


if __name__ == "__main__":
    main()

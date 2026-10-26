#!/usr/bin/env python3
"""Automated adjudication channel (plan REFERENCE_CHANNEL_PLAN_V1.md), part 2: SRAC-UI on the
stored confirmation draws with the channel's labels in place of the master labels.

  naive    B = 0
  defender B_k = Binomial(k, p_u) upper delta_adj/2 quantile, p_u = one-sided
           (1 - delta_adj/2) Clopper-Pearson bound of the held-out clean error rate
  oracle   sum over strata of the upper delta_adj/4 hypergeometric quantile of wrong labels in
           the stratum sample, given the true number of wrong labels in the stratum

usage: run_reference_channel_certificates_v1.py --channel DIR --output DIR [--procs 2]
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import beta, binom, hypergeom

HERE = Path(__file__).resolve().parent; CERT = HERE.parent; ROOT = CERT.parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("ui_v1", HERE / "run_stratified_ui_audit_v1.py")
V1 = importlib.util.module_from_spec(spec); sys.modules["ui_v1"] = V1; spec.loader.exec_module(V1)
H = V1.H
import srac_ui_engine_v1_1 as E  # noqa: E402
import srac_ui_robust_v1 as UR  # noqa: E402

PROTOCOL = CERT / "PROTOCOL_GTSRB_SRAC_UI_CONFIRMATION_V1.json"
FLAGS = ROOT / "results/gtsrb_srac_ui_scan_scores_confirmation_v1/flags.csv"
STORED = CERT / "results_gtsrb_srac_ui_confirmation_v1_1_erratum/decisions.csv"
NS = "gtsrb-srac-ui-confirm-v1|{v}"
DELTA_ADJ = 0.001
# Amendment 1 (runtime, before any output): declared-B decisions with B in the hundreds to
# thousands take up to ~80 s each, so all engines use draws 0-19 and the oracle-B engine
# draws 0-4 of every budget.
DRAWS, ORACLE_DRAWS = 20, 5
# Amendment 3 (runtime): at k = 16000 some oracle-B decisions (B of about 4,000) did not finish
# within two executions of two hours; the oracle-B engine is reported at k = 16000 only for the
# units completed when the second execution was stopped (listed in the amendment record), and
# the remaining units are decided by the other engines without it.
ORACLE_SKIP_K = 16000
COLS = ["victim_id", "seed", "sample_id", "assigned_group", "flag", "is_poison", "victim_eligible"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def setup(v, frame, p_u):
    proto = json.loads(PROTOCOL.read_text()); c = proto["certificate"]
    delta, tau, cap = c["delta_per_decision"], c["recall_target"], c["max_group_clean_flag_rate_cap"]
    pop = H.build_population(frame, proto["protected_groups"]); A = E.n_assertions(pop); alpha = delta / (2 * A)
    obs = frame.channel_poison.to_numpy().astype(bool)
    wrong = obs != pop["poison"]
    W = {f: int(wrong[pop["idx"][f]].sum()) for f in (1, 0)}
    tp = int(pop["recall"] >= tau and pop["max_clean"] <= cap)
    return c, tau, cap, pop, alpha, obs, wrong, W, tp


def run_unit(job):
    """Amendment 2 (execution only): one (victim, k, draw) per unit, saved as a part file,
    so that an interrupted execution resumes; the computation is unchanged."""
    v, frame, p_u, k, draw, part = job
    c, tau, cap, pop, alpha, obs, wrong, W, tp = setup(v, frame, p_u)
    ns = NS.format(v=v)
    n1, n0 = H.allocate(pop["N"][1], pop["N"][0], k, "0.1")
    b_def = int(binom.ppf(1 - DELTA_ADJ / 2, k, p_u))
    b_or = int(sum(hypergeom.ppf(1 - DELTA_ADJ / 4, pop["N"][f], W[f], n) for f, n in ((1, n1), (0, n0))))
    ch = {f: np.random.default_rng(H.stable_seed(ns, "srac_ui", k, draw, f)).choice(pop["idx"][f], size=n, replace=False)
          for f, n in ((1, n1), (0, n0))}
    w_s = int(wrong[ch[1]].sum() + wrong[ch[0]].sum())
    row = dict(victim_id=v, k=k, draw=draw, true_pass=tp, wrong_in_sample=w_s, B_defender=b_def, B_oracle=b_or,
               master=E.decide_ui(pop, ch, alpha, tau, cap),
               naive=UR.decide(pop, ch, obs, alpha, tau, cap, 0),
               defender=UR.decide(pop, ch, obs, alpha, tau, cap, b_def),
               oracle=UR.decide(pop, ch, obs, alpha, tau, cap, b_or) if draw < ORACLE_DRAWS and k != ORACLE_SKIP_K else "")
    pd.DataFrame([row]).to_csv(part, index=False)
    return part


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--channel", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True); ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--parts", type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    crep = json.loads((a.channel / "report.json").read_text())
    lab = pd.read_csv(a.channel / "channel_labels.csv")
    if sha(a.channel / "channel_labels.csv") != crep["channel_labels_sha256"]:
        raise RuntimeError("channel labels changed")
    x, n = crep["heldout_clean_errors"], crep["trusted_heldout"]
    p_u = 1.0 if x == n else float(beta.ppf(1 - DELTA_ADJ / 2, x + 1, n - x))
    f = pd.read_csv(FLAGS, usecols=COLS); f = f[f.victim_eligible.eq(1)]
    f = f.merge(lab[["victim_id", "sample_id", "channel_poison"]], on=["victim_id", "sample_id"], how="left", validate="one_to_one")
    if f.channel_poison.isna().any():
        raise RuntimeError("missing channel labels")
    a.parts.mkdir(parents=True, exist_ok=True)
    budgets = json.loads(PROTOCOL.read_text())["certificate"]["budgets"]
    jobs = [(v, g.reset_index(drop=True), p_u, k, dr, a.parts / f"{v}__{k}__{dr}.csv")
            for v, g in f.groupby("victim_id", sort=True) for k in budgets for dr in range(DRAWS)]
    todo = sorted([j for j in jobs if not j[5].exists()], key=lambda j: (j[4] >= ORACLE_DRAWS, j[3]))
    print(f"{len(todo)} of {len(jobs)} units to compute", flush=True)
    with Pool(a.procs) as p:
        for done in p.imap_unordered(run_unit, todo):
            print("done", done.name, flush=True)
    d = pd.concat([pd.read_csv(j[5], keep_default_na=False) for j in jobs], ignore_index=True)
    st = pd.read_csv(STORED); st = st[st.engine == "srac_ui"][["victim_id", "k", "draw", "decision"]]
    m = d.merge(st, on=["victim_id", "k", "draw"])
    gate = int((m.master != m.decision).sum())
    if gate or len(m) != len(d):  # stored decisions cover draws 0-199; d covers draws 0-19
        raise RuntimeError(f"master-label decisions do not reproduce the stored ones: {gate}")
    a.output.mkdir(parents=True); d.to_csv(a.output / "decisions.csv", index=False)
    summ = []
    for k, g in d.groupby("k"):
        p, fl = g[g.true_pass == 1], g[g.true_pass == 0]
        r = dict(k=int(k), B_defender=int(g.B_defender.iloc[0]), B_oracle_range=[int(g.B_oracle.min()), int(g.B_oracle.max())],
                 wrong_in_sample_mean=float(g.wrong_in_sample.mean()), wrong_in_sample_max=int(g.wrong_in_sample.max()))
        for e in ("master", "naive", "defender", "oracle"):
            pe, fe = p[p[e].isin(["GO", "NO_GO", "ABSTAIN"])], fl[fl[e].isin(["GO", "NO_GO", "ABSTAIN"])]
            r[f"{e}_audits"] = int(len(pe) + len(fe))
            r[f"{e}_go_passing"] = float(pe[e].eq("GO").mean()); r[f"{e}_nogo_failing"] = float(fe[e].eq("NO_GO").mean())
            r[f"{e}_abstain"] = float(pd.concat([pe[e], fe[e]]).eq("ABSTAIN").mean())
            r[f"{e}_false_go"] = int(fe[e].eq("GO").sum()); r[f"{e}_false_nogo"] = int(pe[e].eq("NO_GO").sum())
        r["oracle_B_covers_sample"] = float((g.wrong_in_sample <= g.B_oracle).mean())
        r["defender_B_covers_sample"] = float((g.wrong_in_sample <= g.B_defender).mean())
        summ.append(r)
    rep = dict(status="POST_HOC_PILOT", delta_adj=DELTA_ADJ, heldout_clean_error_upper=p_u, summary=summ,
               channel_labels_sha256=crep["channel_labels_sha256"], script_sha256=sha(__file__),
               robust_sha256=sha(HERE / "srac_ui_robust_v1.py"), decisions_sha256=sha(a.output / "decisions.csv"))
    (a.output / "report.json").write_text(json.dumps(rep, indent=2) + "\n"); print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()

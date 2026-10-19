#!/usr/bin/env python3
"""Post-hoc ablation on block 16: how much of the SRAC-UI gain is the allocation and how much the test.

Developmental (post hoc; the block-16 outcomes were known).  Plan: BLOCK16_SAME_SHARE_ABLATION_PLAN_V1.md.
On exactly the samples that SRAC-UI used in block 16 (flagged share 0.1; same namespace and
stable_seed keys), decide with the interval engine of SRAC under three error allocations:
  frozen    eta = delta / (2 Q), Q = 2 + all non-empty cells (the frozen rule)
  used      eta = delta / (2 Q_used), Q_used counts only cells that enter a decision
  split     one-sided intervals with delta/2 per error direction over the used cells
            (H.etas("onesided_split", delta/2, Q_used)), matching SRAC-UI's guarantee
            (false GO <= delta/2 and false NO-GO <= delta/2)
and compare with the SRAC-UI v1.1 decisions (results_gtsrb_srac_ui_confirmation_v1_1_erratum).
Victim-level GO-rate differences on passing screens, 95% percentile bootstrap over victims
(10,000 resamples, the frozen block-16 seed); NO-GO rates on the failing screen.
"""
from __future__ import annotations
import hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent; CERT = HERE.parent; ROOT = CERT.parent
spec = importlib.util.spec_from_file_location("ui_v1", HERE / "run_stratified_ui_audit_v1.py")
V1 = importlib.util.module_from_spec(spec); sys.modules["ui_v1"] = V1; spec.loader.exec_module(V1)
H = V1.H
PROTOCOL = CERT / "PROTOCOL_GTSRB_SRAC_UI_CONFIRMATION_V1.json"
FLAGS = ROOT / "results/gtsrb_srac_ui_scan_scores_confirmation_v1/flags.csv"
UI = CERT / "results_gtsrb_srac_ui_confirmation_v1_1_erratum/decisions.csv"
OUT = CERT / "results_block16_same_share_ablation_v1"
NS = "gtsrb-srac-ui-confirm-v1|{v}"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    proto = json.loads(PROTOCOL.read_text()); c = proto["certificate"]
    delta, tau, cap = c["delta_per_decision"], c["recall_target"], c["max_group_clean_flag_rate_cap"]
    flags = pd.read_csv(FLAGS); flags = flags[flags.victim_eligible.eq(1)]
    rows = []
    for v, fr in flags.groupby("victim_id", sort=True):
        pop = H.build_population(fr.reset_index(drop=True), proto["protected_groups"])
        q_used = 2 + sum(int(fs > 0) + int(fs > 0 and us > 0) for fs, us in pop["gsize"])
        e_frozen = dict(R=delta / (2 * pop["q"]), G=delta / (2 * pop["q"]))
        e_used = dict(R=delta / (2 * q_used), G=delta / (2 * q_used))
        e_split = H.etas("onesided_split", delta / 2, q_used)
        ns = NS.format(v=v); tp = int(pop["recall"] >= tau and pop["max_clean"] <= cap)
        for k in c["budgets"]:
            n1, n0 = H.allocate(pop["N"][1], pop["N"][0], k, proto["srac_ui"]["flagged_share"])
            for draw in range(c["protected_draws_per_seed_budget"]):
                ch = {f: np.random.default_rng(H.stable_seed(ns, "srac_ui", k, draw, f)).choice(pop["idx"][f], size=n, replace=False)
                      for f, n in ((1, n1), (0, n0))}
                r = dict(victim_id=v, k=k, draw=draw, true_pass=tp, q=pop["q"], q_used=q_used)
                for name, e in (("frozen", e_frozen), ("used", e_used), ("split", e_split)):
                    r[f"interval_{name}"], *_ = H.decide(pop, ch, e, tau, cap)
                rows.append(r)
    d = pd.DataFrame(rows)
    ui = pd.read_csv(UI); ui = ui[ui.engine == "srac_ui"][["victim_id", "k", "draw", "decision"]].rename(columns={"decision": "srac_ui"})
    d = d.merge(ui, on=["victim_id", "k", "draw"], how="left")
    if d.srac_ui.isna().any():
        raise RuntimeError("missing UI decisions")
    seed, draws = int(proto["analysis"]["bootstrap_seed"]), int(proto["analysis"]["bootstrap_draws"])
    res = []
    for k, g in d.groupby("k"):
        p = g[g.true_pass == 1]; f = g[g.true_pass == 0]
        row = dict(k=int(k), srac_ui_go=float(p.srac_ui.eq("GO").mean()), srac_ui_nogo_failing=float(f.srac_ui.eq("NO_GO").mean()))
        for name in ("frozen", "used", "split"):
            col = f"interval_{name}"
            diff = (p.srac_ui.eq("GO").astype(int) - p[col].eq("GO").astype(int)).groupby(p.victim_id).mean().to_numpy()
            rng = np.random.default_rng(seed)
            bm = diff[rng.integers(0, len(diff), size=(draws, len(diff)))].mean(axis=1)
            row[f"{name}_go"] = float(p[col].eq("GO").mean())
            row[f"{name}_nogo_failing"] = float(f[col].eq("NO_GO").mean())
            row[f"ui_minus_{name}"] = float(diff.mean())
            row[f"ui_minus_{name}_ci95"] = [float(np.quantile(bm, .025)), float(np.quantile(bm, .975))]
        row["false_decisions"] = int(sum(((g[c_] == "GO") & (g.true_pass == 0)).sum() + ((g[c_] == "NO_GO") & (g.true_pass == 1)).sum()
                                         for c_ in ("interval_frozen", "interval_used", "interval_split", "srac_ui")))
        res.append(row)
    OUT.mkdir(parents=True)
    d.to_csv(OUT / "decisions.csv", index=False)
    rep = dict(status="POST_HOC_DEVELOPMENTAL", results=res, ui_decisions_sha256=sha(UI), script_sha256=sha(__file__),
               decisions_sha256=sha(OUT / "decisions.csv"))
    (OUT / "report.json").write_text(json.dumps(rep, indent=2) + "\n"); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

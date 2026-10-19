#!/usr/bin/env python3
"""Statistical-unit correction for the SRAC-UI mechanism analysis (post hoc).

The frozen combine step of run_ui_mechanism_ablation_v1.py bootstraps over screens
(victim_id).  Blocks 13 and 14 apply two screens to the same ten checkpoints, so on CIFAR-10
the 21 failing screens come from 14 physical models.  This script leaves the stored decisions
untouched and recomputes every contrast with a cluster bootstrap over physical models
(training seed), keeping the screen-weighted point estimate.  It adds the matched contrast
ui_nobb - suite (neither uses Berger-Boos) and evaluates ui_full - ui_nobb on the same
draws (block 16, draws 0-19).  10,000 resamples, seed 2026101816 (as before).

usage: recompute_mechanism_cluster_ci_v1.py  (writes results_ui_mechanism_ablation_v1_clustered/)
"""
import hashlib, json
from pathlib import Path
import numpy as np, pandas as pd

CERT = Path(__file__).resolve().parents[1]
SRC = CERT / "results_ui_mechanism_ablation_v1/decisions.csv"
OUT = CERT / "results_ui_mechanism_ablation_v1_clustered"
SEED, DRAWS = 2026101816, 10000
CONTRASTS = (("ui_split", "interval_split"), ("ui_full", "ui_split"), ("ui_full", "ui_nobb"),
             ("ui_nobb", "suite"), ("ui_full", "suite"), ("ui_full", "interval_split"))


def cluster_boot(scr: pd.Series, cluster: pd.Series):
    keys = list(dict.fromkeys(cluster.to_numpy()))
    groups = [scr[cluster == k].to_numpy() for k in keys]
    rng = np.random.default_rng(SEED); out = np.empty(DRAWS)
    for i in range(DRAWS):
        pick = rng.integers(0, len(groups), len(groups))
        out[i] = np.concatenate([groups[j] for j in pick]).mean()
    return [float(np.quantile(out, .025)), float(np.quantile(out, .975))]


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    d = pd.read_csv(SRC, dtype={"block": str})
    d["model"] = d.dataset + ":" + d.victim_id.str.extract(r"seed(\d+)")[0]
    rows = []
    for (st, k), g in d.groupby(["set", "k"]):
        for direction, lab, tp in (("GO", "GO", 1), ("NO_GO", "NO_GO", 0)):
            sub = g[g.true_pass == tp]
            for a, b in CONTRASTS:
                s = sub
                if "ui_nobb" in (a, b):
                    s = s[s.ui_nobb.isin(["GO", "NO_GO", "ABSTAIN"])]
                if s.empty:
                    continue
                x = s[a].eq(lab).astype(int) - s[b].eq(lab).astype(int)
                scr = x.groupby(s.victim_id).mean()
                cl = s.groupby("victim_id").model.first().loc[scr.index]
                rows.append(dict(set=st, k=int(k), direction=direction, contrast=f"{a} - {b}", screens=int(len(scr)),
                                 models=int(cl.nunique()), audits=int(len(s)), mean=float(scr.mean()),
                                 ci95=cluster_boot(scr, cl)))
    OUT.mkdir(parents=True)
    rep = dict(status="POST_HOC_STATISTICAL_UNIT_CORRECTION", contrasts=rows,
               source_decisions_sha256=hashlib.sha256(SRC.read_bytes()).hexdigest(),
               script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT / "report.json").write_text(json.dumps(rep, indent=2) + "\n")
    print(pd.DataFrame(rows).query("k == 2000").to_string())


if __name__ == "__main__":
    main()

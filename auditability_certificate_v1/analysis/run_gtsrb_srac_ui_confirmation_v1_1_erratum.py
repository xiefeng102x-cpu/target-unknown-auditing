#!/usr/bin/env python3
"""Block 16 erratum: recompute the srac_ui decisions with engine v1.1 on the same draws.

The frozen block-16 protocol, scores, endpoint script and outputs
(results_gtsrb_srac_ui_confirmation_v1/) are not modified.  This script regenerates the
srac_ui samples exactly as the frozen endpoint script did (same namespace, same
stable_seed keys, same allocation), decides them with analysis/srac_ui_engine_v1_1.py,
keeps the frozen srac and rla_bet rows, and recomputes the pre-declared endpoints
(V, P1, P2) and secondary contrasts with the frozen bootstrap seed.  v1 and v1.1 are
reported side by side.

Usage: --parts DIR [--victims a,b]  (compute; resumable per victim)
       --parts DIR --combine OUTDIR  (assemble; OUTDIR must not exist)
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent; CERT = HERE.parent; ROOT = CERT.parent
PROTOCOL = CERT / "PROTOCOL_GTSRB_SRAC_UI_CONFIRMATION_V1.json"
SCORES = ROOT / "results/gtsrb_srac_ui_scan_scores_confirmation_v1"
FROZEN = CERT / "results_gtsrb_srac_ui_confirmation_v1"
NS = "gtsrb-srac-ui-confirm-v1|{v}"
sys.path.insert(0, str(HERE))
import srac_ui_engine_v1_1 as E  # noqa: E402

spec = importlib.util.spec_from_file_location("ui_v1", HERE / "run_stratified_ui_audit_v1.py")
V1 = importlib.util.module_from_spec(spec); sys.modules["ui_v1"] = V1; spec.loader.exec_module(V1)
H = V1.H


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run_victim(victim, frame, proto):
    c = proto["certificate"]; delta, tau, cap = c["delta_per_decision"], c["recall_target"], c["max_group_clean_flag_rate_cap"]
    pop = H.build_population(frame, proto["protected_groups"])
    alpha = delta / (2 * E.n_assertions(pop)); ns = NS.format(v=victim); share = proto["srac_ui"]["flagged_share"]
    rows = []
    for k in c["budgets"]:
        n1, n0 = H.allocate(pop["N"][1], pop["N"][0], k, share)
        for draw in range(c["protected_draws_per_seed_budget"]):
            ch = {f: np.random.default_rng(H.stable_seed(ns, "srac_ui", k, draw, f)).choice(pop["idx"][f], size=n, replace=False)
                  for f, n in ((1, n1), (0, n0))}
            rows.append(dict(victim_id=victim, engine="srac_ui", k=k, draw=draw, decision=E.decide_ui(pop, ch, alpha, tau, cap)))
    return pd.DataFrame(rows)


def boot(diffs, seed, draws):
    rng = np.random.default_rng(seed)
    m = diffs[rng.integers(0, len(diffs), size=(draws, len(diffs)))].mean(axis=1)
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def endpoints(out, proto):
    a_cfg = proto["analysis"]; contrasts = {}
    passing = out[out.true_pass == 1]
    per = passing.groupby(["victim_id", "engine", "k"]).decision.apply(lambda x: float((x == "GO").mean())).unstack("engine")
    for ref in ("srac", "rla_bet"):
        for k in proto["certificate"]["budgets"]:
            d = (per.xs(k, level="k")["srac_ui"] - per.xs(k, level="k")[ref]).to_numpy()
            ci = boot(d, int(a_cfg["bootstrap_seed"]), int(a_cfg["bootstrap_draws"]))
            contrasts[f"srac_ui_minus_{ref}_k{k}"] = dict(victims=int(len(d)), mean_go_difference=float(d.mean()), ci95=ci, supported=bool(ci[0] > 0))
    ui = out[out.engine == "srac_ui"]
    ep = {"V_validity": bool(ui.false_go.sum() == 0 and ui.false_nogo.sum() == 0),
          "P1_srac_ui_vs_srac_k2000": contrasts["srac_ui_minus_srac_k2000"],
          "P2_srac_ui_vs_rla_k2000": contrasts["srac_ui_minus_rla_bet_k2000"]}
    return ep, contrasts


def summarize(out):
    return out.groupby(["engine", "true_pass", "k"]).agg(n=("decision", "size"), go=("decision", lambda x: float((x == "GO").mean())),
        nogo=("decision", lambda x: float((x == "NO_GO").mean())), false_go=("false_go", "sum"), false_nogo=("false_nogo", "sum"),
        mean_labels=("labels", "mean")).reset_index()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--parts", type=Path, required=True)
    ap.add_argument("--victims", default=""); ap.add_argument("--combine", type=Path, default=None); a = ap.parse_args()
    proto = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if PROTOCOL.with_suffix(".json.sha256").read_text().split()[0] != sha(PROTOCOL):
        raise RuntimeError("protocol sidecar mismatch")
    frozen = pd.read_csv(FROZEN / "decisions.csv")
    rep_v1 = json.loads((FROZEN / "report.json").read_text())
    if sha(FROZEN / "decisions.csv") != rep_v1["sha256"]["decisions.csv"]:
        raise RuntimeError("frozen decisions hash mismatch")
    a.parts.mkdir(parents=True, exist_ok=True)
    if a.combine is None:
        flags = pd.read_csv(SCORES / "flags.csv"); flags = flags[flags.victim_eligible.eq(1)]
        want = [v for v in a.victims.split(",") if v]
        for v, fr in flags.groupby("victim_id", sort=True):
            p = a.parts / f"{v}.csv"
            if (want and v not in want) or p.exists():
                continue
            run_victim(v, fr.reset_index(drop=True)[["victim_id", "seed", "sample_id", "assigned_group", "flag", "is_poison", "victim_eligible"]], proto).to_csv(p, index=False)
            print("part done", v, flush=True)
        return
    if a.combine.exists():
        raise FileExistsError(a.combine)
    new_ui = pd.concat([pd.read_csv(a.parts / f"{v}.csv") for v in sorted(frozen.victim_id.unique())], ignore_index=True)
    key = ["victim_id", "engine", "k", "draw"]
    out = frozen.merge(new_ui.rename(columns={"decision": "decision_v11"}), on=key, how="left")
    if out[out.engine == "srac_ui"].decision_v11.isna().any():
        raise RuntimeError("missing v1.1 decisions")
    out["decision_v1"] = out.decision
    out.loc[out.engine == "srac_ui", "decision"] = out.loc[out.engine == "srac_ui", "decision_v11"]
    out = out.drop(columns=["decision_v11"])
    out["false_go"] = ((out.decision == "GO") & (out.true_pass == 0)).astype(int)
    out["false_nogo"] = ((out.decision == "NO_GO") & (out.true_pass == 1)).astype(int)
    changes = out[out.decision != out.decision_v1]
    ep, contrasts = endpoints(out, proto)
    a.combine.mkdir(parents=True)
    out.drop(columns=["decision_v1"]).to_csv(a.combine / "decisions.csv", index=False)
    changes.to_csv(a.combine / "changes_vs_v1.csv", index=False)
    summarize(out).to_csv(a.combine / "summary.csv", index=False)
    rep = dict(status="COMPLETE_ERRATUM", engine=E.ENGINE_VERSION, created_utc=datetime.now(timezone.utc).isoformat(),
               protocol_sha256=sha(PROTOCOL), frozen_decisions_sha256=sha(FROZEN / "decisions.csv"),
               changed_rows=int(len(changes)),
               transitions=changes.groupby(["k", "decision_v1", "decision"]).size().rename("rows").reset_index().to_dict("records"),
               endpoints_v1=rep_v1["endpoints"], endpoints_v1_1=ep, contrasts_v1_1=contrasts,
               false_decisions=out.groupby("engine")[["false_go", "false_nogo"]].sum().to_dict("index"),
               engine_sha256=sha(HERE / "srac_ui_engine_v1_1.py"), script_sha256=sha(__file__),
               sha256={p.name: sha(p) for p in sorted(a.combine.glob("*.csv"))})
    (a.combine / "report.json").write_text(json.dumps(rep, indent=2) + "\n"); print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()

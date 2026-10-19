#!/usr/bin/env python3
"""Block 16 endpoint: prospective confirmation of the flag-stratified union-intersection
audit (SRAC-UI) against the frozen SRAC engine and the uniform betting comparison RLA.

Frozen before the block-16 victims (GTSRB seeds 255-264) were trained.  Same group-safe
object as block 8 (recall >= 0.80; clean flag rate <= 0.10 in every one of the ten
semantic sets), delta = 0.005 per decision, budgets 2000/8000/16000, 200 draws per victim
and budget, fresh draw namespace "gtsrb-srac-ui-confirm-v1|<victim>".

Engines (code frozen by hash):
  srac_ui   flagged-stratum share 0.1, exact UI test of each linear assertion, Berger-Boos
            gamma = 0.05 alpha, alpha = delta/(2A)        (run_stratified_ui_audit_v1.py)
  srac      frozen SRAC: share 0.5, exact two-sided intervals, eta = delta/(2Q)
  rla_bet   uniform sample, SHANGRLA-style comparison assertions, WoR betting tests,
            anytime stopping                              (run_audit_baselines_v1.py)
Each engine draws its own sample from the namespace; draws are independent across engines.

Primary endpoints (pre-declared):
  V  validity: zero false GO and zero false NO-GO for srac_ui (6,000 decisions).
  P1 power vs srac: victim-level mean GO rate difference srac_ui - srac on policy-passing
     victims at k = 2000; supported if the 95% percentile bootstrap lower bound over victims
     (10,000 resamples, seed 2026101816) is above zero.
  P2 power vs rla_bet: the same contrast srac_ui - rla_bet at k = 2000.
Secondary (descriptive): both contrasts at k = 8000 and 16000; NO-GO rates on failing
victims; mean labels used by rla_bet.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]; CERT = ROOT / "auditability_certificate_v1"
PROTOCOL = CERT / "PROTOCOL_GTSRB_SRAC_UI_CONFIRMATION_V1.json"
SCORES = ROOT / "results/gtsrb_srac_ui_scan_scores_confirmation_v1"
OUTPUT = CERT / "results_gtsrb_srac_ui_confirmation_v1"
UI_SCRIPT = CERT / "analysis/run_stratified_ui_audit_v1.py"
NS = "gtsrb-srac-ui-confirm-v1|{v}"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m


def run_victim(args):
    victim, frame, proto = args
    U = load(UI_SCRIPT, "srac_ui_engine"); BL = U.BL; H = U.H
    c = proto["certificate"]; delta, tau, cap = c["delta_per_decision"], c["recall_target"], c["max_group_clean_flag_rate_cap"]
    pop = H.build_population(frame, proto["protected_groups"]); q = pop["q"]; eta = delta / (2 * q)
    A = U.n_assertions(pop); alpha = delta / (2 * A); N = len(pop["poison"]); asserts = BL.assertions(pop, tau, cap)
    true_pass = int(pop["recall"] >= tau and pop["max_clean"] <= cap)
    ns = NS.format(v=victim); share = proto["srac_ui"]["flagged_share"]; rows = []
    for k in c["budgets"]:
        for draw in range(c["protected_draws_per_seed_budget"]):
            ch_ui = {}; n1, n0 = H.allocate(pop["N"][1], pop["N"][0], k, share)
            for f, n in ((1, n1), (0, n0)):
                ch_ui[f] = np.random.default_rng(H.stable_seed(ns, "srac_ui", k, draw, f)).choice(pop["idx"][f], size=n, replace=False)
            ch_s = {}; n1, n0 = H.allocate(pop["N"][1], pop["N"][0], k, "0.5")
            for f, n in ((1, n1), (0, n0)):
                ch_s[f] = np.random.default_rng(H.stable_seed(ns, "srac", k, draw, f)).choice(pop["idx"][f], size=n, replace=False)
            uni = np.random.default_rng(H.stable_seed(ns, "rla_bet", k, draw)).choice(N, size=k, replace=False)
            d_ui = U.decide_ui(pop, ch_ui, alpha, tau, cap)
            d_s, *_ = H.decide(pop, ch_s, dict(R=eta, G=eta), tau, cap)
            d_r, used = BL.rla_bet(asserts, uni, N, delta)
            for eng, dec, lab in (("srac_ui", d_ui, k), ("srac", d_s, k), ("rla_bet", d_r, used)):
                rows.append(dict(victim_id=victim, engine=eng, k=k, draw=draw, decision=dec, labels=lab, true_pass=true_pass,
                                 recall=pop["recall"], max_clean=pop["max_clean"]))
    return pd.DataFrame(rows)


def boot_lower(diffs: np.ndarray, seed: int, draws: int) -> list[float]:
    rng = np.random.default_rng(seed)
    m = diffs[rng.integers(0, len(diffs), size=(draws, len(diffs)))].mean(axis=1)
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=4); a = ap.parse_args()
    proto = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if proto.get("status") != "FROZEN_BEFORE_VICTIM_TRAINING":
        raise RuntimeError("protocol is not frozen")
    if PROTOCOL.with_suffix(".json.sha256").read_text().split()[0] != sha256(PROTOCOL):
        raise RuntimeError("protocol sidecar mismatch")
    for rel, digest in proto["hashes"]["endpoint_stage"].items():
        if sha256(ROOT / rel) != digest:
            raise RuntimeError(f"endpoint-stage hash mismatch: {rel}")
    man = json.loads((SCORES / "manifest.json").read_text())
    if man.get("protocol_sha256") != sha256(PROTOCOL) or man.get("flags_sha256") != sha256(SCORES / "flags.csv"):
        raise RuntimeError("score handoff mismatch")
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    flags = pd.read_csv(SCORES / "flags.csv"); flags = flags[flags.victim_eligible.eq(1)]
    jobs = [(v, fr.reset_index(drop=True)[["victim_id", "seed", "sample_id", "assigned_group", "flag", "is_poison", "victim_eligible"]], proto)
            for v, fr in flags.groupby("victim_id", sort=True)]
    from multiprocessing import Pool
    with Pool(a.procs) as p:
        out = pd.concat(p.imap_unordered(run_victim, jobs), ignore_index=True)
    out = out.sort_values(["victim_id", "engine", "k", "draw"]).reset_index(drop=True)
    out["false_go"] = ((out.decision == "GO") & (out.true_pass == 0)).astype(int)
    out["false_nogo"] = ((out.decision == "NO_GO") & (out.true_pass == 1)).astype(int)
    OUTPUT.mkdir(parents=True)
    out.to_csv(OUTPUT / "decisions.csv", index=False)
    summ = out.groupby(["engine", "true_pass", "k"]).agg(n=("decision", "size"), go=("decision", lambda x: float((x == "GO").mean())),
        nogo=("decision", lambda x: float((x == "NO_GO").mean())), false_go=("false_go", "sum"), false_nogo=("false_nogo", "sum"),
        mean_labels=("labels", "mean")).reset_index()
    summ.to_csv(OUTPUT / "summary.csv", index=False)
    a_cfg = proto["analysis"]; contrasts = {}
    passing = out[out.true_pass == 1]
    per = passing.groupby(["victim_id", "engine", "k"]).decision.apply(lambda x: float((x == "GO").mean())).unstack("engine")
    for ref in ("srac", "rla_bet"):
        for k in proto["certificate"]["budgets"]:
            if passing.empty:
                contrasts[f"srac_ui_minus_{ref}_k{k}"] = None; continue
            d = (per.xs(k, level="k")["srac_ui"] - per.xs(k, level="k")[ref]).to_numpy()
            ci = boot_lower(d, int(a_cfg["bootstrap_seed"]), int(a_cfg["bootstrap_draws"]))
            contrasts[f"srac_ui_minus_{ref}_k{k}"] = dict(victims=int(len(d)), mean_go_difference=float(d.mean()), ci95=ci,
                                                         supported=bool(ci[0] > 0))
    ui = out[out.engine == "srac_ui"]
    endpoints = {"V_validity": bool(ui.false_go.sum() == 0 and ui.false_nogo.sum() == 0),
                 "P1_srac_ui_vs_srac_k2000": contrasts.get("srac_ui_minus_srac_k2000"),
                 "P2_srac_ui_vs_rla_k2000": contrasts.get("srac_ui_minus_rla_bet_k2000")}
    truth = out.groupby("victim_id")[["recall", "max_clean", "true_pass"]].first().reset_index()
    truth.to_csv(OUTPUT / "truth_by_victim.csv", index=False)
    rep = dict(status="COMPLETE", created_utc=datetime.now(timezone.utc).isoformat(), protocol_sha256=sha256(PROTOCOL),
               flags_sha256=sha256(SCORES / "flags.csv"), eligible_victims=int(truth.shape[0]), passing_victims=int(truth.true_pass.sum()),
               endpoints=endpoints, contrasts=contrasts,
               false_decisions=out.groupby("engine")[["false_go", "false_nogo"]].sum().to_dict("index"),
               sha256={p.name: sha256(p) for p in sorted(OUTPUT.glob("*.csv"))})
    (OUTPUT / "report.json").write_text(json.dumps(rep, indent=2) + "\n")
    print(summ.to_string(index=False)); print(json.dumps(endpoints, indent=2))


if __name__ == "__main__":
    main()

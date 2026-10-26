#!/usr/bin/env python3
"""Write reference SRAC-UI certificate records (aggregated counts -> expected decision).

Each record holds only the counts the certificate reads (no images, labels or record ids):
stratum sizes, sample sizes, sampled poisons, and for every protected group the stratum
sizes and sampled clean counts, plus delta, tau, cap and the decision of engine v1.1, which
is also checked against the stored SRAC-UI confirmation decision.

usage: build_certificate_records_v1.py OUT.json
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent; CERT = HERE.parent
sys.path.insert(0, str(HERE))
import run_directional_budget_v1 as R  # noqa: E402  (H, PROTOCOL, NS, jobs)
import srac_ui_engine_v1_1 as E  # noqa: E402

PICKS = [("gtsrb_semantic_overlap_seed255", 2000, 0), ("gtsrb_semantic_overlap_seed255", 2000, 1),
         ("gtsrb_semantic_overlap_seed255", 8000, 0)]
STORED = CERT / "results_gtsrb_srac_ui_confirmation_v1_1_erratum/decisions.csv"


def main():
    out = Path(sys.argv[1]); H = R.H
    proto = json.loads(R.PROTOCOL.read_text()); c = proto["certificate"]
    delta, tau, cap = c["delta_per_decision"], c["recall_target"], c["max_group_clean_flag_rate_cap"]
    stored = pd.read_csv(STORED); stored = stored[stored.engine == "srac_ui"]
    jobs = dict(R.jobs()); fail_v = None; recs = []
    for v, frame in jobs.items():
        pop = H.build_population(frame, proto["protected_groups"])
        if not (pop["recall"] >= tau and pop["max_clean"] <= cap):
            fail_v = v
    picks = PICKS + [(fail_v, 16000, 0)]
    for v, k, draw in picks:
        frame = jobs[v]; pop = H.build_population(frame, proto["protected_groups"]); Z = pop["poison"]; M = pop["memberships"]
        n1, n0 = H.allocate(pop["N"][1], pop["N"][0], k, "0.1")
        ch = {f: np.random.default_rng(H.stable_seed(R.NS.format(v=v), "srac_ui", k, draw, f)).choice(pop["idx"][f], size=n, replace=False)
              for f, n in ((1, n1), (0, n0))}
        s1, s0 = ch[1], ch[0]
        A = 1 + sum(1 for fs, _ in pop["gsize"] if fs > 0); alpha = delta / (2 * A)
        rgo, rng_ = E.recall_tests(pop["N"][1], n1, int(Z[s1].sum()), pop["N"][0], n0, int(Z[s0].sum()), tau, alpha)
        groups = []; go, ng = rgo, rng_
        for g, gdef in enumerate(proto["protected_groups"]):
            fs, us = pop["gsize"][g]
            if fs == 0:
                continue
            a1 = s1[M[s1, g]]; a0 = s0[M[s0, g]]
            y1, y0 = int((~Z[a1]).sum()), int((~Z[a0]).sum())
            g_go, g_ng = E.group_tests(fs, len(a1), y1, us, len(a0), y0, cap, alpha)
            groups.append(dict(name=gdef["name"], flagged_size=int(fs), unflagged_size=int(us), flagged_sampled=int(len(a1)),
                               flagged_sampled_clean=y1, unflagged_sampled=int(len(a0)), unflagged_sampled_clean=y0,
                               expected_go_test=bool(g_go), expected_nogo_test=bool(g_ng)))
            go &= g_go; ng |= g_ng
        dec = "GO" if go else ("NO_GO" if ng else "ABSTAIN")
        st = stored[(stored.victim_id == v) & (stored.k == k) & (stored.draw == draw)].decision
        if len(st) != 1 or st.iloc[0] != dec:
            raise RuntimeError(f"stored decision mismatch {v} {k} {draw}: {list(st)} vs {dec}")
        recs.append(dict(study="GTSRB SRAC-UI confirmation", population=v, budget=k, draw=draw, delta=delta, tau=tau, cap=cap,
                         assertions=A, alpha_per_test=alpha, flagged_size=int(pop["N"][1]), unflagged_size=int(pop["N"][0]),
                         flagged_sampled=int(n1), unflagged_sampled=int(n0), flagged_sampled_poisons=int(Z[s1].sum()),
                         unflagged_sampled_poisons=int(Z[s0].sum()), expected_recall_go_test=bool(rgo),
                         expected_recall_nogo_test=bool(rng_), groups=groups, expected_decision=dec))
    doc = dict(schema="srac-ui-certificate-records-v1", engine="analysis/srac_ui_engine_v1_1.py",
               engine_sha256=hashlib.sha256((HERE / "srac_ui_engine_v1_1.py").read_bytes()).hexdigest(), records=recs)
    out.write_text(json.dumps(doc, indent=2) + "\n")
    print([(r["population"][-3:], r["budget"], r["draw"], r["expected_decision"]) for r in recs])


if __name__ == "__main__":
    main()

# Corrective protocol: uniform whole-population adjudication V1

Date frozen: 2026-10-13 (Asia/Shanghai), before running the corrective audit
draws reported by this protocol. The victim models, score tables, binary detector
flags, and private truth files are immutable artifacts from earlier blocks. This
is therefore a prospective audit-draw replay, not prospective victim training.

## Reason for the corrective protocol

The earlier clean-member protocol required a registry equal to the complete
clean component of the audit universe. In a binary clean/poison universe, that
requirement makes the complement identifiable as poison and weakens the reason
for a separate recall certificate. It also excludes unknown clean records from
the finite-population proof. This protocol removes that assumption.

## Audit rule

1. Freeze the audit universe, detector scores, and binary screening flags.
2. Draw `k` records uniformly without replacement from the entire audit
   universe, using randomness unavailable when the score table was fixed.
3. Have an independent adjudicator, blinded to detector scores and flags,
   classify only the sampled records as poison or clean.
4. For sampled poison records, bound the population recall of the fixed flag
   rule. For sampled clean records, bound its clean flag rate. Conditional on
   the sampled class counts, each class-specific subset is uniform without
   replacement from its fixed finite population.
5. Use simultaneous Hoeffding-without-replacement bounds. Return GO only when
   the recall lower bound is at least 0.80 and the clean flag-rate upper bound is
   at most 0.10. Return NO-GO when the recall upper bound is below 0.80;
   otherwise ABSTAIN.

No complete clean registry and no poison-count interval are inputs. The primary
replay assumes correct sampled-record adjudication (`B=0`). For a deployment with
labeling error, the manuscript gives worst-case bounds for at most `B` wrong
sample labels and composes the statistical, adjudication, and protected-draw
failure budgets. That extension is theoretical; it is not empirically validated
by the current replay.

## Frozen evaluation

- Blocks: PathMNIST/ResNet-18, PathMNIST/ResNet-50,
  PathMNIST/adaptive-multiscore seeds 3201--3205,
  PathMNIST/legacy-certificate-aware seeds 5301--5305, and
  CIFAR-10/VGG-16-BN; plus a GTSRB/TinyCNN confirmation with seeds 101--110.
  The 5301--5305 block targeted the superseded certificate, not the revised SRAC
  protocol. The GTSRB victim-generation protocol was frozen before training, but
  SRAC was applied to its fixed flags later.
- Detectors: five frozen binary score rules in each of the first five blocks;
  three in GTSRB.
- Budgets: `k in {120, 500, 2000}`.
- Replays: 200 draws per victim, fold, and budget. The same adjudicated draw is
  reused across all fixed detectors available in that block.
- Per-decision error: 0.005. This allocates 0.05 across five detectors by two
  folds in the original blocks and a conservative 0.015 across GTSRB's three
  one-fold detectors. Within a decision, three one-sided tails share the budget.
- Every block and detector is retained. Both false GO and false NO-GO are
  reported, along with GO/NO-GO/ABSTAIN.

The code is `analysis/run_uniform_adjudication_certificate.py`. Source and
output SHA-256 values are written to `results_uniform_adjudication_v1/report.json`.
The GTSRB full-population adapter is
`analysis/prepare_gtsrb_uniform_adjudication.py`; its reconstruction receipt is
`derived_inputs/gtsrb_tinycnn_prefrozen_v1/adapter_receipt.json`.

## Post-audit numerical correction (2026-10-13)

An independent audit found that the first synthetic and exact-boundary scripts
used `delta=0.05`, whereas the frozen-victim decisions use `delta=0.005` after
the five-detector by two-fold allocation. Both diagnostic scripts were rerun at
`delta=0.005`. Their current CSV files supersede the earlier values.

At `tau=0.80`, `alpha_cap=0.10`, and `delta=0.005`, a GO is structurally
impossible unless the audit contains at least 80 poison records and 320 clean
records. Hence `k=120` can never return GO, even for a perfect detector. This is
a policy-and-bound feasibility limit, not an empirical detector failure.

Synthetic cells with `k/N > 0.10` are retained for diagnostics but are marked
`sparse_regime=false` and do not support the sparse-sampling rate-matching claim.
The exact 27-cell boundary audit now reports maximum probabilities
`7.707291268044294e-08` for false GO and
`4.5523956398764605e-05` for false NO-GO. These selected cells test the
implementation; the theorem, not the grid, supplies the uniform guarantee.

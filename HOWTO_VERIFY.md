# Verify this code-only repository

Use Python3.10 or3.11 from the repository root.

```bash
python -m pip install -r requirements-audit.txt
python -B src/verify_srac_release_20261019.py --root .
```

A complete PASS requires manifest integrity, no manuscript/PDF/private payload,
the required statistical regression tests, both corrected UI reference suites, the
public-checkout endpoint smoke tests, Fisher self-tests, and unchanged integrity after
testing. It also requires the robust and directional self-tests and the synthetic
count-to-decision fixtures. This does not validate omitted
experimental result tables, real adjudication channels, or model training.

An integrity-only check is explicitly partial:

```bash
python -B src/verify_srac_release_20261019.py --root . --integrity-only
```

Individual data-independent checks:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 python -B auditability_certificate_v1/analysis/test_srac_ui_engine_v1_1.py
PYTHONDONTWRITEBYTECODE=1 python -B auditability_certificate_v1/analysis/test_srac_ui_engine_v1_1_suite2.py
PYTHONDONTWRITEBYTECODE=1 python -B auditability_certificate_v1/analysis/run_ui_mechanism_ablation_v1.py selftest
PYTHONDONTWRITEBYTECODE=1 python -B auditability_certificate_v1/analysis/srac_ui_robust_v1.py
PYTHONDONTWRITEBYTECODE=1 python -B auditability_certificate_v1/analysis/srac_ui_directional_v1.py
PYTHONDONTWRITEBYTECODE=1 python -B auditability_certificate_v1/analysis/replay_synthetic_certificate_records_v1.py
```

The default pytest selection contains only tests that do not require private
experiment outputs. Other experiment tests and runner entry points require their
documented input files; they are not advertised as data-free reproduction.

## External training dependency

Selected attack/detector adapters use the pinned BackdoorBench revision:

```bash
git clone https://github.com/SCLBD/BackdoorBench.git external/BackdoorBench
git -C external/BackdoorBench checkout f02e3534645f0ee63d6848653062cd6c0d6c400d
```

This is optional for the method-level quick tests. Install that project's own
dependencies and observe its license. Dataset and weight downloads are separate.
Do not add external datasets, weights, or nested Git checkouts to this repository.

## UI label-corruption and population-scaling analyses

The original plan and receipt are distributed as
`auditability_certificate_v1/UI_REDTEAM_SCALING_PLAN_V1.md` and
`auditability_certificate_v1/UI_REDTEAM_SCALING_RECEIPT_V1.json`.
They describe post-hoc developmental analyses, including the disclosed pilot work.
The newly distributed scripts and plan preserve their original bytes and hashes.

Individual checks (the robust self-test is also in the default release verifier):

```bash
python -B auditability_certificate_v1/analysis/srac_ui_robust_v1.py
python -B auditability_certificate_v1/analysis/run_ui_redteam_v1.py --help
python -B auditability_certificate_v1/analysis/run_ui_scaling_v1.py --help
```

The robust self-test checks finite small configurations and reduction to the base
engine at zero label errors. It is not a replay of the real-data red-team results.
Those runs require the controlled frozen flags and protocols for blocks 8, 11,
12, and 13, plus the stored corrected UI decisions for baseline reproduction.
Scaling analysis part C also requires corrected block-16 decisions. These inputs,
models, and experiment outputs are not included in the public code-only repository.

For the red-team plan use `run --draws 20 --blocks 8,11,12,13 --output OUTPUT_DIR`;
the script's default is 200 draws, not the plan's 20. Part-wise runs and the
`combine` command are documented by `--help`. The scaling script uses the fixed
replication and draw counts declared in the plan. Output directories must be new.

## Additional analysis code: 2026-10-26

The following source files are accompanied by their original plans and receipts:

| Analysis | Entry point in `auditability_certificate_v1/analysis/` | Input boundary |
| --- | --- | --- |
| GO-direction risk allocation | `run_go_allocation_v1.py` | Frozen flags and stored interval/UI/betting decisions |
| Fixed-budget SWEETER-style comparison | `run_sweeter_comparison_v1.py` | Same flags/sample order and stored allocation decisions; optional `numba` |
| Audited-size precision checks | `run_float_precision_v1.py` | Frozen flags and stored decisions; optional `mpmath` |
| Automated reference channel | `run_reference_channel_v1.py`, `run_reference_channel_certificates_v1.py` | Official images, frozen flags and channel outputs; training dependencies |
| Directional label-error budgets | `srac_ui_directional_v1.py`, `run_directional_budget_v1.py` | Self-test is data-free; experiment runner needs controlled channel labels and flags |
| Two-stage adjudication simulation | `run_two_stage_adjudication_v1.py` | Controlled confirmation flags and reference-channel file used by its shared loader |

Install optional analysis libraries separately:

```bash
python -m pip install -r requirements-analysis.txt
```

The optional dependency list is not a historical lock file. Basic release verification
does not need these extra libraries, model training, or an accelerator. Training the
reference channel additionally requires `requirements-replay.txt` and official data.

The source runners preserve their existing input layout and do not make controlled
records public. Inspect the corresponding plan and command help before data-dependent
execution. A receipt binds source/input hashes; it does not supply omitted inputs or
prove that a scientific experiment was rerun in this checkout.

The SWEETER-style code compares terminal, fixed-budget tests; it must not be described
as the full sequential SWEETER procedure. Directional robustness certifies GO or ABSTAIN
only. Two-stage adjudication is a simulation with declared error assumptions, not a
verified operational channel.

## Synthetic certificate fixtures

`fixtures/synthetic_certificate_counts_v1.json` contains six hand-constructed examples:
census GO, census recall failure, census group failure, no sample, no poison/no removals,
and an empty universe. The last three exercise conservative ABSTAIN behavior. They have
no experimental sample identities and are not real experiment results.

The standalone replay prints JSON with `synthetic=true`, `records_checked`,
`records_failed`, the decisions, and `passed`. The verifier checks these fields rather
than accepting a process exit code alone. The regression suite also rejects incomplete
self-test schemas.

The source-preserved `build_certificate_records_v1.py` and
`test_certificate_records_v1.py` refer to a separate controlled real-study file at
`auditability_certificate_v1/certificate_records_v1/srac_ui_reference_records_v1.json`.
That file is not distributed because experimental results remain outside this release.
Those two source files are not part of default verification; use the separate synthetic
replay for a self-contained public count-to-decision check.

## API and input boundary

`recall_tests`, `group_tests`, and `decide_ui` in `srac_ui_engine_v1_1.py`
receive a per-assertion level `alpha`, not a family risk `delta`. For the reported
tri-state allocation with `A` active assertions, this is `delta / (2*A)`.
The separately justified GO-direction allocation and GO-only directional variant
must not be substituted for a tri-state guarantee without their stated risk accounting.

The low-level reference engine is not a production input-validation layer. Before
calling it, validate integer counts with `0 <= observed <= sampled <= size`, finite
policy parameters, feasible group/stratum counts, and the consistency of the counts
with the frozen population and memberships. For array-based callers, verify unique,
in-range sample indices in the correct flag strata and the declared uniform
without-replacement sampling design. Overlapping groups are allowed; malformed or
duplicated sample indices do not become valid merely because a test function returns
a decision. A label-error guarantee also requires a valid declared error budget and
its associated calibration risk accounting.

The synthetic replay validates its small count schema, not a general production
population or sampling pipeline. Data-free self-tests check finite configurations.
The precision runner checks selected audited-size calculations. Neither is a formal
floating-point proof for arbitrary `N`, risk levels, or externally supplied inputs.
The frozen reference engine is unchanged in this update.

## Frozen identities

Protocols and receipts record the SHA-256 of the frozen files. Five public files
differ from their frozen bytes only in a path string; `PATH_NORMALIZATION.json`
lists, for each, the frozen hash, the public hash, and the change. Among them is
the corrected UI engine (frozen SHA-256
`ce1275d07607668c2f8243ef8dde6c8f163a7f6a226bdaefe5ac718752a0003f`), whose frozen
interval dependency is distributed byte-identical (SHA-256
`9171814bd8be311461cb6c3a493be1b8faba58c06dbd76f497d51c47bd9d5422`) at
`auditability_certificate_v1/analysis/frozen/cifar10_semantic_overlap_v1/`.
Some endpoint runners re-check the frozen hash of a normalized file
(`run_posthoc_power_study_v1.py`): the block-16 endpoint and the secondary escalation
analyses of blocks 12 and 13. Run them through the compatibility wrapper, which leaves
every frozen file, protocol, and receipt unchanged and accepts a normalized file only if
its bytes match the public hash declared in `PATH_NORMALIZATION.json`:

```bash
python -B auditability_certificate_v1/analysis/public_compat_runner.py \
    auditability_certificate_v1/analysis/run_gtsrb_srac_ui_confirmation.py
```

Every substitution is reported on stderr. The frozen bytes themselves can only be
checked against the controlled archive. The runners still need controlled inputs that
are not public; `auditability_certificate_v1/tests/test_public_endpoint_compat.py`
runs the block-16 and block-12 endpoints end to end on synthetic flags instead.
Historical source hashes are provenance, not proof that omitted data were replayed here.

The first UI reference suite (`test_srac_ui_engine_v1_1.py`) is kept unchanged as the
record bound before the v1.1 recomputation. The second suite also decides the
zero-observation outcome with the engine, adds small risk budgets, an independent
Berger--Boos check, exact expected regression decisions, and a mutation check.

Manuscripts, TeX/PDF files, experimental results, and the earlier repository's
Git history are intentionally excluded.

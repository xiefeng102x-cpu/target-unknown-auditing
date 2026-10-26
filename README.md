# Target-Unknown Auditing — SRAC Code

Source code for target-unknown, group-safe auditing of poison-screening decisions.
The method uses **sparse adjudication**: a protected sample is labeled by an
independent adjudicator; it does not assume a complete clean-member registry.

Repository: [xiefeng102x-cpu/target-unknown-auditing](https://github.com/xiefeng102x-cpu/target-unknown-auditing)

## Public scope

This repository contains code, tests, configurations, and protocol records only.
It does **not** contain manuscript directories, TeX sources, PDFs, experimental
result tables, datasets, model checkpoints, private poison identities, raw scores,
or credentials. Its Git history starts with this code-only upload; no history from
the earlier repository is imported.

## Quick verification

Python3.10 or3.11 is recommended. No GPU, model training, dataset, or paid API is
needed for these checks.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-audit.txt
python -B src/verify_srac_release_20261019.py --root .
```

The verifier checks the code-only manifest and exclusions, runs the data-independent
method and release-policy regression tests, smoke-tests the frozen endpoint runners
on synthetic flags, and exercises both corrected SRAC-UI reference suites, the
SUITE-style Fisher self-test, robust and directional self-tests, and hand-constructed
count-to-decision fixtures.

[Verification instructions](HOWTO_VERIFY.md) · [Artifact scope](ARTIFACT_SCOPE.md)

## Layout

- `auditability_certificate_v1/analysis/`: scalar/group-safe SRAC, corrected UI
  engine, audit comparisons, and controlled-input experiment runners.
- `auditability_certificate_v1/tests/`: method-level and experiment tests.
- `src/`: training, detector adapters, replay utilities, and release verification.
- `configs/`: experiment configurations.
- `fixtures/`: synthetic count fixtures only, not experimental records or results.
- Protocol, plan, and receipt records accompany the corresponding experiment code,
  including the plans of every post-hoc analysis and the protocols of the
  prospective blocks.

The directory name is retained for import compatibility. Protocols and receipts
record the SHA-256 of the frozen files. Five public files differ from their frozen
bytes only in a path string (a dependency path or an absolute workstation path);
[PATH_NORMALIZATION.json](PATH_NORMALIZATION.json) lists their frozen and public
hashes. The frozen interval dependency is distributed byte-identical at
`auditability_certificate_v1/analysis/frozen/cifar10_semantic_overlap_v1/`.
Endpoint runners that re-check such a frozen hash are run through
`auditability_certificate_v1/analysis/public_compat_runner.py`
(see [HOWTO_VERIFY.md](HOWTO_VERIFY.md)).

## Code update: 2026-10-26

The source-only update adds the fixed-budget SWEETER-style comparison, direction-specific
risk allocation, directional label-error budgets, two-stage adjudication simulation,
reference-channel runners, and audited-size floating-point checks, with their original
plans and receipts. Existing public-normalized engines are retained unchanged.

The SWEETER-style comparison is a fixed-budget adaptation, not an evaluation of the
complete sequential method. The directional variant returns **GO or ABSTAIN only**;
it does not inherit a NO-GO guarantee. Two-stage channel results are simulations, not
measured deployment performance. The adjudicator remains a trust assumption, and the
scripts do not establish a generally deployable channel or an end-to-end cost advantage.

The lower bound concerns unknown prevalence in the declared flag-only information model;
it is not a universal cost bound for every source of adjudication information.
The exact certificate addresses the frozen screen and declared groups, not absence of a
backdoor in a retrained model.

Low-level engine functions take a **per-assertion test level `alpha`**, not the
family-level risk `delta`. Callers must validate counts, group membership, stratum
assignment and unique without-replacement sample indices, and apply the declared risk
allocation. These reference routines are not a production input-validation layer;
finite self-tests are not a floating-point correctness proof for arbitrary populations.
See [the API boundary](HOWTO_VERIFY.md#api-and-input-boundary).

## Training and data-dependent analyses

The post-hoc UI label-corruption and population-scaling analyses are accompanied by
`auditability_certificate_v1/UI_REDTEAM_SCALING_PLAN_V1.md` and its original hash
receipt. Their entry points are `run_ui_redteam_v1.py` and `run_ui_scaling_v1.py` in
the analysis directory; `srac_ui_robust_v1.py` provides the declared-label-error
robust tests and a data-free self-test. See HOWTO_VERIFY for commands and required
controlled inputs. These are developmental analyses, not prospective endpoints.

These are separate from quick verification. They require additional dependencies
from `requirements-replay.txt`, official datasets and weights, and, for some
analyses, controlled flags or prior decision tables that are not public here.
A passing method test is not a from-scratch victim-training replay.

`requirements-analysis.txt` adds optional `mpmath` and `numba` dependencies for the
precision and betting-comparison runners. It is a convenience dependency list, not
a lock of the historical analysis environment. Reference-model training also needs
the replay dependencies and official datasets.

The new public reference fixtures are hand-constructed. No real experiment's aggregate
certificate records are distributed. `build_certificate_records_v1.py` and the original
`test_certificate_records_v1.py` preserve source bytes but require the controlled
reference file; they are intentionally outside the default data-free test selection.
The separate synthetic fixture replay is part of default verification.

Selected adapters require BackdoorBench. Acquire it separately under the
appropriate license at the pinned revision recorded by the project, rather than
assuming it is bundled in this new repository. See the reproduction instructions.

## License

Project-owned code uses the MIT License. Data, pretrained weights, and external
software retain their respective licenses. See
[the provenance notice](auditability_certificate_v1/PROJECT_LICENSE_PROVENANCE_2026-10-14.md).

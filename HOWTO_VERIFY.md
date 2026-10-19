# Verify this code-only repository

Use Python3.10 or3.11 from the repository root.

```bash
python -m pip install -r requirements-audit.txt
python -B src/verify_srac_release_20261019.py --root .
```

A complete PASS requires manifest integrity, no manuscript/PDF/private payload,
the required statistical regression tests, corrected UI reference tests, Fisher
self-tests, and unchanged integrity after testing. This does not validate omitted
experimental result tables, real adjudication channels, or model training.

An integrity-only check is explicitly partial:

```bash
python -B src/verify_srac_release_20261019.py --root . --integrity-only
```

Individual data-independent checks:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 python -B auditability_certificate_v1/analysis/test_srac_ui_engine_v1_1.py
PYTHONDONTWRITEBYTECODE=1 python -B auditability_certificate_v1/analysis/run_ui_mechanism_ablation_v1.py selftest
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

## Frozen identities

Protocols and receipts record the SHA-256 of the frozen files. Five public files
differ from their frozen bytes only in a path string; `PATH_NORMALIZATION.json`
lists, for each, the frozen hash, the public hash, and the change. Among them is
the corrected UI engine (frozen SHA-256
`ce1275d07607668c2f8243ef8dde6c8f163a7f6a226bdaefe5ac718752a0003f`), whose frozen
interval dependency is distributed byte-identical (SHA-256
`9171814bd8be311461cb6c3a493be1b8faba58c06dbd76f497d51c47bd9d5422`) at
`auditability_certificate_v1/analysis/frozen/cifar10_semantic_overlap_v1/`.
Endpoint runners that re-check the frozen hash of a normalized file expect the frozen
bytes; they also need controlled inputs that are not public. Historical source hashes
are provenance, not proof that omitted data were replayed here.

Manuscripts, TeX/PDF files, experimental results, and the earlier repository's
Git history are intentionally excluded.

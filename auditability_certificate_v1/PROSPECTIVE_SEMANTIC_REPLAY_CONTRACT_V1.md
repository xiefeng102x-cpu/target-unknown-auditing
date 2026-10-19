# Prospective semantic replay contract v1

## Scope and chronology

This contract was introduced on 2026-10-14 after the frozen
CIFAR-10/VGG-16-BN replay exposed sub-picounit numerical drift in Activation
Clustering. It applies only to experiments whose protocol and implementation are
frozen after this document. It does not reinterpret that earlier byte-level
failure or convert its post-hoc addendum into prospective evidence.

## Required pre-freeze declaration

Each future score-handoff protocol must declare:

1. the semantic-contract version;
2. the complete row-key fields;
3. all fields requiring exact equality;
4. every continuous field allowed a numerical tolerance;
5. an absolute tolerance for each such field, with zero relative tolerance;
6. the comparison implementation hash;
7. the downstream endpoint fields and the rule that derives them.

The default score-table contract is:

```json
{
  "contract_version": "prospective-semantic-replay-v1",
  "key_fields": ["victim_id", "detector", "fold", "sample_id", "role"],
  "exact_fields": ["conventional_flag"],
  "score_fields": ["score"],
  "absolute_tolerance": 1e-10,
  "relative_tolerance": 0.0
}
```

Any protocol may tighten this contract. It may not relax it after endpoint data
are generated.

## Exact invariants

The following remain byte- or value-exact, as appropriate:

- protocol, source, checkpoint, data-source, and freeze-manifest hashes;
- sample identities, poison-plan indices, folds, roles, labels, and row counts;
- private truth and protected-draw artifacts;
- every categorical or integer field;
- detector flags and all downstream GO/NO-GO/ABSTAIN decisions;
- endpoint confusion counts and reported aggregate rates.

The replay rejects duplicate or missing row keys. A small score difference never
excuses a changed detector flag or endpoint decision.

## Tolerant numerical invariants

Only explicitly named continuous diagnostic fields may use the frozen absolute
tolerance. Both values must be finite. Shapes and keys remain exact. The report
records the maximum absolute difference and the number of exceedances for every
field.

Raw emitted files retain their SHA-256 hashes. A semantic PASS with unequal raw
hashes must report both hashes and `byte_exact=false`. Raw hashes are provenance
evidence; semantic comparison establishes endpoint-equivalent replay. Neither is
silently substituted for the other.

## Executable implementation

The reference implementation is
`src/prospective_replay_contract.py`. Its unit tests cover accepted numerical
roundoff, changed keys, changed endpoint flags, material score changes, duplicate
keys, and retention of unequal raw hashes.


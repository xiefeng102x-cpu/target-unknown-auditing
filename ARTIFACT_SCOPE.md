# Code-only artifact scope

Included: project source, method/experiment tests, configurations, protocol records,
a code-only file manifest, dependency specifications, and licensing documentation.

Excluded: manuscript directories and source text, all TeX/PDF files, experimental
result tables, raw data and scores, victim checkpoints, feature tensors, candidate
poison identities, protected sampling keys, credentials, caches, and old Git history.

The public verifier exercises statistical code and code-only publication controls.
It does not train victims or reproduce real-data decisions from absent inputs.
Operational guarantees remain conditional on protected sampling and the declared
adjudication-error model.

The old repository's history is not imported. Removing files from the old
repository's current branch did not erase its historical commits; this new
repository must not be described as retroactively erasing that exposure.

# Frozen CIFAR-10 semantic-overlap endpoint core

`run_protected_group_family_srac.py` is the byte-exact endpoint core used by the
internally frozen CIFAR-10/VGG-16-BN/Blended semantic-family experiment.

- Frozen SHA-256: `9171814bd8be311461cb6c3a493be1b8faba58c06dbd76f497d51c47bd9d5422`
- Protocol SHA-256: `5a53c5798b728d40357d7e8f271cfb4a038d32f4597a98f581432407d3fc40bd`
- Terminal result: `FAIL_CIFAR10_SEMANTIC_OVERLAP_CONFIRMATION`

The active core at `analysis/run_protected_group_family_srac.py` was corrected after
this experiment terminated. It now treats recall as undefined when a group contains
no poison and forces abstention. That post-freeze safety correction does not alter the
sealed score table, decision table, report, or audit receipt. This directory preserves
the executed implementation so the reported endpoint hash remains independently
checkable. It is historical reproduction code, not the recommended active version.

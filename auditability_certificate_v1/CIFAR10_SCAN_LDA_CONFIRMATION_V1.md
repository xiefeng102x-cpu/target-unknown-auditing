# CIFAR-10 SCAn+LDA semantic-overlap confirmation V1

## Status
Internally frozen before seeds 4416--4425 were opened (local hash freeze; no external
timestamp). Protocol: `PROTOCOL_CIFAR10_SCAN_LDA_CONFIRMATION_V1.json` and its `.sha256`
sidecar. Freeze receipt: `CIFAR10_SCAN_LDA_CONFIRMATION_V1_FREEZE_RECEIPT.json`.

## Why this block exists
The semantic-overlap block (seeds 4411--4415) failed its frozen support gate: the
pinned SCAn screen reached recall 0.650--0.884 with near-zero clean flags, so only one
screen met the 0.80 recall policy. The certificate behaved correctly (0 false GO,
0 false NO-GO, including three screens 0.02--0.03 below the recall target), but
positive target-unknown coverage did not transfer. This block tests a stronger,
still target-unknown screen on fresh victims.

## Frozen design (only the screen changes)
- Victims, split, VGG-16-BN, 15-epoch schedule, Blended alpha 0.2, target ship,
  2,250 poisons, eligibility rule: byte-identical to the previous block.
- Screen: pinned SCAn, then inside every SCAn-flagged class a regularised LDA between
  the class's 250 trusted references and its SCAn-flagged members (PCA 32, relative
  ridge 0.1); conformal p-values against the class's 250 other held-out clean
  validation members; add members with p <= 0.05. Final flag = SCAn OR expansion.
- Certificate, protected family, budgets, draws, delta: identical to the previous block.
- Support gate: >=8/10 eligible; >=5 true policy passes; 0 false GO; 0 false NO-GO;
  >=1 victim with GO at k=2000; >=5 victims with GO at k=8000.
- Pre-declared secondary (outside the gate): nested escalation 2k->8k->16k, delta/3 per
  look, one-sided split budget, flagged share 0.25.

## Development disclosure
Selected after inspecting seeds 4401--4405 and 4411--4415 (see protocol). Development
estimates on 4411--4415 (optimistic, same seeds used for selection): recall 0.910--0.977,
max protected-group clean rate <= 0.032, GO at k=8000 in 99.9% of development draws.

## Execution (user's Mac, MPS)
1. `python3 src/run_cifar10_vgg16_bn_blended_scan_lda_confirmation.py --selftest`
2. `caffeinate -i python3 src/run_cifar10_vgg16_bn_blended_scan_lda_confirmation.py --formal --ack OPEN_CIFAR10_VGG16_BLENDED_SCAN_LDA_SEEDS_4416_4425`
   (resume after interruption with `--resume`)
3. `python3 auditability_certificate_v1/analysis/run_cifar10_scan_lda_posttraining_pipeline.py --check`
4. `python3 auditability_certificate_v1/analysis/run_cifar10_scan_lda_posttraining_pipeline.py --execute --ack OPEN_CIFAR10_SCAN_LDA_POSTTRAINING_PIPELINE_V1`
5. `python3 auditability_certificate_v1/analysis/run_cifar10_scan_lda_secondary_escalation.py`

## Interpretation boundary
A pass shows positive target-unknown group-safe coverage on a second dataset, a
14.7M-parameter architecture, and a non-patch attack for this two-stage screen. It
does not establish universal detector, attack, or architecture transfer. A failure is
reported as is; no seed, parameter, or gate is repaired.

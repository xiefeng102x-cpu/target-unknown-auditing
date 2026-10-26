# Execution note for REFERENCE_CHANNEL_PLAN_V1.md

* 2026-10-20 10:14 UTC: a first execution on the author's machine (Linux VM, 4 CPU threads,
  torch 2.7.1+cpu) was stopped by the environment while the first reference model was in
  epoch 9 of 30. No model was saved and no channel label was computed; its partial log is
  kept as results_reference_channel_v1/run.log on that machine, and the log of the completed
  execution as results_reference_channel_v1/run_completed.log.
* The plan and the hashed scripts were then executed unchanged in a second environment
  (2 CPU threads, torch 2.7.1, CPU only), with the same inputs (cache sha256 5d6a4297...,
  flags file and protocol as recorded in report.json).
* Part 2 (run_reference_channel_certificates_v1.py) was hashed before any channel label
  existed (REFERENCE_CHANNEL_PART2_RECEIPT_V1.json).

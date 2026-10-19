# Post-hoc imperfect-adjudication sensitivity V1 (developmental; not confirmation)

Written 2026-10-16 after the GTSRB semantic-overlap V2 and CIFAR-10 SCAn+LDA endpoints were
observed. Inputs are immutable frozen flags, frozen protected families, and the frozen design
(two-sided exact intervals, 50/50 flag allocation, 200 draws per victim and budget,
k in {2000, 8000, 16000}, delta = 0.005). Sampled labels are flipped independently with
probability eps in {0, 0.001, 0.005, 0.01, 0.02}. Variants: naive (B=0, full delta) and
robust (delta/2 for count intervals, delta/2 for the adjudication-error event, B = exact
Binomial(k, eps) upper 1-delta/2 quantile, hull [L(x-B), U(x+B)]). eps=0 must reproduce both
frozen endpoints exactly. All cells are reported; no cell is promoted to a frozen endpoint.

# Post-hoc outcome-aware label red team against the group-safe certificate V1 (developmental)

Written 2026-10-17 at the user's request ("群组安全版本的红队测试"; internal review ARS 2026-10-17,
item R5: the prospective red team R covers only the scalar certificate). Question: when an adversary
who sees the protected draw, the flags, and all true labels corrupts up to B = ceil(rho k) sampled
labels, does the group-safe certificate that ignores errors (B_cert = 0) issue false declarations,
and does the declared-B robust hull stay valid, and at what power cost?

Data: frozen flags, families, draw namespaces, budgets (2000/8000/16000), and 200 draws of blocks
8 (GTSRB semantic, 10 victims), 11 (CIFAR SCAn, 5), 12 (CIFAR SCAn+LDA, 10), 13 (CIFAR Adap-Blend
SCAn+LDA, 10). Certificate: frozen two-sided exact intervals, eta = delta/(2Q), delta = 0.005,
50/50 flag-stratum allocation, tau = 0.80, cap = 0.10.

Adversary: rho in {0.1, 0.5, 1, 2}%. Failing screens -> target false GO, using only GO-helping flips
(sampled unflagged poisons to clean, sampled flagged clean to poison). Passing screens -> target
false NO-GO with the mirror flips. Strategy set: budget split between the two flip types in shares
{0, 1/4, 1/2, 3/4, 1}; within a type, members of the binding group first (GO: largest clean-rate
upper bound; NO-GO: largest lower bound), plus for NO-GO members of the smallest group first; ties
by a fixed seeded permutation. Adversary wins if any strategy reaches its target. The strategy set
is a heuristic lower bound on adversarial power, not the worst case; the robust hull's validity does
not depend on the strategy (it holds for every corruption of at most B labels).

Certificates: naive (B_cert = 0 on corrupted labels) and robust hull with declared B and the full
delta. Also recorded: unattacked robust decision (power cost of declaring B).
Validation gate (must hold before results are used): the unattacked naive decision reproduces every
frozen decision of the four blocks (42,000 decisions).

Execution: two parts (`run --blocks 12,11` and `run --blocks 8,13`), then `combine`. Script hash in
the freeze receipt. All results reported; nothing becomes a frozen endpoint.

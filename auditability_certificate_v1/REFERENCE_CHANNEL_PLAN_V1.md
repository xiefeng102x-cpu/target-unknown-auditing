# An automated adjudication channel without master labels (post hoc pilot)

Status: post hoc pilot, written before any reference model is trained and before any channel
label is computed. The populations, flags, and draws of the SRAC-UI confirmation study and all
its stored decisions were known. Raised by the 2026-10-20 simulated review (M6): the security
use of the certificate requires an adjudicator that does not read the experiment's master
poison labels.

## Channel

* Trusted data: the official GTSRB test split (12,630 images), the trusted pool of the threat
  model. A stratified 90/10 split (numpy.default_rng(2026102010) per class) gives 11,367
  training and 1,263 held-out images.
* Reference ensemble: three GTSRBTinyCNN models (the victim architecture, no backdoor), seeds
  2026102011-2026102013, AdamW, lr 2e-3, wd 1e-4, batch 256, 30 epochs, random 2-pixel shifts
  and +-10% brightness, class-weighted loss, final epoch kept (no selection on audit data).
* Rule (fixed here, not tuned): the channel labels a record **poison** iff the argmax of the
  ensemble's mean softmax differs from the record's assigned label, otherwise **clean**. The
  channel sees only the record's 32x32 image and assigned label; record identifiers and the
  master labels are used only to rebuild the image and, afterwards, to score the channel.
* Applied to every record of the ten confirmation populations (seeds 255-264; 25,108 records
  each), rebuilt with the frozen victim protocol.

## Scoring (hidden truth used only here)

Per population and pooled: clean records labelled poison, poisons labelled clean, overall
error rate, each with exact 95% Clopper-Pearson intervals; error rates by flag stratum; the
held-out clean error of the ensemble (the only error rate a defender can estimate without gold
poisons).

## Certificates on the stored confirmation draws (flagged share 0.1, 200 draws, 3 budgets)

SRAC-UI v1.1 at alpha = delta/(2A) with the channel's labels in place of the master labels:

1. naive: B = 0;
2. defender B: B_k = upper delta_adj/2 quantile of Binomial(k, p_u), where p_u is the
   one-sided (1 - delta_adj/2) Clopper-Pearson upper bound of the held-out clean error rate;
   delta_adj = 0.001. This is what a defender can compute; it ignores errors on poisons;
3. oracle B: for each population and stratum, the upper delta_adj/4 hypergeometric quantile
   of wrong labels in the stratum sample given the true number of wrong labels in the stratum,
   summed over strata; it uses the hidden truth and shows the best case for this channel.

Declared-B decisions use srac_ui_robust_v1 (R = B). Reported for each: GO on passing and NO-GO
on failing screens, false GO and false NO-GO counts, against the stored master-label decisions.
If B is so large that the robust test cannot decide, that is reported as the outcome.

## Interpretation fixed in advance

The certificate needs roughly epsilon <= 1e-3 for useful power at these budgets (Table SIV).
If the channel's error is far above that, the pilot is reported as a negative result: this
channel is not adequate, and the declared-B certificate is expected to abstain rather than err.

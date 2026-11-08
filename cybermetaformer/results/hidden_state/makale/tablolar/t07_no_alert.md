| Model A | Model B | Test recall, A / B | OOD recall, A / B | OOD A − B [95% CI] | p (Holm, plan) |
|---|---|---:|---:|---:|---:|
| CMF (topology-biased attention) | No host mixer | 0.887 / 0.680 | 0.830 / 0.454 | +0.376 [+0.357, +0.395] | 0.0023 |
| Full attention (no topology bias) | No host mixer | 0.895 / 0.680 | 0.831 / 0.454 | +0.378 [+0.359, +0.396] | 0.0023 |
| GCN-like host mixer | No host mixer | 0.886 / 0.680 | 0.809 / 0.454 | +0.355 [+0.336, +0.375] | 0.0023 |
| CMF (topology-biased attention) | Single 78-d token | 0.887 / 0.890 | 0.830 / 0.773 | +0.057 [+0.046, +0.068] | 0.0023 |

Note. Silent host-steps whose host has had no alert in obs[1..t] (N_hist; online-applicable, Section 3.3): n = 54,731 test and 96,486 OOD host-steps. All models: seeds 0–2; the retrospective subset N_ep is compared in Table @tbl:hic_kume. Paired episode bootstrap with 100,000 resamples (Algorithm 2); p (Holm, plan) adjusts over all m = 116 rows of the comparison plan, so its smallest attainable value is m × 2/(n + 1) = 0.0023.

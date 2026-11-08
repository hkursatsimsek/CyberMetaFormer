| Criterion | CMF (mean ± sd) | GRU (mean ± sd) | Δ (CMF − GRU) | Episode bootstrap 95% CI | p (Holm, pair) | p (Holm, plan) | Seed-level 95% t-interval | Seeds favouring CMF |
|---|---:|---:|---:|---|---:|---:|---|---:|
| Test host macro-F1 | 0.9801 ± 0.0001 | 0.9768 ± 0.0003 | +0.0034 | [+0.0029, +0.0039] | < 0.0001 | 0.0023 | [+0.0029, +0.0038] | 5 / 5 |
| OOD host macro-F1 | 0.9417 ± 0.0006 | 0.9264 ± 0.0012 | +0.0153 | [+0.0145, +0.0162] | < 0.0001 | 0.0023 | [+0.0136, +0.0170] | 5 / 5 |
| Relative OOD drop | 0.0392 ± 0.0005 | 0.0516 ± 0.0011 | −0.0124 | [−0.0134, −0.0114] | < 0.0001 | 0.0023 | [−0.0140, −0.0108] | 5 / 5 |
| OOD silent recall | 0.772 ± 0.020 | 0.736 ± 0.009 | +0.035 | [+0.031, +0.040] | < 0.0001 | 0.0023 | [+0.010, +0.061] | 5 / 5 |

Note. Five seeds per model (0–4). The episode bootstrap (100,000 resamples, Algorithm 2) holds the trained models fixed and resamples test and OOD episodes; it does not include retraining variability. The seed-level interval resamples nothing and uses the spread of the five same-index differences; it does not include episode sampling. Neither interval alone is the total uncertainty. Holm (pair) adjusts over the four criteria of this pair; Holm (plan) over all m = 116 rows of the main comparison plan, so the smallest attainable plan-wide p is 0.0023.

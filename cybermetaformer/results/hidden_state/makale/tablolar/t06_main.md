| Criterion | Full-history GRU | CyberMetaFormer | Paired CMF − GRU [95% CI] | p (Holm, plan) |
|---|---:|---:|---:|---:|
| Parameters | 183,588 | 186,665 | Similar budget (B1) | — |
| Test host macro-F1 | 0.9768 | 0.9801 | +0.0034 [+0.0029, +0.0039] | 0.0023 |
| OOD host macro-F1 | 0.9264 | 0.9417 | +0.0153 [+0.0145, +0.0162] | 0.0023 |
| Relative OOD drop | 0.0516 | 0.0392 | −0.0124 [−0.0134, −0.0114] | 0.0023 |
| OOD silent recall | 0.736 | 0.772 | +0.035 [+0.031, +0.040] | 0.0023 |
| OOD ECE (after temperature scaling) | 0.0089 | 0.0034 | Not bootstrapped | — |

Note. CyberMetaFormer: mean of seeds 0–4; full-history GRU: mean of seeds 0–4. Paired episode bootstrap with 100,000 resamples (Algorithm 2); p (Holm, plan) adjusts over all m = 116 rows of the comparison plan, so its smallest attainable value is m × 2/(n + 1) = 0.0023. Intervals use unrounded differences. Lower is better for the relative OOD drop and ECE. ECE uses 15 confidence bins after temperature scaling fitted on the validation set.

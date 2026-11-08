| Split | Model | Root → clean (%) | Root → user (%) | User → clean (%) | False alarm on clean (%) | Worst host (macro-F1) | Silent-onset spells detected (%) | Silent-onset latency, median / p90 (steps) |
|---|---|---:|---:|---:|---:|---|---:|---:|
| Test | CMF | 3.83 ± 0.26 | 0.00 ± 0.00 | 4.15 ± 0.06 | 0.84 ± 0.08 | user1 (0.963) | 95.2 ± 0.3 | 0.6 / 11.0 |
| Test | GRU | 3.99 ± 0.10 | 0.01 ± 0.00 | 4.28 ± 0.05 | 1.06 ± 0.03 | user1 (0.960) | 95.0 ± 0.3 | 1.0 / 9.4 |
| OOD | CMF | 17.52 ± 1.61 | 0.01 ± 0.00 | 3.58 ± 0.06 | 1.75 ± 0.24 | user2 (0.899) | 84.4 ± 1.8 | 2.4 / 22.4 |
| OOD | GRU | 20.47 ± 0.74 | 0.08 ± 0.01 | 3.77 ± 0.08 | 2.36 ± 0.09 | user2 (0.875) | 83.7 ± 1.1 | 2.0 / 19.2 |

Note. Seed mean ± SD over seeds 0–4. Error rates are rows of the pooled, row-normalized confusion: root → clean = P(ŷ = clean | y = root); false alarm on clean = P(ŷ ≠ clean | y = clean). Worst host: the evaluated host with the lowest seed-mean macro-F1 (all hosts in Table @tbl:host). A spell is a maximal run of consecutive compromised steps (y > 0) on one host; it is silent-onset when its first step carries no alert (n = 4,443 test and 7,704 OOD spells; spells with an alert at onset are detected in at least 99.98% of cases by both models, with median latency 0). Detected: the model predicts a compromise (ŷ > 0) at some step of the spell; latency: steps from the onset to that first prediction, over detected spells; median and 90th percentile per seed, averaged over seeds. Latencies use the argmax of the stored float16 logits.

| Split | Model | user → clean (%) | user → user (%) | user → root (%) | root → clean (%) | root → user (%) | root → root (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| Test | CMF | 53.1 ± 0.8 | 43.2 ± 0.6 | 3.8 ± 0.3 | 8.7 ± 0.6 | 0.0 ± 0.0 | 91.3 ± 0.6 |
| Test | GRU | 54.7 ± 0.5 | 41.3 ± 0.6 | 4.1 ± 0.1 | 8.9 ± 0.2 | 0.0 ± 0.0 | 91.1 ± 0.2 |
| OOD | CMF | 69.2 ± 1.1 | 26.1 ± 0.7 | 4.7 ± 0.6 | 21.7 ± 2.0 | 0.0 ± 0.0 | 78.3 ± 2.0 |
| OOD | GRU | 72.7 ± 1.5 | 21.4 ± 1.4 | 5.9 ± 0.4 | 25.2 ± 0.9 | 0.0 ± 0.0 | 74.8 ± 0.9 |

Note. Confusion restricted to silent host-steps (compromised, no alert in the current observation), row-normalized; seed mean ± SD over seeds 0–4, from the stored logits. About 98% of silent host-steps are root compromises (Table @tbl:sessiz), so the root rows dominate silent recall.

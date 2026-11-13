| Split | Model | Root F1 | Root → clean (%) | False alarm on clean (%) | Silent-onset spells detected (%) | Latency, median / p90 (steps) |
|---|---|---:|---:|---:|---:|---:|
| Test | CMF | 0.966 | 3.83 | 0.84 | 95.2 | 0.6 / 11.0 |
| Test | GRU | 0.962 | 3.99 | 1.06 | 95.0 | 1.0 / 9.4 |
| OOD | CMF | 0.826 | 17.52 | 1.75 | 84.4 | 2.4 / 22.4 |
| OOD | GRU | 0.783 | 20.47 | 2.36 | 83.7 | 2.0 / 19.2 |

Note. Seed mean over seeds 0–4 of CyberMetaFormer (CMF) and the full-history GRU. Root F1 from the confusion pooled over the 11 evaluated hosts. Root → clean = P(ŷ = clean | y = root); false alarm on clean = P(ŷ ≠ clean | y = clean). A silent-onset spell is a maximal run of compromised steps on one host whose first step carries no alert; detected: the model predicts a compromise at some step of the spell; latency: steps from the onset to that prediction over detected spells (median and 90th percentile per seed, averaged). Seed SDs, all classes and error types, and per-host results: Tables @tbl:sinif, @tbl:sinif_hata, @tbl:host and @tbl:sessiz_karisiklik.

| Criterion | Full-history GRU | CyberMetaFormer | Δ CMF − GRU | Episode bootstrap 95% CI | Seed-level 95% CI |
|---|---:|---:|---:|---:|---:|
| Test host macro-F1 | 0.9768 ± 0.0003 | 0.9801 ± 0.0001 | +0.0034 | [+0.0029, +0.0039] | [+0.0029, +0.0038] |
| OOD host macro-F1 | 0.9264 ± 0.0012 | 0.9417 ± 0.0006 | +0.0153 | [+0.0145, +0.0162] | [+0.0136, +0.0170] |
| Relative OOD drop | 0.0516 ± 0.0011 | 0.0392 ± 0.0005 | −0.0124 | [−0.0134, −0.0114] | [−0.0140, −0.0108] |
| OOD silent recall | 0.736 ± 0.009 | 0.772 ± 0.020 | +0.035 | [+0.031, +0.040] | [+0.010, +0.061] |

Note. Seed mean ± SD over seeds 0–4 of each model. Episode bootstrap: paired, 100,000 resamples (Algorithm 2), trained models held fixed. Seed-level interval: 95% t-interval (df = 4) of the 5 same-index differences, without episode resampling. The two intervals describe different sources of uncertainty and neither alone is the total uncertainty. Plan-wide Holm-adjusted p = 0.0023 in all four rows, the smallest attainable value over the m = 116 rows of the comparison plan. Lower is better for the relative OOD drop. OOD ECE after temperature scaling: CMF 0.0034, GRU 0.0089. Full tables: Tables @tbl:ana, @tbl:iki_kaynak and @tbl:seedler.

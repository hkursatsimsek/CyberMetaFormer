| Model change | Δ OOD host macro-F1 [95% CI] | Δ OOD silent recall [95% CI] |
|---|---:|---:|
| No host tokens (single 78-d token) | −0.0433 [−0.0451, −0.0415] | −0.051 [−0.057, −0.045] |
| No host mixer | −0.0424 [−0.0441, −0.0408] | −0.185 [−0.195, −0.176] |
| Full attention instead of topology bias | −0.0006 [−0.0010, −0.0002] | −0.002 [−0.005, −0.000] |
| GCN-like host mixer | −0.0134 [−0.0143, −0.0126] | −0.046 [−0.051, −0.040] |
| Short convolution only (no long branch) | −0.0575 [−0.0596, −0.0555] | −0.214 [−0.223, −0.205] |
| SSD branch only | −0.0003 [−0.0007, +0.0001] | +0.004 [+0.001, +0.006] |
| Full-history causal attention instead of SSD | −0.0007 [−0.0011, −0.0003] | +0.010 [+0.007, +0.013] |
| GRU instead of SSD | +0.0011 [+0.0007, +0.0016] | +0.001 [−0.002, +0.004] |
| No memory: window 32, no alert-age channel | −0.0033 [−0.0038, −0.0028] | −0.004 [−0.007, −0.001] |
| B2 scale (2,828,473 parameters) | −0.0035 [−0.0041, −0.0030] | −0.003 [−0.007, −0.000] |

Note. Δ = variant − main model, seeds 0–2 each. B2 is compared with seeds 0–2 of CyberMetaFormer run 2, which was trained on the same hardware (Table @tbl:hostgru); all other variants with seeds 0–2 of the main run. 95% CI from the paired episode bootstrap (100,000 resamples).

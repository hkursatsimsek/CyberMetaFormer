| Model A | Model B | Criterion | A | B | A − B [95% CI] | p (Holm, pair) | p (Holm, family) |
|---|---|---|---:|---:|---:|---:|---:|
| Host-token GRU, topology bias | CMF, run 2 | Test host macro-F1 | 0.9803 | 0.9800 | +0.0002 [+0.0001, +0.0004] | 0.0253 | 0.1179 |
|  |  | OOD host macro-F1 | 0.9419 | 0.9414 | +0.0005 [+0.0001, +0.0009] | 0.0318 | 0.2064 |
|  |  | Relative OOD drop | 0.0391 | 0.0394 | −0.0003 [−0.0007, +0.0002] | 0.2230 | 1.0000 |
|  |  | Test silent recall | 0.904 | 0.906 | −0.002 [−0.004, −0.001] | 0.0086 | 0.0346 |
|  |  | OOD silent recall | 0.764 | 0.785 | −0.021 [−0.024, −0.019] | 0.0001 | 0.0007 |
|  |  | Test N_hist recall | 0.885 | 0.892 | −0.007 [−0.010, −0.003] | 0.0012 | 0.0043 |
|  |  | OOD N_hist recall | 0.826 | 0.836 | −0.010 [−0.014, −0.006] | 0.0001 | 0.0007 |
| Host-token GRU, no bias | CMF, run 2 | Test host macro-F1 | 0.9801 | 0.9800 | +0.0000 [−0.0002, +0.0003] | 1.0000 | 1.0000 |
|  |  | OOD host macro-F1 | 0.9413 | 0.9414 | −0.0001 [−0.0005, +0.0003] | 1.0000 | 1.0000 |
|  |  | Relative OOD drop | 0.0395 | 0.0394 | +0.0001 [−0.0003, +0.0006] | 1.0000 | 1.0000 |
|  |  | Test silent recall | 0.904 | 0.906 | −0.002 [−0.004, −0.001] | 0.0295 | 0.1107 |
|  |  | OOD silent recall | 0.763 | 0.785 | −0.022 [−0.025, −0.020] | 0.0001 | 0.0007 |
|  |  | Test N_hist recall | 0.876 | 0.892 | −0.015 [−0.021, −0.010] | 0.0001 | 0.0007 |
|  |  | OOD N_hist recall | 0.814 | 0.836 | −0.021 [−0.026, −0.017] | 0.0001 | 0.0007 |
| Host-token GRU, topology bias | Host-token GRU, no bias | Test host macro-F1 | 0.9803 | 0.9801 | +0.0002 [+0.0000, +0.0004] | 0.1526 | 0.4510 |
|  |  | OOD host macro-F1 | 0.9419 | 0.9413 | +0.0006 [+0.0002, +0.0009] | 0.0053 | 0.0180 |
|  |  | Relative OOD drop | 0.0391 | 0.0395 | −0.0004 [−0.0008, −0.0000] | 0.1526 | 0.4862 |
|  |  | Test silent recall | 0.904 | 0.904 | +0.000 [−0.001, +0.002] | 0.8615 | 1.0000 |
|  |  | OOD silent recall | 0.764 | 0.763 | +0.001 [−0.001, +0.003] | 0.5860 | 1.0000 |
|  |  | Test N_hist recall | 0.885 | 0.876 | +0.009 [+0.004, +0.014] | 0.0013 | 0.0042 |
|  |  | OOD N_hist recall | 0.826 | 0.814 | +0.011 [+0.007, +0.016] | 0.0001 | 0.0007 |
| Host-token GRU, topology bias | Full-history GRU | Test host macro-F1 | 0.9803 | 0.9768 | +0.0035 [+0.0030, +0.0040] | 0.0001 | 0.0007 |
|  |  | OOD host macro-F1 | 0.9419 | 0.9264 | +0.0155 [+0.0147, +0.0163] | 0.0001 | 0.0007 |
|  |  | Relative OOD drop | 0.0391 | 0.0516 | −0.0124 [−0.0134, −0.0115] | 0.0001 | 0.0007 |
|  |  | Test silent recall | 0.904 | 0.902 | +0.003 [−0.000, +0.006] | 0.1275 | 0.5737 |
|  |  | OOD silent recall | 0.764 | 0.736 | +0.028 [+0.023, +0.033] | 0.0001 | 0.0007 |
|  |  | Test N_hist recall | 0.885 | 0.886 | −0.001 [−0.009, +0.007] | 0.9023 | 1.0000 |
|  |  | OOD N_hist recall | 0.826 | 0.791 | +0.034 [+0.026, +0.044] | 0.0001 | 0.0007 |
| Host-token GRU, no bias | Full-history GRU | Test host macro-F1 | 0.9801 | 0.9768 | +0.0033 [+0.0028, +0.0038] | 0.0001 | 0.0007 |
|  |  | OOD host macro-F1 | 0.9413 | 0.9264 | +0.0149 [+0.0141, +0.0158] | 0.0001 | 0.0007 |
|  |  | Relative OOD drop | 0.0395 | 0.0516 | −0.0120 [−0.0130, −0.0111] | 0.0001 | 0.0007 |
|  |  | Test silent recall | 0.904 | 0.902 | +0.003 [−0.000, +0.006] | 0.0892 | 0.7136 |
|  |  | OOD silent recall | 0.763 | 0.736 | +0.027 [+0.022, +0.031] | 0.0001 | 0.0007 |
|  |  | Test N_hist recall | 0.876 | 0.886 | −0.009 [−0.019, −0.001] | 0.0752 | 0.4510 |
|  |  | OOD N_hist recall | 0.814 | 0.791 | +0.023 [+0.014, +0.032] | 0.0001 | 0.0007 |

Note. All models seeds 0–4. Paired episode bootstrap (100,000 resamples, Algorithm 2); Holm (pair) over the 7 criteria of a pair, Holm (family) over the m = 35 rows of this table. The host-token GRU references are compared with CyberMetaFormer run 2, trained on the same hardware (Table @tbl:hostgru). Lower is better for the relative OOD drop.

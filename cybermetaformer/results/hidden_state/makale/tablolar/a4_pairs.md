**Item 4 family** (paired episode bootstrap, 100,000 resamples; plan-wide Holm over m = 35 rows)

| A | B | Seeds A / B | Criterion | A | B | A − B | 95% CI | p (Holm, pair) | p (Holm, plan) |
|---|---|---|---|---:|---:|---:|---|---:|---:|
| Host-token GRU, topology bias | CMF, run 2 | 5 / 5 | Test host macro-F1 | 0.9803 | 0.9800 | +0.0002 | [+0.0001, +0.0004] | 0.0253 | 0.1179 |
| Host-token GRU, topology bias | CMF, run 2 | 5 / 5 | OOD host macro-F1 | 0.9419 | 0.9414 | +0.0005 | [+0.0001, +0.0009] | 0.0318 | 0.2064 |
| Host-token GRU, topology bias | CMF, run 2 | 5 / 5 | Relative OOD drop | 0.0391 | 0.0394 | −0.0003 | [−0.0007, +0.0002] | 0.2230 | 1.0000 |
| Host-token GRU, topology bias | CMF, run 2 | 5 / 5 | Test silent recall | 0.904 | 0.906 | −0.002 | [−0.004, −0.001] | 0.0086 | 0.0346 |
| Host-token GRU, topology bias | CMF, run 2 | 5 / 5 | OOD silent recall | 0.764 | 0.785 | −0.021 | [−0.024, −0.019] | 0.0001 | 0.0007 |
| Host-token GRU, topology bias | CMF, run 2 | 5 / 5 | Test N_hist recall | 0.885 | 0.892 | −0.007 | [−0.010, −0.003] | 0.0012 | 0.0043 |
| Host-token GRU, topology bias | CMF, run 2 | 5 / 5 | OOD N_hist recall | 0.826 | 0.836 | −0.010 | [−0.014, −0.006] | 0.0001 | 0.0007 |
| Host-token GRU, no bias | CMF, run 2 | 5 / 5 | Test host macro-F1 | 0.9801 | 0.9800 | +0.0000 | [−0.0002, +0.0003] | 1.0000 | 1.0000 |
| Host-token GRU, no bias | CMF, run 2 | 5 / 5 | OOD host macro-F1 | 0.9413 | 0.9414 | −0.0001 | [−0.0005, +0.0003] | 1.0000 | 1.0000 |
| Host-token GRU, no bias | CMF, run 2 | 5 / 5 | Relative OOD drop | 0.0395 | 0.0394 | +0.0001 | [−0.0003, +0.0006] | 1.0000 | 1.0000 |
| Host-token GRU, no bias | CMF, run 2 | 5 / 5 | Test silent recall | 0.904 | 0.906 | −0.002 | [−0.004, −0.001] | 0.0295 | 0.1107 |
| Host-token GRU, no bias | CMF, run 2 | 5 / 5 | OOD silent recall | 0.763 | 0.785 | −0.022 | [−0.025, −0.020] | 0.0001 | 0.0007 |
| Host-token GRU, no bias | CMF, run 2 | 5 / 5 | Test N_hist recall | 0.876 | 0.892 | −0.015 | [−0.021, −0.010] | 0.0001 | 0.0007 |
| Host-token GRU, no bias | CMF, run 2 | 5 / 5 | OOD N_hist recall | 0.814 | 0.836 | −0.021 | [−0.026, −0.017] | 0.0001 | 0.0007 |
| Host-token GRU, topology bias | Host-token GRU, no bias | 5 / 5 | Test host macro-F1 | 0.9803 | 0.9801 | +0.0002 | [+0.0000, +0.0004] | 0.1526 | 0.4510 |
| Host-token GRU, topology bias | Host-token GRU, no bias | 5 / 5 | OOD host macro-F1 | 0.9419 | 0.9413 | +0.0006 | [+0.0002, +0.0009] | 0.0053 | 0.0180 |
| Host-token GRU, topology bias | Host-token GRU, no bias | 5 / 5 | Relative OOD drop | 0.0391 | 0.0395 | −0.0004 | [−0.0008, −0.0000] | 0.1526 | 0.4862 |
| Host-token GRU, topology bias | Host-token GRU, no bias | 5 / 5 | Test silent recall | 0.904 | 0.904 | +0.000 | [−0.001, +0.002] | 0.8615 | 1.0000 |
| Host-token GRU, topology bias | Host-token GRU, no bias | 5 / 5 | OOD silent recall | 0.764 | 0.763 | +0.001 | [−0.001, +0.003] | 0.5860 | 1.0000 |
| Host-token GRU, topology bias | Host-token GRU, no bias | 5 / 5 | Test N_hist recall | 0.885 | 0.876 | +0.009 | [+0.004, +0.014] | 0.0013 | 0.0042 |
| Host-token GRU, topology bias | Host-token GRU, no bias | 5 / 5 | OOD N_hist recall | 0.826 | 0.814 | +0.011 | [+0.007, +0.016] | 0.0001 | 0.0007 |
| Host-token GRU, topology bias | Full-history GRU | 5 / 5 | Test host macro-F1 | 0.9803 | 0.9768 | +0.0035 | [+0.0030, +0.0040] | 0.0001 | 0.0007 |
| Host-token GRU, topology bias | Full-history GRU | 5 / 5 | OOD host macro-F1 | 0.9419 | 0.9264 | +0.0155 | [+0.0147, +0.0163] | 0.0001 | 0.0007 |
| Host-token GRU, topology bias | Full-history GRU | 5 / 5 | Relative OOD drop | 0.0391 | 0.0516 | −0.0124 | [−0.0134, −0.0115] | 0.0001 | 0.0007 |
| Host-token GRU, topology bias | Full-history GRU | 5 / 5 | Test silent recall | 0.904 | 0.902 | +0.003 | [−0.000, +0.006] | 0.1275 | 0.5737 |
| Host-token GRU, topology bias | Full-history GRU | 5 / 5 | OOD silent recall | 0.764 | 0.736 | +0.028 | [+0.023, +0.033] | 0.0001 | 0.0007 |
| Host-token GRU, topology bias | Full-history GRU | 5 / 5 | Test N_hist recall | 0.885 | 0.886 | −0.001 | [−0.009, +0.007] | 0.9023 | 1.0000 |
| Host-token GRU, topology bias | Full-history GRU | 5 / 5 | OOD N_hist recall | 0.826 | 0.791 | +0.034 | [+0.026, +0.044] | 0.0001 | 0.0007 |
| Host-token GRU, no bias | Full-history GRU | 5 / 5 | Test host macro-F1 | 0.9801 | 0.9768 | +0.0033 | [+0.0028, +0.0038] | 0.0001 | 0.0007 |
| Host-token GRU, no bias | Full-history GRU | 5 / 5 | OOD host macro-F1 | 0.9413 | 0.9264 | +0.0149 | [+0.0141, +0.0158] | 0.0001 | 0.0007 |
| Host-token GRU, no bias | Full-history GRU | 5 / 5 | Relative OOD drop | 0.0395 | 0.0516 | −0.0120 | [−0.0130, −0.0111] | 0.0001 | 0.0007 |
| Host-token GRU, no bias | Full-history GRU | 5 / 5 | Test silent recall | 0.904 | 0.902 | +0.003 | [−0.000, +0.006] | 0.0892 | 0.7136 |
| Host-token GRU, no bias | Full-history GRU | 5 / 5 | OOD silent recall | 0.763 | 0.736 | +0.027 | [+0.022, +0.031] | 0.0001 | 0.0007 |
| Host-token GRU, no bias | Full-history GRU | 5 / 5 | Test N_hist recall | 0.876 | 0.886 | −0.009 | [−0.019, −0.001] | 0.0752 | 0.4510 |
| Host-token GRU, no bias | Full-history GRU | 5 / 5 | OOD N_hist recall | 0.814 | 0.791 | +0.023 | [+0.014, +0.032] | 0.0001 | 0.0007 |

**Related ablations from the main comparison plan (5-seed rule)** (paired episode bootstrap, 100,000 resamples; plan-wide Holm over m = 116 rows)

| A | B | Seeds A / B | Criterion | A | B | A − B | 95% CI | p (Holm, pair) | p (Holm, plan) |
|---|---|---|---|---:|---:|---:|---|---:|---:|
| CMF, no topology bias | CMF | 3 / 3 | Test host macro-F1 | 0.9800 | 0.9801 | −0.0001 | [−0.0003, +0.0001] | 0.2754 | 1.0000 |
| CMF, no topology bias | CMF | 3 / 3 | OOD host macro-F1 | 0.9411 | 0.9417 | −0.0006 | [−0.0010, −0.0002] | 0.0148 | 0.1186 |
| CMF, no topology bias | CMF | 3 / 3 | OOD silent recall | 0.774 | 0.776 | −0.002 | [−0.005, −0.000] | 0.0946 | 0.8044 |
| CMF, GRU for SSD | CMF | 3 / 3 | Test host macro-F1 | 0.9806 | 0.9801 | +0.0005 | [+0.0002, +0.0007] | 0.0008 | 0.0110 |
| CMF, GRU for SSD | CMF | 3 / 3 | OOD host macro-F1 | 0.9428 | 0.9417 | +0.0011 | [+0.0007, +0.0016] | < 0.0001 | 0.0023 |
| CMF, GRU for SSD | CMF | 3 / 3 | OOD silent recall | 0.777 | 0.776 | +0.001 | [−0.002, +0.004] | 0.3873 | 1.0000 |
| CMF, no host mixer | CMF | 3 / 3 | Test host macro-F1 | 0.9598 | 0.9801 | −0.0203 | [−0.0215, −0.0192] | < 0.0001 | 0.0023 |
| CMF, no host mixer | CMF | 3 / 3 | OOD host macro-F1 | 0.8993 | 0.9417 | −0.0424 | [−0.0441, −0.0408] | < 0.0001 | 0.0023 |
| CMF, no host mixer | CMF | 3 / 3 | OOD silent recall | 0.591 | 0.776 | −0.185 | [−0.195, −0.176] | < 0.0001 | 0.0023 |

**No-alert subset, host mixing (same plan)** (paired episode bootstrap, 100,000 resamples; plan-wide Holm over m = 116 rows)

| A | B | Seeds A / B | Criterion | A | B | A − B | 95% CI | p (Holm, pair) | p (Holm, plan) |
|---|---|---|---|---:|---:|---:|---|---:|---:|
| CMF | CMF, no host mixer | 3 / 3 | Test N_hist recall | 0.887 | 0.680 | +0.208 | [+0.187, +0.228] | < 0.0001 | 0.0023 |
| CMF | CMF, no host mixer | 3 / 3 | OOD N_hist recall | 0.830 | 0.454 | +0.376 | [+0.357, +0.395] | < 0.0001 | 0.0023 |
| CMF, no topology bias | CMF, no host mixer | 3 / 3 | Test N_hist recall | 0.895 | 0.680 | +0.215 | [+0.195, +0.236] | < 0.0001 | 0.0023 |
| CMF, no topology bias | CMF, no host mixer | 3 / 3 | OOD N_hist recall | 0.831 | 0.454 | +0.378 | [+0.359, +0.396] | < 0.0001 | 0.0023 |

Note. A − B uses each model's seed mean over the seeds listed; ablations are compared with CyberMetaFormer seeds 0–2. Both models see the same episode weights in every resample; test and OOD episodes are resampled independently (Algorithm 2). Two-sided p = 2(min(#Δ ≤ 0, #Δ ≥ 0) + 1)/(n + 1); the smallest attainable raw p is 2/(n + 1). Lower is better for the relative OOD drop.

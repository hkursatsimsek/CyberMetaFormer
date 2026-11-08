| Model | Seeds | Test N_hist | OOD N_hist | Test N_ep | OOD N_ep |
|---|---:|---:|---:|---:|---:|
| CMF (topology-biased attention) | 5 | 0.884 ± 0.008 | 0.826 ± 0.014 | 0.901 ± 0.008 | 0.851 ± 0.011 |
| CMF, full attention (no topology bias) | 3 | 0.895 ± 0.010 | 0.831 ± 0.023 | 0.910 ± 0.011 | 0.853 ± 0.019 |
| CMF, GCN-like host mixer | 3 | 0.886 ± 0.004 | 0.809 ± 0.010 | 0.902 ± 0.004 | 0.837 ± 0.014 |
| CMF, no host mixer | 3 | 0.680 ± 0.019 | 0.454 ± 0.038 | 0.709 ± 0.018 | 0.488 ± 0.043 |
| CMF, single 78-d token | 3 | 0.890 ± 0.002 | 0.773 ± 0.009 | 0.910 ± 0.002 | 0.795 ± 0.009 |
| Host-token GRU, topology-biased attention | 5 | 0.885 ± 0.003 | 0.826 ± 0.010 | 0.903 ± 0.003 | 0.849 ± 0.011 |
| Host-token GRU, attention without bias | 5 | 0.876 ± 0.008 | 0.814 ± 0.020 | 0.896 ± 0.008 | 0.842 ± 0.020 |
| Full-history GRU (flat input) | 5 | 0.886 ± 0.003 | 0.791 ± 0.010 | 0.906 ± 0.004 | 0.818 ± 0.011 |
| Δ CMF − no host mixer (seeds 0–2) | — | +0.208 [+0.187, +0.228] | +0.376 [+0.357, +0.395] | +0.195 [+0.175, +0.217] | +0.365 [+0.341, +0.389] |
| Δ CMF − full-history GRU (seeds 0–4) | — | −0.002 [−0.011, +0.007] | +0.035 [+0.026, +0.044] | −0.005 [−0.015, +0.005] | +0.033 [+0.022, +0.044] |

Note. Recall = share of the silent host-steps of the subset predicted as compromised (user or root). N_hist: no alert on the host in obs[1..t] (n = 54,731 test, 96,486 OOD host-steps); N_ep: no alert on the host in the whole episode (n = 50,884 test, 69,088 OOD). Seed mean ± SD over the seeds listed; the CMF row uses seeds 0–4, so its values differ slightly from the seed 0–2 values in Table @tbl:hic. Δ rows: paired episode bootstrap (10,000 resamples, Algorithm 2) with the seeds shared by both models, Holm over the 10 comparisons of each subset; p (Holm) ≤ 0.002 except CMF − full-history GRU, Test N_hist (p = 1.00); CMF − full-history GRU, Test N_ep (p = 0.63).

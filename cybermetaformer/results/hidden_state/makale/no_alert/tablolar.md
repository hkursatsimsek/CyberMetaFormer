# No-alert subsets (M4 item 1)

- Alert: A(h,τ) = exploit-activity bit (b1) OR security flag g1 (b3) in obs[τ], τ = 1..T (features.alerts(kind='export'))
- Silent: y(h,t) > 0 and not A(h,t)
- N_hist(h,t): no alert on h up to t (online)
- N_ep(h): no alert on h in the whole episode

## Prevalence

| Measure | Train | Test | OOD |
|---|---:|---:|---:|
| Compromised / all host-steps | 0.224 | 0.223 | 0.113 |
| Silent / compromised host-steps | 0.408 | 0.404 | 0.593 |
| Root share among silent | 0.980 | 0.979 | 0.975 |
| N_hist / silent | 0.260 | 0.253 | 0.301 |
| N_ep / silent | 0.242 | 0.235 | 0.215 |
| N_ep / N_hist | 0.931 | 0.930 | 0.716 |
| Previous alert within 32 steps* | 0.558 | 0.559 | 0.513 |
| Previous alert within 64 steps* | 0.891 | 0.894 | 0.885 |
| Silent host-steps (n) | 1,759,365 | 216,495 | 321,008 |
| N_hist host-steps (n) | 457,746 | 54,731 | 96,486 |
| N_ep host-steps (n) | 426,051 | 50,884 | 69,088 |
| N_hist (episode, host) pairs | 7,229 | 861 | 1,891 |
| N_ep (episode, host) pairs | 5,584 | 676 | 1,022 |
| Episodes | 16,000 | 2,000 | 4,000 |

*Among silent compromises whose host had an earlier alert.

## Recall (seed mean ± sd)

| Model | Seeds | Test N_hist | OOD N_hist | Test N_ep | OOD N_ep |
|---|---:|---:|---:|---:|---:|
| CMF (topology-biased host attention) | 5 | 0.884 ± 0.008 | 0.826 ± 0.014 | 0.901 ± 0.008 | 0.851 ± 0.011 |
| CMF, full host attention (no topology bias) | 3 | 0.895 ± 0.010 | 0.831 ± 0.023 | 0.910 ± 0.011 | 0.853 ± 0.019 |
| CMF, GCN-like host mixer | 3 | 0.886 ± 0.004 | 0.809 ± 0.010 | 0.902 ± 0.004 | 0.837 ± 0.014 |
| CMF, no host mixer | 3 | 0.680 ± 0.019 | 0.454 ± 0.038 | 0.709 ± 0.018 | 0.488 ± 0.043 |
| CMF, single 78-d token | 3 | 0.890 ± 0.002 | 0.773 ± 0.009 | 0.910 ± 0.002 | 0.795 ± 0.009 |
| Full-history GRU | 5 | 0.886 ± 0.003 | 0.791 ± 0.010 | 0.906 ± 0.004 | 0.818 ± 0.011 |
| Host-token GRU, topology-biased host attention | 5 | 0.885 ± 0.003 | 0.826 ± 0.010 | 0.903 ± 0.003 | 0.849 ± 0.011 |
| Host-token GRU, full host attention | 5 | 0.876 ± 0.008 | 0.814 ± 0.020 | 0.896 ± 0.008 | 0.842 ± 0.020 |

## Paired episode bootstrap, hist (10000 resamples; Holm within this table)

Both models of a pair enter with the seeds they share (seed mean).

| A | B | Seeds | Split | A − B | 95% CI | p (Holm) |
|---|---|---|---|---:|---|---:|
| CMF (topology-biased host attention) | CMF, no host mixer | 0, 1, 2 | test | +0.208 | [+0.187, +0.228] | 0.0020 |
| CMF (topology-biased host attention) | CMF, no host mixer | 0, 1, 2 | ood | +0.376 | [+0.357, +0.395] | 0.0020 |
| CMF, full host attention (no topology bias) | CMF, no host mixer | 0, 1, 2 | test | +0.215 | [+0.195, +0.236] | 0.0020 |
| CMF, full host attention (no topology bias) | CMF, no host mixer | 0, 1, 2 | ood | +0.378 | [+0.359, +0.397] | 0.0020 |
| CMF, GCN-like host mixer | CMF, no host mixer | 0, 1, 2 | test | +0.206 | [+0.186, +0.227] | 0.0020 |
| CMF, GCN-like host mixer | CMF, no host mixer | 0, 1, 2 | ood | +0.355 | [+0.337, +0.375] | 0.0020 |
| CMF (topology-biased host attention) | CMF, single 78-d token | 0, 1, 2 | test | -0.003 | [-0.014, +0.008] | 1.0000 |
| CMF (topology-biased host attention) | CMF, single 78-d token | 0, 1, 2 | ood | +0.057 | [+0.046, +0.068] | 0.0020 |
| CMF (topology-biased host attention) | Full-history GRU | 0, 1, 2, 3, 4 | test | -0.002 | [-0.011, +0.007] | 1.0000 |
| CMF (topology-biased host attention) | Full-history GRU | 0, 1, 2, 3, 4 | ood | +0.035 | [+0.026, +0.044] | 0.0020 |

## Paired episode bootstrap, ep (10000 resamples; Holm within this table)

Both models of a pair enter with the seeds they share (seed mean).

| A | B | Seeds | Split | A − B | 95% CI | p (Holm) |
|---|---|---|---|---:|---|---:|
| CMF (topology-biased host attention) | CMF, no host mixer | 0, 1, 2 | test | +0.195 | [+0.175, +0.217] | 0.0020 |
| CMF (topology-biased host attention) | CMF, no host mixer | 0, 1, 2 | ood | +0.365 | [+0.341, +0.389] | 0.0020 |
| CMF, full host attention (no topology bias) | CMF, no host mixer | 0, 1, 2 | test | +0.201 | [+0.181, +0.223] | 0.0020 |
| CMF, full host attention (no topology bias) | CMF, no host mixer | 0, 1, 2 | ood | +0.366 | [+0.342, +0.389] | 0.0020 |
| CMF, GCN-like host mixer | CMF, no host mixer | 0, 1, 2 | test | +0.193 | [+0.172, +0.214] | 0.0020 |
| CMF, GCN-like host mixer | CMF, no host mixer | 0, 1, 2 | ood | +0.349 | [+0.326, +0.373] | 0.0020 |
| CMF (topology-biased host attention) | CMF, single 78-d token | 0, 1, 2 | test | -0.006 | [-0.017, +0.005] | 0.6291 |
| CMF (topology-biased host attention) | CMF, single 78-d token | 0, 1, 2 | ood | +0.057 | [+0.043, +0.071] | 0.0020 |
| CMF (topology-biased host attention) | Full-history GRU | 0, 1, 2, 3, 4 | test | -0.005 | [-0.015, +0.005] | 0.6291 |
| CMF (topology-biased host attention) | Full-history GRU | 0, 1, 2, 3, 4 | ood | +0.033 | [+0.022, +0.044] | 0.0020 |

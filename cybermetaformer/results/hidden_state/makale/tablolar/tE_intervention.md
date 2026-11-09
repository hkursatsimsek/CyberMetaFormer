| Intervention | Host set S | Model | Δ silent recall | Δ N_hist recall | Decision change, silent | Δ recall, alerted breaches | Δ false alarm, clean |
|---|---|---|---:|---:|---:|---:|---:|
| Empty (t = 0 observation) | Related hosts | CMF | −0.314 ± 0.161 | −0.586 ± 0.222 | 0.399 ± 0.131 | −0.000 ± 0.000 | +0.108 ± 0.046 |
|  |  | CMF, no topology bias | −0.187 ± 0.084 | −0.436 ± 0.215 | 0.305 ± 0.065 | −0.000 ± 0.000 | +0.158 ± 0.035 |
|  |  | Full-history GRU | −0.449 ± 0.044 | −0.604 ± 0.042 | 0.506 ± 0.039 | −0.029 ± 0.013 | +0.066 ± 0.009 |
| Empty (t = 0 observation) | Random unrelated hosts | CMF | +0.049 ± 0.005 | +0.036 ± 0.008 | 0.067 ± 0.004 | +0.000 ± 0.000 | +0.096 ± 0.008 |
|  |  | CMF, no topology bias | +0.024 ± 0.023 | −0.005 ± 0.046 | 0.078 ± 0.015 | −0.000 ± 0.000 | +0.090 ± 0.013 |
|  |  | Full-history GRU | −0.008 ± 0.007 | −0.023 ± 0.010 | 0.122 ± 0.005 | −0.000 ± 0.000 | +0.115 ± 0.013 |
| Swap (other episode, same policies) | Related hosts | CMF | −0.177 ± 0.031 | −0.202 ± 0.016 | 0.267 ± 0.027 | −0.000 ± 0.000 | +0.076 ± 0.007 |
|  |  | CMF, no topology bias | −0.159 ± 0.016 | −0.197 ± 0.014 | 0.246 ± 0.010 | −0.000 ± 0.000 | +0.081 ± 0.009 |
|  |  | Full-history GRU | −0.210 ± 0.005 | −0.212 ± 0.006 | 0.296 ± 0.003 | −0.024 ± 0.006 | +0.090 ± 0.003 |
| Swap (other episode, same policies) | Random unrelated hosts | CMF | −0.035 ± 0.011 | −0.022 ± 0.007 | 0.105 ± 0.007 | −0.000 ± 0.000 | +0.043 ± 0.005 |
|  |  | CMF, no topology bias | −0.025 ± 0.009 | −0.014 ± 0.006 | 0.096 ± 0.008 | −0.000 ± 0.000 | +0.046 ± 0.006 |
|  |  | Full-history GRU | −0.049 ± 0.003 | −0.032 ± 0.006 | 0.122 ± 0.004 | −0.007 ± 0.001 | +0.044 ± 0.003 |
| Shuffle (in time) | Related hosts | CMF | −0.100 ± 0.025 | −0.151 ± 0.031 | 0.196 ± 0.018 | −0.000 ± 0.000 | +0.070 ± 0.008 |
|  |  | CMF, no topology bias | −0.099 ± 0.015 | −0.174 ± 0.032 | 0.190 ± 0.020 | −0.000 ± 0.000 | +0.069 ± 0.005 |
|  |  | Full-history GRU | −0.078 ± 0.016 | −0.073 ± 0.021 | 0.181 ± 0.011 | −0.008 ± 0.002 | +0.095 ± 0.005 |
| Shuffle (in time) | Random unrelated hosts | CMF | −0.063 ± 0.025 | −0.191 ± 0.049 | 0.123 ± 0.019 | −0.000 ± 0.000 | +0.031 ± 0.005 |
|  |  | CMF, no topology bias | −0.057 ± 0.020 | −0.180 ± 0.042 | 0.117 ± 0.013 | −0.000 ± 0.000 | +0.032 ± 0.006 |
|  |  | Full-history GRU | −0.058 ± 0.012 | −0.140 ± 0.025 | 0.128 ± 0.009 | −0.006 ± 0.001 | +0.045 ± 0.007 |

Note. Test episodes of the training policies (2,000). The target host's own observation columns are kept; the columns of a host set S are replaced at every step. Empty: S's own t = 0 observation; swap: S's columns from a random other episode with the same Red and Blue policy (keeps the marginal and the alert count, breaks the joint distribution); shuffle: S's columns permuted in time within the episode (t ≥ 1). S = related hosts (attack-graph neighbours and same subnet in the training topology) or three random equal-size sets of unrelated hosts (pooled). Targets: the 8 label hosts ent0, ent1, ent2, opserv, user1, user2, user3, user4; host-steps pooled over targets (211,741 silent, 54,731 N_hist). Silent and N_hist are defined by the target's own observation and are unchanged by the intervention. Δ = intervened − original, mean ± SD over seeds 0–4. Decision change: share of the silent target host-steps whose predicted class changes. Alerted breaches: compromised target host-steps with an alert in the current observation (original recall CMF 1.000, CMF, no topology bias 1.000, Full-history GRU 0.999); false alarm: P(ŷ ≠ clean | y = clean) on the targets (original CMF 0.013, CMF, no topology bias 0.014, Full-history GRU 0.016).

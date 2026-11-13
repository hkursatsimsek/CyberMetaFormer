# Intervention experiments: does the target-host estimate use other hosts' observations?

Test set (2,000 episodes, training policies). The target host's own observation columns are kept; the columns of a host set S are replaced at every step: empty = S's own t = 0 observation; swap = S's columns from a random other episode with the same Red and Blue policy; shuffle = S's columns permuted in time within the episode (t ≥ 1). S = related hosts (attack-graph neighbours and same subnet in the training topology) or 3 random equal-size sets of unrelated hosts (pooled). Targets: ent0, ent1, ent2, opserv, user1, user2, user3, user4; host-steps pooled over targets. Silent and N_hist are defined by the target's own observation and are unchanged by the intervention. Δ = intervened − original; mean ± sd over 5 seeds; CIs from a paired episode bootstrap of the seed mean (100,000 resamples), Holm over all 114 rows. Interventions act on observations, not on the simulator, so the results show how much the estimates depend on other hosts' inputs; they are not counterfactual evidence about the environment.

## Host sets

| Target | Related | Random draws |
|---|---|---|
| ent0 | def, ent1, ent2, user1, user2 | ophost1, ophost2, opserv, user0, user4 / ophost2, opserv, user0, user3, user4 / ophost1, opserv, user0, user3, user4 |
| ent1 | def, ent0, ent2, user3, user4 | ophost0, ophost1, opserv, user0, user1 / ophost0, ophost1, ophost2, user1, user2 / ophost1, ophost2, opserv, user0, user2 |
| ent2 | def, ent0, ent1, opserv | ophost2, user1, user2, user4 / ophost2, user0, user1, user4 / ophost2, user0, user1, user4 |
| opserv | ent2, ophost0, ophost1, ophost2 | def, user1, user3, user4 / ent0, user1, user2, user3 / ent0, ent1, user2, user3 |
| user1 | ent0, user0, user2, user3, user4 | def, ent1, ent2, ophost0, ophost1 / def, ent2, ophost1, ophost2, opserv / ent1, ent2, ophost0, ophost1, ophost2 |
| user2 | ent0, user0, user1, user3, user4 | def, ophost0, ophost1, ophost2, opserv / def, ent2, ophost0, ophost2, opserv / ent1, ent2, ophost0, ophost1, opserv |
| user3 | ent1, user0, user1, user2, user4 | def, ent0, ent2, ophost0, ophost1 / ent0, ent2, ophost0, ophost1, opserv / def, ent0, ophost1, ophost2, opserv |
| user4 | ent1, user0, user1, user2, user3 | def, ophost0, ophost1, ophost2, opserv / def, ophost0, ophost1, ophost2, opserv / ent0, ent2, ophost1, ophost2, opserv |

## Target host-steps

| Target | Silent recall | N_hist recall | Recall on alerted breaches | False-alarm rate on clean steps |
|---|---|---|---|---|
| ent0 | 21,917 | 4,899 | 38,037 | 140,046 |
| ent1 | 29,754 | 8,462 | 34,916 | 135,330 |
| ent2 | 23,469 | 7,655 | 45,365 | 131,166 |
| opserv | 6,520 | 0 | 61,417 | 132,063 |
| user1 | 31,945 | 8,654 | 30,505 | 137,550 |
| user2 | 32,665 | 8,940 | 30,842 | 136,493 |
| user3 | 32,307 | 8,150 | 32,113 | 135,580 |
| user4 | 33,164 | 7,971 | 31,172 | 135,664 |

## Original values

| Model | Silent recall | N_hist recall | Recall on alerted breaches | False-alarm rate on clean steps |
|---|---|---|---|---|
| CyberMetaFormer | 0.902 ± 0.006 | 0.884 ± 0.008 | 1.000 ± 0.000 | 0.013 ± 0.001 |
| CMF, full attention (no topology bias) | 0.903 ± 0.006 | 0.885 ± 0.016 | 1.000 ± 0.000 | 0.014 ± 0.001 |
| GRU (full history) | 0.899 ± 0.002 | 0.886 ± 0.003 | 0.999 ± 0.000 | 0.016 ± 0.000 |

## Effects (Δ = intervened − original; mean ± sd over seeds)

| Intervention | Hosts | Model | Δ silent recall | Δ N_hist recall | Decision change (silent) | ΔP(breach) (silent) | Δ recall, alerted | Δ false alarm, clean |
|---|---|---|---|---|---|---|---|---|
| empty (t = 0 observation) | related | CyberMetaFormer | -0.314 ± 0.161 | -0.586 ± 0.222 | 0.399 ± 0.131 | -0.308 ± 0.150 | -0.000 ± 0.000 | +0.108 ± 0.046 |
| empty (t = 0 observation) | related | CMF, full attention (no topology bias) | -0.187 ± 0.084 | -0.436 ± 0.215 | 0.305 ± 0.065 | -0.183 ± 0.075 | -0.000 ± 0.000 | +0.158 ± 0.035 |
| empty (t = 0 observation) | related | GRU (full history) | -0.449 ± 0.044 | -0.604 ± 0.042 | 0.506 ± 0.039 | -0.421 ± 0.043 | -0.029 ± 0.013 | +0.066 ± 0.009 |
| empty (t = 0 observation) | random | CyberMetaFormer | +0.049 ± 0.005 | +0.036 ± 0.008 | 0.067 ± 0.004 | +0.047 ± 0.005 | +0.000 ± 0.000 | +0.096 ± 0.008 |
| empty (t = 0 observation) | random | CMF, full attention (no topology bias) | +0.024 ± 0.023 | -0.005 ± 0.046 | 0.078 ± 0.015 | +0.023 ± 0.022 | -0.000 ± 0.000 | +0.090 ± 0.013 |
| empty (t = 0 observation) | random | GRU (full history) | -0.008 ± 0.007 | -0.023 ± 0.010 | 0.122 ± 0.005 | +0.000 ± 0.007 | -0.000 ± 0.000 | +0.115 ± 0.013 |
| swap (other episode, same policies) | related | CyberMetaFormer | -0.177 ± 0.031 | -0.202 ± 0.016 | 0.267 ± 0.027 | -0.173 ± 0.029 | -0.000 ± 0.000 | +0.076 ± 0.007 |
| swap (other episode, same policies) | related | CMF, full attention (no topology bias) | -0.159 ± 0.016 | -0.197 ± 0.014 | 0.246 ± 0.010 | -0.152 ± 0.014 | -0.000 ± 0.000 | +0.081 ± 0.009 |
| swap (other episode, same policies) | related | GRU (full history) | -0.210 ± 0.005 | -0.212 ± 0.006 | 0.296 ± 0.003 | -0.197 ± 0.005 | -0.024 ± 0.006 | +0.090 ± 0.003 |
| swap (other episode, same policies) | random | CyberMetaFormer | -0.035 ± 0.011 | -0.022 ± 0.007 | 0.105 ± 0.007 | -0.037 ± 0.010 | -0.000 ± 0.000 | +0.043 ± 0.005 |
| swap (other episode, same policies) | random | CMF, full attention (no topology bias) | -0.025 ± 0.009 | -0.014 ± 0.006 | 0.096 ± 0.008 | -0.026 ± 0.009 | -0.000 ± 0.000 | +0.046 ± 0.006 |
| swap (other episode, same policies) | random | GRU (full history) | -0.049 ± 0.003 | -0.032 ± 0.006 | 0.122 ± 0.004 | -0.045 ± 0.003 | -0.007 ± 0.001 | +0.044 ± 0.003 |
| shuffle (in time) | related | CyberMetaFormer | -0.100 ± 0.025 | -0.151 ± 0.031 | 0.196 ± 0.018 | -0.098 ± 0.024 | -0.000 ± 0.000 | +0.070 ± 0.008 |
| shuffle (in time) | related | CMF, full attention (no topology bias) | -0.099 ± 0.015 | -0.174 ± 0.032 | 0.190 ± 0.020 | -0.096 ± 0.015 | -0.000 ± 0.000 | +0.069 ± 0.005 |
| shuffle (in time) | related | GRU (full history) | -0.078 ± 0.016 | -0.073 ± 0.021 | 0.181 ± 0.011 | -0.071 ± 0.015 | -0.008 ± 0.002 | +0.095 ± 0.005 |
| shuffle (in time) | random | CyberMetaFormer | -0.063 ± 0.025 | -0.191 ± 0.049 | 0.123 ± 0.019 | -0.065 ± 0.025 | -0.000 ± 0.000 | +0.031 ± 0.005 |
| shuffle (in time) | random | CMF, full attention (no topology bias) | -0.057 ± 0.020 | -0.180 ± 0.042 | 0.117 ± 0.013 | -0.057 ± 0.019 | -0.000 ± 0.000 | +0.032 ± 0.006 |
| shuffle (in time) | random | GRU (full history) | -0.058 ± 0.012 | -0.140 ± 0.025 | 0.128 ± 0.009 | -0.053 ± 0.012 | -0.006 ± 0.001 | +0.045 ± 0.007 |

## Intervened − original (paired episode bootstrap)

| Intervention | Hosts | Model | Criterion | Δ [95% CI] | p (Holm) |
|---|---|---|---|---|---|
| empty (t = 0 observation) | related | CyberMetaFormer | Silent recall | -0.3136 [-0.3252, -0.3019] | 0.002 |
| empty (t = 0 observation) | random | CyberMetaFormer | Silent recall | +0.0486 [+0.0437, +0.0535] | 0.002 |
| empty (t = 0 observation) | related | CMF, full attention (no topology bias) | Silent recall | -0.1868 [-0.1984, -0.1753] | 0.002 |
| empty (t = 0 observation) | random | CMF, full attention (no topology bias) | Silent recall | +0.0239 [+0.0193, +0.0286] | 0.002 |
| empty (t = 0 observation) | related | GRU (full history) | Silent recall | -0.4489 [-0.4627, -0.4350] | 0.002 |
| empty (t = 0 observation) | random | GRU (full history) | Silent recall | -0.0081 [-0.0176, +0.0011] | 0.953 |
| empty (t = 0 observation) | related | CyberMetaFormer | N_hist recall | -0.5860 [-0.6082, -0.5636] | 0.002 |
| empty (t = 0 observation) | random | CyberMetaFormer | N_hist recall | +0.0360 [+0.0243, +0.0482] | 0.002 |
| empty (t = 0 observation) | related | CMF, full attention (no topology bias) | N_hist recall | -0.4363 [-0.4633, -0.4092] | 0.002 |
| empty (t = 0 observation) | random | CMF, full attention (no topology bias) | N_hist recall | -0.0054 [-0.0181, +0.0076] | 1.000 |
| empty (t = 0 observation) | related | GRU (full history) | N_hist recall | -0.6043 [-0.6269, -0.5812] | 0.002 |
| empty (t = 0 observation) | random | GRU (full history) | N_hist recall | -0.0227 [-0.0422, -0.0033] | 0.347 |
| empty (t = 0 observation) | related | CyberMetaFormer | Recall on alerted breaches | -0.0000 [-0.0000, -0.0000] | 0.002 |
| empty (t = 0 observation) | random | CyberMetaFormer | Recall on alerted breaches | +0.0000 [-0.0000, +0.0000] | 1.000 |
| empty (t = 0 observation) | related | CMF, full attention (no topology bias) | Recall on alerted breaches | -0.0000 [-0.0000, -0.0000] | 0.028 |
| empty (t = 0 observation) | random | CMF, full attention (no topology bias) | Recall on alerted breaches | -0.0000 [-0.0000, +0.0000] | 1.000 |
| empty (t = 0 observation) | related | GRU (full history) | Recall on alerted breaches | -0.0289 [-0.0320, -0.0259] | 0.002 |
| empty (t = 0 observation) | random | GRU (full history) | Recall on alerted breaches | -0.0004 [-0.0008, -0.0002] | 0.050 |
| empty (t = 0 observation) | related | CyberMetaFormer | False-alarm rate on clean steps | +0.1083 [+0.1034, +0.1132] | 0.002 |
| empty (t = 0 observation) | random | CyberMetaFormer | False-alarm rate on clean steps | +0.0959 [+0.0909, +0.1009] | 0.002 |
| empty (t = 0 observation) | related | CMF, full attention (no topology bias) | False-alarm rate on clean steps | +0.1581 [+0.1520, +0.1642] | 0.002 |
| empty (t = 0 observation) | random | CMF, full attention (no topology bias) | False-alarm rate on clean steps | +0.0898 [+0.0853, +0.0944] | 0.002 |
| empty (t = 0 observation) | related | GRU (full history) | False-alarm rate on clean steps | +0.0656 [+0.0623, +0.0690] | 0.002 |
| empty (t = 0 observation) | random | GRU (full history) | False-alarm rate on clean steps | +0.1145 [+0.1095, +0.1195] | 0.002 |
| swap (other episode, same policies) | related | CyberMetaFormer | Silent recall | -0.1767 [-0.1911, -0.1625] | 0.002 |
| swap (other episode, same policies) | random | CyberMetaFormer | Silent recall | -0.0351 [-0.0412, -0.0290] | 0.002 |
| swap (other episode, same policies) | related | CMF, full attention (no topology bias) | Silent recall | -0.1588 [-0.1726, -0.1452] | 0.002 |
| swap (other episode, same policies) | random | CMF, full attention (no topology bias) | Silent recall | -0.0250 [-0.0306, -0.0195] | 0.002 |
| swap (other episode, same policies) | related | GRU (full history) | Silent recall | -0.2095 [-0.2254, -0.1938] | 0.002 |
| swap (other episode, same policies) | random | GRU (full history) | Silent recall | -0.0487 [-0.0555, -0.0420] | 0.002 |
| swap (other episode, same policies) | related | CyberMetaFormer | N_hist recall | -0.2017 [-0.2329, -0.1710] | 0.002 |
| swap (other episode, same policies) | random | CyberMetaFormer | N_hist recall | -0.0223 [-0.0327, -0.0118] | 0.002 |
| swap (other episode, same policies) | related | CMF, full attention (no topology bias) | N_hist recall | -0.1971 [-0.2285, -0.1665] | 0.002 |
| swap (other episode, same policies) | random | CMF, full attention (no topology bias) | N_hist recall | -0.0140 [-0.0245, -0.0035] | 0.165 |
| swap (other episode, same policies) | related | GRU (full history) | N_hist recall | -0.2124 [-0.2436, -0.1823] | 0.002 |
| swap (other episode, same policies) | random | GRU (full history) | N_hist recall | -0.0319 [-0.0424, -0.0214] | 0.002 |
| swap (other episode, same policies) | related | CyberMetaFormer | Recall on alerted breaches | -0.0000 [-0.0000, -0.0000] | 0.004 |
| swap (other episode, same policies) | random | CyberMetaFormer | Recall on alerted breaches | -0.0000 [-0.0000, -0.0000] | 0.015 |
| swap (other episode, same policies) | related | CMF, full attention (no topology bias) | Recall on alerted breaches | -0.0000 [-0.0000, -0.0000] | 0.002 |
| swap (other episode, same policies) | random | CMF, full attention (no topology bias) | Recall on alerted breaches | -0.0000 [-0.0000, -0.0000] | 0.002 |
| swap (other episode, same policies) | related | GRU (full history) | Recall on alerted breaches | -0.0239 [-0.0262, -0.0218] | 0.002 |
| swap (other episode, same policies) | random | GRU (full history) | Recall on alerted breaches | -0.0066 [-0.0072, -0.0060] | 0.002 |
| swap (other episode, same policies) | related | CyberMetaFormer | False-alarm rate on clean steps | +0.0756 [+0.0718, +0.0795] | 0.002 |
| swap (other episode, same policies) | random | CyberMetaFormer | False-alarm rate on clean steps | +0.0434 [+0.0404, +0.0464] | 0.002 |
| swap (other episode, same policies) | related | CMF, full attention (no topology bias) | False-alarm rate on clean steps | +0.0813 [+0.0775, +0.0852] | 0.002 |
| swap (other episode, same policies) | random | CMF, full attention (no topology bias) | False-alarm rate on clean steps | +0.0459 [+0.0429, +0.0491] | 0.002 |
| swap (other episode, same policies) | related | GRU (full history) | False-alarm rate on clean steps | +0.0895 [+0.0857, +0.0933] | 0.002 |
| swap (other episode, same policies) | random | GRU (full history) | False-alarm rate on clean steps | +0.0443 [+0.0415, +0.0473] | 0.002 |
| shuffle (in time) | related | CyberMetaFormer | Silent recall | -0.0998 [-0.1091, -0.0906] | 0.002 |
| shuffle (in time) | random | CyberMetaFormer | Silent recall | -0.0630 [-0.0714, -0.0549] | 0.002 |
| shuffle (in time) | related | CMF, full attention (no topology bias) | Silent recall | -0.0986 [-0.1078, -0.0896] | 0.002 |
| shuffle (in time) | random | CMF, full attention (no topology bias) | Silent recall | -0.0566 [-0.0644, -0.0491] | 0.002 |
| shuffle (in time) | related | GRU (full history) | Silent recall | -0.0776 [-0.0864, -0.0689] | 0.002 |
| shuffle (in time) | random | GRU (full history) | Silent recall | -0.0580 [-0.0651, -0.0509] | 0.002 |
| shuffle (in time) | related | CyberMetaFormer | N_hist recall | -0.1511 [-0.1772, -0.1258] | 0.002 |
| shuffle (in time) | random | CyberMetaFormer | N_hist recall | -0.1912 [-0.2151, -0.1678] | 0.002 |
| shuffle (in time) | related | CMF, full attention (no topology bias) | N_hist recall | -0.1741 [-0.1988, -0.1503] | 0.002 |
| shuffle (in time) | random | CMF, full attention (no topology bias) | N_hist recall | -0.1797 [-0.2021, -0.1578] | 0.002 |
| shuffle (in time) | related | GRU (full history) | N_hist recall | -0.0733 [-0.0923, -0.0547] | 0.002 |
| shuffle (in time) | random | GRU (full history) | N_hist recall | -0.1399 [-0.1575, -0.1227] | 0.002 |
| shuffle (in time) | related | CyberMetaFormer | Recall on alerted breaches | -0.0000 [-0.0000, +0.0000] | 1.000 |
| shuffle (in time) | random | CyberMetaFormer | Recall on alerted breaches | -0.0000 [-0.0000, +0.0000] | 0.895 |
| shuffle (in time) | related | CMF, full attention (no topology bias) | Recall on alerted breaches | -0.0000 [-0.0000, +0.0000] | 1.000 |
| shuffle (in time) | random | CMF, full attention (no topology bias) | Recall on alerted breaches | -0.0000 [-0.0000, -0.0000] | 0.010 |
| shuffle (in time) | related | GRU (full history) | Recall on alerted breaches | -0.0079 [-0.0086, -0.0073] | 0.002 |
| shuffle (in time) | random | GRU (full history) | Recall on alerted breaches | -0.0063 [-0.0067, -0.0059] | 0.002 |
| shuffle (in time) | related | CyberMetaFormer | False-alarm rate on clean steps | +0.0696 [+0.0656, +0.0737] | 0.002 |
| shuffle (in time) | random | CyberMetaFormer | False-alarm rate on clean steps | +0.0307 [+0.0282, +0.0331] | 0.002 |
| shuffle (in time) | related | CMF, full attention (no topology bias) | False-alarm rate on clean steps | +0.0690 [+0.0650, +0.0730] | 0.002 |
| shuffle (in time) | random | CMF, full attention (no topology bias) | False-alarm rate on clean steps | +0.0320 [+0.0296, +0.0346] | 0.002 |
| shuffle (in time) | related | GRU (full history) | False-alarm rate on clean steps | +0.0955 [+0.0907, +0.1003] | 0.002 |
| shuffle (in time) | random | GRU (full history) | False-alarm rate on clean steps | +0.0447 [+0.0420, +0.0475] | 0.002 |

## Related − random hosts (same intervention)

| Intervention | Hosts | Model | Criterion | Δ [95% CI] | p (Holm) |
|---|---|---|---|---|---|
| empty (t = 0 observation) | related − random | CyberMetaFormer | Silent recall | -0.3622 [-0.3743, -0.3500] | 0.002 |
| empty (t = 0 observation) | related − random | CMF, full attention (no topology bias) | Silent recall | -0.2107 [-0.2219, -0.1996] | 0.002 |
| empty (t = 0 observation) | related − random | GRU (full history) | Silent recall | -0.4407 [-0.4556, -0.4258] | 0.002 |
| empty (t = 0 observation) | related − random | CyberMetaFormer | N_hist recall | -0.6220 [-0.6435, -0.6004] | 0.002 |
| empty (t = 0 observation) | related − random | CMF, full attention (no topology bias) | N_hist recall | -0.4309 [-0.4569, -0.4049] | 0.002 |
| empty (t = 0 observation) | related − random | GRU (full history) | N_hist recall | -0.5817 [-0.6080, -0.5547] | 0.002 |
| swap (other episode, same policies) | related − random | CyberMetaFormer | Silent recall | -0.1416 [-0.1550, -0.1284] | 0.002 |
| swap (other episode, same policies) | related − random | CMF, full attention (no topology bias) | Silent recall | -0.1338 [-0.1467, -0.1211] | 0.002 |
| swap (other episode, same policies) | related − random | GRU (full history) | Silent recall | -0.1609 [-0.1753, -0.1465] | 0.002 |
| swap (other episode, same policies) | related − random | CyberMetaFormer | N_hist recall | -0.1793 [-0.2095, -0.1500] | 0.002 |
| swap (other episode, same policies) | related − random | CMF, full attention (no topology bias) | N_hist recall | -0.1831 [-0.2133, -0.1540] | 0.002 |
| swap (other episode, same policies) | related − random | GRU (full history) | N_hist recall | -0.1805 [-0.2110, -0.1510] | 0.002 |
| shuffle (in time) | related − random | CyberMetaFormer | Silent recall | -0.0368 [-0.0476, -0.0259] | 0.002 |
| shuffle (in time) | related − random | CMF, full attention (no topology bias) | Silent recall | -0.0421 [-0.0519, -0.0322] | 0.002 |
| shuffle (in time) | related − random | GRU (full history) | Silent recall | -0.0197 [-0.0282, -0.0111] | 0.002 |
| shuffle (in time) | related − random | CyberMetaFormer | N_hist recall | +0.0401 [+0.0057, +0.0744] | 0.347 |
| shuffle (in time) | related − random | CMF, full attention (no topology bias) | N_hist recall | +0.0055 [-0.0254, +0.0361] | 1.000 |
| shuffle (in time) | related − random | GRU (full history) | N_hist recall | +0.0666 [+0.0451, +0.0880] | 0.002 |

## Difference of effects between models (Δ_a − Δ_b)

| Intervention | Hosts | a | b | Criterion | Δ [95% CI] | p (Holm) |
|---|---|---|---|---|---|---|
| empty (t = 0 observation) | related | CyberMetaFormer | GRU (full history) | Silent recall | +0.1353 [+0.1241, +0.1464] | 0.002 |
| empty (t = 0 observation) | random | CyberMetaFormer | GRU (full history) | Silent recall | +0.0567 [+0.0485, +0.0652] | 0.002 |
| empty (t = 0 observation) | related | CyberMetaFormer | CMF, full attention (no topology bias) | Silent recall | -0.1268 [-0.1346, -0.1190] | 0.002 |
| empty (t = 0 observation) | random | CyberMetaFormer | CMF, full attention (no topology bias) | Silent recall | +0.0247 [+0.0218, +0.0276] | 0.002 |
| empty (t = 0 observation) | related | CyberMetaFormer | GRU (full history) | N_hist recall | +0.0183 [-0.0059, +0.0426] | 1.000 |
| empty (t = 0 observation) | random | CyberMetaFormer | GRU (full history) | N_hist recall | +0.0586 [+0.0417, +0.0762] | 0.002 |
| empty (t = 0 observation) | related | CyberMetaFormer | CMF, full attention (no topology bias) | N_hist recall | -0.1497 [-0.1674, -0.1319] | 0.002 |
| empty (t = 0 observation) | random | CyberMetaFormer | CMF, full attention (no topology bias) | N_hist recall | +0.0414 [+0.0332, +0.0498] | 0.002 |
| swap (other episode, same policies) | related | CyberMetaFormer | GRU (full history) | Silent recall | +0.0328 [+0.0265, +0.0393] | 0.002 |
| swap (other episode, same policies) | random | CyberMetaFormer | GRU (full history) | Silent recall | +0.0136 [+0.0104, +0.0168] | 0.002 |
| swap (other episode, same policies) | related | CyberMetaFormer | CMF, full attention (no topology bias) | Silent recall | -0.0179 [-0.0215, -0.0144] | 0.002 |
| swap (other episode, same policies) | random | CyberMetaFormer | CMF, full attention (no topology bias) | Silent recall | -0.0101 [-0.0120, -0.0082] | 0.002 |
| swap (other episode, same policies) | related | CyberMetaFormer | GRU (full history) | N_hist recall | +0.0108 [-0.0042, +0.0260] | 1.000 |
| swap (other episode, same policies) | random | CyberMetaFormer | GRU (full history) | N_hist recall | +0.0096 [+0.0006, +0.0185] | 0.475 |
| swap (other episode, same policies) | related | CyberMetaFormer | CMF, full attention (no topology bias) | N_hist recall | -0.0045 [-0.0123, +0.0030] | 1.000 |
| swap (other episode, same policies) | random | CyberMetaFormer | CMF, full attention (no topology bias) | N_hist recall | -0.0083 [-0.0130, -0.0038] | 0.009 |
| shuffle (in time) | related | CyberMetaFormer | GRU (full history) | Silent recall | -0.0222 [-0.0289, -0.0155] | 0.002 |
| shuffle (in time) | random | CyberMetaFormer | GRU (full history) | Silent recall | -0.0050 [-0.0095, -0.0006] | 0.349 |
| shuffle (in time) | related | CyberMetaFormer | CMF, full attention (no topology bias) | Silent recall | -0.0011 [-0.0046, +0.0024] | 1.000 |
| shuffle (in time) | random | CyberMetaFormer | CMF, full attention (no topology bias) | Silent recall | -0.0064 [-0.0086, -0.0042] | 0.002 |
| shuffle (in time) | related | CyberMetaFormer | GRU (full history) | N_hist recall | -0.0777 [-0.0969, -0.0590] | 0.002 |
| shuffle (in time) | random | CyberMetaFormer | GRU (full history) | N_hist recall | -0.0512 [-0.0646, -0.0381] | 0.002 |
| shuffle (in time) | related | CyberMetaFormer | CMF, full attention (no topology bias) | N_hist recall | +0.0231 [+0.0134, +0.0329] | 0.002 |
| shuffle (in time) | random | CyberMetaFormer | CMF, full attention (no topology bias) | N_hist recall | -0.0115 [-0.0183, -0.0050] | 0.012 |

## Input shift control

Novel rows = share of intervened full observations (78-dim) never seen in the training split; novel transitions = share of consecutive per-host token pairs of the intervened hosts never seen in training. Alerts = alerts on the intervened hosts per episode, before and after.

| Intervention | Hosts | Novel rows | Novel transitions | Alerts before | Alerts after |
|---|---|---|---|---|---|
| none (original test set) | all | 0.258 | 0.000 | – | – |
| empty (t = 0 observation) | related | 0.763 | 0.000 | 65.6 | 0.0 |
| empty (t = 0 observation) | random | 0.615 | 0.000 | 54.8 | 0.0 |
| swap (other episode, same policies) | related | 0.400 | 0.000 | 65.6 | 65.6 |
| swap (other episode, same policies) | random | 0.371 | 0.000 | 54.8 | 54.8 |
| shuffle (in time) | related | 0.376 | 0.147 | 65.6 | 65.6 |
| shuffle (in time) | random | 0.371 | 0.139 | 54.8 | 54.8 |

## Δ silent recall per target (seed mean; random = mean over draws)

| Model | Target | empty related | empty random | swap related | swap random | shuffle related | shuffle random |
|---|---|---|---|---|---|---|---|
| CyberMetaFormer | ent0 | -0.227 | +0.049 | -0.222 | -0.080 | -0.151 | -0.081 |
| CyberMetaFormer | ent1 | -0.339 | +0.029 | -0.330 | -0.083 | -0.266 | -0.090 |
| CyberMetaFormer | ent2 | -0.689 | +0.025 | -0.411 | -0.069 | -0.436 | -0.037 |
| CyberMetaFormer | opserv | -0.000 | -0.000 | -0.000 | -0.000 | +0.000 | -0.000 |
| CyberMetaFormer | user1 | -0.312 | +0.064 | -0.116 | -0.004 | -0.014 | -0.031 |
| CyberMetaFormer | user2 | -0.279 | +0.048 | -0.123 | -0.004 | -0.015 | -0.093 |
| CyberMetaFormer | user3 | -0.262 | +0.074 | -0.087 | -0.017 | +0.017 | -0.038 |
| CyberMetaFormer | user4 | -0.230 | +0.053 | -0.077 | -0.024 | +0.022 | -0.083 |
| CMF, full attention (no topology bias) | ent0 | -0.179 | -0.027 | -0.218 | -0.065 | -0.141 | -0.087 |
| CMF, full attention (no topology bias) | ent1 | -0.277 | -0.031 | -0.325 | -0.078 | -0.247 | -0.089 |
| CMF, full attention (no topology bias) | ent2 | -0.712 | -0.010 | -0.397 | -0.070 | -0.417 | -0.037 |
| CMF, full attention (no topology bias) | opserv | +0.000 | +0.000 | -0.000 | -0.000 | -0.000 | -0.000 |
| CMF, full attention (no topology bias) | user1 | -0.114 | +0.057 | -0.092 | +0.008 | -0.020 | -0.023 |
| CMF, full attention (no topology bias) | user2 | -0.080 | +0.043 | -0.094 | +0.005 | -0.024 | -0.078 |
| CMF, full attention (no topology bias) | user3 | -0.065 | +0.061 | -0.064 | -0.003 | +0.012 | -0.034 |
| CMF, full attention (no topology bias) | user4 | -0.070 | +0.047 | -0.053 | -0.008 | +0.011 | -0.066 |
| GRU (full history) | ent0 | -0.392 | -0.032 | -0.287 | -0.103 | -0.140 | -0.093 |
| GRU (full history) | ent1 | -0.566 | -0.019 | -0.388 | -0.099 | -0.227 | -0.090 |
| GRU (full history) | ent2 | -0.621 | -0.021 | -0.378 | -0.064 | -0.288 | -0.020 |
| GRU (full history) | opserv | -0.013 | -0.011 | -0.199 | -0.036 | -0.145 | -0.066 |
| GRU (full history) | user1 | -0.419 | -0.031 | -0.139 | -0.012 | -0.013 | -0.013 |
| GRU (full history) | user2 | -0.473 | +0.006 | -0.164 | -0.018 | -0.015 | -0.091 |
| GRU (full history) | user3 | -0.449 | -0.009 | -0.114 | -0.036 | +0.022 | -0.029 |
| GRU (full history) | user4 | -0.350 | +0.035 | -0.087 | -0.037 | +0.038 | -0.070 |

| Intervention | Host set S | Δ_A − Δ_B | Criterion | Difference [95% CI] | p (Holm) |
|---|---|---|---|---:|---:|
| Empty (t = 0 observation) | Related hosts | CMF − Full-history GRU | Silent recall | +0.135 [+0.124, +0.146] | 0.0023 |
|  |  | CMF − CMF, no topology bias | Silent recall | −0.127 [−0.135, −0.119] | 0.0023 |
|  |  | CMF − Full-history GRU | N_hist recall | +0.018 [−0.006, +0.043] | 1.0000 |
|  |  | CMF − CMF, no topology bias | N_hist recall | −0.150 [−0.167, −0.132] | 0.0023 |
| Empty (t = 0 observation) | Random unrelated hosts | CMF − Full-history GRU | Silent recall | +0.057 [+0.048, +0.065] | 0.0023 |
|  |  | CMF − CMF, no topology bias | Silent recall | +0.025 [+0.022, +0.028] | 0.0023 |
|  |  | CMF − Full-history GRU | N_hist recall | +0.059 [+0.042, +0.076] | 0.0023 |
|  |  | CMF − CMF, no topology bias | N_hist recall | +0.041 [+0.033, +0.050] | 0.0023 |
| Swap (other episode, same policies) | Related hosts | CMF − Full-history GRU | Silent recall | +0.033 [+0.027, +0.039] | 0.0023 |
|  |  | CMF − CMF, no topology bias | Silent recall | −0.018 [−0.021, −0.014] | 0.0023 |
|  |  | CMF − Full-history GRU | N_hist recall | +0.011 [−0.004, +0.026] | 1.0000 |
|  |  | CMF − CMF, no topology bias | N_hist recall | −0.005 [−0.012, +0.003] | 1.0000 |
| Swap (other episode, same policies) | Random unrelated hosts | CMF − Full-history GRU | Silent recall | +0.014 [+0.010, +0.017] | 0.0023 |
|  |  | CMF − CMF, no topology bias | Silent recall | −0.010 [−0.012, −0.008] | 0.0023 |
|  |  | CMF − Full-history GRU | N_hist recall | +0.010 [+0.001, +0.019] | 0.4753 |
|  |  | CMF − CMF, no topology bias | N_hist recall | −0.008 [−0.013, −0.004] | 0.0087 |
| Shuffle (in time) | Related hosts | CMF − Full-history GRU | Silent recall | −0.022 [−0.029, −0.016] | 0.0023 |
|  |  | CMF − CMF, no topology bias | Silent recall | −0.001 [−0.005, +0.002] | 1.0000 |
|  |  | CMF − Full-history GRU | N_hist recall | −0.078 [−0.097, −0.059] | 0.0023 |
|  |  | CMF − CMF, no topology bias | N_hist recall | +0.023 [+0.013, +0.033] | 0.0023 |
| Shuffle (in time) | Random unrelated hosts | CMF − Full-history GRU | Silent recall | −0.005 [−0.009, −0.001] | 0.3492 |
|  |  | CMF − CMF, no topology bias | Silent recall | −0.006 [−0.009, −0.004] | 0.0023 |
|  |  | CMF − Full-history GRU | N_hist recall | −0.051 [−0.065, −0.038] | 0.0023 |
|  |  | CMF − CMF, no topology bias | N_hist recall | −0.012 [−0.018, −0.005] | 0.0122 |

Note. Difference between two models of the intervention effect Δ = intervened − original (Table @tbl:interventions); a positive value means that A loses less than B. Paired episode bootstrap of the seed mean (100,000 resamples, Algorithm 2), Holm over all m = 114 rows of this analysis (smallest attainable adjusted p = 0.0023).

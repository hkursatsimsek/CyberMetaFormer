| Set | Episodes changed | Remove-mask steps alerted when corrected | Model | Δ host macro-F1 | Δ silent recall | Δ N_hist recall | Δ remove-mask recall | Δ silent recall, own definition |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| Test (training policies) | 628 / 2,000 | 0.880 of 36,282 | CMF | −0.0002 [−0.0004, −0.0001] | −0.001 [−0.002, −0.000] | −0.003 [−0.005, −0.001] | −0.004 [−0.008, +0.001] | −0.018 [−0.020, −0.015] |
|  |  |  | Full-history GRU | −0.0000 [−0.0002, +0.0002] | −0.000 [−0.002, +0.001] | −0.003 [−0.006, −0.001] | −0.004 [−0.010, +0.001] | −0.017 [−0.020, −0.015] |
| ε = 0.3 (main OOD) | 3,477 / 4,000 | 0.486 of 80,524 | CMF | −0.0012 [−0.0017, −0.0006] | −0.007 [−0.008, −0.006] | −0.011 [−0.014, −0.009] | −0.009 [−0.011, −0.006] | −0.040 [−0.043, −0.037] |
|  |  |  | Full-history GRU | −0.0009 [−0.0015, −0.0003] | −0.017 [−0.019, −0.014] | −0.014 [−0.018, −0.010] | −0.029 [−0.035, −0.023] | −0.055 [−0.060, −0.052] |
| ε = 0.5 | 3,855 / 4,000 | 0.642 of 110,248 | CMF | −0.0020 [−0.0024, −0.0016] | −0.010 [−0.012, −0.008] | −0.023 [−0.028, −0.019] | −0.015 [−0.018, −0.012] | −0.119 [−0.125, −0.113] |
|  |  |  | Full-history GRU | −0.0017 [−0.0022, −0.0012] | −0.026 [−0.030, −0.023] | −0.042 [−0.048, −0.036] | −0.039 [−0.045, −0.033] | −0.154 [−0.161, −0.146] |
| ε = 0.7 | 3,965 / 4,000 | 0.819 of 170,146 | CMF | −0.0015 [−0.0018, −0.0012] | −0.007 [−0.009, −0.005] | −0.029 [−0.035, −0.024] | −0.010 [−0.012, −0.007] | −0.269 [−0.280, −0.258] |
|  |  |  | Full-history GRU | −0.0011 [−0.0015, −0.0008] | −0.022 [−0.025, −0.019] | −0.058 [−0.066, −0.049] | −0.026 [−0.030, −0.022] | −0.339 [−0.351, −0.328] |
| Remove instead of restore | 4,000 / 4,000 | 1.000 of 1,792,986 | CMF | +0.0145 [+0.0137, +0.0152] | +0.034 [+0.033, +0.036] | +0.406 [+0.395, +0.416] | +0.000 [+0.000, +0.000] | −0.037 [−0.046, −0.029] |
|  |  |  | Full-history GRU | +0.0207 [+0.0197, +0.0217] | +0.049 [+0.047, +0.051] | +0.486 [+0.472, +0.500] | +0.009 [+0.008, +0.009] | −0.019 [−0.027, −0.012] |

Note. Δ = corrected − flawed for the same model, seed mean over seeds 0–4, 95% CI from the paired episode bootstrap of the seed mean (10,000 resamples, Algorithm 2), Holm over all m = 100 rows of this analysis (smallest attainable adjusted p = 0.0200). Episodes changed: episodes whose defender observation differs between the copies. Remove-mask host-steps: silent host-steps whose root access is masked by a failed remove in the flawed observation; in the corrected copy the given share of them carries an alert (the root flag), so it is no longer silent. Own definition: silent host-steps recomputed from each copy's own observation, which removes those host-steps from the silent set of the corrected copy.

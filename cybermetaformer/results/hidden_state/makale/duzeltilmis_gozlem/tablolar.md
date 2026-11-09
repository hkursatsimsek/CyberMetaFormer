# Flawed vs. corrected observation (remove flag)

Models trained on the flawed data; evaluated on both copies. Mean ± sd over 5 seeds; per-member temperature from v1 validation. Silent, N_hist and remove-mask slices are defined on the flawed observation (identical host-steps in both copies); "own def." recomputes silent and N_hist from each copy's observation. Differences: paired episode bootstrap (10,000 resamples), Holm over all 100 rows.

## Sets

| Set | Episodes | Changed episodes | Silent host-steps | N_hist | Remove-mask | Silent steps alerted when corrected | Remove-mask steps alerted when corrected |
|---|---|---|---|---|---|---|---|
| Test | 2,000 | 628 | 216,495 | 54,731 | 36,282 | 0.147 | 0.880 |
| OOD (ε = 0.3) | 4,000 | 3,477 | 321,008 | 96,486 | 80,524 | 0.122 | 0.486 |
| ε = 0.5 | 4,000 | 3,855 | 260,683 | 79,367 | 110,248 | 0.272 | 0.642 |
| ε = 0.7 | 4,000 | 3,965 | 262,539 | 64,484 | 170,146 | 0.531 | 0.819 |
| react_remove | 4,000 | 4,000 | 1,957,382 | 164,396 | 1,792,986 | 0.916 | 1.000 |

## Metrics

| Set | Model | Copy | Host macro-F1 | Silent recall | N_hist recall | Remove-mask recall | Silent recall (own def.) | N_hist recall (own def.) |
|---|---|---|---|---|---|---|---|---|
| Test | CyberMetaFormer | flawed | 0.980 ± 0.000 | 0.904 ± 0.006 | 0.884 ± 0.008 | 0.970 ± 0.002 | 0.904 ± 0.006 | 0.884 ± 0.008 |
| Test | CyberMetaFormer | corrected | 0.980 ± 0.000 | 0.903 ± 0.005 | 0.881 ± 0.007 | 0.966 ± 0.002 | 0.887 ± 0.006 | 0.881 ± 0.007 |
| Test | GRU (full history) | flawed | 0.977 ± 0.000 | 0.902 ± 0.002 | 0.886 ± 0.003 | 0.963 ± 0.001 | 0.902 ± 0.002 | 0.886 ± 0.003 |
| Test | GRU (full history) | corrected | 0.977 ± 0.000 | 0.901 ± 0.003 | 0.883 ± 0.003 | 0.958 ± 0.002 | 0.884 ± 0.003 | 0.883 ± 0.003 |
| OOD (ε = 0.3) | CyberMetaFormer | flawed | 0.942 ± 0.001 | 0.772 ± 0.020 | 0.826 ± 0.014 | 0.941 ± 0.011 | 0.772 ± 0.020 | 0.826 ± 0.014 |
| OOD (ε = 0.3) | CyberMetaFormer | corrected | 0.941 ± 0.001 | 0.764 ± 0.017 | 0.815 ± 0.009 | 0.933 ± 0.007 | 0.732 ± 0.019 | 0.815 ± 0.009 |
| OOD (ε = 0.3) | GRU (full history) | flawed | 0.926 ± 0.001 | 0.736 ± 0.009 | 0.791 ± 0.010 | 0.920 ± 0.008 | 0.736 ± 0.009 | 0.791 ± 0.010 |
| OOD (ε = 0.3) | GRU (full history) | corrected | 0.925 ± 0.001 | 0.720 ± 0.008 | 0.778 ± 0.009 | 0.891 ± 0.005 | 0.681 ± 0.009 | 0.778 ± 0.009 |
| ε = 0.5 | CyberMetaFormer | flawed | 0.942 ± 0.001 | 0.718 ± 0.022 | 0.727 ± 0.011 | 0.917 ± 0.012 | 0.718 ± 0.022 | 0.727 ± 0.011 |
| ε = 0.5 | CyberMetaFormer | corrected | 0.940 ± 0.001 | 0.708 ± 0.019 | 0.704 ± 0.007 | 0.902 ± 0.008 | 0.599 ± 0.026 | 0.704 ± 0.007 |
| ε = 0.5 | GRU (full history) | flawed | 0.924 ± 0.001 | 0.685 ± 0.014 | 0.677 ± 0.017 | 0.895 ± 0.008 | 0.685 ± 0.014 | 0.677 ± 0.017 |
| ε = 0.5 | GRU (full history) | corrected | 0.922 ± 0.001 | 0.658 ± 0.012 | 0.635 ± 0.013 | 0.856 ± 0.007 | 0.531 ± 0.016 | 0.635 ± 0.013 |
| ε = 0.7 | CyberMetaFormer | flawed | 0.961 ± 0.001 | 0.775 ± 0.015 | 0.626 ± 0.015 | 0.940 ± 0.008 | 0.775 ± 0.015 | 0.626 ± 0.015 |
| ε = 0.7 | CyberMetaFormer | corrected | 0.960 ± 0.001 | 0.768 ± 0.012 | 0.596 ± 0.009 | 0.930 ± 0.003 | 0.506 ± 0.025 | 0.596 ± 0.009 |
| ε = 0.7 | GRU (full history) | flawed | 0.946 ± 0.001 | 0.741 ± 0.011 | 0.554 ± 0.018 | 0.925 ± 0.004 | 0.741 ± 0.011 | 0.554 ± 0.018 |
| ε = 0.7 | GRU (full history) | corrected | 0.945 ± 0.001 | 0.719 ± 0.010 | 0.496 ± 0.019 | 0.899 ± 0.003 | 0.402 ± 0.022 | 0.496 ± 0.019 |
| react_remove | CyberMetaFormer | flawed | 0.969 ± 0.006 | 0.959 ± 0.013 | 0.516 ± 0.147 | 1.000 ± 0.001 | 0.959 ± 0.013 | 0.516 ± 0.147 |
| react_remove | CyberMetaFormer | corrected | 0.984 ± 0.000 | 0.993 ± 0.001 | 0.922 ± 0.013 | 1.000 ± 0.000 | 0.922 ± 0.013 | 0.922 ± 0.013 |
| react_remove | GRU (full history) | flawed | 0.962 ± 0.006 | 0.945 ± 0.012 | 0.440 ± 0.046 | 0.991 ± 0.009 | 0.945 ± 0.012 | 0.440 ± 0.046 |
| react_remove | GRU (full history) | corrected | 0.982 ± 0.000 | 0.994 ± 0.000 | 0.926 ± 0.004 | 1.000 ± 0.000 | 0.926 ± 0.004 | 0.926 ± 0.004 |

## Corrected − flawed (same model)

| Set | Model | Criterion | Δ [95% CI] | p (Holm) |
|---|---|---|---|---|
| Test | CyberMetaFormer | Host macro-F1 | -0.0002 [-0.0004, -0.0001] | 0.020 |
| Test | CyberMetaFormer | Silent recall | -0.0009 [-0.0017, -0.0000] | 0.462 |
| Test | CyberMetaFormer | N_hist recall | -0.0027 [-0.0051, -0.0008] | 0.025 |
| Test | CyberMetaFormer | Remove-mask recall | -0.0040 [-0.0081, +0.0007] | 0.826 |
| Test | CyberMetaFormer | Silent recall (own def.) | -0.0176 [-0.0202, -0.0151] | 0.020 |
| Test | CyberMetaFormer | N_hist recall (own def.) | -0.0027 [-0.0051, -0.0008] | 0.025 |
| Test | GRU (full history) | Host macro-F1 | -0.0000 [-0.0002, +0.0002] | 1.000 |
| Test | GRU (full history) | Silent recall | -0.0004 [-0.0016, +0.0008] | 1.000 |
| Test | GRU (full history) | N_hist recall | -0.0028 [-0.0056, -0.0005] | 0.133 |
| Test | GRU (full history) | Remove-mask recall | -0.0042 [-0.0098, +0.0011] | 0.944 |
| Test | GRU (full history) | Silent recall (own def.) | -0.0174 [-0.0203, -0.0147] | 0.020 |
| Test | GRU (full history) | N_hist recall (own def.) | -0.0028 [-0.0056, -0.0005] | 0.133 |
| OOD (ε = 0.3) | CyberMetaFormer | Host macro-F1 | -0.0012 [-0.0017, -0.0006] | 0.020 |
| OOD (ε = 0.3) | CyberMetaFormer | Silent recall | -0.0071 [-0.0084, -0.0059] | 0.020 |
| OOD (ε = 0.3) | CyberMetaFormer | N_hist recall | -0.0111 [-0.0135, -0.0088] | 0.020 |
| OOD (ε = 0.3) | CyberMetaFormer | Remove-mask recall | -0.0086 [-0.0113, -0.0059] | 0.020 |
| OOD (ε = 0.3) | CyberMetaFormer | Silent recall (own def.) | -0.0398 [-0.0427, -0.0371] | 0.020 |
| OOD (ε = 0.3) | CyberMetaFormer | N_hist recall (own def.) | -0.0111 [-0.0135, -0.0088] | 0.020 |
| OOD (ε = 0.3) | GRU (full history) | Host macro-F1 | -0.0009 [-0.0015, -0.0003] | 0.063 |
| OOD (ε = 0.3) | GRU (full history) | Silent recall | -0.0166 [-0.0191, -0.0142] | 0.020 |
| OOD (ε = 0.3) | GRU (full history) | N_hist recall | -0.0136 [-0.0176, -0.0096] | 0.020 |
| OOD (ε = 0.3) | GRU (full history) | Remove-mask recall | -0.0292 [-0.0353, -0.0232] | 0.020 |
| OOD (ε = 0.3) | GRU (full history) | Silent recall (own def.) | -0.0555 [-0.0596, -0.0515] | 0.020 |
| OOD (ε = 0.3) | GRU (full history) | N_hist recall (own def.) | -0.0136 [-0.0176, -0.0096] | 0.020 |
| ε = 0.5 | CyberMetaFormer | Host macro-F1 | -0.0020 [-0.0024, -0.0016] | 0.020 |
| ε = 0.5 | CyberMetaFormer | Silent recall | -0.0102 [-0.0121, -0.0084] | 0.020 |
| ε = 0.5 | CyberMetaFormer | N_hist recall | -0.0232 [-0.0275, -0.0191] | 0.020 |
| ε = 0.5 | CyberMetaFormer | Remove-mask recall | -0.0150 [-0.0182, -0.0119] | 0.020 |
| ε = 0.5 | CyberMetaFormer | Silent recall (own def.) | -0.1191 [-0.1253, -0.1131] | 0.020 |
| ε = 0.5 | CyberMetaFormer | N_hist recall (own def.) | -0.0232 [-0.0275, -0.0191] | 0.020 |
| ε = 0.5 | GRU (full history) | Host macro-F1 | -0.0017 [-0.0022, -0.0012] | 0.020 |
| ε = 0.5 | GRU (full history) | Silent recall | -0.0264 [-0.0295, -0.0232] | 0.020 |
| ε = 0.5 | GRU (full history) | N_hist recall | -0.0419 [-0.0483, -0.0357] | 0.020 |
| ε = 0.5 | GRU (full history) | Remove-mask recall | -0.0390 [-0.0448, -0.0333] | 0.020 |
| ε = 0.5 | GRU (full history) | Silent recall (own def.) | -0.1537 [-0.1611, -0.1463] | 0.020 |
| ε = 0.5 | GRU (full history) | N_hist recall (own def.) | -0.0419 [-0.0483, -0.0357] | 0.020 |
| ε = 0.7 | CyberMetaFormer | Host macro-F1 | -0.0015 [-0.0018, -0.0012] | 0.020 |
| ε = 0.7 | CyberMetaFormer | Silent recall | -0.0066 [-0.0086, -0.0047] | 0.020 |
| ε = 0.7 | CyberMetaFormer | N_hist recall | -0.0293 [-0.0349, -0.0239] | 0.020 |
| ε = 0.7 | CyberMetaFormer | Remove-mask recall | -0.0099 [-0.0123, -0.0074] | 0.020 |
| ε = 0.7 | CyberMetaFormer | Silent recall (own def.) | -0.2687 [-0.2795, -0.2581] | 0.020 |
| ε = 0.7 | CyberMetaFormer | N_hist recall (own def.) | -0.0293 [-0.0349, -0.0239] | 0.020 |
| ε = 0.7 | GRU (full history) | Host macro-F1 | -0.0011 [-0.0015, -0.0008] | 0.020 |
| ε = 0.7 | GRU (full history) | Silent recall | -0.0221 [-0.0251, -0.0192] | 0.020 |
| ε = 0.7 | GRU (full history) | N_hist recall | -0.0575 [-0.0658, -0.0493] | 0.020 |
| ε = 0.7 | GRU (full history) | Remove-mask recall | -0.0257 [-0.0295, -0.0220] | 0.020 |
| ε = 0.7 | GRU (full history) | Silent recall (own def.) | -0.3395 [-0.3507, -0.3284] | 0.020 |
| ε = 0.7 | GRU (full history) | N_hist recall (own def.) | -0.0575 [-0.0658, -0.0493] | 0.020 |
| react_remove | CyberMetaFormer | Host macro-F1 | +0.0145 [+0.0137, +0.0152] | 0.020 |
| react_remove | CyberMetaFormer | Silent recall | +0.0343 [+0.0328, +0.0359] | 0.020 |
| react_remove | CyberMetaFormer | N_hist recall | +0.4057 [+0.3955, +0.4161] | 0.020 |
| react_remove | CyberMetaFormer | Remove-mask recall | +0.0003 [+0.0002, +0.0003] | 0.020 |
| react_remove | CyberMetaFormer | Silent recall (own def.) | -0.0373 [-0.0461, -0.0292] | 0.020 |
| react_remove | CyberMetaFormer | N_hist recall (own def.) | +0.4057 [+0.3955, +0.4161] | 0.020 |
| react_remove | GRU (full history) | Host macro-F1 | +0.0207 [+0.0197, +0.0217] | 0.020 |
| react_remove | GRU (full history) | Silent recall | +0.0488 [+0.0467, +0.0508] | 0.020 |
| react_remove | GRU (full history) | N_hist recall | +0.4859 [+0.4716, +0.5002] | 0.020 |
| react_remove | GRU (full history) | Remove-mask recall | +0.0087 [+0.0081, +0.0092] | 0.020 |
| react_remove | GRU (full history) | Silent recall (own def.) | -0.0194 [-0.0274, -0.0116] | 0.020 |
| react_remove | GRU (full history) | N_hist recall (own def.) | +0.4859 [+0.4716, +0.5002] | 0.020 |

## CyberMetaFormer − GRU (full history), per copy

| Set | Copy | Criterion | Δ [95% CI] | p (Holm) |
|---|---|---|---|---|
| Test | flawed | Host macro-F1 | +0.0034 [+0.0029, +0.0039] | 0.020 |
| Test | flawed | Silent recall | +0.0028 [-0.0004, +0.0059] | 0.778 |
| Test | flawed | N_hist recall | -0.0019 [-0.0112, +0.0074] | 1.000 |
| Test | flawed | Remove-mask recall | +0.0070 [+0.0025, +0.0124] | 0.042 |
| Test | corrected | Host macro-F1 | +0.0031 [+0.0027, +0.0036] | 0.020 |
| Test | corrected | Silent recall | +0.0023 [-0.0008, +0.0054] | 0.951 |
| Test | corrected | N_hist recall | -0.0019 [-0.0110, +0.0071] | 1.000 |
| Test | corrected | Remove-mask recall | +0.0072 [+0.0031, +0.0119] | 0.020 |
| OOD (ε = 0.3) | flawed | Host macro-F1 | +0.0153 [+0.0145, +0.0162] | 0.020 |
| OOD (ε = 0.3) | flawed | Silent recall | +0.0355 [+0.0309, +0.0402] | 0.020 |
| OOD (ε = 0.3) | flawed | N_hist recall | +0.0348 [+0.0261, +0.0436] | 0.020 |
| OOD (ε = 0.3) | flawed | Remove-mask recall | +0.0211 [+0.0148, +0.0273] | 0.020 |
| OOD (ε = 0.3) | corrected | Host macro-F1 | +0.0151 [+0.0142, +0.0159] | 0.020 |
| OOD (ε = 0.3) | corrected | Silent recall | +0.0449 [+0.0401, +0.0498] | 0.020 |
| OOD (ε = 0.3) | corrected | N_hist recall | +0.0372 [+0.0282, +0.0464] | 0.020 |
| OOD (ε = 0.3) | corrected | Remove-mask recall | +0.0417 [+0.0344, +0.0490] | 0.020 |
| ε = 0.5 | flawed | Host macro-F1 | +0.0179 [+0.0170, +0.0188] | 0.020 |
| ε = 0.5 | flawed | Silent recall | +0.0337 [+0.0281, +0.0394] | 0.020 |
| ε = 0.5 | flawed | N_hist recall | +0.0508 [+0.0385, +0.0630] | 0.020 |
| ε = 0.5 | flawed | Remove-mask recall | +0.0214 [+0.0157, +0.0271] | 0.020 |
| ε = 0.5 | corrected | Host macro-F1 | +0.0175 [+0.0167, +0.0184] | 0.020 |
| ε = 0.5 | corrected | Silent recall | +0.0498 [+0.0440, +0.0555] | 0.020 |
| ε = 0.5 | corrected | N_hist recall | +0.0695 [+0.0569, +0.0823] | 0.020 |
| ε = 0.5 | corrected | Remove-mask recall | +0.0454 [+0.0392, +0.0518] | 0.020 |
| ε = 0.7 | flawed | Host macro-F1 | +0.0148 [+0.0142, +0.0155] | 0.020 |
| ε = 0.7 | flawed | Silent recall | +0.0336 [+0.0290, +0.0380] | 0.020 |
| ε = 0.7 | flawed | N_hist recall | +0.0720 [+0.0590, +0.0850] | 0.020 |
| ε = 0.7 | flawed | Remove-mask recall | +0.0152 [+0.0118, +0.0187] | 0.020 |
| ε = 0.7 | corrected | Host macro-F1 | +0.0145 [+0.0138, +0.0151] | 0.020 |
| ε = 0.7 | corrected | Silent recall | +0.0491 [+0.0443, +0.0538] | 0.020 |
| ε = 0.7 | corrected | N_hist recall | +0.1002 [+0.0863, +0.1141] | 0.020 |
| ε = 0.7 | corrected | Remove-mask recall | +0.0311 [+0.0270, +0.0352] | 0.020 |
| react_remove | flawed | Host macro-F1 | +0.0077 [+0.0072, +0.0082] | 0.020 |
| react_remove | flawed | Silent recall | +0.0141 [+0.0132, +0.0150] | 0.020 |
| react_remove | flawed | N_hist recall | +0.0763 [+0.0679, +0.0845] | 0.020 |
| react_remove | flawed | Remove-mask recall | +0.0084 [+0.0079, +0.0089] | 0.020 |
| react_remove | corrected | Host macro-F1 | +0.0015 [+0.0012, +0.0019] | 0.020 |
| react_remove | corrected | Silent recall | -0.0003 [-0.0008, +0.0001] | 1.000 |
| react_remove | corrected | N_hist recall | -0.0039 [-0.0096, +0.0017] | 1.000 |
| react_remove | corrected | Remove-mask recall | +0.0000 [+0.0000, +0.0000] | 0.063 |

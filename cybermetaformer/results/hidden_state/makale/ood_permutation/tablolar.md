# Host permutation OOD

Models trained on the original host order; evaluated with permuted hosts. P1 = ophost0←ophost1, ophost1←ophost2, ophost2←ophost0, user1↔user2, user3↔user4 (attack-graph edges and subnets preserved); P2 = user1↔user3, user2↔user4 (subnets preserved, user→enterprise edges misaligned); P3 = only the host-embedding rows permuted as in P2, data unchanged (host-token models). Observation columns (4 bits, scan history, decoy count) and labels move together; slot a←b holds host b's data. Decoy capacity is host-specific (user1/2: 4, user3: 2, user4: 1 at the start), so in P1 and P2 it moves with the host; in P1s and P2s the decoy-count columns stay with the slot. Mean ± sd over 5 seeds (argmax; temperature does not change it). Differences: paired episode bootstrap (100,000 resamples), Holm over all 330 rows.

## Sets

| Set | Episodes | Host-steps | Clean | User | Root | Silent | N_hist |
|---|---|---|---|---|---|---|---|
| Test | 2,000 | 2,400,000 | 1,864,216 | 57,167 | 478,617 | 216,495 | 54,731 |
| OOD (ε = 0.3) | 4,000 | 4,800,000 | 4,259,100 | 153,611 | 387,289 | 321,008 | 96,486 |

## Metrics

| Set | Model | Condition | Host macro-F1 | Silent recall | N_hist recall |
|---|---|---|---|---|---|
| Test | CyberMetaFormer | identity | 0.980 ± 0.000 | 0.904 ± 0.006 | 0.884 ± 0.008 |
| Test | CyberMetaFormer | P1 (within class) | 0.976 ± 0.001 | 0.852 ± 0.020 | 0.833 ± 0.014 |
| Test | CyberMetaFormer | P1, decoy count stays in slot | 0.980 ± 0.000 | 0.904 ± 0.006 | 0.883 ± 0.007 |
| Test | CyberMetaFormer | P2 (across classes) | 0.963 ± 0.003 | 0.719 ± 0.038 | 0.614 ± 0.057 |
| Test | CyberMetaFormer | P2, decoy count stays in slot | 0.973 ± 0.003 | 0.829 ± 0.029 | 0.701 ± 0.042 |
| Test | CyberMetaFormer | P3 (E_host only) | 0.971 ± 0.002 | 0.804 ± 0.022 | 0.788 ± 0.038 |
| Test | CMF, full attention (no topology bias) | identity | 0.980 ± 0.000 | 0.905 ± 0.006 | 0.885 ± 0.016 |
| Test | CMF, full attention (no topology bias) | P1 (within class) | 0.975 ± 0.001 | 0.838 ± 0.012 | 0.818 ± 0.028 |
| Test | CMF, full attention (no topology bias) | P1, decoy count stays in slot | 0.980 ± 0.000 | 0.904 ± 0.006 | 0.884 ± 0.017 |
| Test | CMF, full attention (no topology bias) | P2 (across classes) | 0.966 ± 0.007 | 0.770 ± 0.067 | 0.713 ± 0.091 |
| Test | CMF, full attention (no topology bias) | P2, decoy count stays in slot | 0.977 ± 0.003 | 0.877 ± 0.032 | 0.813 ± 0.065 |
| Test | CMF, full attention (no topology bias) | P3 (E_host only) | 0.966 ± 0.007 | 0.770 ± 0.067 | 0.713 ± 0.091 |
| Test | Host-token GRU, topology-biased attention | identity | 0.980 ± 0.000 | 0.904 ± 0.002 | 0.885 ± 0.003 |
| Test | Host-token GRU, topology-biased attention | P1 (within class) | 0.978 ± 0.001 | 0.870 ± 0.021 | 0.858 ± 0.018 |
| Test | Host-token GRU, topology-biased attention | P1, decoy count stays in slot | 0.980 ± 0.000 | 0.904 ± 0.002 | 0.885 ± 0.004 |
| Test | Host-token GRU, topology-biased attention | P2 (across classes) | 0.957 ± 0.005 | 0.667 ± 0.059 | 0.570 ± 0.039 |
| Test | Host-token GRU, topology-biased attention | P2, decoy count stays in slot | 0.970 ± 0.003 | 0.797 ± 0.033 | 0.671 ± 0.013 |
| Test | Host-token GRU, topology-biased attention | P3 (E_host only) | 0.970 ± 0.004 | 0.793 ± 0.041 | 0.789 ± 0.045 |
| Test | Host-token GRU, attention without bias | identity | 0.980 ± 0.000 | 0.904 ± 0.007 | 0.876 ± 0.008 |
| Test | Host-token GRU, attention without bias | P1 (within class) | 0.976 ± 0.001 | 0.860 ± 0.023 | 0.827 ± 0.019 |
| Test | Host-token GRU, attention without bias | P1, decoy count stays in slot | 0.980 ± 0.000 | 0.903 ± 0.008 | 0.874 ± 0.009 |
| Test | Host-token GRU, attention without bias | P2 (across classes) | 0.967 ± 0.004 | 0.760 ± 0.038 | 0.696 ± 0.063 |
| Test | Host-token GRU, attention without bias | P2, decoy count stays in slot | 0.979 ± 0.001 | 0.892 ± 0.016 | 0.838 ± 0.043 |
| Test | Host-token GRU, attention without bias | P3 (E_host only) | 0.967 ± 0.004 | 0.760 ± 0.038 | 0.696 ± 0.063 |
| Test | GRU (full history) | identity | 0.977 ± 0.000 | 0.902 ± 0.002 | 0.886 ± 0.003 |
| Test | GRU (full history) | P1 (within class) | 0.976 ± 0.000 | 0.906 ± 0.005 | 0.888 ± 0.004 |
| Test | GRU (full history) | P1, decoy count stays in slot | 0.977 ± 0.000 | 0.902 ± 0.003 | 0.887 ± 0.003 |
| Test | GRU (full history) | P2 (across classes) | 0.937 ± 0.006 | 0.835 ± 0.021 | 0.795 ± 0.019 |
| Test | GRU (full history) | P2, decoy count stays in slot | 0.956 ± 0.001 | 0.734 ± 0.004 | 0.659 ± 0.011 |
| OOD (ε = 0.3) | CyberMetaFormer | identity | 0.942 ± 0.001 | 0.772 ± 0.020 | 0.826 ± 0.014 |
| OOD (ε = 0.3) | CyberMetaFormer | P1 (within class) | 0.934 ± 0.001 | 0.712 ± 0.023 | 0.770 ± 0.015 |
| OOD (ε = 0.3) | CyberMetaFormer | P1, decoy count stays in slot | 0.938 ± 0.001 | 0.761 ± 0.021 | 0.818 ± 0.015 |
| OOD (ε = 0.3) | CyberMetaFormer | P2 (across classes) | 0.913 ± 0.005 | 0.585 ± 0.031 | 0.550 ± 0.048 |
| OOD (ε = 0.3) | CyberMetaFormer | P2, decoy count stays in slot | 0.924 ± 0.003 | 0.668 ± 0.028 | 0.627 ± 0.032 |
| OOD (ε = 0.3) | CyberMetaFormer | P3 (E_host only) | 0.927 ± 0.003 | 0.670 ± 0.019 | 0.720 ± 0.041 |
| OOD (ε = 0.3) | CMF, full attention (no topology bias) | identity | 0.941 ± 0.001 | 0.766 ± 0.017 | 0.817 ± 0.028 |
| OOD (ε = 0.3) | CMF, full attention (no topology bias) | P1 (within class) | 0.932 ± 0.002 | 0.695 ± 0.023 | 0.746 ± 0.039 |
| OOD (ε = 0.3) | CMF, full attention (no topology bias) | P1, decoy count stays in slot | 0.937 ± 0.001 | 0.752 ± 0.018 | 0.810 ± 0.031 |
| OOD (ε = 0.3) | CMF, full attention (no topology bias) | P2 (across classes) | 0.920 ± 0.011 | 0.628 ± 0.057 | 0.645 ± 0.090 |
| OOD (ε = 0.3) | CMF, full attention (no topology bias) | P2, decoy count stays in slot | 0.932 ± 0.005 | 0.720 ± 0.030 | 0.736 ± 0.056 |
| OOD (ε = 0.3) | CMF, full attention (no topology bias) | P3 (E_host only) | 0.920 ± 0.011 | 0.628 ± 0.057 | 0.645 ± 0.090 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | identity | 0.942 ± 0.001 | 0.764 ± 0.007 | 0.826 ± 0.010 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | P1 (within class) | 0.935 ± 0.000 | 0.718 ± 0.012 | 0.786 ± 0.013 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | P1, decoy count stays in slot | 0.938 ± 0.001 | 0.754 ± 0.007 | 0.820 ± 0.010 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | P2 (across classes) | 0.906 ± 0.002 | 0.545 ± 0.018 | 0.511 ± 0.019 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | P2, decoy count stays in slot | 0.921 ± 0.003 | 0.632 ± 0.019 | 0.596 ± 0.010 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | P3 (E_host only) | 0.926 ± 0.002 | 0.662 ± 0.013 | 0.718 ± 0.021 |
| OOD (ε = 0.3) | Host-token GRU, attention without bias | identity | 0.941 ± 0.001 | 0.763 ± 0.017 | 0.814 ± 0.020 |
| OOD (ε = 0.3) | Host-token GRU, attention without bias | P1 (within class) | 0.933 ± 0.001 | 0.708 ± 0.021 | 0.747 ± 0.017 |
| OOD (ε = 0.3) | Host-token GRU, attention without bias | P1, decoy count stays in slot | 0.937 ± 0.001 | 0.751 ± 0.018 | 0.806 ± 0.019 |
| OOD (ε = 0.3) | Host-token GRU, attention without bias | P2 (across classes) | 0.921 ± 0.003 | 0.633 ± 0.020 | 0.631 ± 0.034 |
| OOD (ε = 0.3) | Host-token GRU, attention without bias | P2, decoy count stays in slot | 0.935 ± 0.002 | 0.735 ± 0.024 | 0.762 ± 0.044 |
| OOD (ε = 0.3) | Host-token GRU, attention without bias | P3 (E_host only) | 0.921 ± 0.003 | 0.633 ± 0.020 | 0.631 ± 0.034 |
| OOD (ε = 0.3) | GRU (full history) | identity | 0.926 ± 0.001 | 0.736 ± 0.009 | 0.791 ± 0.010 |
| OOD (ε = 0.3) | GRU (full history) | P1 (within class) | 0.925 ± 0.002 | 0.748 ± 0.011 | 0.796 ± 0.013 |
| OOD (ε = 0.3) | GRU (full history) | P1, decoy count stays in slot | 0.925 ± 0.001 | 0.734 ± 0.009 | 0.790 ± 0.010 |
| OOD (ε = 0.3) | GRU (full history) | P2 (across classes) | 0.877 ± 0.006 | 0.761 ± 0.017 | 0.735 ± 0.029 |
| OOD (ε = 0.3) | GRU (full history) | P2, decoy count stays in slot | 0.900 ± 0.001 | 0.582 ± 0.006 | 0.533 ± 0.006 |

## Permuted − identity (same model)

| Set | Model | Condition | Criterion | Δ [95% CI] | p (Holm) |
|---|---|---|---|---|---|
| Test | CyberMetaFormer | P1 (within class) | Host macro-F1 | -0.0038 [-0.0043, -0.0034] | 0.007 |
| Test | CyberMetaFormer | P1 (within class) | Silent recall | -0.0523 [-0.0572, -0.0474] | 0.007 |
| Test | CyberMetaFormer | P1 (within class) | N_hist recall | -0.0504 [-0.0630, -0.0384] | 0.007 |
| Test | CyberMetaFormer | P1, decoy count stays in slot | Host macro-F1 | -0.0003 [-0.0005, -0.0001] | 0.186 |
| Test | CyberMetaFormer | P1, decoy count stays in slot | Silent recall | -0.0004 [-0.0018, +0.0009] | 1.000 |
| Test | CyberMetaFormer | P1, decoy count stays in slot | N_hist recall | -0.0006 [-0.0048, +0.0036] | 1.000 |
| Test | CyberMetaFormer | P2 (across classes) | Host macro-F1 | -0.0175 [-0.0187, -0.0164] | 0.007 |
| Test | CyberMetaFormer | P2 (across classes) | Silent recall | -0.1851 [-0.1939, -0.1763] | 0.007 |
| Test | CyberMetaFormer | P2 (across classes) | N_hist recall | -0.2699 [-0.2948, -0.2450] | 0.007 |
| Test | CyberMetaFormer | P2, decoy count stays in slot | Host macro-F1 | -0.0074 [-0.0082, -0.0066] | 0.007 |
| Test | CyberMetaFormer | P2, decoy count stays in slot | Silent recall | -0.0749 [-0.0826, -0.0675] | 0.007 |
| Test | CyberMetaFormer | P2, decoy count stays in slot | N_hist recall | -0.1824 [-0.2061, -0.1593] | 0.007 |
| Test | CyberMetaFormer | P3 (E_host only) | Host macro-F1 | -0.0088 [-0.0095, -0.0080] | 0.007 |
| Test | CyberMetaFormer | P3 (E_host only) | Silent recall | -0.1007 [-0.1078, -0.0936] | 0.007 |
| Test | CyberMetaFormer | P3 (E_host only) | N_hist recall | -0.0954 [-0.1112, -0.0800] | 0.007 |
| Test | CMF, full attention (no topology bias) | P1 (within class) | Host macro-F1 | -0.0049 [-0.0055, -0.0044] | 0.007 |
| Test | CMF, full attention (no topology bias) | P1 (within class) | Silent recall | -0.0670 [-0.0725, -0.0614] | 0.007 |
| Test | CMF, full attention (no topology bias) | P1 (within class) | N_hist recall | -0.0670 [-0.0802, -0.0545] | 0.007 |
| Test | CMF, full attention (no topology bias) | P1, decoy count stays in slot | Host macro-F1 | -0.0003 [-0.0005, -0.0001] | 0.303 |
| Test | CMF, full attention (no topology bias) | P1, decoy count stays in slot | Silent recall | -0.0006 [-0.0021, +0.0008] | 1.000 |
| Test | CMF, full attention (no topology bias) | P1, decoy count stays in slot | N_hist recall | -0.0013 [-0.0064, +0.0033] | 1.000 |
| Test | CMF, full attention (no topology bias) | P2 (across classes) | Host macro-F1 | -0.0138 [-0.0147, -0.0129] | 0.007 |
| Test | CMF, full attention (no topology bias) | P2 (across classes) | Silent recall | -0.1351 [-0.1425, -0.1277] | 0.007 |
| Test | CMF, full attention (no topology bias) | P2 (across classes) | N_hist recall | -0.1718 [-0.1885, -0.1554] | 0.007 |
| Test | CMF, full attention (no topology bias) | P2, decoy count stays in slot | Host macro-F1 | -0.0030 [-0.0034, -0.0026] | 0.007 |
| Test | CMF, full attention (no topology bias) | P2, decoy count stays in slot | Silent recall | -0.0277 [-0.0311, -0.0245] | 0.007 |
| Test | CMF, full attention (no topology bias) | P2, decoy count stays in slot | N_hist recall | -0.0722 [-0.0829, -0.0617] | 0.007 |
| Test | CMF, full attention (no topology bias) | P3 (E_host only) | Host macro-F1 | -0.0138 [-0.0147, -0.0129] | 0.007 |
| Test | CMF, full attention (no topology bias) | P3 (E_host only) | Silent recall | -0.1351 [-0.1425, -0.1277] | 0.007 |
| Test | CMF, full attention (no topology bias) | P3 (E_host only) | N_hist recall | -0.1718 [-0.1885, -0.1554] | 0.007 |
| Test | Host-token GRU, topology-biased attention | P1 (within class) | Host macro-F1 | -0.0027 [-0.0031, -0.0023] | 0.007 |
| Test | Host-token GRU, topology-biased attention | P1 (within class) | Silent recall | -0.0344 [-0.0381, -0.0308] | 0.007 |
| Test | Host-token GRU, topology-biased attention | P1 (within class) | N_hist recall | -0.0273 [-0.0355, -0.0196] | 0.007 |
| Test | Host-token GRU, topology-biased attention | P1, decoy count stays in slot | Host macro-F1 | -0.0002 [-0.0004, -0.0000] | 1.000 |
| Test | Host-token GRU, topology-biased attention | P1, decoy count stays in slot | Silent recall | -0.0002 [-0.0012, +0.0007] | 1.000 |
| Test | Host-token GRU, topology-biased attention | P1, decoy count stays in slot | N_hist recall | +0.0001 [-0.0031, +0.0036] | 1.000 |
| Test | Host-token GRU, topology-biased attention | P2 (across classes) | Host macro-F1 | -0.0230 [-0.0245, -0.0215] | 0.007 |
| Test | Host-token GRU, topology-biased attention | P2 (across classes) | Silent recall | -0.2370 [-0.2472, -0.2270] | 0.007 |
| Test | Host-token GRU, topology-biased attention | P2 (across classes) | N_hist recall | -0.3147 [-0.3439, -0.2858] | 0.007 |
| Test | Host-token GRU, topology-biased attention | P2, decoy count stays in slot | Host macro-F1 | -0.0101 [-0.0111, -0.0091] | 0.007 |
| Test | Host-token GRU, topology-biased attention | P2, decoy count stays in slot | Silent recall | -0.1074 [-0.1170, -0.0982] | 0.007 |
| Test | Host-token GRU, topology-biased attention | P2, decoy count stays in slot | N_hist recall | -0.2139 [-0.2418, -0.1867] | 0.007 |
| Test | Host-token GRU, topology-biased attention | P3 (E_host only) | Host macro-F1 | -0.0101 [-0.0110, -0.0091] | 0.007 |
| Test | Host-token GRU, topology-biased attention | P3 (E_host only) | Silent recall | -0.1116 [-0.1198, -0.1033] | 0.007 |
| Test | Host-token GRU, topology-biased attention | P3 (E_host only) | N_hist recall | -0.0958 [-0.1119, -0.0800] | 0.007 |
| Test | Host-token GRU, attention without bias | P1 (within class) | Host macro-F1 | -0.0037 [-0.0042, -0.0032] | 0.007 |
| Test | Host-token GRU, attention without bias | P1 (within class) | Silent recall | -0.0446 [-0.0493, -0.0399] | 0.007 |
| Test | Host-token GRU, attention without bias | P1 (within class) | N_hist recall | -0.0490 [-0.0614, -0.0371] | 0.007 |
| Test | Host-token GRU, attention without bias | P1, decoy count stays in slot | Host macro-F1 | -0.0004 [-0.0005, -0.0002] | 0.011 |
| Test | Host-token GRU, attention without bias | P1, decoy count stays in slot | Silent recall | -0.0009 [-0.0020, +0.0003] | 1.000 |
| Test | Host-token GRU, attention without bias | P1, decoy count stays in slot | N_hist recall | -0.0019 [-0.0059, +0.0018] | 1.000 |
| Test | Host-token GRU, attention without bias | P2 (across classes) | Host macro-F1 | -0.0133 [-0.0145, -0.0122] | 0.007 |
| Test | Host-token GRU, attention without bias | P2 (across classes) | Silent recall | -0.1440 [-0.1537, -0.1343] | 0.007 |
| Test | Host-token GRU, attention without bias | P2 (across classes) | N_hist recall | -0.1800 [-0.2025, -0.1578] | 0.007 |
| Test | Host-token GRU, attention without bias | P2, decoy count stays in slot | Host macro-F1 | -0.0015 [-0.0018, -0.0012] | 0.007 |
| Test | Host-token GRU, attention without bias | P2, decoy count stays in slot | Silent recall | -0.0117 [-0.0140, -0.0095] | 0.007 |
| Test | Host-token GRU, attention without bias | P2, decoy count stays in slot | N_hist recall | -0.0383 [-0.0459, -0.0309] | 0.007 |
| Test | Host-token GRU, attention without bias | P3 (E_host only) | Host macro-F1 | -0.0133 [-0.0145, -0.0122] | 0.007 |
| Test | Host-token GRU, attention without bias | P3 (E_host only) | Silent recall | -0.1440 [-0.1537, -0.1343] | 0.007 |
| Test | Host-token GRU, attention without bias | P3 (E_host only) | N_hist recall | -0.1800 [-0.2025, -0.1578] | 0.007 |
| Test | GRU (full history) | P1 (within class) | Host macro-F1 | -0.0007 [-0.0011, -0.0003] | 0.020 |
| Test | GRU (full history) | P1 (within class) | Silent recall | +0.0047 [+0.0019, +0.0076] | 0.045 |
| Test | GRU (full history) | P1 (within class) | N_hist recall | +0.0025 [-0.0055, +0.0109] | 1.000 |
| Test | GRU (full history) | P1, decoy count stays in slot | Host macro-F1 | -0.0003 [-0.0006, +0.0001] | 1.000 |
| Test | GRU (full history) | P1, decoy count stays in slot | Silent recall | +0.0004 [-0.0023, +0.0030] | 1.000 |
| Test | GRU (full history) | P1, decoy count stays in slot | N_hist recall | +0.0011 [-0.0068, +0.0091] | 1.000 |
| Test | GRU (full history) | P2 (across classes) | Host macro-F1 | -0.0397 [-0.0419, -0.0376] | 0.007 |
| Test | GRU (full history) | P2 (across classes) | Silent recall | -0.0666 [-0.0779, -0.0555] | 0.007 |
| Test | GRU (full history) | P2 (across classes) | N_hist recall | -0.0906 [-0.1150, -0.0668] | 0.007 |
| Test | GRU (full history) | P2, decoy count stays in slot | Host macro-F1 | -0.0210 [-0.0228, -0.0193] | 0.007 |
| Test | GRU (full history) | P2, decoy count stays in slot | Silent recall | -0.1674 [-0.1804, -0.1548] | 0.007 |
| Test | GRU (full history) | P2, decoy count stays in slot | N_hist recall | -0.2267 [-0.2561, -0.1979] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | P1 (within class) | Host macro-F1 | -0.0078 [-0.0086, -0.0071] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | P1 (within class) | Silent recall | -0.0593 [-0.0642, -0.0546] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | P1 (within class) | N_hist recall | -0.0555 [-0.0658, -0.0454] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | P1, decoy count stays in slot | Host macro-F1 | -0.0041 [-0.0047, -0.0035] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | P1, decoy count stays in slot | Silent recall | -0.0110 [-0.0141, -0.0079] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | P1, decoy count stays in slot | N_hist recall | -0.0078 [-0.0143, -0.0016] | 0.640 |
| OOD (ε = 0.3) | CyberMetaFormer | P2 (across classes) | Host macro-F1 | -0.0283 [-0.0296, -0.0270] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | P2 (across classes) | Silent recall | -0.1864 [-0.1940, -0.1788] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | P2 (across classes) | N_hist recall | -0.2761 [-0.2954, -0.2571] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | P2, decoy count stays in slot | Host macro-F1 | -0.0174 [-0.0186, -0.0162] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | P2, decoy count stays in slot | Silent recall | -0.1036 [-0.1109, -0.0965] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | P2, decoy count stays in slot | N_hist recall | -0.1986 [-0.2175, -0.1801] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | P3 (E_host only) | Host macro-F1 | -0.0143 [-0.0153, -0.0134] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | P3 (E_host only) | Silent recall | -0.1021 [-0.1080, -0.0963] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | P3 (E_host only) | N_hist recall | -0.1064 [-0.1185, -0.0943] | 0.007 |
| OOD (ε = 0.3) | CMF, full attention (no topology bias) | P1 (within class) | Host macro-F1 | -0.0091 [-0.0099, -0.0083] | 0.007 |
| OOD (ε = 0.3) | CMF, full attention (no topology bias) | P1 (within class) | Silent recall | -0.0706 [-0.0756, -0.0657] | 0.007 |
| OOD (ε = 0.3) | CMF, full attention (no topology bias) | P1 (within class) | N_hist recall | -0.0710 [-0.0820, -0.0602] | 0.007 |
| OOD (ε = 0.3) | CMF, full attention (no topology bias) | P1, decoy count stays in slot | Host macro-F1 | -0.0043 [-0.0049, -0.0037] | 0.007 |
| OOD (ε = 0.3) | CMF, full attention (no topology bias) | P1, decoy count stays in slot | Silent recall | -0.0136 [-0.0168, -0.0104] | 0.007 |
| OOD (ε = 0.3) | CMF, full attention (no topology bias) | P1, decoy count stays in slot | N_hist recall | -0.0074 [-0.0141, -0.0009] | 1.000 |
| OOD (ε = 0.3) | CMF, full attention (no topology bias) | P2 (across classes) | Host macro-F1 | -0.0214 [-0.0223, -0.0205] | 0.007 |
| OOD (ε = 0.3) | CMF, full attention (no topology bias) | P2 (across classes) | Silent recall | -0.1377 [-0.1435, -0.1318] | 0.007 |
| OOD (ε = 0.3) | CMF, full attention (no topology bias) | P2 (across classes) | N_hist recall | -0.1725 [-0.1854, -0.1598] | 0.007 |
| OOD (ε = 0.3) | CMF, full attention (no topology bias) | P2, decoy count stays in slot | Host macro-F1 | -0.0090 [-0.0097, -0.0083] | 0.007 |
| OOD (ε = 0.3) | CMF, full attention (no topology bias) | P2, decoy count stays in slot | Silent recall | -0.0456 [-0.0496, -0.0417] | 0.007 |
| OOD (ε = 0.3) | CMF, full attention (no topology bias) | P2, decoy count stays in slot | N_hist recall | -0.0811 [-0.0912, -0.0712] | 0.007 |
| OOD (ε = 0.3) | CMF, full attention (no topology bias) | P3 (E_host only) | Host macro-F1 | -0.0214 [-0.0223, -0.0205] | 0.007 |
| OOD (ε = 0.3) | CMF, full attention (no topology bias) | P3 (E_host only) | Silent recall | -0.1377 [-0.1435, -0.1318] | 0.007 |
| OOD (ε = 0.3) | CMF, full attention (no topology bias) | P3 (E_host only) | N_hist recall | -0.1725 [-0.1854, -0.1598] | 0.007 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | P1 (within class) | Host macro-F1 | -0.0072 [-0.0080, -0.0065] | 0.007 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | P1 (within class) | Silent recall | -0.0464 [-0.0509, -0.0419] | 0.007 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | P1 (within class) | N_hist recall | -0.0397 [-0.0489, -0.0308] | 0.007 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | P1, decoy count stays in slot | Host macro-F1 | -0.0038 [-0.0044, -0.0032] | 0.007 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | P1, decoy count stays in slot | Silent recall | -0.0105 [-0.0136, -0.0075] | 0.007 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | P1, decoy count stays in slot | N_hist recall | -0.0052 [-0.0112, +0.0007] | 1.000 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | P2 (across classes) | Host macro-F1 | -0.0359 [-0.0374, -0.0344] | 0.007 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | P2 (across classes) | Silent recall | -0.2196 [-0.2286, -0.2107] | 0.007 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | P2 (across classes) | N_hist recall | -0.3144 [-0.3356, -0.2934] | 0.007 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | P2, decoy count stays in slot | Host macro-F1 | -0.0213 [-0.0227, -0.0199] | 0.007 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | P2, decoy count stays in slot | Silent recall | -0.1320 [-0.1405, -0.1238] | 0.007 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | P2, decoy count stays in slot | N_hist recall | -0.2292 [-0.2494, -0.2094] | 0.007 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | P3 (E_host only) | Host macro-F1 | -0.0154 [-0.0165, -0.0143] | 0.007 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | P3 (E_host only) | Silent recall | -0.1021 [-0.1087, -0.0956] | 0.007 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | P3 (E_host only) | N_hist recall | -0.1073 [-0.1207, -0.0939] | 0.007 |
| OOD (ε = 0.3) | Host-token GRU, attention without bias | P1 (within class) | Host macro-F1 | -0.0086 [-0.0095, -0.0078] | 0.007 |
| OOD (ε = 0.3) | Host-token GRU, attention without bias | P1 (within class) | Silent recall | -0.0551 [-0.0600, -0.0503] | 0.007 |
| OOD (ε = 0.3) | Host-token GRU, attention without bias | P1 (within class) | N_hist recall | -0.0672 [-0.0791, -0.0556] | 0.007 |
| OOD (ε = 0.3) | Host-token GRU, attention without bias | P1, decoy count stays in slot | Host macro-F1 | -0.0041 [-0.0047, -0.0035] | 0.007 |
| OOD (ε = 0.3) | Host-token GRU, attention without bias | P1, decoy count stays in slot | Silent recall | -0.0119 [-0.0149, -0.0090] | 0.007 |
| OOD (ε = 0.3) | Host-token GRU, attention without bias | P1, decoy count stays in slot | N_hist recall | -0.0081 [-0.0144, -0.0020] | 0.480 |
| OOD (ε = 0.3) | Host-token GRU, attention without bias | P2 (across classes) | Host macro-F1 | -0.0207 [-0.0220, -0.0195] | 0.007 |
| OOD (ε = 0.3) | Host-token GRU, attention without bias | P2 (across classes) | Silent recall | -0.1305 [-0.1376, -0.1234] | 0.007 |
| OOD (ε = 0.3) | Host-token GRU, attention without bias | P2 (across classes) | N_hist recall | -0.1836 [-0.2008, -0.1665] | 0.007 |
| OOD (ε = 0.3) | Host-token GRU, attention without bias | P2, decoy count stays in slot | Host macro-F1 | -0.0067 [-0.0074, -0.0060] | 0.007 |
| OOD (ε = 0.3) | Host-token GRU, attention without bias | P2, decoy count stays in slot | Silent recall | -0.0283 [-0.0318, -0.0250] | 0.007 |
| OOD (ε = 0.3) | Host-token GRU, attention without bias | P2, decoy count stays in slot | N_hist recall | -0.0525 [-0.0613, -0.0439] | 0.007 |
| OOD (ε = 0.3) | Host-token GRU, attention without bias | P3 (E_host only) | Host macro-F1 | -0.0207 [-0.0220, -0.0195] | 0.007 |
| OOD (ε = 0.3) | Host-token GRU, attention without bias | P3 (E_host only) | Silent recall | -0.1305 [-0.1376, -0.1234] | 0.007 |
| OOD (ε = 0.3) | Host-token GRU, attention without bias | P3 (E_host only) | N_hist recall | -0.1836 [-0.2008, -0.1665] | 0.007 |
| OOD (ε = 0.3) | GRU (full history) | P1 (within class) | Host macro-F1 | -0.0018 [-0.0026, -0.0010] | 0.007 |
| OOD (ε = 0.3) | GRU (full history) | P1 (within class) | Silent recall | +0.0120 [+0.0078, +0.0162] | 0.007 |
| OOD (ε = 0.3) | GRU (full history) | P1 (within class) | N_hist recall | +0.0051 [-0.0025, +0.0127] | 1.000 |
| OOD (ε = 0.3) | GRU (full history) | P1, decoy count stays in slot | Host macro-F1 | -0.0011 [-0.0019, -0.0003] | 0.270 |
| OOD (ε = 0.3) | GRU (full history) | P1, decoy count stays in slot | Silent recall | -0.0022 [-0.0062, +0.0018] | 1.000 |
| OOD (ε = 0.3) | GRU (full history) | P1, decoy count stays in slot | N_hist recall | -0.0014 [-0.0090, +0.0061] | 1.000 |
| OOD (ε = 0.3) | GRU (full history) | P2 (across classes) | Host macro-F1 | -0.0497 [-0.0515, -0.0478] | 0.007 |
| OOD (ε = 0.3) | GRU (full history) | P2 (across classes) | Silent recall | +0.0245 [+0.0137, +0.0352] | 0.007 |
| OOD (ε = 0.3) | GRU (full history) | P2 (across classes) | N_hist recall | -0.0561 [-0.0763, -0.0360] | 0.007 |
| OOD (ε = 0.3) | GRU (full history) | P2, decoy count stays in slot | Host macro-F1 | -0.0269 [-0.0287, -0.0251] | 0.007 |
| OOD (ε = 0.3) | GRU (full history) | P2, decoy count stays in slot | Silent recall | -0.1538 [-0.1636, -0.1442] | 0.007 |
| OOD (ε = 0.3) | GRU (full history) | P2, decoy count stays in slot | N_hist recall | -0.2580 [-0.2792, -0.2373] | 0.007 |

## Model pairs per condition (a − b)

| Set | a | b | Condition | Criterion | Δ [95% CI] | p (Holm) |
|---|---|---|---|---|---|---|
| Test | CyberMetaFormer | GRU (full history) | identity | Host macro-F1 | +0.0034 [+0.0029, +0.0039] | 0.007 |
| Test | CyberMetaFormer | GRU (full history) | identity | Silent recall | +0.0028 [-0.0003, +0.0059] | 1.000 |
| Test | CyberMetaFormer | GRU (full history) | identity | N_hist recall | -0.0019 [-0.0114, +0.0073] | 1.000 |
| Test | CyberMetaFormer | GRU (full history) | P1 (within class) | Host macro-F1 | +0.0002 [-0.0004, +0.0009] | 1.000 |
| Test | CyberMetaFormer | GRU (full history) | P1 (within class) | Silent recall | -0.0542 [-0.0595, -0.0489] | 0.007 |
| Test | CyberMetaFormer | GRU (full history) | P1 (within class) | N_hist recall | -0.0547 [-0.0678, -0.0423] | 0.007 |
| Test | CyberMetaFormer | GRU (full history) | P1, decoy count stays in slot | Host macro-F1 | +0.0033 [+0.0028, +0.0038] | 0.007 |
| Test | CyberMetaFormer | GRU (full history) | P1, decoy count stays in slot | Silent recall | +0.0020 [-0.0007, +0.0047] | 1.000 |
| Test | CyberMetaFormer | GRU (full history) | P1, decoy count stays in slot | N_hist recall | -0.0036 [-0.0111, +0.0036] | 1.000 |
| Test | CyberMetaFormer | GRU (full history) | P2 (across classes) | Host macro-F1 | +0.0256 [+0.0235, +0.0276] | 0.007 |
| Test | CyberMetaFormer | GRU (full history) | P2 (across classes) | Silent recall | -0.1157 [-0.1281, -0.1031] | 0.007 |
| Test | CyberMetaFormer | GRU (full history) | P2 (across classes) | N_hist recall | -0.1812 [-0.2050, -0.1571] | 0.007 |
| Test | CyberMetaFormer | GRU (full history) | P2, decoy count stays in slot | Host macro-F1 | +0.0170 [+0.0156, +0.0185] | 0.007 |
| Test | CyberMetaFormer | GRU (full history) | P2, decoy count stays in slot | Silent recall | +0.0953 [+0.0853, +0.1055] | 0.007 |
| Test | CyberMetaFormer | GRU (full history) | P2, decoy count stays in slot | N_hist recall | +0.0423 [+0.0238, +0.0612] | 0.007 |
| Test | CyberMetaFormer | CMF, full attention (no topology bias) | identity | Host macro-F1 | +0.0002 [+0.0001, +0.0004] | 0.270 |
| Test | CyberMetaFormer | CMF, full attention (no topology bias) | identity | Silent recall | -0.0003 [-0.0018, +0.0011] | 1.000 |
| Test | CyberMetaFormer | CMF, full attention (no topology bias) | identity | N_hist recall | -0.0016 [-0.0062, +0.0030] | 1.000 |
| Test | CyberMetaFormer | CMF, full attention (no topology bias) | P1 (within class) | Host macro-F1 | +0.0013 [+0.0010, +0.0016] | 0.007 |
| Test | CyberMetaFormer | CMF, full attention (no topology bias) | P1 (within class) | Silent recall | +0.0144 [+0.0116, +0.0172] | 0.007 |
| Test | CyberMetaFormer | CMF, full attention (no topology bias) | P1 (within class) | N_hist recall | +0.0150 [+0.0066, +0.0236] | 0.030 |
| Test | CyberMetaFormer | CMF, full attention (no topology bias) | P1, decoy count stays in slot | Host macro-F1 | +0.0002 [+0.0000, +0.0005] | 0.786 |
| Test | CyberMetaFormer | CMF, full attention (no topology bias) | P1, decoy count stays in slot | Silent recall | -0.0001 [-0.0017, +0.0014] | 1.000 |
| Test | CyberMetaFormer | CMF, full attention (no topology bias) | P1, decoy count stays in slot | N_hist recall | -0.0009 [-0.0059, +0.0042] | 1.000 |
| Test | CyberMetaFormer | CMF, full attention (no topology bias) | P2 (across classes) | Host macro-F1 | -0.0035 [-0.0041, -0.0028] | 0.007 |
| Test | CyberMetaFormer | CMF, full attention (no topology bias) | P2 (across classes) | Silent recall | -0.0503 [-0.0568, -0.0440] | 0.007 |
| Test | CyberMetaFormer | CMF, full attention (no topology bias) | P2 (across classes) | N_hist recall | -0.0997 [-0.1168, -0.0829] | 0.007 |
| Test | CyberMetaFormer | CMF, full attention (no topology bias) | P2, decoy count stays in slot | Host macro-F1 | -0.0042 [-0.0047, -0.0037] | 0.007 |
| Test | CyberMetaFormer | CMF, full attention (no topology bias) | P2, decoy count stays in slot | Silent recall | -0.0475 [-0.0527, -0.0425] | 0.007 |
| Test | CyberMetaFormer | CMF, full attention (no topology bias) | P2, decoy count stays in slot | N_hist recall | -0.1118 [-0.1275, -0.0968] | 0.007 |
| Test | CyberMetaFormer | CMF, full attention (no topology bias) | P3 (E_host only) | Host macro-F1 | +0.0053 [+0.0048, +0.0058] | 0.007 |
| Test | CyberMetaFormer | CMF, full attention (no topology bias) | P3 (E_host only) | Silent recall | +0.0341 [+0.0303, +0.0379] | 0.007 |
| Test | CyberMetaFormer | CMF, full attention (no topology bias) | P3 (E_host only) | N_hist recall | +0.0748 [+0.0639, +0.0858] | 0.007 |
| Test | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | identity | Host macro-F1 | +0.0002 [+0.0000, +0.0004] | 1.000 |
| Test | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | identity | Silent recall | +0.0001 [-0.0013, +0.0016] | 1.000 |
| Test | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | identity | N_hist recall | +0.0089 [+0.0043, +0.0136] | 0.011 |
| Test | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P1 (within class) | Host macro-F1 | +0.0012 [+0.0008, +0.0015] | 0.007 |
| Test | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P1 (within class) | Silent recall | +0.0103 [+0.0076, +0.0131] | 0.007 |
| Test | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P1 (within class) | N_hist recall | +0.0306 [+0.0222, +0.0394] | 0.007 |
| Test | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P1, decoy count stays in slot | Host macro-F1 | +0.0004 [+0.0001, +0.0006] | 0.019 |
| Test | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P1, decoy count stays in slot | Silent recall | +0.0008 [-0.0007, +0.0023] | 1.000 |
| Test | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P1, decoy count stays in slot | N_hist recall | +0.0110 [+0.0065, +0.0158] | 0.007 |
| Test | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P2 (across classes) | Host macro-F1 | -0.0095 [-0.0106, -0.0084] | 0.007 |
| Test | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P2 (across classes) | Silent recall | -0.0928 [-0.1027, -0.0832] | 0.007 |
| Test | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P2 (across classes) | N_hist recall | -0.1258 [-0.1499, -0.1024] | 0.007 |
| Test | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P2, decoy count stays in slot | Host macro-F1 | -0.0084 [-0.0092, -0.0075] | 0.007 |
| Test | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P2, decoy count stays in slot | Silent recall | -0.0956 [-0.1042, -0.0873] | 0.007 |
| Test | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P2, decoy count stays in slot | N_hist recall | -0.1667 [-0.1910, -0.1432] | 0.007 |
| Test | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P3 (E_host only) | Host macro-F1 | +0.0035 [+0.0030, +0.0040] | 0.007 |
| Test | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P3 (E_host only) | Silent recall | +0.0326 [+0.0282, +0.0372] | 0.007 |
| Test | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P3 (E_host only) | N_hist recall | +0.0931 [+0.0785, +0.1084] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | GRU (full history) | identity | Host macro-F1 | +0.0153 [+0.0145, +0.0162] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | GRU (full history) | identity | Silent recall | +0.0355 [+0.0310, +0.0400] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | GRU (full history) | identity | N_hist recall | +0.0347 [+0.0261, +0.0436] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | GRU (full history) | P1 (within class) | Host macro-F1 | +0.0093 [+0.0083, +0.0103] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | GRU (full history) | P1 (within class) | Silent recall | -0.0358 [-0.0415, -0.0302] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | GRU (full history) | P1 (within class) | N_hist recall | -0.0258 [-0.0369, -0.0149] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | GRU (full history) | P1, decoy count stays in slot | Host macro-F1 | +0.0124 [+0.0116, +0.0132] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | GRU (full history) | P1, decoy count stays in slot | Silent recall | +0.0267 [+0.0219, +0.0315] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | GRU (full history) | P1, decoy count stays in slot | N_hist recall | +0.0283 [+0.0195, +0.0371] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | GRU (full history) | P2 (across classes) | Host macro-F1 | +0.0367 [+0.0348, +0.0387] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | GRU (full history) | P2 (across classes) | Silent recall | -0.1754 [-0.1858, -0.1649] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | GRU (full history) | P2 (across classes) | N_hist recall | -0.1853 [-0.2046, -0.1658] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | GRU (full history) | P2, decoy count stays in slot | Host macro-F1 | +0.0248 [+0.0234, +0.0262] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | GRU (full history) | P2, decoy count stays in slot | Silent recall | +0.0857 [+0.0784, +0.0932] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | GRU (full history) | P2, decoy count stays in slot | N_hist recall | +0.0941 [+0.0800, +0.1083] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | CMF, full attention (no topology bias) | identity | Host macro-F1 | +0.0006 [+0.0003, +0.0009] | 0.012 |
| OOD (ε = 0.3) | CyberMetaFormer | CMF, full attention (no topology bias) | identity | Silent recall | +0.0060 [+0.0041, +0.0079] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | CMF, full attention (no topology bias) | identity | N_hist recall | +0.0089 [+0.0051, +0.0129] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | CMF, full attention (no topology bias) | P1 (within class) | Host macro-F1 | +0.0019 [+0.0015, +0.0023] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | CMF, full attention (no topology bias) | P1 (within class) | Silent recall | +0.0173 [+0.0151, +0.0195] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | CMF, full attention (no topology bias) | P1 (within class) | N_hist recall | +0.0245 [+0.0194, +0.0297] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | CMF, full attention (no topology bias) | P1, decoy count stays in slot | Host macro-F1 | +0.0008 [+0.0005, +0.0012] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | CMF, full attention (no topology bias) | P1, decoy count stays in slot | Silent recall | +0.0086 [+0.0067, +0.0105] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | CMF, full attention (no topology bias) | P1, decoy count stays in slot | N_hist recall | +0.0085 [+0.0046, +0.0124] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | CMF, full attention (no topology bias) | P2 (across classes) | Host macro-F1 | -0.0063 [-0.0072, -0.0053] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | CMF, full attention (no topology bias) | P2 (across classes) | Silent recall | -0.0427 [-0.0480, -0.0375] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | CMF, full attention (no topology bias) | P2 (across classes) | N_hist recall | -0.0947 [-0.1076, -0.0822] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | CMF, full attention (no topology bias) | P2, decoy count stays in slot | Host macro-F1 | -0.0078 [-0.0085, -0.0070] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | CMF, full attention (no topology bias) | P2, decoy count stays in slot | Silent recall | -0.0520 [-0.0570, -0.0472] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | CMF, full attention (no topology bias) | P2, decoy count stays in slot | N_hist recall | -0.1086 [-0.1208, -0.0967] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | CMF, full attention (no topology bias) | P3 (E_host only) | Host macro-F1 | +0.0077 [+0.0071, +0.0083] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | CMF, full attention (no topology bias) | P3 (E_host only) | Silent recall | +0.0415 [+0.0384, +0.0447] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | CMF, full attention (no topology bias) | P3 (E_host only) | N_hist recall | +0.0750 [+0.0675, +0.0826] | 0.007 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | identity | Host macro-F1 | +0.0006 [+0.0002, +0.0009] | 0.064 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | identity | Silent recall | +0.0012 [-0.0010, +0.0033] | 1.000 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | identity | N_hist recall | +0.0114 [+0.0069, +0.0159] | 0.007 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P1 (within class) | Host macro-F1 | +0.0020 [+0.0015, +0.0025] | 0.007 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P1 (within class) | Silent recall | +0.0099 [+0.0071, +0.0127] | 0.007 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P1 (within class) | N_hist recall | +0.0389 [+0.0320, +0.0460] | 0.007 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P1, decoy count stays in slot | Host macro-F1 | +0.0009 [+0.0005, +0.0012] | 0.007 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P1, decoy count stays in slot | Silent recall | +0.0026 [+0.0004, +0.0047] | 1.000 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P1, decoy count stays in slot | N_hist recall | +0.0143 [+0.0097, +0.0191] | 0.007 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P2 (across classes) | Host macro-F1 | -0.0146 [-0.0160, -0.0132] | 0.007 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P2 (across classes) | Silent recall | -0.0880 [-0.0956, -0.0806] | 0.007 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P2 (across classes) | N_hist recall | -0.1194 [-0.1358, -0.1034] | 0.007 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P2, decoy count stays in slot | Host macro-F1 | -0.0141 [-0.0152, -0.0129] | 0.007 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P2, decoy count stays in slot | Silent recall | -0.1025 [-0.1101, -0.0952] | 0.007 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P2, decoy count stays in slot | N_hist recall | -0.1653 [-0.1823, -0.1486] | 0.007 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P3 (E_host only) | Host macro-F1 | +0.0059 [+0.0052, +0.0066] | 0.007 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P3 (E_host only) | Silent recall | +0.0295 [+0.0261, +0.0330] | 0.007 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P3 (E_host only) | N_hist recall | +0.0877 [+0.0784, +0.0974] | 0.007 |

## Difference of permutation effects: (a_P − a_id) − (b_P − b_id)

| Set | a | b | Condition | Criterion | Δ [95% CI] | p (Holm) |
|---|---|---|---|---|---|---|
| Test | CyberMetaFormer | GRU (full history) | P1 (within class) | Host macro-F1 | -0.0031 [-0.0037, -0.0025] | 0.007 |
| Test | CyberMetaFormer | GRU (full history) | P1 (within class) | Silent recall | -0.0570 [-0.0628, -0.0513] | 0.007 |
| Test | CyberMetaFormer | GRU (full history) | P1 (within class) | N_hist recall | -0.0528 [-0.0687, -0.0381] | 0.007 |
| Test | CyberMetaFormer | GRU (full history) | P1, decoy count stays in slot | Host macro-F1 | -0.0000 [-0.0004, +0.0003] | 1.000 |
| Test | CyberMetaFormer | GRU (full history) | P1, decoy count stays in slot | Silent recall | -0.0007 [-0.0037, +0.0021] | 1.000 |
| Test | CyberMetaFormer | GRU (full history) | P1, decoy count stays in slot | N_hist recall | -0.0016 [-0.0113, +0.0074] | 1.000 |
| Test | CyberMetaFormer | GRU (full history) | P2 (across classes) | Host macro-F1 | +0.0222 [+0.0202, +0.0243] | 0.007 |
| Test | CyberMetaFormer | GRU (full history) | P2 (across classes) | Silent recall | -0.1185 [-0.1312, -0.1057] | 0.007 |
| Test | CyberMetaFormer | GRU (full history) | P2 (across classes) | N_hist recall | -0.1793 [-0.2042, -0.1541] | 0.007 |
| Test | CyberMetaFormer | GRU (full history) | P2, decoy count stays in slot | Host macro-F1 | +0.0136 [+0.0122, +0.0151] | 0.007 |
| Test | CyberMetaFormer | GRU (full history) | P2, decoy count stays in slot | Silent recall | +0.0925 [+0.0824, +0.1028] | 0.007 |
| Test | CyberMetaFormer | GRU (full history) | P2, decoy count stays in slot | N_hist recall | +0.0442 [+0.0251, +0.0635] | 0.007 |
| Test | CyberMetaFormer | CMF, full attention (no topology bias) | P1 (within class) | Host macro-F1 | +0.0011 [+0.0008, +0.0014] | 0.007 |
| Test | CyberMetaFormer | CMF, full attention (no topology bias) | P1 (within class) | Silent recall | +0.0147 [+0.0120, +0.0174] | 0.007 |
| Test | CyberMetaFormer | CMF, full attention (no topology bias) | P1 (within class) | N_hist recall | +0.0167 [+0.0086, +0.0249] | 0.007 |
| Test | CyberMetaFormer | CMF, full attention (no topology bias) | P1, decoy count stays in slot | Host macro-F1 | +0.0000 [-0.0002, +0.0002] | 1.000 |
| Test | CyberMetaFormer | CMF, full attention (no topology bias) | P1, decoy count stays in slot | Silent recall | +0.0002 [-0.0010, +0.0014] | 1.000 |
| Test | CyberMetaFormer | CMF, full attention (no topology bias) | P1, decoy count stays in slot | N_hist recall | +0.0008 [-0.0036, +0.0052] | 1.000 |
| Test | CyberMetaFormer | CMF, full attention (no topology bias) | P2 (across classes) | Host macro-F1 | -0.0037 [-0.0044, -0.0030] | 0.007 |
| Test | CyberMetaFormer | CMF, full attention (no topology bias) | P2 (across classes) | Silent recall | -0.0500 [-0.0566, -0.0436] | 0.007 |
| Test | CyberMetaFormer | CMF, full attention (no topology bias) | P2 (across classes) | N_hist recall | -0.0980 [-0.1156, -0.0809] | 0.007 |
| Test | CyberMetaFormer | CMF, full attention (no topology bias) | P2, decoy count stays in slot | Host macro-F1 | -0.0044 [-0.0049, -0.0039] | 0.007 |
| Test | CyberMetaFormer | CMF, full attention (no topology bias) | P2, decoy count stays in slot | Silent recall | -0.0472 [-0.0525, -0.0422] | 0.007 |
| Test | CyberMetaFormer | CMF, full attention (no topology bias) | P2, decoy count stays in slot | N_hist recall | -0.1102 [-0.1261, -0.0947] | 0.007 |
| Test | CyberMetaFormer | CMF, full attention (no topology bias) | P3 (E_host only) | Host macro-F1 | +0.0050 [+0.0046, +0.0055] | 0.007 |
| Test | CyberMetaFormer | CMF, full attention (no topology bias) | P3 (E_host only) | Silent recall | +0.0344 [+0.0305, +0.0383] | 0.007 |
| Test | CyberMetaFormer | CMF, full attention (no topology bias) | P3 (E_host only) | N_hist recall | +0.0764 [+0.0652, +0.0878] | 0.007 |
| Test | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P1 (within class) | Host macro-F1 | +0.0010 [+0.0007, +0.0013] | 0.007 |
| Test | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P1 (within class) | Silent recall | +0.0102 [+0.0078, +0.0126] | 0.007 |
| Test | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P1 (within class) | N_hist recall | +0.0217 [+0.0142, +0.0296] | 0.007 |
| Test | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P1, decoy count stays in slot | Host macro-F1 | +0.0002 [+0.0000, +0.0003] | 1.000 |
| Test | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P1, decoy count stays in slot | Silent recall | +0.0006 [-0.0002, +0.0015] | 1.000 |
| Test | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P1, decoy count stays in slot | N_hist recall | +0.0021 [-0.0007, +0.0050] | 1.000 |
| Test | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P2 (across classes) | Host macro-F1 | -0.0097 [-0.0108, -0.0086] | 0.007 |
| Test | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P2 (across classes) | Silent recall | -0.0930 [-0.1030, -0.0833] | 0.007 |
| Test | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P2 (across classes) | N_hist recall | -0.1348 [-0.1594, -0.1108] | 0.007 |
| Test | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P2, decoy count stays in slot | Host macro-F1 | -0.0085 [-0.0094, -0.0077] | 0.007 |
| Test | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P2, decoy count stays in slot | Silent recall | -0.0957 [-0.1045, -0.0873] | 0.007 |
| Test | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P2, decoy count stays in slot | N_hist recall | -0.1756 [-0.2001, -0.1518] | 0.007 |
| Test | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P3 (E_host only) | Host macro-F1 | +0.0033 [+0.0028, +0.0038] | 0.007 |
| Test | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P3 (E_host only) | Silent recall | +0.0325 [+0.0282, +0.0370] | 0.007 |
| Test | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P3 (E_host only) | N_hist recall | +0.0842 [+0.0698, +0.0991] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | GRU (full history) | P1 (within class) | Host macro-F1 | -0.0061 [-0.0071, -0.0050] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | GRU (full history) | P1 (within class) | Silent recall | -0.0713 [-0.0771, -0.0656] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | GRU (full history) | P1 (within class) | N_hist recall | -0.0605 [-0.0719, -0.0494] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | GRU (full history) | P1, decoy count stays in slot | Host macro-F1 | -0.0029 [-0.0039, -0.0020] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | GRU (full history) | P1, decoy count stays in slot | Silent recall | -0.0088 [-0.0134, -0.0042] | 0.010 |
| OOD (ε = 0.3) | CyberMetaFormer | GRU (full history) | P1, decoy count stays in slot | N_hist recall | -0.0064 [-0.0153, +0.0024] | 1.000 |
| OOD (ε = 0.3) | CyberMetaFormer | GRU (full history) | P2 (across classes) | Host macro-F1 | +0.0214 [+0.0194, +0.0234] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | GRU (full history) | P2 (across classes) | Silent recall | -0.2109 [-0.2216, -0.2001] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | GRU (full history) | P2 (across classes) | N_hist recall | -0.2200 [-0.2399, -0.2001] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | GRU (full history) | P2, decoy count stays in slot | Host macro-F1 | +0.0095 [+0.0080, +0.0109] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | GRU (full history) | P2, decoy count stays in slot | Silent recall | +0.0502 [+0.0430, +0.0576] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | GRU (full history) | P2, decoy count stays in slot | N_hist recall | +0.0594 [+0.0445, +0.0742] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | CMF, full attention (no topology bias) | P1 (within class) | Host macro-F1 | +0.0013 [+0.0009, +0.0016] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | CMF, full attention (no topology bias) | P1 (within class) | Silent recall | +0.0113 [+0.0093, +0.0133] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | CMF, full attention (no topology bias) | P1 (within class) | N_hist recall | +0.0156 [+0.0110, +0.0202] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | CMF, full attention (no topology bias) | P1, decoy count stays in slot | Host macro-F1 | +0.0002 [-0.0000, +0.0005] | 1.000 |
| OOD (ε = 0.3) | CyberMetaFormer | CMF, full attention (no topology bias) | P1, decoy count stays in slot | Silent recall | +0.0026 [+0.0012, +0.0040] | 0.018 |
| OOD (ε = 0.3) | CyberMetaFormer | CMF, full attention (no topology bias) | P1, decoy count stays in slot | N_hist recall | -0.0004 [-0.0032, +0.0023] | 1.000 |
| OOD (ε = 0.3) | CyberMetaFormer | CMF, full attention (no topology bias) | P2 (across classes) | Host macro-F1 | -0.0069 [-0.0078, -0.0059] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | CMF, full attention (no topology bias) | P2 (across classes) | Silent recall | -0.0487 [-0.0541, -0.0434] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | CMF, full attention (no topology bias) | P2 (across classes) | N_hist recall | -0.1036 [-0.1168, -0.0908] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | CMF, full attention (no topology bias) | P2, decoy count stays in slot | Host macro-F1 | -0.0084 [-0.0092, -0.0076] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | CMF, full attention (no topology bias) | P2, decoy count stays in slot | Silent recall | -0.0580 [-0.0630, -0.0532] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | CMF, full attention (no topology bias) | P2, decoy count stays in slot | N_hist recall | -0.1176 [-0.1301, -0.1054] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | CMF, full attention (no topology bias) | P3 (E_host only) | Host macro-F1 | +0.0071 [+0.0065, +0.0076] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | CMF, full attention (no topology bias) | P3 (E_host only) | Silent recall | +0.0355 [+0.0325, +0.0386] | 0.007 |
| OOD (ε = 0.3) | CyberMetaFormer | CMF, full attention (no topology bias) | P3 (E_host only) | N_hist recall | +0.0661 [+0.0590, +0.0732] | 0.007 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P1 (within class) | Host macro-F1 | +0.0014 [+0.0010, +0.0018] | 0.007 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P1 (within class) | Silent recall | +0.0087 [+0.0065, +0.0110] | 0.007 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P1 (within class) | N_hist recall | +0.0275 [+0.0218, +0.0334] | 0.007 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P1, decoy count stays in slot | Host macro-F1 | +0.0003 [+0.0001, +0.0006] | 0.708 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P1, decoy count stays in slot | Silent recall | +0.0014 [+0.0000, +0.0028] | 1.000 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P1, decoy count stays in slot | N_hist recall | +0.0029 [+0.0001, +0.0057] | 1.000 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P2 (across classes) | Host macro-F1 | -0.0152 [-0.0166, -0.0138] | 0.007 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P2 (across classes) | Silent recall | -0.0891 [-0.0966, -0.0818] | 0.007 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P2 (across classes) | N_hist recall | -0.1308 [-0.1468, -0.1151] | 0.007 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P2, decoy count stays in slot | Host macro-F1 | -0.0146 [-0.0158, -0.0135] | 0.007 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P2, decoy count stays in slot | Silent recall | -0.1037 [-0.1110, -0.0965] | 0.007 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P2, decoy count stays in slot | N_hist recall | -0.1767 [-0.1933, -0.1605] | 0.007 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P3 (E_host only) | Host macro-F1 | +0.0053 [+0.0047, +0.0059] | 0.007 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P3 (E_host only) | Silent recall | +0.0284 [+0.0251, +0.0317] | 0.007 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | Host-token GRU, attention without bias | P3 (E_host only) | N_hist recall | +0.0763 [+0.0674, +0.0855] | 0.007 |

## Error types (rate within the true class; mean ± sd over seeds)

| Set | Model | Condition | root→clean | root→user | user→clean | clean false alarm |
|---|---|---|---|---|---|---|
| Test | CyberMetaFormer | identity | 0.038 ± 0.003 | 0.000 ± 0.000 | 0.042 ± 0.001 | 0.008 ± 0.001 |
| Test | CyberMetaFormer | P1 (within class) | 0.062 ± 0.009 | 0.000 ± 0.000 | 0.043 ± 0.000 | 0.006 ± 0.001 |
| Test | CyberMetaFormer | P1, decoy count stays in slot | 0.039 ± 0.003 | 0.000 ± 0.000 | 0.042 ± 0.001 | 0.008 ± 0.001 |
| Test | CyberMetaFormer | P2 (across classes) | 0.121 ± 0.017 | 0.000 ± 0.000 | 0.048 ± 0.003 | 0.007 ± 0.001 |
| Test | CyberMetaFormer | P2, decoy count stays in slot | 0.072 ± 0.013 | 0.000 ± 0.000 | 0.042 ± 0.001 | 0.008 ± 0.001 |
| Test | CyberMetaFormer | P3 (E_host only) | 0.083 ± 0.010 | 0.000 ± 0.000 | 0.047 ± 0.003 | 0.006 ± 0.001 |
| Test | CMF, full attention (no topology bias) | identity | 0.038 ± 0.003 | 0.000 ± 0.000 | 0.041 ± 0.000 | 0.008 ± 0.001 |
| Test | CMF, full attention (no topology bias) | P1 (within class) | 0.068 ± 0.005 | 0.000 ± 0.000 | 0.043 ± 0.000 | 0.006 ± 0.001 |
| Test | CMF, full attention (no topology bias) | P1, decoy count stays in slot | 0.038 ± 0.003 | 0.000 ± 0.000 | 0.041 ± 0.000 | 0.008 ± 0.001 |
| Test | CMF, full attention (no topology bias) | P2 (across classes) | 0.099 ± 0.030 | 0.000 ± 0.000 | 0.051 ± 0.009 | 0.008 ± 0.003 |
| Test | CMF, full attention (no topology bias) | P2, decoy count stays in slot | 0.051 ± 0.014 | 0.000 ± 0.000 | 0.041 ± 0.000 | 0.009 ± 0.001 |
| Test | CMF, full attention (no topology bias) | P3 (E_host only) | 0.099 ± 0.030 | 0.000 ± 0.000 | 0.051 ± 0.009 | 0.008 ± 0.003 |
| Test | Host-token GRU, topology-biased attention | identity | 0.038 ± 0.001 | 0.000 ± 0.000 | 0.041 ± 0.000 | 0.007 ± 0.000 |
| Test | Host-token GRU, topology-biased attention | P1 (within class) | 0.054 ± 0.010 | 0.000 ± 0.000 | 0.043 ± 0.001 | 0.007 ± 0.001 |
| Test | Host-token GRU, topology-biased attention | P1, decoy count stays in slot | 0.038 ± 0.001 | 0.000 ± 0.000 | 0.041 ± 0.000 | 0.008 ± 0.000 |
| Test | Host-token GRU, topology-biased attention | P2 (across classes) | 0.145 ± 0.026 | 0.001 ± 0.000 | 0.049 ± 0.003 | 0.006 ± 0.001 |
| Test | Host-token GRU, topology-biased attention | P2, decoy count stays in slot | 0.087 ± 0.015 | 0.000 ± 0.000 | 0.042 ± 0.000 | 0.008 ± 0.000 |
| Test | Host-token GRU, topology-biased attention | P3 (E_host only) | 0.088 ± 0.018 | 0.000 ± 0.000 | 0.048 ± 0.003 | 0.006 ± 0.001 |
| Test | Host-token GRU, attention without bias | identity | 0.038 ± 0.003 | 0.000 ± 0.000 | 0.041 ± 0.000 | 0.008 ± 0.001 |
| Test | Host-token GRU, attention without bias | P1 (within class) | 0.058 ± 0.010 | 0.000 ± 0.000 | 0.043 ± 0.000 | 0.007 ± 0.001 |
| Test | Host-token GRU, attention without bias | P1, decoy count stays in slot | 0.039 ± 0.004 | 0.000 ± 0.000 | 0.041 ± 0.000 | 0.008 ± 0.001 |
| Test | Host-token GRU, attention without bias | P2 (across classes) | 0.103 ± 0.017 | 0.000 ± 0.000 | 0.048 ± 0.002 | 0.007 ± 0.001 |
| Test | Host-token GRU, attention without bias | P2, decoy count stays in slot | 0.044 ± 0.007 | 0.000 ± 0.000 | 0.041 ± 0.000 | 0.008 ± 0.001 |
| Test | Host-token GRU, attention without bias | P3 (E_host only) | 0.103 ± 0.017 | 0.000 ± 0.000 | 0.048 ± 0.002 | 0.007 ± 0.001 |
| Test | GRU (full history) | identity | 0.040 ± 0.001 | 0.000 ± 0.000 | 0.043 ± 0.000 | 0.009 ± 0.000 |
| Test | GRU (full history) | P1 (within class) | 0.038 ± 0.002 | 0.000 ± 0.000 | 0.042 ± 0.001 | 0.011 ± 0.000 |
| Test | GRU (full history) | P1, decoy count stays in slot | 0.040 ± 0.001 | 0.000 ± 0.000 | 0.043 ± 0.001 | 0.010 ± 0.000 |
| Test | GRU (full history) | P2 (across classes) | 0.070 ± 0.010 | 0.001 ± 0.000 | 0.042 ± 0.001 | 0.059 ± 0.012 |
| Test | GRU (full history) | P2, decoy count stays in slot | 0.116 ± 0.002 | 0.001 ± 0.000 | 0.047 ± 0.002 | 0.017 ± 0.002 |
| OOD (ε = 0.3) | CyberMetaFormer | identity | 0.175 ± 0.016 | 0.000 ± 0.000 | 0.036 ± 0.001 | 0.016 ± 0.002 |
| OOD (ε = 0.3) | CyberMetaFormer | P1 (within class) | 0.224 ± 0.019 | 0.000 ± 0.000 | 0.037 ± 0.001 | 0.015 ± 0.002 |
| OOD (ε = 0.3) | CyberMetaFormer | P1, decoy count stays in slot | 0.184 ± 0.017 | 0.000 ± 0.000 | 0.036 ± 0.000 | 0.018 ± 0.002 |
| OOD (ε = 0.3) | CyberMetaFormer | P2 (across classes) | 0.329 ± 0.026 | 0.000 ± 0.000 | 0.039 ± 0.002 | 0.014 ± 0.002 |
| OOD (ε = 0.3) | CyberMetaFormer | P2, decoy count stays in slot | 0.261 ± 0.023 | 0.000 ± 0.000 | 0.036 ± 0.001 | 0.018 ± 0.002 |
| OOD (ε = 0.3) | CyberMetaFormer | P3 (E_host only) | 0.259 ± 0.015 | 0.000 ± 0.000 | 0.039 ± 0.002 | 0.014 ± 0.001 |
| OOD (ε = 0.3) | CMF, full attention (no topology bias) | identity | 0.180 ± 0.014 | 0.000 ± 0.000 | 0.036 ± 0.001 | 0.016 ± 0.001 |
| OOD (ε = 0.3) | CMF, full attention (no topology bias) | P1 (within class) | 0.238 ± 0.019 | 0.000 ± 0.000 | 0.037 ± 0.001 | 0.014 ± 0.001 |
| OOD (ε = 0.3) | CMF, full attention (no topology bias) | P1, decoy count stays in slot | 0.191 ± 0.015 | 0.000 ± 0.000 | 0.036 ± 0.001 | 0.018 ± 0.001 |
| OOD (ε = 0.3) | CMF, full attention (no topology bias) | P2 (across classes) | 0.293 ± 0.047 | 0.000 ± 0.000 | 0.039 ± 0.002 | 0.015 ± 0.003 |
| OOD (ε = 0.3) | CMF, full attention (no topology bias) | P2, decoy count stays in slot | 0.218 ± 0.025 | 0.000 ± 0.000 | 0.036 ± 0.001 | 0.018 ± 0.001 |
| OOD (ε = 0.3) | CMF, full attention (no topology bias) | P3 (E_host only) | 0.293 ± 0.047 | 0.000 ± 0.000 | 0.039 ± 0.002 | 0.015 ± 0.003 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | identity | 0.181 ± 0.006 | 0.000 ± 0.000 | 0.036 ± 0.000 | 0.015 ± 0.001 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | P1 (within class) | 0.219 ± 0.010 | 0.000 ± 0.000 | 0.037 ± 0.001 | 0.015 ± 0.001 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | P1, decoy count stays in slot | 0.190 ± 0.006 | 0.000 ± 0.000 | 0.036 ± 0.000 | 0.017 ± 0.001 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | P2 (across classes) | 0.362 ± 0.015 | 0.001 ± 0.000 | 0.040 ± 0.001 | 0.013 ± 0.002 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | P2, decoy count stays in slot | 0.291 ± 0.015 | 0.000 ± 0.000 | 0.036 ± 0.000 | 0.016 ± 0.001 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | P3 (E_host only) | 0.265 ± 0.010 | 0.000 ± 0.000 | 0.039 ± 0.001 | 0.014 ± 0.002 |
| OOD (ε = 0.3) | Host-token GRU, attention without bias | identity | 0.182 ± 0.014 | 0.000 ± 0.000 | 0.036 ± 0.001 | 0.015 ± 0.002 |
| OOD (ε = 0.3) | Host-token GRU, attention without bias | P1 (within class) | 0.228 ± 0.018 | 0.000 ± 0.000 | 0.037 ± 0.001 | 0.015 ± 0.002 |
| OOD (ε = 0.3) | Host-token GRU, attention without bias | P1, decoy count stays in slot | 0.192 ± 0.015 | 0.000 ± 0.000 | 0.036 ± 0.001 | 0.017 ± 0.002 |
| OOD (ε = 0.3) | Host-token GRU, attention without bias | P2 (across classes) | 0.289 ± 0.017 | 0.000 ± 0.000 | 0.039 ± 0.001 | 0.015 ± 0.003 |
| OOD (ε = 0.3) | Host-token GRU, attention without bias | P2, decoy count stays in slot | 0.206 ± 0.020 | 0.000 ± 0.000 | 0.036 ± 0.001 | 0.018 ± 0.001 |
| OOD (ε = 0.3) | Host-token GRU, attention without bias | P3 (E_host only) | 0.289 ± 0.017 | 0.000 ± 0.000 | 0.039 ± 0.001 | 0.015 ± 0.003 |
| OOD (ε = 0.3) | GRU (full history) | identity | 0.205 ± 0.007 | 0.001 ± 0.000 | 0.038 ± 0.001 | 0.021 ± 0.001 |
| OOD (ε = 0.3) | GRU (full history) | P1 (within class) | 0.195 ± 0.009 | 0.001 ± 0.000 | 0.037 ± 0.001 | 0.024 ± 0.001 |
| OOD (ε = 0.3) | GRU (full history) | P1, decoy count stays in slot | 0.207 ± 0.007 | 0.001 ± 0.000 | 0.038 ± 0.001 | 0.022 ± 0.001 |
| OOD (ε = 0.3) | GRU (full history) | P2 (across classes) | 0.187 ± 0.014 | 0.001 ± 0.000 | 0.033 ± 0.001 | 0.076 ± 0.010 |
| OOD (ε = 0.3) | GRU (full history) | P2, decoy count stays in slot | 0.333 ± 0.005 | 0.001 ± 0.000 | 0.039 ± 0.001 | 0.024 ± 0.001 |

## Silent recall by host slot group (mean ± sd over seeds)

| Set | Model | Condition | user1–2 | user3–4 | ent0 | ent1 | ent2 | op |
|---|---|---|---|---|---|---|---|---|
| Test | silent host-steps | identity | 64,610 | 65,471 | 21,917 | 29,754 | 23,469 | 11,274 |
| Test | CyberMetaFormer | identity | 0.893 ± 0.008 | 0.888 ± 0.007 | 0.881 ± 0.010 | 0.914 ± 0.004 | 0.945 ± 0.004 | 1.000 ± 0.000 |
| Test | CyberMetaFormer | P1 (within class) | 0.878 ± 0.009 | 0.743 ± 0.046 | 0.870 ± 0.014 | 0.904 ± 0.007 | 0.933 ± 0.009 | 1.000 ± 0.000 |
| Test | CyberMetaFormer | P1, decoy count stays in slot | 0.893 ± 0.007 | 0.887 ± 0.008 | 0.881 ± 0.010 | 0.915 ± 0.004 | 0.943 ± 0.005 | 1.000 ± 0.000 |
| Test | CyberMetaFormer | P2 (across classes) | 0.443 ± 0.093 | 0.745 ± 0.063 | 0.855 ± 0.007 | 0.899 ± 0.007 | 0.928 ± 0.007 | 1.000 ± 0.000 |
| Test | CyberMetaFormer | P2, decoy count stays in slot | 0.771 ± 0.043 | 0.766 ± 0.051 | 0.873 ± 0.012 | 0.911 ± 0.013 | 0.944 ± 0.004 | 1.000 ± 0.000 |
| Test | CyberMetaFormer | P3 (E_host only) | 0.880 ± 0.008 | 0.586 ± 0.065 | 0.867 ± 0.017 | 0.897 ± 0.005 | 0.929 ± 0.007 | 1.000 ± 0.000 |
| Test | CMF, full attention (no topology bias) | identity | 0.892 ± 0.006 | 0.891 ± 0.006 | 0.875 ± 0.008 | 0.915 ± 0.011 | 0.946 ± 0.003 | 1.000 ± 0.000 |
| Test | CMF, full attention (no topology bias) | P1 (within class) | 0.871 ± 0.007 | 0.705 ± 0.027 | 0.864 ± 0.005 | 0.905 ± 0.011 | 0.929 ± 0.006 | 1.000 ± 0.000 |
| Test | CMF, full attention (no topology bias) | P1, decoy count stays in slot | 0.892 ± 0.006 | 0.889 ± 0.006 | 0.876 ± 0.008 | 0.916 ± 0.011 | 0.944 ± 0.004 | 1.000 ± 0.000 |
| Test | CMF, full attention (no topology bias) | P2 (across classes) | 0.536 ± 0.153 | 0.824 ± 0.075 | 0.854 ± 0.013 | 0.898 ± 0.011 | 0.918 ± 0.016 | 1.000 ± 0.000 |
| Test | CMF, full attention (no topology bias) | P2, decoy count stays in slot | 0.848 ± 0.048 | 0.843 ± 0.063 | 0.874 ± 0.007 | 0.916 ± 0.010 | 0.944 ± 0.003 | 1.000 ± 0.000 |
| Test | CMF, full attention (no topology bias) | P3 (E_host only) | 0.824 ± 0.075 | 0.536 ± 0.153 | 0.854 ± 0.013 | 0.898 ± 0.011 | 0.918 ± 0.016 | 1.000 ± 0.000 |
| Test | Host-token GRU, topology-biased attention | identity | 0.892 ± 0.004 | 0.891 ± 0.004 | 0.874 ± 0.005 | 0.915 ± 0.004 | 0.945 ± 0.002 | 1.000 ± 0.000 |
| Test | Host-token GRU, topology-biased attention | P1 (within class) | 0.889 ± 0.011 | 0.785 ± 0.060 | 0.867 ± 0.004 | 0.911 ± 0.003 | 0.940 ± 0.003 | 1.000 ± 0.000 |
| Test | Host-token GRU, topology-biased attention | P1, decoy count stays in slot | 0.893 ± 0.003 | 0.889 ± 0.005 | 0.874 ± 0.005 | 0.915 ± 0.005 | 0.944 ± 0.001 | 1.000 ± 0.000 |
| Test | Host-token GRU, topology-biased attention | P2 (across classes) | 0.303 ± 0.129 | 0.710 ± 0.090 | 0.852 ± 0.019 | 0.902 ± 0.008 | 0.936 ± 0.005 | 1.000 ± 0.001 |
| Test | Host-token GRU, topology-biased attention | P2, decoy count stays in slot | 0.725 ± 0.050 | 0.708 ± 0.062 | 0.860 ± 0.013 | 0.909 ± 0.009 | 0.944 ± 0.002 | 1.000 ± 0.000 |
| Test | Host-token GRU, topology-biased attention | P3 (E_host only) | 0.895 ± 0.016 | 0.527 ± 0.121 | 0.862 ± 0.008 | 0.912 ± 0.008 | 0.937 ± 0.006 | 1.000 ± 0.001 |
| Test | Host-token GRU, attention without bias | identity | 0.891 ± 0.008 | 0.891 ± 0.009 | 0.876 ± 0.009 | 0.913 ± 0.011 | 0.946 ± 0.004 | 1.000 ± 0.000 |
| Test | Host-token GRU, attention without bias | P1 (within class) | 0.889 ± 0.011 | 0.748 ± 0.059 | 0.874 ± 0.011 | 0.912 ± 0.011 | 0.940 ± 0.006 | 1.000 ± 0.000 |
| Test | Host-token GRU, attention without bias | P1, decoy count stays in slot | 0.891 ± 0.009 | 0.889 ± 0.009 | 0.876 ± 0.009 | 0.913 ± 0.011 | 0.943 ± 0.005 | 1.000 ± 0.000 |
| Test | Host-token GRU, attention without bias | P2 (across classes) | 0.433 ± 0.098 | 0.878 ± 0.025 | 0.873 ± 0.011 | 0.914 ± 0.010 | 0.934 ± 0.008 | 1.000 ± 0.000 |
| Test | Host-token GRU, attention without bias | P2, decoy count stays in slot | 0.875 ± 0.018 | 0.868 ± 0.034 | 0.875 ± 0.010 | 0.914 ± 0.011 | 0.945 ± 0.004 | 1.000 ± 0.000 |
| Test | Host-token GRU, attention without bias | P3 (E_host only) | 0.878 ± 0.025 | 0.433 ± 0.098 | 0.873 ± 0.011 | 0.914 ± 0.010 | 0.934 ± 0.008 | 1.000 ± 0.000 |
| Test | GRU (full history) | identity | 0.891 ± 0.006 | 0.892 ± 0.002 | 0.858 ± 0.011 | 0.911 ± 0.011 | 0.937 ± 0.008 | 1.000 ± 0.000 |
| Test | GRU (full history) | P1 (within class) | 0.900 ± 0.008 | 0.896 ± 0.013 | 0.860 ± 0.008 | 0.916 ± 0.014 | 0.937 ± 0.011 | 1.000 ± 0.000 |
| Test | GRU (full history) | P1, decoy count stays in slot | 0.892 ± 0.006 | 0.892 ± 0.003 | 0.863 ± 0.007 | 0.910 ± 0.017 | 0.936 ± 0.010 | 1.000 ± 0.000 |
| Test | GRU (full history) | P2 (across classes) | 0.702 ± 0.035 | 0.866 ± 0.062 | 0.802 ± 0.045 | 0.929 ± 0.013 | 0.954 ± 0.006 | 0.999 ± 0.001 |
| Test | GRU (full history) | P2, decoy count stays in slot | 0.645 ± 0.013 | 0.620 ± 0.007 | 0.812 ± 0.016 | 0.861 ± 0.023 | 0.934 ± 0.006 | 1.000 ± 0.000 |
| OOD (ε = 0.3) | silent host-steps | identity | 87,972 | 90,315 | 32,724 | 46,538 | 47,218 | 16,241 |
| OOD (ε = 0.3) | CyberMetaFormer | identity | 0.736 ± 0.026 | 0.730 ± 0.019 | 0.759 ± 0.027 | 0.792 ± 0.020 | 0.831 ± 0.014 | 0.994 ± 0.002 |
| OOD (ε = 0.3) | CyberMetaFormer | P1 (within class) | 0.713 ± 0.026 | 0.562 ± 0.024 | 0.743 ± 0.034 | 0.776 ± 0.025 | 0.819 ± 0.017 | 0.993 ± 0.002 |
| OOD (ε = 0.3) | CyberMetaFormer | P1, decoy count stays in slot | 0.719 ± 0.027 | 0.708 ± 0.022 | 0.759 ± 0.026 | 0.790 ± 0.022 | 0.831 ± 0.014 | 0.993 ± 0.002 |
| OOD (ε = 0.3) | CyberMetaFormer | P2 (across classes) | 0.253 ± 0.076 | 0.586 ± 0.056 | 0.717 ± 0.017 | 0.759 ± 0.020 | 0.816 ± 0.013 | 0.993 ± 0.001 |
| OOD (ε = 0.3) | CyberMetaFormer | P2, decoy count stays in slot | 0.559 ± 0.047 | 0.547 ± 0.037 | 0.743 ± 0.029 | 0.779 ± 0.025 | 0.830 ± 0.015 | 0.993 ± 0.002 |
| OOD (ε = 0.3) | CyberMetaFormer | P3 (E_host only) | 0.742 ± 0.007 | 0.388 ± 0.062 | 0.739 ± 0.026 | 0.768 ± 0.015 | 0.817 ± 0.013 | 0.993 ± 0.001 |
| OOD (ε = 0.3) | CMF, full attention (no topology bias) | identity | 0.728 ± 0.019 | 0.726 ± 0.024 | 0.746 ± 0.019 | 0.784 ± 0.018 | 0.828 ± 0.014 | 0.993 ± 0.001 |
| OOD (ε = 0.3) | CMF, full attention (no topology bias) | P1 (within class) | 0.693 ± 0.028 | 0.532 ± 0.035 | 0.728 ± 0.023 | 0.767 ± 0.016 | 0.814 ± 0.016 | 0.992 ± 0.001 |
| OOD (ε = 0.3) | CMF, full attention (no topology bias) | P1, decoy count stays in slot | 0.707 ± 0.020 | 0.700 ± 0.025 | 0.744 ± 0.020 | 0.783 ± 0.018 | 0.827 ± 0.014 | 0.992 ± 0.001 |
| OOD (ε = 0.3) | CMF, full attention (no topology bias) | P2 (across classes) | 0.363 ± 0.099 | 0.645 ± 0.093 | 0.712 ± 0.034 | 0.752 ± 0.030 | 0.798 ± 0.031 | 0.992 ± 0.001 |
| OOD (ε = 0.3) | CMF, full attention (no topology bias) | P2, decoy count stays in slot | 0.658 ± 0.064 | 0.635 ± 0.060 | 0.743 ± 0.020 | 0.783 ± 0.017 | 0.827 ± 0.013 | 0.993 ± 0.001 |
| OOD (ε = 0.3) | CMF, full attention (no topology bias) | P3 (E_host only) | 0.645 ± 0.093 | 0.363 ± 0.099 | 0.712 ± 0.034 | 0.752 ± 0.030 | 0.798 ± 0.031 | 0.992 ± 0.001 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | identity | 0.731 ± 0.007 | 0.726 ± 0.010 | 0.732 ± 0.009 | 0.776 ± 0.010 | 0.833 ± 0.011 | 0.991 ± 0.002 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | P1 (within class) | 0.726 ± 0.012 | 0.574 ± 0.027 | 0.725 ± 0.006 | 0.770 ± 0.006 | 0.828 ± 0.011 | 0.989 ± 0.003 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | P1, decoy count stays in slot | 0.715 ± 0.007 | 0.706 ± 0.011 | 0.730 ± 0.009 | 0.774 ± 0.009 | 0.832 ± 0.010 | 0.990 ± 0.003 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | P2 (across classes) | 0.135 ± 0.031 | 0.563 ± 0.064 | 0.708 ± 0.027 | 0.750 ± 0.017 | 0.824 ± 0.014 | 0.990 ± 0.002 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | P2, decoy count stays in slot | 0.502 ± 0.035 | 0.497 ± 0.035 | 0.715 ± 0.017 | 0.756 ± 0.019 | 0.831 ± 0.010 | 0.990 ± 0.002 |
| OOD (ε = 0.3) | Host-token GRU, topology-biased attention | P3 (E_host only) | 0.761 ± 0.037 | 0.344 ± 0.041 | 0.721 ± 0.013 | 0.771 ± 0.009 | 0.826 ± 0.014 | 0.990 ± 0.002 |
| OOD (ε = 0.3) | Host-token GRU, attention without bias | identity | 0.731 ± 0.013 | 0.723 ± 0.019 | 0.741 ± 0.033 | 0.774 ± 0.035 | 0.825 ± 0.013 | 0.991 ± 0.003 |
| OOD (ε = 0.3) | Host-token GRU, attention without bias | P1 (within class) | 0.722 ± 0.015 | 0.542 ± 0.034 | 0.737 ± 0.035 | 0.769 ± 0.038 | 0.821 ± 0.013 | 0.991 ± 0.003 |
| OOD (ε = 0.3) | Host-token GRU, attention without bias | P1, decoy count stays in slot | 0.713 ± 0.014 | 0.702 ± 0.023 | 0.738 ± 0.033 | 0.771 ± 0.036 | 0.824 ± 0.014 | 0.991 ± 0.003 |
| OOD (ε = 0.3) | Host-token GRU, attention without bias | P2 (across classes) | 0.276 ± 0.024 | 0.727 ± 0.040 | 0.732 ± 0.037 | 0.763 ± 0.044 | 0.817 ± 0.017 | 0.991 ± 0.002 |
| OOD (ε = 0.3) | Host-token GRU, attention without bias | P2, decoy count stays in slot | 0.697 ± 0.021 | 0.657 ± 0.056 | 0.738 ± 0.034 | 0.772 ± 0.037 | 0.823 ± 0.014 | 0.991 ± 0.003 |
| OOD (ε = 0.3) | Host-token GRU, attention without bias | P3 (E_host only) | 0.727 ± 0.040 | 0.276 ± 0.024 | 0.732 ± 0.037 | 0.763 ± 0.044 | 0.817 ± 0.017 | 0.991 ± 0.002 |
| OOD (ε = 0.3) | GRU (full history) | identity | 0.704 ± 0.018 | 0.701 ± 0.015 | 0.689 ± 0.024 | 0.739 ± 0.031 | 0.807 ± 0.027 | 0.982 ± 0.005 |
| OOD (ε = 0.3) | GRU (full history) | P1 (within class) | 0.726 ± 0.031 | 0.719 ± 0.031 | 0.680 ± 0.025 | 0.750 ± 0.031 | 0.810 ± 0.024 | 0.983 ± 0.007 |
| OOD (ε = 0.3) | GRU (full history) | P1, decoy count stays in slot | 0.704 ± 0.023 | 0.694 ± 0.014 | 0.685 ± 0.026 | 0.739 ± 0.035 | 0.811 ± 0.022 | 0.983 ± 0.007 |
| OOD (ε = 0.3) | GRU (full history) | P2 (across classes) | 0.590 ± 0.060 | 0.851 ± 0.043 | 0.633 ± 0.059 | 0.827 ± 0.037 | 0.869 ± 0.014 | 0.974 ± 0.011 |
| OOD (ε = 0.3) | GRU (full history) | P2, decoy count stays in slot | 0.487 ± 0.025 | 0.462 ± 0.018 | 0.584 ± 0.046 | 0.629 ± 0.050 | 0.806 ± 0.024 | 0.981 ± 0.007 |

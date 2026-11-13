# M4 revision, item 4 pairwise tests (paired episode bootstrap)

100000 samples, bootstrap seed 0. Each model enters with the mean over the given seeds (column: number of seeds). a − b; 95% CI. p Holm: pair = over the criteria of the pair; plan = over all rows. Smallest p 2 / (n + 1). For the relative drop smaller is better. `python -m cybermetaformer.hidden_state.compare --plan m4`.
Plan-wide Holm family: 35 rows; smallest raw p 2e-05, smallest plan-wide corrected p 0.0007.

## A4: host-token GRU − CMF (M4 Pro, against ana_m4)

| a | b | seeds a / b | criterion | a | b | a − b | 95% CI | p Holm (pair) | p Holm (plan) |
|---|---|---|---|---:|---:|---:|---|---:|---:|
| cybermetaformer-gru_host_topo_k101 | cybermetaformer-ana_m4_k101 | 5 / 5 | test_host_macro_f1 | 0.9803 | 0.9800 | +0.0002 | [+0.0001, +0.0004] | 0.0253 | 0.1179 |
| cybermetaformer-gru_host_topo_k101 | cybermetaformer-ana_m4_k101 | 5 / 5 | ood_host_macro_f1 | 0.9419 | 0.9414 | +0.0005 | [+0.0001, +0.0009] | 0.0318 | 0.2064 |
| cybermetaformer-gru_host_topo_k101 | cybermetaformer-ana_m4_k101 | 5 / 5 | ood_rel_drop | 0.0391 | 0.0394 | -0.0003 | [-0.0007, +0.0002] | 0.2230 | 1.0000 |
| cybermetaformer-gru_host_topo_k101 | cybermetaformer-ana_m4_k101 | 5 / 5 | test_silent_recall | 0.9042 | 0.9065 | -0.0022 | [-0.0037, -0.0008] | 0.0086 | 0.0346 |
| cybermetaformer-gru_host_topo_k101 | cybermetaformer-ana_m4_k101 | 5 / 5 | ood_silent_recall | 0.7642 | 0.7854 | -0.0213 | [-0.0238, -0.0188] | 0.0001 | 0.0007 |
| cybermetaformer-gru_host_topo_k101 | cybermetaformer-ana_m4_k101 | 5 / 5 | test_never_recall | 0.8851 | 0.8917 | -0.0066 | [-0.0102, -0.0031] | 0.0012 | 0.0043 |
| cybermetaformer-gru_host_topo_k101 | cybermetaformer-ana_m4_k101 | 5 / 5 | ood_never_recall | 0.8257 | 0.8356 | -0.0099 | [-0.0144, -0.0055] | 0.0001 | 0.0007 |
| cybermetaformer-gru_host_attn_k101 | cybermetaformer-ana_m4_k101 | 5 / 5 | test_host_macro_f1 | 0.9801 | 0.9800 | +0.0000 | [-0.0002, +0.0003] | 1.0000 | 1.0000 |
| cybermetaformer-gru_host_attn_k101 | cybermetaformer-ana_m4_k101 | 5 / 5 | ood_host_macro_f1 | 0.9413 | 0.9414 | -0.0001 | [-0.0005, +0.0003] | 1.0000 | 1.0000 |
| cybermetaformer-gru_host_attn_k101 | cybermetaformer-ana_m4_k101 | 5 / 5 | ood_rel_drop | 0.0395 | 0.0394 | +0.0001 | [-0.0003, +0.0006] | 1.0000 | 1.0000 |
| cybermetaformer-gru_host_attn_k101 | cybermetaformer-ana_m4_k101 | 5 / 5 | test_silent_recall | 0.9041 | 0.9065 | -0.0024 | [-0.0041, -0.0007] | 0.0295 | 0.1107 |
| cybermetaformer-gru_host_attn_k101 | cybermetaformer-ana_m4_k101 | 5 / 5 | ood_silent_recall | 0.7630 | 0.7854 | -0.0224 | [-0.0250, -0.0199] | 0.0001 | 0.0007 |
| cybermetaformer-gru_host_attn_k101 | cybermetaformer-ana_m4_k101 | 5 / 5 | test_never_recall | 0.8762 | 0.8917 | -0.0155 | [-0.0208, -0.0105] | 0.0001 | 0.0007 |
| cybermetaformer-gru_host_attn_k101 | cybermetaformer-ana_m4_k101 | 5 / 5 | ood_never_recall | 0.8142 | 0.8356 | -0.0214 | [-0.0257, -0.0170] | 0.0001 | 0.0007 |

## A4: topology bias within the host-token GRU (M4 Pro)

| a | b | seeds a / b | criterion | a | b | a − b | 95% CI | p Holm (pair) | p Holm (plan) |
|---|---|---|---|---:|---:|---:|---|---:|---:|
| cybermetaformer-gru_host_topo_k101 | cybermetaformer-gru_host_attn_k101 | 5 / 5 | test_host_macro_f1 | 0.9803 | 0.9801 | +0.0002 | [+0.0000, +0.0004] | 0.1526 | 0.4510 |
| cybermetaformer-gru_host_topo_k101 | cybermetaformer-gru_host_attn_k101 | 5 / 5 | ood_host_macro_f1 | 0.9419 | 0.9413 | +0.0006 | [+0.0002, +0.0009] | 0.0053 | 0.0180 |
| cybermetaformer-gru_host_topo_k101 | cybermetaformer-gru_host_attn_k101 | 5 / 5 | ood_rel_drop | 0.0391 | 0.0395 | -0.0004 | [-0.0008, -0.0000] | 0.1526 | 0.4862 |
| cybermetaformer-gru_host_topo_k101 | cybermetaformer-gru_host_attn_k101 | 5 / 5 | test_silent_recall | 0.9042 | 0.9041 | +0.0001 | [-0.0013, +0.0016] | 0.8615 | 1.0000 |
| cybermetaformer-gru_host_topo_k101 | cybermetaformer-gru_host_attn_k101 | 5 / 5 | ood_silent_recall | 0.7642 | 0.7630 | +0.0012 | [-0.0010, +0.0033] | 0.5860 | 1.0000 |
| cybermetaformer-gru_host_topo_k101 | cybermetaformer-gru_host_attn_k101 | 5 / 5 | test_never_recall | 0.8851 | 0.8762 | +0.0089 | [+0.0043, +0.0136] | 0.0013 | 0.0042 |
| cybermetaformer-gru_host_topo_k101 | cybermetaformer-gru_host_attn_k101 | 5 / 5 | ood_never_recall | 0.8257 | 0.8142 | +0.0114 | [+0.0070, +0.0159] | 0.0001 | 0.0007 |

## A4: host-token GRU − full-history GRU (across machines)

| a | b | seeds a / b | criterion | a | b | a − b | 95% CI | p Holm (pair) | p Holm (plan) |
|---|---|---|---|---:|---:|---:|---|---:|---:|
| cybermetaformer-gru_host_topo_k101 | gru_full_k101 | 5 / 5 | test_host_macro_f1 | 0.9803 | 0.9768 | +0.0035 | [+0.0030, +0.0040] | 0.0001 | 0.0007 |
| cybermetaformer-gru_host_topo_k101 | gru_full_k101 | 5 / 5 | ood_host_macro_f1 | 0.9419 | 0.9264 | +0.0155 | [+0.0147, +0.0163] | 0.0001 | 0.0007 |
| cybermetaformer-gru_host_topo_k101 | gru_full_k101 | 5 / 5 | ood_rel_drop | 0.0391 | 0.0516 | -0.0124 | [-0.0134, -0.0115] | 0.0001 | 0.0007 |
| cybermetaformer-gru_host_topo_k101 | gru_full_k101 | 5 / 5 | test_silent_recall | 0.9042 | 0.9015 | +0.0027 | [-0.0002, +0.0055] | 0.1275 | 0.5737 |
| cybermetaformer-gru_host_topo_k101 | gru_full_k101 | 5 / 5 | ood_silent_recall | 0.7642 | 0.7361 | +0.0280 | [+0.0235, +0.0326] | 0.0001 | 0.0007 |
| cybermetaformer-gru_host_topo_k101 | gru_full_k101 | 5 / 5 | test_never_recall | 0.8851 | 0.8856 | -0.0005 | [-0.0087, +0.0074] | 0.9023 | 1.0000 |
| cybermetaformer-gru_host_topo_k101 | gru_full_k101 | 5 / 5 | ood_never_recall | 0.8257 | 0.7912 | +0.0345 | [+0.0256, +0.0435] | 0.0001 | 0.0007 |
| cybermetaformer-gru_host_attn_k101 | gru_full_k101 | 5 / 5 | test_host_macro_f1 | 0.9801 | 0.9768 | +0.0033 | [+0.0028, +0.0038] | 0.0001 | 0.0007 |
| cybermetaformer-gru_host_attn_k101 | gru_full_k101 | 5 / 5 | ood_host_macro_f1 | 0.9413 | 0.9264 | +0.0149 | [+0.0141, +0.0158] | 0.0001 | 0.0007 |
| cybermetaformer-gru_host_attn_k101 | gru_full_k101 | 5 / 5 | ood_rel_drop | 0.0395 | 0.0516 | -0.0120 | [-0.0130, -0.0111] | 0.0001 | 0.0007 |
| cybermetaformer-gru_host_attn_k101 | gru_full_k101 | 5 / 5 | test_silent_recall | 0.9041 | 0.9015 | +0.0026 | [-0.0004, +0.0055] | 0.0892 | 0.7136 |
| cybermetaformer-gru_host_attn_k101 | gru_full_k101 | 5 / 5 | ood_silent_recall | 0.7630 | 0.7361 | +0.0269 | [+0.0223, +0.0314] | 0.0001 | 0.0007 |
| cybermetaformer-gru_host_attn_k101 | gru_full_k101 | 5 / 5 | test_never_recall | 0.8762 | 0.8856 | -0.0095 | [-0.0186, -0.0005] | 0.0752 | 0.4510 |
| cybermetaformer-gru_host_attn_k101 | gru_full_k101 | 5 / 5 | ood_never_recall | 0.8142 | 0.7912 | +0.0230 | [+0.0140, +0.0322] | 0.0001 | 0.0007 |

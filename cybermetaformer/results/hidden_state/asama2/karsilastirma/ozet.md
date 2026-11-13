# M2 pairwise tests (paired episode bootstrap)

10000 samples, bootstrap seed 0. Each model enters with the mean over its seeds. a − b; 95% CI. p Holm: pair = over the criteria of the pair; plan = over all rows. Smallest p 2 / (n + 1). For the relative drop smaller is better. `python -m cybermetaformer.hidden_state.compare --plan m2`.

## k=8: best M1 at the same k

| a | b | criterion | a | b | a − b | 95% CI | p Holm (pair) | p Holm (plan) |
|---|---|---|---:|---:|---:|---|---:|---:|
| transformer_k8 | xgb_k8 | test_host_macro_f1 | 0.9367 | 0.9412 | -0.0045 | [-0.0056, -0.0035] | 0.0004 | 0.0100 |
| transformer_k8 | xgb_k8 | ood_host_macro_f1 | 0.8261 | 0.8395 | -0.0134 | [-0.0148, -0.0120] | 0.0004 | 0.0100 |
| patchtst_k8 | xgb_k8 | test_host_macro_f1 | 0.9321 | 0.9412 | -0.0091 | [-0.0102, -0.0081] | 0.0004 | 0.0100 |
| patchtst_k8 | xgb_k8 | ood_host_macro_f1 | 0.8239 | 0.8395 | -0.0156 | [-0.0173, -0.0139] | 0.0004 | 0.0100 |
| itransformer_k8 | xgb_k8 | test_host_macro_f1 | 0.9418 | 0.9412 | +0.0006 | [-0.0003, +0.0014] | 0.3736 | 0.6631 |
| itransformer_k8 | xgb_k8 | ood_host_macro_f1 | 0.8386 | 0.8395 | -0.0010 | [-0.0026, +0.0005] | 0.3736 | 0.6631 |
| tsmixer_k8 | xgb_k8 | test_host_macro_f1 | 0.9374 | 0.9412 | -0.0038 | [-0.0048, -0.0028] | 0.0004 | 0.0100 |
| tsmixer_k8 | xgb_k8 | ood_host_macro_f1 | 0.8277 | 0.8395 | -0.0118 | [-0.0134, -0.0103] | 0.0004 | 0.0100 |
| timesnet_k8 | xgb_k8 | test_host_macro_f1 | 0.9360 | 0.9412 | -0.0052 | [-0.0061, -0.0043] | 0.0004 | 0.0100 |
| timesnet_k8 | xgb_k8 | ood_host_macro_f1 | 0.8161 | 0.8395 | -0.0234 | [-0.0252, -0.0217] | 0.0004 | 0.0100 |
| timemixer_k8 | xgb_k8 | test_host_macro_f1 | 0.9368 | 0.9412 | -0.0044 | [-0.0054, -0.0034] | 0.0004 | 0.0100 |
| timemixer_k8 | xgb_k8 | ood_host_macro_f1 | 0.8226 | 0.8395 | -0.0170 | [-0.0186, -0.0154] | 0.0004 | 0.0100 |

## k=32: best M1 at the same k

| a | b | criterion | a | b | a − b | 95% CI | p Holm (pair) | p Holm (plan) |
|---|---|---|---:|---:|---:|---|---:|---:|
| transformer_k32 | gru_k32 | test_host_macro_f1 | 0.9628 | 0.9658 | -0.0029 | [-0.0036, -0.0023] | 0.0004 | 0.0100 |
| transformer_k32 | gru_k32 | ood_host_macro_f1 | 0.8648 | 0.8871 | -0.0223 | [-0.0237, -0.0208] | 0.0004 | 0.0100 |
| patchtst_k32 | gru_k32 | test_host_macro_f1 | 0.9545 | 0.9658 | -0.0113 | [-0.0123, -0.0103] | 0.0004 | 0.0100 |
| patchtst_k32 | gru_k32 | ood_host_macro_f1 | 0.8505 | 0.8871 | -0.0366 | [-0.0384, -0.0348] | 0.0004 | 0.0100 |
| itransformer_k32 | gru_k32 | test_host_macro_f1 | 0.9663 | 0.9658 | +0.0005 | [-0.0002, +0.0012] | 0.1658 | 0.6631 |
| itransformer_k32 | gru_k32 | ood_host_macro_f1 | 0.8780 | 0.8871 | -0.0091 | [-0.0107, -0.0075] | 0.0004 | 0.0100 |
| tsmixer_k32 | gru_k32 | test_host_macro_f1 | 0.9612 | 0.9658 | -0.0046 | [-0.0052, -0.0039] | 0.0004 | 0.0100 |
| tsmixer_k32 | gru_k32 | ood_host_macro_f1 | 0.8626 | 0.8871 | -0.0245 | [-0.0259, -0.0230] | 0.0004 | 0.0100 |
| timesnet_k32 | gru_k32 | test_host_macro_f1 | 0.9236 | 0.9658 | -0.0421 | [-0.0443, -0.0401] | 0.0004 | 0.0100 |
| timesnet_k32 | gru_k32 | ood_host_macro_f1 | 0.8220 | 0.8871 | -0.0651 | [-0.0667, -0.0635] | 0.0004 | 0.0100 |
| timemixer_k32 | gru_k32 | test_host_macro_f1 | 0.9608 | 0.9658 | -0.0050 | [-0.0057, -0.0043] | 0.0004 | 0.0100 |
| timemixer_k32 | gru_k32 | ood_host_macro_f1 | 0.8657 | 0.8871 | -0.0214 | [-0.0229, -0.0197] | 0.0004 | 0.0100 |

## k=64: best windowed M1

| a | b | criterion | a | b | a − b | 95% CI | p Holm (pair) | p Holm (plan) |
|---|---|---|---:|---:|---:|---|---:|---:|
| transformer_k64 | gru_k32 | test_host_macro_f1 | 0.9714 | 0.9658 | +0.0057 | [+0.0048, +0.0066] | 0.0004 | 0.0100 |
| transformer_k64 | gru_k32 | ood_host_macro_f1 | 0.8918 | 0.8871 | +0.0047 | [+0.0031, +0.0063] | 0.0004 | 0.0100 |
| itransformer_k64 | gru_k32 | test_host_macro_f1 | 0.9704 | 0.9658 | +0.0047 | [+0.0036, +0.0057] | 0.0004 | 0.0100 |
| itransformer_k64 | gru_k32 | ood_host_macro_f1 | 0.8862 | 0.8871 | -0.0009 | [-0.0029, +0.0010] | 0.3792 | 0.6631 |
| tsmixer_k64 | gru_k32 | test_host_macro_f1 | 0.9667 | 0.9658 | +0.0009 | [+0.0000, +0.0019] | 0.0452 | 0.2260 |
| tsmixer_k64 | gru_k32 | ood_host_macro_f1 | 0.8772 | 0.8871 | -0.0099 | [-0.0116, -0.0082] | 0.0004 | 0.0100 |
| timemixer_k64 | gru_k32 | test_host_macro_f1 | 0.9692 | 0.9658 | +0.0035 | [+0.0027, +0.0043] | 0.0004 | 0.0100 |
| timemixer_k64 | gru_k32 | ood_host_macro_f1 | 0.8930 | 0.8871 | +0.0059 | [+0.0044, +0.0075] | 0.0004 | 0.0100 |

## RQ3: full-history GRU

| a | b | criterion | a | b | a − b | 95% CI | p Holm (pair) | p Holm (plan) |
|---|---|---|---:|---:|---:|---|---:|---:|
| transformer_k64 | gru_full_k101 | ood_host_macro_f1 | 0.8918 | 0.9264 | -0.0346 | [-0.0363, -0.0328] | 0.0006 | 0.0100 |
| transformer_k64 | gru_full_k101 | ood_rel_drop | 0.0820 | 0.0516 | +0.0304 | [+0.0284, +0.0322] | 0.0006 | 0.0100 |
| transformer_k64 | gru_full_k101 | ood_silent_recall | 0.6367 | 0.7361 | -0.0994 | [-0.1053, -0.0934] | 0.0006 | 0.0100 |
| patchtst_k32 | gru_full_k101 | ood_host_macro_f1 | 0.8505 | 0.9264 | -0.0759 | [-0.0782, -0.0736] | 0.0006 | 0.0100 |
| patchtst_k32 | gru_full_k101 | ood_rel_drop | 0.1090 | 0.0516 | +0.0574 | [+0.0547, +0.0601] | 0.0006 | 0.0100 |
| patchtst_k32 | gru_full_k101 | ood_silent_recall | 0.4877 | 0.7361 | -0.2485 | [-0.2575, -0.2394] | 0.0006 | 0.0100 |
| itransformer_k64 | gru_full_k101 | ood_host_macro_f1 | 0.8862 | 0.9264 | -0.0402 | [-0.0424, -0.0380] | 0.0006 | 0.0100 |
| itransformer_k64 | gru_full_k101 | ood_rel_drop | 0.0868 | 0.0516 | +0.0353 | [+0.0327, +0.0376] | 0.0006 | 0.0100 |
| itransformer_k64 | gru_full_k101 | ood_silent_recall | 0.6358 | 0.7361 | -0.1003 | [-0.1069, -0.0936] | 0.0006 | 0.0100 |
| tsmixer_k64 | gru_full_k101 | ood_host_macro_f1 | 0.8772 | 0.9264 | -0.0492 | [-0.0510, -0.0473] | 0.0006 | 0.0100 |
| tsmixer_k64 | gru_full_k101 | ood_rel_drop | 0.0926 | 0.0516 | +0.0410 | [+0.0389, +0.0431] | 0.0006 | 0.0100 |
| tsmixer_k64 | gru_full_k101 | ood_silent_recall | 0.5908 | 0.7361 | -0.1453 | [-0.1522, -0.1385] | 0.0006 | 0.0100 |
| timesnet_k8 | gru_full_k101 | ood_host_macro_f1 | 0.8161 | 0.9264 | -0.1103 | [-0.1126, -0.1079] | 0.0006 | 0.0100 |
| timesnet_k8 | gru_full_k101 | ood_rel_drop | 0.1281 | 0.0516 | +0.0765 | [+0.0733, +0.0794] | 0.0006 | 0.0100 |
| timesnet_k8 | gru_full_k101 | ood_silent_recall | 0.3637 | 0.7361 | -0.3725 | [-0.3809, -0.3640] | 0.0006 | 0.0100 |
| timemixer_k64 | gru_full_k101 | ood_host_macro_f1 | 0.8930 | 0.9264 | -0.0334 | [-0.0353, -0.0314] | 0.0006 | 0.0100 |
| timemixer_k64 | gru_full_k101 | ood_rel_drop | 0.0786 | 0.0516 | +0.0270 | [+0.0249, +0.0291] | 0.0006 | 0.0100 |
| timemixer_k64 | gru_full_k101 | ood_silent_recall | 0.6714 | 0.7361 | -0.0648 | [-0.0705, -0.0590] | 0.0006 | 0.0100 |

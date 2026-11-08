# M3 ikili sınamaları (eşli bölüm bootstrap'ı)

10000 örnek, bootstrap seed 0. Her model kendi seed'lerinin ortalamasıyla girer. a − b; GA %95. p Holm: çift içi ölçütler üzerinden; plan: bütün satırlar üzerinden. En küçük p 2 / (n + 1). Göreli düşüşte küçük daha iyi. `python -m cybermetaformer.hidden_state.compare --plan m3`.

## RQ3: ana model − çıtalar

| a | b | ölçüt | a | b | a − b | %95 GA | p Holm (çift) | p Holm (plan) |
|---|---|---|---:|---:|---:|---|---:|---:|
| cybermetaformer_k101 | gru_full_k101 | test_host_macro_f1 | 0.9801 | 0.9768 | +0.0033 | [+0.0028, +0.0038] | 0.0008 | 0.0232 |
| cybermetaformer_k101 | gru_full_k101 | ood_host_macro_f1 | 0.9417 | 0.9264 | +0.0153 | [+0.0145, +0.0162] | 0.0008 | 0.0232 |
| cybermetaformer_k101 | gru_full_k101 | ood_rel_drop | 0.0392 | 0.0516 | -0.0124 | [-0.0134, -0.0114] | 0.0008 | 0.0232 |
| cybermetaformer_k101 | gru_full_k101 | ood_silent_recall | 0.7761 | 0.7361 | +0.0400 | [+0.0352, +0.0449] | 0.0008 | 0.0232 |
| cybermetaformer_k101 | gru_k32 | test_host_macro_f1 | 0.9801 | 0.9658 | +0.0143 | [+0.0133, +0.0154] | 0.0006 | 0.0232 |
| cybermetaformer_k101 | gru_k32 | ood_host_macro_f1 | 0.9417 | 0.8871 | +0.0546 | [+0.0530, +0.0563] | 0.0006 | 0.0232 |
| cybermetaformer_k101 | gru_k32 | ood_rel_drop | 0.0392 | 0.0815 | -0.0423 | [-0.0443, -0.0403] | 0.0006 | 0.0232 |
| cybermetaformer_k101 | timemixer_k64 | test_host_macro_f1 | 0.9801 | 0.9692 | +0.0109 | [+0.0100, +0.0118] | 0.0008 | 0.0232 |
| cybermetaformer_k101 | timemixer_k64 | ood_host_macro_f1 | 0.9417 | 0.8930 | +0.0487 | [+0.0466, +0.0506] | 0.0008 | 0.0232 |
| cybermetaformer_k101 | timemixer_k64 | ood_rel_drop | 0.0392 | 0.0786 | -0.0395 | [-0.0416, -0.0372] | 0.0008 | 0.0232 |
| cybermetaformer_k101 | timemixer_k64 | ood_silent_recall | 0.7761 | 0.6714 | +0.1048 | [+0.0988, +0.1109] | 0.0008 | 0.0232 |
| cybermetaformer_k101 | transformer_k64 | test_host_macro_f1 | 0.9801 | 0.9714 | +0.0086 | [+0.0079, +0.0094] | 0.0008 | 0.0232 |
| cybermetaformer_k101 | transformer_k64 | ood_host_macro_f1 | 0.9417 | 0.8918 | +0.0499 | [+0.0480, +0.0517] | 0.0008 | 0.0232 |
| cybermetaformer_k101 | transformer_k64 | ood_rel_drop | 0.0392 | 0.0820 | -0.0428 | [-0.0448, -0.0408] | 0.0008 | 0.0232 |
| cybermetaformer_k101 | transformer_k64 | ood_silent_recall | 0.7761 | 0.6367 | +0.1394 | [+0.1330, +0.1460] | 0.0008 | 0.0232 |
| cybermetaformer_k101 | patchtst_k32 | test_host_macro_f1 | 0.9801 | 0.9545 | +0.0256 | [+0.0242, +0.0271] | 0.0008 | 0.0232 |
| cybermetaformer_k101 | patchtst_k32 | ood_host_macro_f1 | 0.9417 | 0.8505 | +0.0912 | [+0.0889, +0.0935] | 0.0008 | 0.0232 |
| cybermetaformer_k101 | patchtst_k32 | ood_rel_drop | 0.0392 | 0.1090 | -0.0698 | [-0.0726, -0.0670] | 0.0008 | 0.0232 |
| cybermetaformer_k101 | patchtst_k32 | ood_silent_recall | 0.7761 | 0.4877 | +0.2885 | [+0.2793, +0.2975] | 0.0008 | 0.0232 |

## Ablasyon çekirdeği (M3 Pro, ana modele karşı)

| a | b | ölçüt | a | b | a − b | %95 GA | p Holm (çift) | p Holm (plan) |
|---|---|---|---:|---:|---:|---|---:|---:|
| cybermetaformer-duz78_k101 | cybermetaformer_k101 | test_host_macro_f1 | 0.9720 | 0.9801 | -0.0081 | [-0.0091, -0.0071] | 0.0006 | 0.0232 |
| cybermetaformer-duz78_k101 | cybermetaformer_k101 | ood_host_macro_f1 | 0.8984 | 0.9417 | -0.0433 | [-0.0451, -0.0414] | 0.0006 | 0.0232 |
| cybermetaformer-duz78_k101 | cybermetaformer_k101 | ood_silent_recall | 0.7255 | 0.7761 | -0.0506 | [-0.0566, -0.0448] | 0.0006 | 0.0232 |
| cybermetaformer-host_yok_k101 | cybermetaformer_k101 | test_host_macro_f1 | 0.9598 | 0.9801 | -0.0203 | [-0.0215, -0.0192] | 0.0006 | 0.0232 |
| cybermetaformer-host_yok_k101 | cybermetaformer_k101 | ood_host_macro_f1 | 0.8993 | 0.9417 | -0.0424 | [-0.0441, -0.0408] | 0.0006 | 0.0232 |
| cybermetaformer-host_yok_k101 | cybermetaformer_k101 | ood_silent_recall | 0.5908 | 0.7761 | -0.1854 | [-0.1946, -0.1764] | 0.0006 | 0.0232 |
| cybermetaformer-host_tam_k101 | cybermetaformer_k101 | test_host_macro_f1 | 0.9800 | 0.9801 | -0.0001 | [-0.0003, +0.0001] | 0.2708 | 1.0000 |
| cybermetaformer-host_tam_k101 | cybermetaformer_k101 | ood_host_macro_f1 | 0.9411 | 0.9417 | -0.0006 | [-0.0010, -0.0002] | 0.0156 | 0.1248 |
| cybermetaformer-host_tam_k101 | cybermetaformer_k101 | ood_silent_recall | 0.7738 | 0.7761 | -0.0023 | [-0.0046, -0.0000] | 0.0924 | 0.7853 |
| cybermetaformer-host_gcn_k101 | cybermetaformer_k101 | test_host_macro_f1 | 0.9770 | 0.9801 | -0.0031 | [-0.0036, -0.0026] | 0.0006 | 0.0232 |
| cybermetaformer-host_gcn_k101 | cybermetaformer_k101 | ood_host_macro_f1 | 0.9283 | 0.9417 | -0.0134 | [-0.0143, -0.0126] | 0.0006 | 0.0232 |
| cybermetaformer-host_gcn_k101 | cybermetaformer_k101 | ood_silent_recall | 0.7303 | 0.7761 | -0.0458 | [-0.0512, -0.0404] | 0.0006 | 0.0232 |
| cybermetaformer-konv_dikkat_k101 | cybermetaformer_k101 | test_host_macro_f1 | 0.9801 | 0.9801 | +0.0000 | [-0.0002, +0.0002] | 0.9005 | 1.0000 |
| cybermetaformer-konv_dikkat_k101 | cybermetaformer_k101 | ood_host_macro_f1 | 0.9410 | 0.9417 | -0.0007 | [-0.0011, -0.0003] | 0.0036 | 0.0486 |
| cybermetaformer-konv_dikkat_k101 | cybermetaformer_k101 | ood_silent_recall | 0.7861 | 0.7761 | +0.0099 | [+0.0072, +0.0126] | 0.0006 | 0.0232 |
| cybermetaformer-hafiza_yok_k101 | cybermetaformer_k101 | test_host_macro_f1 | 0.9798 | 0.9801 | -0.0003 | [-0.0006, -0.0001] | 0.0068 | 0.0850 |
| cybermetaformer-hafiza_yok_k101 | cybermetaformer_k101 | ood_host_macro_f1 | 0.9384 | 0.9417 | -0.0033 | [-0.0038, -0.0028] | 0.0006 | 0.0232 |
| cybermetaformer-hafiza_yok_k101 | cybermetaformer_k101 | ood_silent_recall | 0.7721 | 0.7761 | -0.0041 | [-0.0071, -0.0010] | 0.0084 | 0.1932 |
| cybermetaformer-yalniz_yas_k101 | cybermetaformer_k101 | test_host_macro_f1 | 0.9795 | 0.9801 | -0.0006 | [-0.0009, -0.0003] | 0.0006 | 0.0232 |
| cybermetaformer-yalniz_yas_k101 | cybermetaformer_k101 | ood_host_macro_f1 | 0.9390 | 0.9417 | -0.0027 | [-0.0032, -0.0022] | 0.0006 | 0.0232 |
| cybermetaformer-yalniz_yas_k101 | cybermetaformer_k101 | ood_silent_recall | 0.7890 | 0.7761 | +0.0129 | [+0.0100, +0.0158] | 0.0006 | 0.0232 |
| cybermetaformer-yalniz_ssd_k101 | cybermetaformer_k101 | test_host_macro_f1 | 0.9801 | 0.9801 | +0.0000 | [-0.0002, +0.0002] | 0.9867 | 1.0000 |
| cybermetaformer-yalniz_ssd_k101 | cybermetaformer_k101 | ood_host_macro_f1 | 0.9414 | 0.9417 | -0.0003 | [-0.0007, +0.0002] | 0.4168 | 1.0000 |
| cybermetaformer-yalniz_ssd_k101 | cybermetaformer_k101 | ood_silent_recall | 0.7819 | 0.7761 | +0.0057 | [+0.0034, +0.0081] | 0.0006 | 0.0232 |
| cybermetaformer-konum_yok_k101 | cybermetaformer_k101 | test_host_macro_f1 | 0.9802 | 0.9801 | +0.0001 | [-0.0002, +0.0003] | 0.4192 | 1.0000 |
| cybermetaformer-konum_yok_k101 | cybermetaformer_k101 | ood_host_macro_f1 | 0.9424 | 0.9417 | +0.0007 | [+0.0004, +0.0011] | 0.0006 | 0.0232 |
| cybermetaformer-konum_yok_k101 | cybermetaformer_k101 | ood_silent_recall | 0.7655 | 0.7761 | -0.0107 | [-0.0131, -0.0083] | 0.0006 | 0.0232 |

## Ablasyon eki (M3 Pro, ana modele karşı)

| a | b | ölçüt | a | b | a − b | %95 GA | p Holm (çift) | p Holm (plan) |
|---|---|---|---:|---:|---:|---|---:|---:|
| cybermetaformer-kol_konv_k101 | cybermetaformer_k101 | test_host_macro_f1 | 0.9656 | 0.9801 | -0.0145 | [-0.0157, -0.0133] | 0.0006 | 0.0232 |
| cybermetaformer-kol_konv_k101 | cybermetaformer_k101 | ood_host_macro_f1 | 0.8842 | 0.9417 | -0.0575 | [-0.0596, -0.0555] | 0.0006 | 0.0232 |
| cybermetaformer-kol_konv_k101 | cybermetaformer_k101 | ood_silent_recall | 0.5623 | 0.7761 | -0.2139 | [-0.2227, -0.2050] | 0.0006 | 0.0232 |
| cybermetaformer-kol_ssd_k101 | cybermetaformer_k101 | test_host_macro_f1 | 0.9799 | 0.9801 | -0.0002 | [-0.0005, +0.0000] | 0.1008 | 0.8063 |
| cybermetaformer-kol_ssd_k101 | cybermetaformer_k101 | ood_host_macro_f1 | 0.9414 | 0.9417 | -0.0003 | [-0.0007, +0.0001] | 0.1680 | 1.0000 |
| cybermetaformer-kol_ssd_k101 | cybermetaformer_k101 | ood_silent_recall | 0.7800 | 0.7761 | +0.0039 | [+0.0014, +0.0063] | 0.0054 | 0.0486 |
| cybermetaformer-konv_gru_k101 | cybermetaformer_k101 | test_host_macro_f1 | 0.9806 | 0.9801 | +0.0005 | [+0.0002, +0.0007] | 0.0008 | 0.0232 |
| cybermetaformer-konv_gru_k101 | cybermetaformer_k101 | ood_host_macro_f1 | 0.9428 | 0.9417 | +0.0011 | [+0.0007, +0.0016] | 0.0006 | 0.0232 |
| cybermetaformer-konv_gru_k101 | cybermetaformer_k101 | ood_silent_recall | 0.7774 | 0.7761 | +0.0012 | [-0.0016, +0.0040] | 0.3888 | 1.0000 |
| cybermetaformer-havuz_mlp_k101 | cybermetaformer_k101 | test_host_macro_f1 | 0.9790 | 0.9801 | -0.0011 | [-0.0014, -0.0008] | 0.0006 | 0.0232 |
| cybermetaformer-havuz_mlp_k101 | cybermetaformer_k101 | ood_host_macro_f1 | 0.9354 | 0.9417 | -0.0063 | [-0.0069, -0.0057] | 0.0006 | 0.0232 |
| cybermetaformer-havuz_mlp_k101 | cybermetaformer_k101 | ood_silent_recall | 0.7514 | 0.7761 | -0.0247 | [-0.0283, -0.0213] | 0.0006 | 0.0232 |

## RQ4 (H4): hiç uyarı görülmemiş dilim

| a | b | ölçüt | a | b | a − b | %95 GA | p Holm (çift) | p Holm (plan) |
|---|---|---|---:|---:|---:|---|---:|---:|
| cybermetaformer_k101 | cybermetaformer-host_yok_k101 | test_never_recall | 0.8871 | 0.6795 | +0.2076 | [+0.1870, +0.2285] | 0.0004 | 0.0232 |
| cybermetaformer_k101 | cybermetaformer-host_yok_k101 | ood_never_recall | 0.8298 | 0.4537 | +0.3761 | [+0.3575, +0.3951] | 0.0004 | 0.0232 |
| cybermetaformer_k101 | cybermetaformer-duz78_k101 | test_never_recall | 0.8871 | 0.8899 | -0.0029 | [-0.0135, +0.0079] | 0.5953 | 1.0000 |
| cybermetaformer_k101 | cybermetaformer-duz78_k101 | ood_never_recall | 0.8298 | 0.7729 | +0.0569 | [+0.0460, +0.0682] | 0.0004 | 0.0232 |
| cybermetaformer-host_tam_k101 | cybermetaformer-host_yok_k101 | test_never_recall | 0.8949 | 0.6795 | +0.2154 | [+0.1955, +0.2362] | 0.0004 | 0.0232 |
| cybermetaformer-host_tam_k101 | cybermetaformer-host_yok_k101 | ood_never_recall | 0.8313 | 0.4537 | +0.3776 | [+0.3590, +0.3967] | 0.0004 | 0.0232 |
| cybermetaformer-host_gcn_k101 | cybermetaformer-host_yok_k101 | test_never_recall | 0.8856 | 0.6795 | +0.2061 | [+0.1861, +0.2271] | 0.0004 | 0.0232 |
| cybermetaformer-host_gcn_k101 | cybermetaformer-host_yok_k101 | ood_never_recall | 0.8091 | 0.4537 | +0.3554 | [+0.3367, +0.3746] | 0.0004 | 0.0232 |

## Alan genelleme ve artırma (M4 Pro, ana_m4'e karşı)

| a | b | ölçüt | a | b | a − b | %95 GA | p Holm (çift) | p Holm (plan) |
|---|---|---|---:|---:|---:|---|---:|---:|
| cybermetaformer-dg_groupdro_k101 | cybermetaformer-ana_m4_k101 | test_host_macro_f1 | 0.9756 | 0.9799 | -0.0043 | [-0.0047, -0.0039] | 0.0008 | 0.0232 |
| cybermetaformer-dg_groupdro_k101 | cybermetaformer-ana_m4_k101 | ood_host_macro_f1 | 0.9299 | 0.9413 | -0.0114 | [-0.0122, -0.0107] | 0.0008 | 0.0232 |
| cybermetaformer-dg_groupdro_k101 | cybermetaformer-ana_m4_k101 | ood_rel_drop | 0.0469 | 0.0394 | +0.0075 | [+0.0067, +0.0083] | 0.0008 | 0.0232 |
| cybermetaformer-dg_groupdro_k101 | cybermetaformer-ana_m4_k101 | ood_silent_recall | 0.7360 | 0.7884 | -0.0524 | [-0.0566, -0.0485] | 0.0008 | 0.0232 |
| cybermetaformer-dg_vrex_k101 | cybermetaformer-ana_m4_k101 | test_host_macro_f1 | 0.9800 | 0.9799 | +0.0000 | [-0.0003, +0.0002] | 0.9455 | 1.0000 |
| cybermetaformer-dg_vrex_k101 | cybermetaformer-ana_m4_k101 | ood_host_macro_f1 | 0.9408 | 0.9413 | -0.0005 | [-0.0009, -0.0001] | 0.0336 | 0.2464 |
| cybermetaformer-dg_vrex_k101 | cybermetaformer-ana_m4_k101 | ood_rel_drop | 0.0399 | 0.0394 | +0.0005 | [+0.0000, +0.0010] | 0.0664 | 0.6307 |
| cybermetaformer-dg_vrex_k101 | cybermetaformer-ana_m4_k101 | ood_silent_recall | 0.7643 | 0.7884 | -0.0241 | [-0.0265, -0.0218] | 0.0008 | 0.0232 |
| cybermetaformer-dg_coral_k101 | cybermetaformer-ana_m4_k101 | test_host_macro_f1 | 0.9800 | 0.9799 | +0.0000 | [-0.0003, +0.0003] | 0.8217 | 1.0000 |
| cybermetaformer-dg_coral_k101 | cybermetaformer-ana_m4_k101 | ood_host_macro_f1 | 0.9416 | 0.9413 | +0.0003 | [-0.0001, +0.0007] | 0.3408 | 1.0000 |
| cybermetaformer-dg_coral_k101 | cybermetaformer-ana_m4_k101 | ood_rel_drop | 0.0391 | 0.0394 | -0.0003 | [-0.0007, +0.0002] | 0.4520 | 1.0000 |
| cybermetaformer-dg_coral_k101 | cybermetaformer-ana_m4_k101 | ood_silent_recall | 0.7821 | 0.7884 | -0.0063 | [-0.0085, -0.0040] | 0.0008 | 0.0232 |
| cybermetaformer-dg_dann_k101 | cybermetaformer-ana_m4_k101 | test_host_macro_f1 | 0.9791 | 0.9799 | -0.0009 | [-0.0011, -0.0006] | 0.0008 | 0.0232 |
| cybermetaformer-dg_dann_k101 | cybermetaformer-ana_m4_k101 | ood_host_macro_f1 | 0.9384 | 0.9413 | -0.0029 | [-0.0035, -0.0024] | 0.0008 | 0.0232 |
| cybermetaformer-dg_dann_k101 | cybermetaformer-ana_m4_k101 | ood_rel_drop | 0.0415 | 0.0394 | +0.0021 | [+0.0015, +0.0028] | 0.0008 | 0.0232 |
| cybermetaformer-dg_dann_k101 | cybermetaformer-ana_m4_k101 | ood_silent_recall | 0.7777 | 0.7884 | -0.0107 | [-0.0135, -0.0079] | 0.0008 | 0.0232 |
| cybermetaformer-art_uyari_k101 | cybermetaformer-ana_m4_k101 | test_host_macro_f1 | 0.9789 | 0.9799 | -0.0010 | [-0.0014, -0.0007] | 0.0008 | 0.0232 |
| cybermetaformer-art_uyari_k101 | cybermetaformer-ana_m4_k101 | ood_host_macro_f1 | 0.9396 | 0.9413 | -0.0017 | [-0.0024, -0.0010] | 0.0008 | 0.0232 |
| cybermetaformer-art_uyari_k101 | cybermetaformer-ana_m4_k101 | ood_rel_drop | 0.0401 | 0.0394 | +0.0007 | [-0.0001, +0.0015] | 0.0662 | 0.9929 |
| cybermetaformer-art_uyari_k101 | cybermetaformer-ana_m4_k101 | ood_silent_recall | 0.7461 | 0.7884 | -0.0422 | [-0.0468, -0.0378] | 0.0008 | 0.0232 |
| cybermetaformer-art_zaman_k101 | cybermetaformer-ana_m4_k101 | test_host_macro_f1 | 0.9801 | 0.9799 | +0.0002 | [-0.0000, +0.0004] | 0.1080 | 1.0000 |
| cybermetaformer-art_zaman_k101 | cybermetaformer-ana_m4_k101 | ood_host_macro_f1 | 0.9423 | 0.9413 | +0.0010 | [+0.0006, +0.0014] | 0.0008 | 0.0232 |
| cybermetaformer-art_zaman_k101 | cybermetaformer-ana_m4_k101 | ood_rel_drop | 0.0386 | 0.0394 | -0.0008 | [-0.0013, -0.0003] | 0.0016 | 0.0232 |
| cybermetaformer-art_zaman_k101 | cybermetaformer-ana_m4_k101 | ood_silent_recall | 0.7677 | 0.7884 | -0.0207 | [-0.0229, -0.0184] | 0.0008 | 0.0232 |

## Aksiyon girdisi

| a | b | ölçüt | a | b | a − b | %95 GA | p Holm (çift) | p Holm (plan) |
|---|---|---|---:|---:|---:|---|---:|---:|
| cybermetaformer-aksiyon_k101 | cybermetaformer-ana_m4_k101 | test_host_macro_f1 | 0.9810 | 0.9799 | +0.0010 | [+0.0007, +0.0014] | 0.0008 | 0.0232 |
| cybermetaformer-aksiyon_k101 | cybermetaformer-ana_m4_k101 | ood_host_macro_f1 | 0.9497 | 0.9413 | +0.0084 | [+0.0075, +0.0093] | 0.0008 | 0.0232 |
| cybermetaformer-aksiyon_k101 | cybermetaformer-ana_m4_k101 | ood_silent_recall | 0.7731 | 0.7884 | -0.0153 | [-0.0188, -0.0118] | 0.0008 | 0.0232 |
| cybermetaformer-aksiyon_k101 | cybermetaformer-ana_m4_k101 | test_silent_recall | 0.9083 | 0.9060 | +0.0023 | [+0.0004, +0.0042] | 0.0202 | 0.4242 |
| gru_full-aksiyon_k101 | gru_full_k101 | test_host_macro_f1 | 0.9782 | 0.9767 | +0.0016 | [+0.0011, +0.0021] | 0.0008 | 0.0232 |
| gru_full-aksiyon_k101 | gru_full_k101 | ood_host_macro_f1 | 0.9394 | 0.9264 | +0.0130 | [+0.0120, +0.0141] | 0.0008 | 0.0232 |
| gru_full-aksiyon_k101 | gru_full_k101 | ood_silent_recall | 0.7461 | 0.7364 | +0.0097 | [+0.0051, +0.0144] | 0.0008 | 0.0232 |
| gru_full-aksiyon_k101 | gru_full_k101 | test_silent_recall | 0.9080 | 0.9028 | +0.0052 | [+0.0027, +0.0078] | 0.0008 | 0.0232 |

## Ölçek ve makineler arası tekrar

| a | b | ölçüt | a | b | a − b | %95 GA | p Holm (çift) | p Holm (plan) |
|---|---|---|---:|---:|---:|---|---:|---:|
| cybermetaformer-b2_k101 | cybermetaformer-ana_m4_k101 | test_host_macro_f1 | 0.9788 | 0.9799 | -0.0012 | [-0.0015, -0.0009] | 0.0006 | 0.0232 |
| cybermetaformer-b2_k101 | cybermetaformer-ana_m4_k101 | ood_host_macro_f1 | 0.9378 | 0.9413 | -0.0035 | [-0.0041, -0.0030] | 0.0006 | 0.0232 |
| cybermetaformer-b2_k101 | cybermetaformer-ana_m4_k101 | ood_silent_recall | 0.7849 | 0.7884 | -0.0035 | [-0.0065, -0.0005] | 0.0248 | 0.4960 |
| cybermetaformer-ana_m4_k101 | cybermetaformer_k101 | test_host_macro_f1 | 0.9799 | 0.9801 | -0.0001 | [-0.0003, +0.0001] | 0.1814 | 1.0000 |
| cybermetaformer-ana_m4_k101 | cybermetaformer_k101 | ood_host_macro_f1 | 0.9413 | 0.9417 | -0.0004 | [-0.0007, -0.0000] | 0.0808 | 0.7271 |
| cybermetaformer-ana_m4_k101 | cybermetaformer_k101 | ood_silent_recall | 0.7884 | 0.7761 | +0.0122 | [+0.0102, +0.0143] | 0.0006 | 0.0232 |

## (ii) saldırgan kayması: yalnız bline ile eğitim

| a | b | ölçüt | a | b | a − b | %95 GA | p Holm (çift) | p Holm (plan) |
|---|---|---|---:|---:|---:|---|---:|---:|
| cybermetaformer-saldirgan_bline_k101 | gru_full-saldirgan_bline_k101 | test_shift_host_macro_f1 | 0.8286 | 0.5753 | +0.2534 | [+0.2485, +0.2582] | 0.0006 | 0.0232 |
| cybermetaformer-saldirgan_bline_k101 | gru_full-saldirgan_bline_k101 | ood_test_shift_host_macro_f1 | 0.7237 | 0.5132 | +0.2105 | [+0.2074, +0.2134] | 0.0006 | 0.0232 |
| cybermetaformer-saldirgan_bline_k101 | gru_full-saldirgan_bline_k101 | ood_test_shift_silent_recall | 0.2066 | 0.0614 | +0.1452 | [+0.1364, +0.1539] | 0.0006 | 0.0232 |
| cybermetaformer-saldirgan_bline_k101 | timemixer-saldirgan_bline_k64 | test_shift_host_macro_f1 | 0.8286 | 0.5159 | +0.3127 | [+0.3070, +0.3184] | 0.0006 | 0.0232 |
| cybermetaformer-saldirgan_bline_k101 | timemixer-saldirgan_bline_k64 | ood_test_shift_host_macro_f1 | 0.7237 | 0.4627 | +0.2609 | [+0.2577, +0.2640] | 0.0006 | 0.0232 |
| cybermetaformer-saldirgan_bline_k101 | timemixer-saldirgan_bline_k64 | ood_test_shift_silent_recall | 0.2066 | 0.0976 | +0.1090 | [+0.0996, +0.1183] | 0.0006 | 0.0232 |

## (ii) saldırgan kayması: yalnız meander ile eğitim

| a | b | ölçüt | a | b | a − b | %95 GA | p Holm (çift) | p Holm (plan) |
|---|---|---|---:|---:|---:|---|---:|---:|
| cybermetaformer-saldirgan_meander_k101 | gru_full-saldirgan_meander_k101 | test_shift_host_macro_f1 | 0.8409 | 0.8498 | -0.0088 | [-0.0118, -0.0057] | 0.0006 | 0.0232 |
| cybermetaformer-saldirgan_meander_k101 | gru_full-saldirgan_meander_k101 | ood_test_shift_host_macro_f1 | 0.7377 | 0.7254 | +0.0123 | [+0.0084, +0.0165] | 0.0006 | 0.0232 |
| cybermetaformer-saldirgan_meander_k101 | gru_full-saldirgan_meander_k101 | ood_test_shift_silent_recall | 0.2966 | 0.3690 | -0.0725 | [-0.0859, -0.0589] | 0.0006 | 0.0232 |
| cybermetaformer-saldirgan_meander_k101 | timemixer-saldirgan_meander_k64 | test_shift_host_macro_f1 | 0.8409 | 0.8153 | +0.0256 | [+0.0230, +0.0285] | 0.0006 | 0.0232 |
| cybermetaformer-saldirgan_meander_k101 | timemixer-saldirgan_meander_k64 | ood_test_shift_host_macro_f1 | 0.7377 | 0.6820 | +0.0557 | [+0.0524, +0.0591] | 0.0006 | 0.0232 |
| cybermetaformer-saldirgan_meander_k101 | timemixer-saldirgan_meander_k64 | ood_test_shift_silent_recall | 0.2966 | 0.2283 | +0.0682 | [+0.0564, +0.0799] | 0.0006 | 0.0232 |

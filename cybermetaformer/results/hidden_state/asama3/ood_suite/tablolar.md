# M3 OOD package (i), (iii), (iv)

Saved weights; temperature only on the v1 val. **Member**: seed mean (own temperature); **ensemble**: 3 seeds, topluluk_T. ε=0.1 the training policy of v1 test, ε=0.3 `ood_test`, 0.5 / 0.7 / analyse / remove `minicage_v1_shift` (4000 episodes per policy), bug `minicage_v1_bugged`. `python -m cybermetaformer.hidden_state.ood_suite analyze`.

| Set | Episodes |
|---|---:|
| test | 2000 |
| eps0.1 | 400 |
| ood | 4000 |
| eps0.5 | 4000 |
| eps0.7 | 4000 |
| analyse | 4000 |
| remove | 4000 |
| bug_test | 2000 |
| bug_ood | 4000 |

## Host macro-F1

| Model | Configuration | test | eps0.1 | ood | eps0.5 | eps0.7 | analyse | remove | bug_test | bug_ood |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CyberMetaFormer | uye | 0.9801 | 0.9597 | 0.9417 | 0.9419 | 0.9610 | 0.8809 | 0.9664 | 0.9792 | 0.9407 |
| CyberMetaFormer | topluluk_T | 0.9804 | 0.9604 | 0.9433 | 0.9441 | 0.9627 | 0.8814 | 0.9651 | 0.9798 | 0.9422 |
| Full GRU | uye | 0.9767 | 0.9548 | 0.9264 | 0.9239 | 0.9460 | 0.8621 | 0.9607 | 0.9761 | 0.9281 |
| Full GRU | topluluk_T | 0.9779 | 0.9574 | 0.9322 | 0.9301 | 0.9511 | 0.8690 | 0.9648 | 0.9772 | 0.9344 |
| CMF (M4 copy) | uye | 0.9799 | 0.9594 | 0.9413 | 0.9407 | 0.9598 | 0.8770 | 0.9677 | 0.9792 | 0.9395 |
| CMF (M4 copy) | topluluk_T | 0.9804 | 0.9604 | 0.9427 | 0.9425 | 0.9614 | 0.8775 | 0.9673 | 0.9797 | 0.9412 |
| CMF + action | uye | 0.9810 | 0.9643 | 0.9497 | 0.9494 | 0.9653 | 0.8948 | 0.9812 | 0.9804 | 0.9504 |
| CMF + action | topluluk_T | 0.9813 | 0.9650 | 0.9519 | 0.9520 | 0.9675 | 0.9021 | 0.9827 | 0.9808 | 0.9532 |
| Full GRU + action | uye | 0.9782 | 0.9601 | 0.9394 | 0.9385 | 0.9548 | 0.8812 | 0.9770 | 0.9773 | 0.9389 |
| Full GRU + action | topluluk_T | 0.9791 | 0.9618 | 0.9439 | 0.9432 | 0.9585 | 0.8869 | 0.9804 | 0.9785 | 0.9445 |

## Silent recall

| Model | Configuration | test | eps0.1 | ood | eps0.5 | eps0.7 | analyse | remove | bug_test | bug_ood |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CyberMetaFormer | uye | 0.9059 | 0.8656 | 0.7761 | 0.7238 | 0.7800 | 0.6373 | 0.9532 | 0.9089 | 0.7845 |
| CyberMetaFormer | topluluk_T | 0.9072 | 0.8676 | 0.7808 | 0.7265 | 0.7798 | 0.6420 | 0.9509 | 0.9100 | 0.7895 |
| Full GRU | uye | 0.9028 | 0.8577 | 0.7364 | 0.6824 | 0.7365 | 0.7056 | 0.9428 | 0.9023 | 0.7502 |
| Full GRU | topluluk_T | 0.9072 | 0.8666 | 0.7460 | 0.6824 | 0.7310 | 0.7077 | 0.9504 | 0.9072 | 0.7581 |
| CMF (M4 copy) | uye | 0.9060 | 0.8633 | 0.7883 | 0.7449 | 0.7969 | 0.6383 | 0.9559 | 0.9073 | 0.7935 |
| CMF (M4 copy) | topluluk_T | 0.9066 | 0.8646 | 0.7923 | 0.7474 | 0.7982 | 0.6398 | 0.9544 | 0.9089 | 0.7965 |
| CMF + action | uye | 0.9082 | 0.8671 | 0.7730 | 0.7224 | 0.7884 | 0.5556 | 0.9878 | 0.9095 | 0.7787 |
| CMF + action | topluluk_T | 0.9087 | 0.8689 | 0.7784 | 0.7250 | 0.7889 | 0.5512 | 0.9897 | 0.9104 | 0.7844 |
| Full GRU + action | uye | 0.9080 | 0.8638 | 0.7461 | 0.6794 | 0.7411 | 0.5936 | 0.9782 | 0.9068 | 0.7531 |
| Full GRU + action | topluluk_T | 0.9115 | 0.8666 | 0.7501 | 0.6767 | 0.7347 | 0.5841 | 0.9847 | 0.9116 | 0.7581 |

## ECE

| Model | Configuration | test | eps0.1 | ood | eps0.5 | eps0.7 | analyse | remove | bug_test | bug_ood |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CyberMetaFormer | uye | 0.0007 | 0.0020 | 0.0031 | 0.0035 | 0.0024 | 0.0065 | 0.0104 | 0.0017 | 0.0033 |
| CyberMetaFormer | topluluk_T | 0.0005 | 0.0015 | 0.0025 | 0.0022 | 0.0014 | 0.0044 | 0.0099 | 0.0013 | 0.0021 |
| Full GRU | uye | 0.0009 | 0.0036 | 0.0088 | 0.0100 | 0.0078 | 0.0093 | 0.0097 | 0.0010 | 0.0071 |
| Full GRU | topluluk_T | 0.0007 | 0.0026 | 0.0052 | 0.0048 | 0.0037 | 0.0064 | 0.0060 | 0.0008 | 0.0039 |
| CMF (M4 copy) | uye | 0.0008 | 0.0018 | 0.0036 | 0.0038 | 0.0024 | 0.0064 | 0.0082 | 0.0016 | 0.0040 |
| CMF (M4 copy) | topluluk_T | 0.0005 | 0.0015 | 0.0030 | 0.0028 | 0.0012 | 0.0064 | 0.0086 | 0.0013 | 0.0031 |
| CMF + action | uye | 0.0009 | 0.0016 | 0.0022 | 0.0030 | 0.0029 | 0.0024 | 0.0008 | 0.0018 | 0.0025 |
| CMF + action | topluluk_T | 0.0006 | 0.0011 | 0.0007 | 0.0014 | 0.0014 | 0.0013 | 0.0013 | 0.0012 | 0.0015 |
| Full GRU + action | uye | 0.0011 | 0.0034 | 0.0070 | 0.0093 | 0.0086 | 0.0050 | 0.0028 | 0.0011 | 0.0057 |
| Full GRU + action | topluluk_T | 0.0010 | 0.0030 | 0.0041 | 0.0058 | 0.0054 | 0.0026 | 0.0026 | 0.0010 | 0.0028 |

## Remove-mask recall

| Model | Configuration | test | eps0.1 | ood | eps0.5 | eps0.7 | analyse | remove | bug_test | bug_ood |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CyberMetaFormer | uye | 0.9700 | 0.9657 | 0.9429 | 0.9187 | 0.9420 |  | 0.9996 |  |  |
| CyberMetaFormer | topluluk_T | 0.9687 | 0.9667 | 0.9486 | 0.9229 | 0.9436 |  | 1.0000 |  |  |
| Full GRU | uye | 0.9630 | 0.9542 | 0.9233 | 0.8948 | 0.9244 |  | 0.9900 |  |  |
| Full GRU | topluluk_T | 0.9631 | 0.9627 | 0.9385 | 0.9072 | 0.9288 |  | 0.9995 |  |  |
| CMF (M4 copy) | uye | 0.9710 | 0.9658 | 0.9490 | 0.9283 | 0.9494 |  | 1.0000 |  |  |
| CMF (M4 copy) | topluluk_T | 0.9709 | 0.9696 | 0.9543 | 0.9319 | 0.9517 |  | 1.0000 |  |  |
| CMF + action | uye | 0.9731 | 0.9618 | 0.9477 | 0.9217 | 0.9470 |  | 0.9987 |  |  |
| CMF + action | topluluk_T | 0.9726 | 0.9578 | 0.9535 | 0.9271 | 0.9492 |  | 0.9998 |  |  |
| Full GRU + action | uye | 0.9661 | 0.9487 | 0.9234 | 0.8873 | 0.9194 |  | 0.9933 |  |  |
| Full GRU + action | topluluk_T | 0.9669 | 0.9526 | 0.9335 | 0.8917 | 0.9206 |  | 0.9998 |  |  |

## Green-mask recall

| Model | Configuration | test | eps0.1 | ood | eps0.5 | eps0.7 | analyse | remove | bug_test | bug_ood |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CyberMetaFormer | uye | 0.8923 | 0.8549 | 0.7196 | 0.5913 | 0.4820 | 0.6527 | 0.4406 |  |  |
| CyberMetaFormer | topluluk_T | 0.8935 | 0.8565 | 0.7241 | 0.5938 | 0.4813 | 0.6562 | 0.4080 |  |  |
| Full GRU | uye | 0.8882 | 0.8402 | 0.6704 | 0.5241 | 0.3833 | 0.7134 | 0.4195 |  |  |
| Full GRU | topluluk_T | 0.8931 | 0.8481 | 0.6778 | 0.5144 | 0.3598 | 0.7149 | 0.4057 |  |  |
| CMF (M4 copy) | uye | 0.8933 | 0.8522 | 0.7335 | 0.6194 | 0.5205 | 0.6533 | 0.4736 |  |  |
| CMF (M4 copy) | topluluk_T | 0.8932 | 0.8525 | 0.7383 | 0.6210 | 0.5215 | 0.6544 | 0.4527 |  |  |
| CMF + action | uye | 0.8931 | 0.8554 | 0.7130 | 0.5806 | 0.4948 | 0.5744 | 0.8615 |  |  |
| CMF + action | topluluk_T | 0.8928 | 0.8559 | 0.7177 | 0.5830 | 0.4915 | 0.5713 | 0.8729 |  |  |
| Full GRU + action | uye | 0.8946 | 0.8499 | 0.6892 | 0.5379 | 0.4107 | 0.6108 | 0.8040 |  |  |
| Full GRU + action | topluluk_T | 0.8994 | 0.8519 | 0.6914 | 0.5324 | 0.3913 | 0.6032 | 0.8132 |  |  |

## Miss (5%) recall

| Model | Configuration | test | eps0.1 | ood | eps0.5 | eps0.7 | analyse | remove | bug_test | bug_ood |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CyberMetaFormer | uye | 0.8940 | 0.8597 | 0.7214 | 0.5637 | 0.4812 | 0.6104 | 0.4583 |  |  |
| CyberMetaFormer | topluluk_T | 0.8970 | 0.8626 | 0.7258 | 0.5638 | 0.4729 | 0.6174 | 0.4258 |  |  |
| Full GRU | uye | 0.8949 | 0.8643 | 0.6798 | 0.5309 | 0.4026 | 0.6920 | 0.4395 |  |  |
| Full GRU | topluluk_T | 0.9006 | 0.8749 | 0.6881 | 0.5228 | 0.3786 | 0.6951 | 0.4275 |  |  |
| CMF (M4 copy) | uye | 0.8921 | 0.8575 | 0.7363 | 0.5958 | 0.5085 | 0.6121 | 0.4794 |  |  |
| CMF (M4 copy) | topluluk_T | 0.8943 | 0.8600 | 0.7375 | 0.5975 | 0.5054 | 0.6144 | 0.4632 |  |  |
| CMF + action | uye | 0.8987 | 0.8641 | 0.7172 | 0.5692 | 0.4989 | 0.5227 | 0.8807 |  |  |
| CMF + action | topluluk_T | 0.9010 | 0.8696 | 0.7234 | 0.5669 | 0.4975 | 0.5161 | 0.8905 |  |  |
| Full GRU + action | uye | 0.8993 | 0.8669 | 0.6823 | 0.5093 | 0.4162 | 0.5636 | 0.8277 |  |  |
| Full GRU + action | topluluk_T | 0.9020 | 0.8708 | 0.6836 | 0.4969 | 0.3945 | 0.5506 | 0.8321 |  |  |

## Paired tests (member, seed mean; a − b)

Paired episode bootstrap, 4000 samples; p Holm over all rows.

| a | b | set | criterion | a − b | 95% CI | p Holm |
|---|---|---|---|---:|---|---:|
| CyberMetaFormer | Full GRU | test | host_macro_f1 | +0.0034 | [+0.0029, +0.0040] | 0.0270 |
| CyberMetaFormer | Full GRU | test | silent_recall | +0.0031 | [-0.0001, +0.0061] | 0.2909 |
| CyberMetaFormer | Full GRU | eps0.1 | host_macro_f1 | +0.0049 | [+0.0035, +0.0063] | 0.0270 |
| CyberMetaFormer | Full GRU | eps0.1 | silent_recall | +0.0078 | [+0.0001, +0.0155] | 0.2909 |
| CyberMetaFormer | Full GRU | ood | host_macro_f1 | +0.0153 | [+0.0144, +0.0163] | 0.0270 |
| CyberMetaFormer | Full GRU | ood | silent_recall | +0.0398 | [+0.0346, +0.0449] | 0.0270 |
| CyberMetaFormer | Full GRU | eps0.5 | host_macro_f1 | +0.0180 | [+0.0171, +0.0189] | 0.0270 |
| CyberMetaFormer | Full GRU | eps0.5 | silent_recall | +0.0414 | [+0.0354, +0.0474] | 0.0270 |
| CyberMetaFormer | Full GRU | eps0.7 | host_macro_f1 | +0.0150 | [+0.0143, +0.0157] | 0.0270 |
| CyberMetaFormer | Full GRU | eps0.7 | silent_recall | +0.0436 | [+0.0384, +0.0484] | 0.0270 |
| CyberMetaFormer | Full GRU | analyse | host_macro_f1 | +0.0188 | [+0.0178, +0.0199] | 0.0270 |
| CyberMetaFormer | Full GRU | analyse | silent_recall | -0.0684 | [-0.0724, -0.0642] | 0.0270 |
| CyberMetaFormer | Full GRU | remove | host_macro_f1 | +0.0058 | [+0.0053, +0.0063] | 0.0270 |
| CyberMetaFormer | Full GRU | remove | silent_recall | +0.0104 | [+0.0095, +0.0114] | 0.0270 |
| CyberMetaFormer | Full GRU | bug_test | host_macro_f1 | +0.0031 | [+0.0025, +0.0037] | 0.0270 |
| CyberMetaFormer | Full GRU | bug_test | silent_recall | +0.0067 | [+0.0029, +0.0103] | 0.0270 |
| CyberMetaFormer | Full GRU | bug_ood | host_macro_f1 | +0.0126 | [+0.0116, +0.0136] | 0.0270 |
| CyberMetaFormer | Full GRU | bug_ood | silent_recall | +0.0343 | [+0.0295, +0.0394] | 0.0270 |
| CMF + action | CMF (M4 copy) | test | host_macro_f1 | +0.0010 | [+0.0007, +0.0014] | 0.0270 |
| CMF + action | CMF (M4 copy) | test | silent_recall | +0.0023 | [+0.0003, +0.0041] | 0.1980 |
| CMF + action | CMF (M4 copy) | eps0.1 | host_macro_f1 | +0.0049 | [+0.0035, +0.0065] | 0.0270 |
| CMF + action | CMF (M4 copy) | eps0.1 | silent_recall | +0.0038 | [-0.0009, +0.0081] | 0.3759 |
| CMF + action | CMF (M4 copy) | ood | host_macro_f1 | +0.0084 | [+0.0075, +0.0092] | 0.0270 |
| CMF + action | CMF (M4 copy) | ood | silent_recall | -0.0153 | [-0.0187, -0.0119] | 0.0270 |
| CMF + action | CMF (M4 copy) | eps0.5 | host_macro_f1 | +0.0087 | [+0.0078, +0.0096] | 0.0270 |
| CMF + action | CMF (M4 copy) | eps0.5 | silent_recall | -0.0225 | [-0.0269, -0.0182] | 0.0270 |
| CMF + action | CMF (M4 copy) | eps0.7 | host_macro_f1 | +0.0055 | [+0.0048, +0.0062] | 0.0270 |
| CMF + action | CMF (M4 copy) | eps0.7 | silent_recall | -0.0085 | [-0.0124, -0.0047] | 0.0270 |
| CMF + action | CMF (M4 copy) | analyse | host_macro_f1 | +0.0178 | [+0.0166, +0.0191] | 0.0270 |
| CMF + action | CMF (M4 copy) | analyse | silent_recall | -0.0827 | [-0.0876, -0.0779] | 0.0270 |
| CMF + action | CMF (M4 copy) | remove | host_macro_f1 | +0.0135 | [+0.0127, +0.0144] | 0.0270 |
| CMF + action | CMF (M4 copy) | remove | silent_recall | +0.0319 | [+0.0303, +0.0336] | 0.0270 |
| CMF + action | CMF (M4 copy) | bug_test | host_macro_f1 | +0.0012 | [+0.0009, +0.0016] | 0.0270 |
| CMF + action | CMF (M4 copy) | bug_test | silent_recall | +0.0022 | [+0.0003, +0.0041] | 0.2000 |
| CMF + action | CMF (M4 copy) | bug_ood | host_macro_f1 | +0.0109 | [+0.0100, +0.0118] | 0.0270 |
| CMF + action | CMF (M4 copy) | bug_ood | silent_recall | -0.0148 | [-0.0183, -0.0114] | 0.0270 |
| Full GRU + action | Full GRU | test | host_macro_f1 | +0.0016 | [+0.0011, +0.0021] | 0.0270 |
| Full GRU + action | Full GRU | test | silent_recall | +0.0052 | [+0.0027, +0.0077] | 0.0270 |
| Full GRU + action | Full GRU | eps0.1 | host_macro_f1 | +0.0053 | [+0.0036, +0.0070] | 0.0270 |
| Full GRU + action | Full GRU | eps0.1 | silent_recall | +0.0060 | [-0.0010, +0.0132] | 0.3759 |
| Full GRU + action | Full GRU | ood | host_macro_f1 | +0.0130 | [+0.0120, +0.0141] | 0.0270 |
| Full GRU + action | Full GRU | ood | silent_recall | +0.0098 | [+0.0052, +0.0145] | 0.0270 |
| Full GRU + action | Full GRU | eps0.5 | host_macro_f1 | +0.0146 | [+0.0135, +0.0156] | 0.0270 |
| Full GRU + action | Full GRU | eps0.5 | silent_recall | -0.0029 | [-0.0087, +0.0029] | 0.4519 |
| Full GRU + action | Full GRU | eps0.7 | host_macro_f1 | +0.0088 | [+0.0081, +0.0095] | 0.0270 |
| Full GRU + action | Full GRU | eps0.7 | silent_recall | +0.0046 | [+0.0003, +0.0089] | 0.2904 |
| Full GRU + action | Full GRU | analyse | host_macro_f1 | +0.0191 | [+0.0179, +0.0203] | 0.0270 |
| Full GRU + action | Full GRU | analyse | silent_recall | -0.1121 | [-0.1169, -0.1071] | 0.0270 |
| Full GRU + action | Full GRU | remove | host_macro_f1 | +0.0164 | [+0.0155, +0.0173] | 0.0270 |
| Full GRU + action | Full GRU | remove | silent_recall | +0.0354 | [+0.0336, +0.0373] | 0.0270 |
| Full GRU + action | Full GRU | bug_test | host_macro_f1 | +0.0011 | [+0.0007, +0.0016] | 0.0270 |
| Full GRU + action | Full GRU | bug_test | silent_recall | +0.0045 | [+0.0013, +0.0078] | 0.0750 |
| Full GRU + action | Full GRU | bug_ood | host_macro_f1 | +0.0109 | [+0.0098, +0.0119] | 0.0270 |
| Full GRU + action | Full GRU | bug_ood | silent_recall | +0.0029 | [-0.0017, +0.0076] | 0.4519 |

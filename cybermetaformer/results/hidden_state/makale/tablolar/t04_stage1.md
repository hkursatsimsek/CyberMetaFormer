| Model | History | Test F1 | OOD F1 | Relative OOD drop | Silent recall, test / OOD |
|---|---:|---:|---:|---:|---:|
| Majority class | — | 0.285 | 0.311 | −0.091 | 0.000 / 0.000 |
| Alert-persistence rule | 1 | 0.543 | 0.412 | 0.242 | 0.000 / 0.000 |
| Original MLP baseline | 8 | 0.929 | 0.816 | 0.122 | — |
| XGBoost† | 32 | 0.963 | 0.878 | 0.088 | 0.826 / 0.508 |
| XGB-TF | full | 0.957 | 0.864 | 0.097 | 0.798 / 0.480 |
| MLP | 32 | 0.956 | 0.850 | 0.110 | 0.810 / 0.511 |
| 1D-CNN | 32 | 0.962 | 0.866 | 0.100 | 0.843 / 0.602 |
| LSTM | 32 | 0.965 | 0.881 | 0.087 | 0.848 / 0.605 |
| GRU | 32 | 0.966 | 0.887 | 0.081 | 0.858 / 0.651 |
| LSTM (full history) | full | 0.976 | 0.920 | 0.058 | 0.897 / 0.719 |
| GRU (full history) | full | 0.977 | 0.926 | 0.052 | 0.902 / 0.736 |

Note. Each model at the history length k selected by validation host macro-F1; five-seed means unless marked († one seed). The two rules are deterministic. Relative OOD drop = (F1_test − F1_OOD) / F1_test; it is negative when the OOD score is higher. The original MLP baseline is the eight-observation MLP of the reference implementation, reproduced on this dataset; its silent recall was not recorded.

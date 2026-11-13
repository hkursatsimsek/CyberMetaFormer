| Model | History | Test F1 | OOD F1 | OOD silent recall | Seeds |
|---|---:|---:|---:|---:|---:|
| Original MLP baseline | 8 | 0.929 | 0.816 | — | 1 |
| XGBoost | 32 | 0.963 | 0.878 | 0.508 | 1 |
| LSTM | full | 0.976 | 0.920 | 0.719 | 5 |
| GRU | full | 0.977 | 0.926 | 0.736 | 5 |
| Transformer | 64 | 0.971 | 0.892 | 0.637 | 3 |
| iTransformer | 64 | 0.970 | 0.886 | 0.636 | 3 |
| TimeMixer | 64 | 0.969 | 0.893 | 0.671 | 3 |
| PatchTST | 32 | 0.955 | 0.850 | 0.488 | 3 |

Note. Host macro-F1 and OOD silent recall, seed means; each model at the history length k selected by validation host macro-F1 (full: the whole episode so far). Seed counts and tuning budgets differ between the families (Section 3.6). The original MLP baseline is the eight-observation MLP of the reference implementation reproduced on this dataset; its silent recall was not recorded. All Stage 1 and Stage 2 models, including the rules, the windowed recurrent models, TSMixer and TimesNet: Tables @tbl:asama1 and @tbl:asama2.

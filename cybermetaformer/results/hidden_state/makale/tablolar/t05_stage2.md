| Architecture | History | Test F1 | OOD F1 | Silent recall, test / OOD | Params |
|---|---:|---:|---:|---:|---:|
| Transformer | 64 | 0.971 | 0.892 | 0.875 / 0.637 | 2,399,268 |
| iTransformer | 64 | 0.970 | 0.886 | 0.872 / 0.636 | 383,972 |
| TSMixer | 64 | 0.967 | 0.877 | 0.856 / 0.591 | 619,470 |
| TimeMixer | 64 | 0.969 | 0.893 | 0.876 / 0.671 | 144,516 |
| TimesNet | 8 | 0.936 | 0.816 | 0.717 / 0.364 | 301,572 |
| PatchTST | 32 | 0.955 | 0.850 | 0.791 / 0.488 | 385,444 |
| GRU (windowed) | 32 | 0.966 | 0.887 | 0.858 / 0.651 | 183,588 |
| GRU (full history) | full | 0.977 | 0.926 | 0.902 / 0.736 | 183,588 |

Note. Stage 2 models: three-seed means at the k selected by validation host macro-F1. GRU references from Stage 1 (five seeds).

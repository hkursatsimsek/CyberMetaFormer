| Model | Input | Silent recall, ε = 0.1 | Silent recall, OOD | Silent recall, periodic analyse | Analyse / ε = 0.1 | Host macro-F1, periodic analyse |
|---|---|---:|---:|---:|---:|---:|
| CyberMetaFormer (main) | Host tokens | 0.866 ± 0.015 | 0.776 ± 0.025 | 0.637 ± 0.041 | 0.74 | 0.881 ± 0.005 |
| Full attention instead of topology bias | Host tokens | 0.868 ± 0.008 | 0.774 ± 0.015 | 0.650 ± 0.017 | 0.75 | 0.881 ± 0.002 |
| GCN-like host mixer | Host tokens | 0.842 ± 0.009 | 0.730 ± 0.005 | 0.601 ± 0.011 | 0.71 | 0.879 ± 0.001 |
| No host mixer | Host tokens | 0.720 ± 0.012 | 0.591 ± 0.018 | 0.449 ± 0.008 | 0.62 | 0.806 ± 0.003 |
| No memory: window 32, no alert-age channel | Host tokens | 0.869 ± 0.002 | 0.772 ± 0.009 | 0.676 ± 0.005 | 0.78 | 0.878 ± 0.001 |
| Window 32, alert-age channel kept | Host tokens | 0.873 ± 0.012 | 0.789 ± 0.002 | 0.669 ± 0.022 | 0.77 | 0.879 ± 0.004 |
| Short convolution only (no long branch) | Host tokens | 0.712 ± 0.010 | 0.562 ± 0.032 | 0.615 ± 0.006 | 0.86 | 0.858 ± 0.010 |
| SSD branch only | Host tokens | 0.870 ± 0.010 | 0.780 ± 0.024 | 0.625 ± 0.012 | 0.72 | 0.879 ± 0.002 |
| No position embedding | Host tokens | 0.857 ± 0.002 | 0.765 ± 0.003 | 0.633 ± 0.019 | 0.74 | 0.881 ± 0.002 |
| No host tokens (single 78-d token) | Flat 78-d | 0.860 ± 0.004 | 0.726 ± 0.009 | 0.711 ± 0.009 | 0.83 | 0.863 ± 0.001 |
| Full-history GRU | Flat 78-d | 0.858 ± 0.005 | 0.736 ± 0.005 | 0.706 ± 0.002 | 0.82 | 0.862 ± 0.004 |

Note. Seed mean ± SD over seeds 0–2 (the ablation seeds), from the stored logits (argmax). ε = 0.1: test episodes of the training policy; OOD: ε = 0.3; periodic analyse: react_restore_analyse. With seeds 0–4 the periodic-analyse silent recall is 0.625 for CyberMetaFormer and 0.703 for the full-history GRU (Table @tbl:failure_analysis).

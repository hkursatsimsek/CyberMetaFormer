| Model | Params | CPU latency (ms / step) | OOD host macro-F1 |
|---|---:|---:|---:|
| CyberMetaFormer (B1) | 186,665 | 0.86 | 0.942 |
| CyberMetaFormer (B2) | 2,828,473 | 1.44 | 0.938 |
| GRU (full history) | 183,588 | 0.04 | 0.926 |
| TimeMixer (k = 64) | 144,516 | 1.43 | 0.893 |
| Transformer (k = 64) | 2,399,268 | 0.90 | 0.892 |

Note. Earlier measurement (Stage 3 efficiency run): batch size 1 on an Apple M3 Pro CPU, 6 threads, PyTorch 2.14.1; median of 50 timed calls after 10 warm-up calls (CyberMetaFormer: 91 consecutive step-mode calls). Full-history GRU and CyberMetaFormer run in step mode. OOD F1: CyberMetaFormer seeds 0–4, B2 seeds 0–2, GRU seeds 0–4, Stage 2 models seeds 0–2.

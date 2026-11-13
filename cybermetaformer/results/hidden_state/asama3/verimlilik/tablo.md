# Efficiency, stage 3 (CyberMetaFormer + M1/M2 final models, same session)

Measured on: apple-m3-pro, macOS-27.0.1-arm64-arm-64bit, torch 2.14.1, CPU 6 threads. Latency: median at bs=1 (50 repeats); throughput: MPS bs≈8192. Full-history models (full GRU/LSTM, CMF) in the online step mode: RNN single cell update, CMF `CyberMetaFormer.step` (conv buffer + SSD state; cost independent of t). CMF throughput: 82 episodes × 100 targets in one pass. ★ = on the Pareto front (CPU latency × OOD F1). Best k per model by val; F1 values are seed means (CMF 3, M1/M2 ozet.json). Ensemble rows (ood_suite, topluluk_T, 3 seeds): cost = member × 3, not part of the Pareto front.

| Model | k | Parameters | Latency CPU (ms) | Latency MPS (ms) | Throughput MPS (µs/sample) | Test F1 | OOD F1 | OOD silent recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CyberMetaFormer ensemble (3) | tam | 559 995 | 2.57 | 8.06 | 67.02 | 0.980 | 0.943 | 0.781 |
| CyberMetaFormer ★ | tam | 186 665 | 0.86 | 2.69 | 22.34 | 0.980 | 0.942 | 0.776 |
| CyberMetaFormer B2 | tam | 2 828 473 | 1.44 | 3.01 | 74.35 | 0.979 | 0.938 | 0.785 |
| M1: GRU (full history) ensemble (3) | tam | 550 764 | 0.13 | 1.13 | 1.02 | 0.978 | 0.932 | 0.746 |
| M1: GRU (full history) ★ | tam | 183 588 | 0.04 | 0.38 | 0.34 | 0.977 | 0.926 | 0.736 |
| M1: LSTM (full history) | tam | 353 316 | 0.05 | 0.29 | 0.44 | 0.976 | 0.920 | 0.719 |
| TimeMixer | 64 | 144 516 | 1.43 | 4.79 | 31.46 | 0.969 | 0.893 | 0.671 |
| Transformer | 64 | 2 399 268 | 0.90 | 1.02 | 149.46 | 0.971 | 0.892 | 0.637 |
| M1: GRU | 32 | 183 588 | 0.53 | 5.19 | 9.30 | 0.966 | 0.887 | 0.651 |
| iTransformer | 64 | 383 972 | 1.05 | 1.61 | 53.15 | 0.970 | 0.886 | 0.636 |
| M1: LSTM | 32 | 353 316 | 0.49 | 0.69 | 11.27 | 0.965 | 0.881 | 0.605 |
| TSMixer | 64 | 619 470 | 0.33 | 0.68 | 58.27 | 0.967 | 0.877 | 0.591 |
| M1: 1D-CNN | 32 | 342 948 | 0.36 | 0.59 | 9.44 | 0.962 | 0.866 | 0.602 |
| PatchTST | 32 | 385 444 | 2.59 | 1.08 | 83.77 | 0.955 | 0.850 | 0.488 |
| M1: MLP ★ | 32 | 1 559 588 | 0.04 | 0.29 | 0.76 | 0.956 | 0.850 | 0.511 |
| TimesNet | 8 | 301 572 | 1.23 | 2.89 | 15.89 | 0.936 | 0.816 | 0.364 |

## CMF online inference: step mode and full recomputation

Without the step mode the whole history (S=101) would be recomputed at every step; "full recomputation" is its worst case (last step). Deviation: largest absolute logit difference of the step mode from forward_episode (4 test episodes).

| Model | Step CPU (ms) | Step MPS (ms) | Full recomputation CPU (ms) | Full recomputation MPS (ms) | Deviation CPU | Deviation MPS |
|---|---:|---:|---:|---:|---:|---:|
| CyberMetaFormer | 0.86 | 2.69 | 6.62 | 3.67 | 5.7e-06 | 7.6e-06 |
| CyberMetaFormer B2 | 1.44 | 3.01 | 15.38 | 6.76 | 6.6e-05 | 5.6e-05 |

## Pareto front (all cells)

| Model | k | Latency CPU (ms) | OOD F1 |
|---|---:|---:|---:|
| M1: LSTM | 1 | 0.03 | 0.810 |
| M1: MLP | 16 | 0.03 | 0.840 |
| M1: MLP | 32 | 0.04 | 0.850 |
| M1: GRU (full history) | tam | 0.04 | 0.926 |
| CyberMetaFormer | tam | 0.86 | 0.942 |

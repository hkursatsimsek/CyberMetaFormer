# Verimlilik (final modeller, HPO boyutları)

Ölçüm: apple-m3-pro, macOS-27.0.1-arm64-arm-64bit, torch 2.14.1, CPU 6 iş parçacığı. Gecikme bs=1 medyanı (50 tekrar); verim MPS bs=8192. Tam geçmiş RNN: adım başına tek hücre güncellemesi (aynı ayarla k=1). ★ = Pareto sınırında (CPU gecikmesi × OOD F1). Model başına val'e göre en iyi k; bütün hücreler `olcum_*.json`.

| Model | k | Parametre | Gecikme CPU (ms) | Gecikme MPS (ms) | Verim MPS (µs/örnek) | Test F1 | OOD F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| M1: GRU (tam geçmiş) ★ | tam | 183 588 | 0.04 | 0.34 | 0.35 | 0.977 | 0.926 |
| M1: LSTM (tam geçmiş) | tam | 353 316 | 0.05 | 0.29 | 0.43 | 0.976 | 0.920 |
| TimeMixer | 64 | 144 516 | 1.48 | 5.10 | 32.96 | 0.969 | 0.893 |
| Transformer | 64 | 2 399 268 | 1.22 | 1.09 | 155.85 | 0.971 | 0.892 |
| M1: GRU | 32 | 183 588 | 0.52 | 5.33 | 9.35 | 0.966 | 0.887 |
| iTransformer | 64 | 383 972 | 0.94 | 1.67 | 52.75 | 0.970 | 0.886 |
| M1: LSTM | 32 | 353 316 | 0.47 | 0.72 | 11.28 | 0.965 | 0.881 |
| TSMixer | 64 | 619 470 | 0.34 | 0.67 | 61.41 | 0.967 | 0.877 |
| M1: 1D-CNN | 32 | 342 948 | 0.37 | 0.60 | 9.49 | 0.962 | 0.866 |
| PatchTST | 32 | 385 444 | 2.92 | 1.12 | 86.83 | 0.955 | 0.850 |
| M1: MLP ★ | 32 | 1 559 588 | 0.04 | 0.24 | 0.80 | 0.956 | 0.850 |
| TimesNet | 8 | 301 572 | 1.29 | 3.09 | 16.48 | 0.936 | 0.816 |

## Pareto sınırı (bütün hücreler)

| Model | k | Gecikme CPU (ms) | OOD F1 |
|---|---:|---:|---:|
| M1: LSTM | 1 | 0.03 | 0.810 |
| M1: MLP | 16 | 0.03 | 0.840 |
| M1: MLP | 32 | 0.04 | 0.850 |
| M1: GRU (tam geçmiş) | tam | 0.04 | 0.926 |

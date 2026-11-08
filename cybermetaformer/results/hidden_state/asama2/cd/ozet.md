# Kritik fark diyagramları (Friedman + Nemenyi, bloklar = bölümler)

Her model her bölümde seed'lerinin ortalamasıyla girer; sıra 1 = en iyi. Pozitif etiketi olmayan bölümler düşüldü. `python -m cybermetaformer.hidden_state.cd`.

## k8

**Test:** 2000 bölüm, Friedman p = 0, CD = 0.373.

| Sıra | Model | Ortalama sıra | Bölüm F1 ortalaması | Bölüm F1 medyanı |
|---:|---|---:|---:|---:|
| 1 | iTransformer | 5.16 | 0.905 | 0.919 |
| 2 | XGBoost | 5.29 | 0.906 | 0.921 |
| 3 | MLP | 5.79 | 0.903 | 0.915 |
| 4 | GRU | 6.13 | 0.901 | 0.913 |
| 5 | 1D-CNN | 6.37 | 0.901 | 0.911 |
| 6 | LSTM | 6.58 | 0.898 | 0.910 |
| 7 | TSMixer | 6.58 | 0.899 | 0.909 |
| 8 | TimeMixer | 6.59 | 0.899 | 0.909 |
| 9 | TimesNet | 6.79 | 0.898 | 0.906 |
| 10 | Transformer | 6.82 | 0.899 | 0.911 |
| 11 | XGB-TF | 7.14 | 0.893 | 0.909 |
| 12 | PatchTST | 8.76 | 0.885 | 0.896 |

**OOD:** 4000 bölüm, Friedman p = 0, CD = 0.263.

| Sıra | Model | Ortalama sıra | Bölüm F1 ortalaması | Bölüm F1 medyanı |
|---:|---|---:|---:|---:|
| 1 | XGBoost | 3.60 | 0.828 | 0.832 |
| 2 | XGB-TF | 5.09 | 0.813 | 0.819 |
| 3 | 1D-CNN | 5.49 | 0.808 | 0.810 |
| 4 | iTransformer | 5.50 | 0.809 | 0.812 |
| 5 | MLP | 5.86 | 0.803 | 0.804 |
| 6 | TSMixer | 6.64 | 0.799 | 0.803 |
| 7 | GRU | 6.66 | 0.799 | 0.801 |
| 8 | Transformer | 7.43 | 0.794 | 0.799 |
| 9 | PatchTST | 7.60 | 0.787 | 0.791 |
| 10 | TimeMixer | 7.97 | 0.791 | 0.794 |
| 11 | TimesNet | 8.07 | 0.785 | 0.786 |
| 12 | LSTM | 8.07 | 0.789 | 0.791 |

## k32

**Test:** 2000 bölüm, Friedman p = 0, CD = 0.373.

| Sıra | Model | Ortalama sıra | Bölüm F1 ortalaması | Bölüm F1 medyanı |
|---:|---|---:|---:|---:|
| 1 | GRU | 4.94 | 0.932 | 0.950 |
| 2 | iTransformer | 5.03 | 0.932 | 0.949 |
| 3 | LSTM | 5.33 | 0.929 | 0.946 |
| 4 | Transformer | 5.54 | 0.929 | 0.946 |
| 5 | XGBoost | 5.80 | 0.928 | 0.950 |
| 6 | 1D-CNN | 5.88 | 0.927 | 0.944 |
| 7 | TimeMixer | 5.96 | 0.925 | 0.941 |
| 8 | TSMixer | 6.31 | 0.925 | 0.941 |
| 9 | MLP | 7.13 | 0.920 | 0.933 |
| 10 | PatchTST | 8.07 | 0.913 | 0.923 |
| 11 | XGB-TF | 8.15 | 0.910 | 0.928 |
| 12 | TimesNet | 9.87 | 0.857 | 0.866 |

**OOD:** 4000 bölüm, Friedman p = 0, CD = 0.263.

| Sıra | Model | Ortalama sıra | Bölüm F1 ortalaması | Bölüm F1 medyanı |
|---:|---|---:|---:|---:|
| 1 | XGBoost | 3.91 | 0.870 | 0.874 |
| 2 | GRU | 4.77 | 0.852 | 0.855 |
| 3 | LSTM | 5.23 | 0.848 | 0.853 |
| 4 | iTransformer | 5.54 | 0.847 | 0.850 |
| 5 | Transformer | 6.35 | 0.842 | 0.846 |
| 6 | XGB-TF | 6.68 | 0.840 | 0.847 |
| 7 | 1D-CNN | 6.70 | 0.839 | 0.842 |
| 8 | TSMixer | 6.83 | 0.837 | 0.841 |
| 9 | TimeMixer | 6.90 | 0.836 | 0.840 |
| 10 | PatchTST | 7.46 | 0.830 | 0.837 |
| 11 | MLP | 8.04 | 0.830 | 0.834 |
| 12 | TimesNet | 9.59 | 0.811 | 0.813 |

## enIyiK

**Test:** 2000 bölüm, Friedman p = 0, CD = 0.444.

| Sıra | Model | Ortalama sıra | Bölüm F1 ortalaması | Bölüm F1 medyanı |
|---:|---|---:|---:|---:|
| 1 | GRU (tam geçmiş) | 4.88 | 0.951 | 0.977 |
| 2 | LSTM (tam geçmiş) | 5.02 | 0.950 | 0.976 |
| 3 | Transformer k=64 | 6.04 | 0.941 | 0.962 |
| 4 | iTransformer k=64 | 6.24 | 0.940 | 0.959 |
| 5 | TimeMixer k=64 | 6.42 | 0.938 | 0.954 |
| 6 | GRU k=32 | 6.89 | 0.932 | 0.950 |
| 7 | LSTM k=32 | 7.34 | 0.929 | 0.946 |
| 8 | TSMixer k=64 | 7.40 | 0.934 | 0.951 |
| 9 | XGBoost k=32 | 7.51 | 0.928 | 0.950 |
| 10 | 1D-CNN k=32 | 7.80 | 0.927 | 0.944 |
| 11 | MLP k=32 | 8.95 | 0.920 | 0.933 |
| 12 | XGB-TF k=tam | 9.68 | 0.917 | 0.935 |
| 13 | PatchTST k=32 | 9.86 | 0.913 | 0.923 |
| 14 | TimesNet k=8 | 10.95 | 0.898 | 0.906 |

**OOD:** 4000 bölüm, Friedman p = 0, CD = 0.314.

| Sıra | Model | Ortalama sıra | Bölüm F1 ortalaması | Bölüm F1 medyanı |
|---:|---|---:|---:|---:|
| 1 | GRU (tam geçmiş) | 2.85 | 0.925 | 0.929 |
| 2 | LSTM (tam geçmiş) | 3.07 | 0.922 | 0.927 |
| 3 | Transformer k=64 | 5.97 | 0.880 | 0.881 |
| 4 | TimeMixer k=64 | 6.23 | 0.874 | 0.877 |
| 5 | iTransformer k=64 | 6.50 | 0.878 | 0.880 |
| 6 | XGBoost k=32 | 6.67 | 0.870 | 0.874 |
| 7 | XGB-TF k=tam | 6.90 | 0.876 | 0.879 |
| 8 | TSMixer k=64 | 7.45 | 0.865 | 0.869 |
| 9 | GRU k=32 | 7.78 | 0.852 | 0.855 |
| 10 | LSTM k=32 | 8.19 | 0.848 | 0.853 |
| 11 | 1D-CNN k=32 | 9.62 | 0.839 | 0.842 |
| 12 | PatchTST k=32 | 10.04 | 0.830 | 0.837 |
| 13 | MLP k=32 | 10.78 | 0.830 | 0.834 |
| 14 | TimesNet k=8 | 12.96 | 0.785 | 0.786 |

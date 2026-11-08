| Seed | Model | Test host macro-F1 | OOD host macro-F1 | Relative OOD drop | OOD silent recall | OOD N_hist recall |
|---|---|---:|---:|---:|---:|---:|
| 0 | CMF | 0.9801 | 0.9424 | 0.0385 | 0.760 | 0.819 |
| 0 | GRU | 0.9770 | 0.9275 | 0.0506 | 0.734 | 0.790 |
| 1 | CMF | 0.9801 | 0.9417 | 0.0392 | 0.805 | 0.846 |
| 1 | GRU | 0.9768 | 0.9260 | 0.0520 | 0.742 | 0.788 |
| 2 | CMF | 0.9801 | 0.9411 | 0.0398 | 0.763 | 0.824 |
| 2 | GRU | 0.9762 | 0.9256 | 0.0519 | 0.733 | 0.789 |
| 3 | CMF | 0.9801 | 0.9413 | 0.0396 | 0.758 | 0.810 |
| 3 | GRU | 0.9770 | 0.9278 | 0.0503 | 0.748 | 0.808 |
| 4 | CMF | 0.9803 | 0.9422 | 0.0388 | 0.772 | 0.831 |
| 4 | GRU | 0.9769 | 0.9250 | 0.0531 | 0.724 | 0.782 |
| Mean ± sd | CMF | 0.9801 ± 0.0001 | 0.9417 ± 0.0006 | 0.0392 ± 0.0005 | 0.772 ± 0.020 | 0.826 ± 0.014 |
| Mean ± sd | GRU | 0.9768 ± 0.0003 | 0.9264 ± 0.0012 | 0.0516 ± 0.0011 | 0.736 ± 0.009 | 0.791 ± 0.010 |
| CMF − GRU | Mean ± sd | +0.0034 ± 0.0003 | +0.0153 ± 0.0014 | −0.0124 ± 0.0013 | +0.035 ± 0.020 | +0.035 ± 0.021 |
| CMF − GRU | Min – max | +0.0031 – +0.0039 | +0.0135 – +0.0172 | −0.0143 – −0.0108 | +0.010 – +0.063 | +0.002 – +0.058 |
| CMF − GRU | Seeds favouring CMF | 5 / 5 | 5 / 5 | 5 / 5 | 5 / 5 | 5 / 5 |
| CMF − GRU | Seed-level 95% t-interval (df = 4) | [+0.0029, +0.0038] | [+0.0136, +0.0170] | [−0.0140, −0.0108] | [+0.010, +0.061] | [+0.008, +0.061] |

Note. Both models trained with seeds 0–4 (main CyberMetaFormer run). Differences pair runs by seed index; the index carries no shared randomness between the two architectures, so the pairing is a bookkeeping convention. The t-interval treats the five seeds as the sample and describes retraining variability only; it is a seed-level interval, not an episode-level one, and with five seeds it is wide. Lower is better for the relative OOD drop.

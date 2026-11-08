| Model | Setting | Seeds | Split | Episodes | max \|Δz\| | max \|Δp\| | Argmax flips | Precision floor max \|Δz\| |
|---|---|---|---|---:|---:|---:|---:|---:|
| CyberMetaFormer (B1) | CPU, FP64 | 0, 1, 2, 3, 4 | Test | 2,000 | 5.6e-13 | 4.2e-14 | 0 / 12,000,000 | — |
| CyberMetaFormer (B1) | CPU, FP64 | 0, 1, 2, 3, 4 | OOD | 4,000 | 6.9e-13 | 1.3e-13 | 0 / 24,000,000 | — |
| CyberMetaFormer (B1) | CPU, FP32 | 0, 1, 2, 3, 4 | Test | 2,000 | 9.7e-05 | 8.3e-06 | 0 / 12,000,000 | 9.7e-05 |
| CyberMetaFormer (B1) | CPU, FP32 | 0, 1, 2, 3, 4 | OOD | 4,000 | 2.4e-04 | 2.8e-05 | 0 / 24,000,000 | 2.4e-04 |
| CyberMetaFormer (B1) | MPS, FP32 | 0, 1, 2, 3, 4 | Test | 2,000 | 1.4e-04 | 1.4e-05 | 0 / 12,000,000 | 1.6e-04 |
| CyberMetaFormer (B1) | MPS, FP32 | 0, 1, 2, 3, 4 | OOD | 4,000 | 2.0e-04 | 6.0e-05 | 0 / 24,000,000 | 2.4e-04 |

Note. Δ compares the step mode (Algorithm 1, recurrent SSD and convolution buffer) with the episode mode (parallel form, Eq. 7) on the same device and precision, over all episodes, steps t = 1, …, T and the 12 label hosts; maxima and flip counts are taken over seeds. Probabilities use each run's validation temperature. The precision floor is the largest difference between the episode-mode logits in that setting and the CPU FP64 episode-mode logits.

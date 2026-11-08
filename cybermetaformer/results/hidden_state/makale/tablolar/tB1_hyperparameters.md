| Hyperparameter | Value |
|---|---:|
| Token dimension d | 64 |
| MetaFormer blocks | 3 |
| Host-attention heads | 4 |
| Channel-MLP expansion ratio | 2 |
| SSD state size N | 16 |
| Causal depthwise kernel (steps) | 4 |
| Dropout | 0.141 |
| Learning rate | 2.4 × 10⁻³ |
| Weight decay | 6.2 × 10⁻³ |
| Model parameters | 186,665 |
| Validation host macro-F1 (best trial) | 0.9773 |

Note. Best of 15 trials (11 completed, 4 pruned; trial 13); each trial trained on 4,000 episodes for 10 epochs.

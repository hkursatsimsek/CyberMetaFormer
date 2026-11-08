| Stage | Operation | Input shape | Output shape | Parameters |
|---|---|---|---|---:|
| Input | Scaled Blue observation $\hat{o}_t$ | (B, S, 78) | — | — |
| Tokenization | Host tokens $x_{t,h}$ (Eq. 1) | (B, S, 78) | (B, S, H, 6) | 0 |
|  | Alert-age features $[\phi_{t,h};\nu_{t,h}]$ (Eq. 2) | (B, S, 78) | (B, S, H, 8) | 0 |
| Embedding | Input projection $W_{\mathrm{in}}$ (14→d) | (B, S, H, 14) | (B, S, H, d) | 960 |
|  | Host, subnet and position embeddings (Eq. 3) | — | (B, S, H, d) | 832 + 192 + 6,464 |
| Block ℓ (×3) | LayerNorm $\mathrm{LN}_1$ | (B, S, H, d) | (B, S, H, d) | 128 |
|  | Causal depthwise convolution branch, K = 4 (Eq. 4) | (B·H, S, d) | (B·H, S, d) | 4,480 |
|  | SSD branch, 4 heads, N = 16, P = 16 (Eqs. 5–7) | (B·H, S, d) | (B·H, S, d) | 21,072 |
|  | Softmax branch gate (Eq. 8) | (B·H, S, d) | (B·H, S, 2) | 130 |
|  | LayerNorm $\mathrm{LN}_2$ + topology-biased host attention (Eq. 9) | (B, S, H, d) | (B, S, H, d) | 128 + 16,656 |
|  | LayerNorm $\mathrm{LN}_3$ + channel MLP, d→2d→d (Eq. 10) | (B, S, H, d) | (B, S, H, d) | 128 + 16,576 |
|  | Block total |  |  | 59,298 |
| Output | Final LayerNorm + shared head on 12 label hosts (Eq. 11) | (B, S, H, d) | (B, S, 12, 3) | 128 + 195 |
| **Total** |  | (B, S, 78) | (B, S, 12, 3) | **186,665** |

Note. B: episodes per batch; S = T + 1 = 101 steps; H = 13 hosts; d = 64. Inside the temporal mixer every host is an independent sequence (B·H rows) with shared weights; inside the host mixer every time step is an independent set of 13 tokens (B·S rows). Online inference keeps, per episode, a 13×4 last-event table and, per block, a convolution buffer and an SSD state of shapes (13, 3, 64) and (13, 4, 16, 16) (over the 3 blocks: 47,424 floats, 189,696 bytes in FP32, without the integer last-event table).

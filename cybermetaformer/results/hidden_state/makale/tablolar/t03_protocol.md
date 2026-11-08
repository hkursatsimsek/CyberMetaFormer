| Setting | Stage 1 | Stage 2 | CyberMetaFormer |
|---|---|---|---|
| Candidate history lengths k | 1, 4, 8, 16, 32; full history: GRU, LSTM, MLP, XGB-TF; XGB-TF also 64 | 1, 4, 8, 16, 32, 64‡ | Full history |
| Random seeds | 5; 1 for XGBoost and the full-history MLP (marked †) | 3 | 5 (main model, seeds 0–4); 5 (host-token GRU references); 3 (ablations, seeds 0–2) |
| Hyperparameter search trials | 25 per model at k = 8 and 32 | 15 per model at k = 8 and 32 | 15 (main model only; variants reuse its setting) |
| Maximum final epochs | 30 (neural networks) | 30 | 30 |
| Early stopping | Patience 5 epochs; XGBoost 50 rounds | Patience 5 epochs | Patience 5 epochs |
| Model selection | Validation host macro-F1 | Validation host macro-F1 | Validation host macro-F1 |
| Paired statistical test | Paired episode bootstrap, 10,000 resamples; Holm | Paired episode bootstrap, 10,000 resamples; Holm | Paired episode bootstrap, 100,000 resamples; Holm |

Note. ‡ PatchTST not at k = 1, 4, 16, 64 (1 not applicable; 4, 16, 64 outside the compute budget); TimesNet not at k = 1, 4, 16, 64 (1 not applicable; 4, 16, 64 outside the compute budget); TimeMixer not at k = 1 (1 not applicable). Full history means obs[0..t] at every step t. The hyperparameter search used 4,000 training episodes and 10 epochs per trial for CyberMetaFormer.

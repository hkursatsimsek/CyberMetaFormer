| Set | Observation | CMF host macro-F1 | GRU host macro-F1 | CMF silent recall | GRU silent recall | CMF − GRU silent recall [95% CI] | CMF N_hist recall | GRU N_hist recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Test (training policies) | Flawed (as in training) | 0.9801 | 0.9768 | 0.904 | 0.902 | +0.003 [−0.000, +0.006] | 0.884 | 0.886 |
|  | Corrected | 0.9799 | 0.9768 | 0.903 | 0.901 | +0.002 [−0.001, +0.005] | 0.881 | 0.883 |
| ε = 0.3 (main OOD) | Flawed (as in training) | 0.9417 | 0.9264 | 0.772 | 0.736 | +0.036 [+0.031, +0.040] | 0.826 | 0.791 |
|  | Corrected | 0.9405 | 0.9255 | 0.764 | 0.720 | +0.045 [+0.040, +0.050] | 0.815 | 0.778 |
| ε = 0.5 | Flawed (as in training) | 0.9419 | 0.9240 | 0.718 | 0.685 | +0.034 [+0.028, +0.039] | 0.727 | 0.677 |
|  | Corrected | 0.9398 | 0.9223 | 0.708 | 0.658 | +0.050 [+0.044, +0.056] | 0.704 | 0.635 |
| ε = 0.7 | Flawed (as in training) | 0.9611 | 0.9463 | 0.775 | 0.741 | +0.034 [+0.029, +0.038] | 0.626 | 0.554 |
|  | Corrected | 0.9596 | 0.9451 | 0.768 | 0.719 | +0.049 [+0.044, +0.054] | 0.596 | 0.496 |
| Remove instead of restore | Flawed (as in training) | 0.9694 | 0.9617 | 0.959 | 0.945 | +0.014 [+0.013, +0.015] | 0.516 | 0.440 |
|  | Corrected | 0.9839 | 0.9823 | 0.993 | 0.994 | −0.000 [−0.001, +0.000] | 0.922 | 0.926 |

Note. Both models are trained on the flawed data (seeds 0–4) and evaluated without retraining on two copies of the same episodes: the flawed observation of the training data and a copy re-simulated with the same seeds and the recorded defender actions in which a failed remove no longer shows the "removed" flag (and an invalid restore of user0 no longer clears its flags); true states, attacker observations, actions and rewards are byte-identical. Silent and N_hist host-steps are defined on the flawed observation, so both copies are scored on the same host-steps. Mean over seeds 0–4, each with its own validation temperature; CMF − GRU: paired episode bootstrap of the seed mean (10,000 resamples, Algorithm 2), Holm over all m = 100 rows of this analysis (smallest attainable adjusted p = 0.0200). Per-model corrected − flawed differences, the remove-mask slice and the share of host-steps that the correction turns into alerted ones: Table @tbl:duzeltilmis_fark.

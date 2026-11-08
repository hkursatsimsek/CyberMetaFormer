| Training → test attacker | CyberMetaFormer | Full-history GRU | TimeMixer (k = 64) |
|---|---:|---:|---:|
| B-line → Meander, test | 0.829 ± 0.037 | 0.575 ± 0.004 | 0.516 ± 0.010 |
| B-line → Meander, OOD | 0.724 ± 0.042 | 0.513 ± 0.006 | 0.463 ± 0.008 |
| Meander → B-line, test | 0.841 ± 0.012 | 0.850 ± 0.009 | 0.815 ± 0.004 |
| Meander → B-line, OOD | 0.738 ± 0.009 | 0.725 ± 0.014 | 0.682 ± 0.007 |

Note. Host macro-F1, mean ± SD over seeds 0–2. Each model is trained on one attacker and evaluated on the test and OOD episodes of the other attacker.

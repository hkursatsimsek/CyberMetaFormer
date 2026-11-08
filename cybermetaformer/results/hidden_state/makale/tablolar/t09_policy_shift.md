| Policy / condition | Episodes | CMF F1 | GRU F1 | CMF silent recall | GRU silent recall |
|---|---:|---:|---:|---:|---:|
| ε = 0.1 (training policy, test episodes) | 400 | 0.960 | 0.955 | 0.864 | 0.855 |
| ε = 0.3 (main OOD) | 4,000 | 0.942 | 0.926 | 0.772 | 0.736 |
| ε = 0.5 | 4,000 | 0.942 | 0.924 | 0.718 | 0.685 |
| ε = 0.7 | 4,000 | 0.961 | 0.946 | 0.775 | 0.741 |
| Periodic analyse on idle steps | 4,000 | 0.881 | 0.862 | 0.625 | 0.703 |
| Remove instead of restore | 4,000 | 0.969 | 0.962 | 0.959 | 0.945 |
| Earlier simulator version, OOD | 4,000 | 0.941 | 0.927 | 0.778 | 0.748 |

Note. Host macro-F1 and silent recall of each seed (0–4), averaged; each seed uses its own validation temperature. ε = 0.1: the test episodes of the training policy react_restore_eps0.1. ε = 0.5 and 0.7, periodic analyse (react_restore_analyse) and remove (react_remove): new episodes, both attackers, no training on them. Earlier simulator version: OOD-policy episodes generated with the simulator's documented bugs left in place (separate generation seed).

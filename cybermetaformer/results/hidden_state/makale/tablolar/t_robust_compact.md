| Evaluation condition | CMF F1 | GRU F1 | CMF silent recall | GRU silent recall |
|---|---:|---:|---:|---:|
| OOD defender policy (ε = 0.3) | 0.942 | 0.926 | 0.772 | 0.736 |
| Periodic analyse on idle steps | 0.881 | 0.862 | 0.625 | 0.703 |
| Host permutation P2s | 0.924 | 0.900 | 0.668 | 0.582 |
| Rewired simulator topology | 0.928 | 0.901 | 0.679 | 0.589 |
| Rewired, relation matrix updated | 0.942 | — | 0.772 | — |
| Corrected remove flag (ε = 0.3) | 0.941 | 0.925 | 0.764 | 0.720 |

Note. Seed mean over seeds 0–4; all models trained on the original data and evaluated without retraining. Periodic analyse: the defender analyses hosts on idle steps (react_restore_analyse). P2s: host permutation across attack-graph roles (user1 ↔ user3, user2 ↔ user4) with the decoy-count columns kept in their slot. Rewired: new OOD-policy episodes with user1/2 → ent1 and user3/4 → ent0; relation matrix updated: the topology bias of CMF rebuilt from the rewired links, weights unchanged (the flat GRU has no relation matrix). Corrected remove flag: the OOD episodes re-simulated so that a failed remove no longer shows the "removed" flag; silent host-steps are defined on the flawed observation. All policies and paired differences: Tables @tbl:politika, @tbl:failure_analysis, @tbl:host_kayma, @tbl:host_kayma_fark, @tbl:duzeltilmis and @tbl:duzeltilmis_fark.

| Shift type | Condition | What changes | Episodes | Retraining | Results |
|---|---|---|---:|---|---|
| Policy parameter | ε = 0.3 (main OOD test) | Random-action probability of the restore-based defender raised from 0.1 to 0.3 | 4,000 | No | Tables @tbl:ana, @tbl:politika |
|  | ε = 0.5 and 0.7 | Same parameter, larger values | 4,000 each | No | Table @tbl:politika |
| Observation mechanism | Periodic analyse on idle steps | Idle steps are used to analyse hosts, which re-emits the root flag of compromised hosts (alert refresh) | 4,000 | No | Tables @tbl:politika, @tbl:basarisizlik |
|  | Remove instead of restore | Removal replaces restore; a failed remove keeps root access but shows the "removed" flag | 4,000 | No | Tables @tbl:politika, @tbl:basarisizlik |
|  | Corrected remove flag | Same episodes and defender actions re-simulated; a failed remove no longer shows "removed" | test 2,000; four shift sets 4,000 each | No | Table @tbl:duzeltilmis |
| Simulator version | Earlier simulator version | Documented simulator bugs left in place; OOD policy | 4,000 | No | Table @tbl:politika |
| Attacker | B-line ↔ Meander transfer | Model trained on one attacker and tested on the other | test 1,000; OOD 2,000 | Yes: one model per training attacker | Table @tbl:saldirgan |
| Host identity | Host permutations P1, P1s, P2, P2s, P3 | Observation columns and labels of hosts permuted within or across attack-graph roles; P3 permutes only the host embeddings | test 2,000; OOD 4,000 | No | Table @tbl:host_kayma |
| Topology | Rewired user → enterprise links | Simulator links user1/2 → ent1 and user3/4 → ent0; host count, subnets and observation layout unchanged | 4,000 per policy (ε = 0.1, 0.3) | No; relation matrix rebuilt in one condition | Table @tbl:host_kayma |

Note. Shifts relative to the training data, which contain the five training defender policies (Table @tbl:politikalar), both attackers and the original MiniCAGE topology. Policy and mechanism shifts and the earlier simulator version use new episodes of both attackers; the corrected copy re-simulates the episodes of the test, OOD and shift sets with the recorded defender actions. Not evaluated: the full CybORG engine and networks with a different number of hosts (Section 5.3).

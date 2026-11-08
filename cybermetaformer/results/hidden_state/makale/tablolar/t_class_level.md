| Split | Model | Class | Precision | Recall | F1 | Host-averaged F1 |
|---|---|---|---:|---:|---:|---:|
| Test | CMF | clean | 0.988 ± 0.001 | 0.992 ± 0.001 | 0.990 ± 0.000 | 0.988 ± 0.000 |
| Test | CMF | user | 0.992 ± 0.000 | 0.955 ± 0.001 | 0.973 ± 0.000 | 0.978 ± 0.000 |
| Test | CMF | root | 0.971 ± 0.003 | 0.962 ± 0.003 | 0.966 ± 0.000 | 0.974 ± 0.000 |
| Test | GRU | clean | 0.987 ± 0.000 | 0.989 ± 0.000 | 0.988 ± 0.000 | 0.987 ± 0.000 |
| Test | GRU | user | 0.991 ± 0.000 | 0.952 ± 0.001 | 0.971 ± 0.000 | 0.973 ± 0.001 |
| Test | GRU | root | 0.963 ± 0.001 | 0.960 ± 0.001 | 0.962 ± 0.000 | 0.970 ± 0.000 |
| OOD | CMF | clean | 0.981 ± 0.002 | 0.983 ± 0.002 | 0.982 ± 0.000 | 0.981 ± 0.000 |
| OOD | CMF | user | 0.994 ± 0.000 | 0.961 ± 0.000 | 0.977 ± 0.000 | 0.972 ± 0.000 |
| OOD | CMF | root | 0.827 ± 0.016 | 0.825 ± 0.016 | 0.826 ± 0.002 | 0.872 ± 0.002 |
| OOD | GRU | clean | 0.978 ± 0.001 | 0.976 ± 0.001 | 0.977 ± 0.000 | 0.976 ± 0.000 |
| OOD | GRU | user | 0.990 ± 0.001 | 0.956 ± 0.001 | 0.972 ± 0.000 | 0.965 ± 0.000 |
| OOD | GRU | root | 0.773 ± 0.005 | 0.795 ± 0.007 | 0.783 ± 0.002 | 0.837 ± 0.004 |

Note. Seed mean ± SD over seeds 0–4 of each model. Precision, recall and F1 of each class from the confusion pooled over the 11 evaluated hosts and all host-steps of the split; host-averaged F1 averages the per-host F1 of the class over the hosts in which it occurs. The 11 evaluated hosts are the label hosts with at least one compromised step (def is never compromised; user0, the attacker's foothold, is not a label host). Computed from the stored per-episode confusion matrices without new inference.

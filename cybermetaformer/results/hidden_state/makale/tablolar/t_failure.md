| Policy | Silent host-steps, user / root | Model | Host macro-F1 | Silent recall | Silent root → clean | Silent user → clean | Root → clean, all root steps |
|---|---:|---|---:|---:|---:|---:|---:|
| ε = 0.1 (training policy) | 1,061 / 48,602 | CMF | 0.960 ± 0.001 | 0.864 ± 0.011 | 0.126 | 0.614 | 0.120 |
|  |  | Host-token GRU, topology bias | 0.961 ± 0.001 | 0.864 ± 0.006 | 0.126 | 0.609 | 0.120 |
|  |  | Host-token GRU, no topology bias | 0.961 ± 0.000 | 0.868 ± 0.011 | 0.121 | 0.608 | 0.116 |
|  |  | Full-history GRU | 0.955 ± 0.000 | 0.855 ± 0.005 | 0.134 | 0.640 | 0.128 |
| ε = 0.3 (main OOD) | 7,889 / 313,119 | CMF | 0.942 ± 0.001 | 0.772 ± 0.020 | 0.217 | 0.692 | 0.175 |
|  |  | Host-token GRU, topology bias | 0.942 ± 0.001 | 0.764 ± 0.007 | 0.224 | 0.691 | 0.181 |
|  |  | Host-token GRU, no topology bias | 0.941 ± 0.001 | 0.763 ± 0.017 | 0.225 | 0.696 | 0.182 |
|  |  | Full-history GRU | 0.926 ± 0.001 | 0.736 ± 0.009 | 0.252 | 0.727 | 0.205 |
| Periodic analyse on idle steps | 14,020 / 114,356 | CMF | 0.881 ± 0.005 | 0.625 ± 0.034 | 0.353 | 0.552 | 0.300 |
|  |  | Host-token GRU, topology bias | 0.881 ± 0.003 | 0.624 ± 0.011 | 0.354 | 0.553 | 0.301 |
|  |  | Host-token GRU, no topology bias | 0.881 ± 0.004 | 0.639 ± 0.021 | 0.338 | 0.552 | 0.287 |
|  |  | Full-history GRU | 0.862 ± 0.003 | 0.703 ± 0.006 | 0.264 | 0.567 | 0.232 |
| Remove instead of restore | 1,869 / 1,955,513 | CMF | 0.969 ± 0.006 | 0.959 ± 0.013 | 0.040 | 0.681 | 0.040 |
|  |  | Host-token GRU, topology bias | 0.973 ± 0.009 | 0.967 ± 0.019 | 0.032 | 0.657 | 0.032 |
|  |  | Host-token GRU, no topology bias | 0.968 ± 0.006 | 0.955 ± 0.012 | 0.045 | 0.673 | 0.045 |
|  |  | Full-history GRU | 0.962 ± 0.006 | 0.945 ± 0.012 | 0.054 | 0.733 | 0.054 |

Note. Seed mean (± SD for F1 and silent recall) over seeds 0–4, from the stored logits (argmax), no retraining; the error-rate columns have seed SD ≤ 0.038. Error rates are within the true class: silent root → clean is the share of silent root host-steps predicted clean; the last column uses all root host-steps. ε = 0.1: the test episodes of the training policy (400); the other policies: 4,000 episodes each, both attackers. Slices by attacker, subnet and alert age: Table @tbl:analyse_dilim.

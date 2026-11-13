# M3 explainability: host mixer and the analyse loss

`explain_m3.py`. Silent recall and host macro-F1 per member (argmax; temperature does not change predictions), mean over 3 seeds ± std (ddof=1). eps0.1 = the `react_restore_eps0.1` episodes of v1 test; OOD = `ood_test` (ε=0.3); analyse = the `react_restore_analyse` episodes of `minicage_v1_shift` (4,000).

## 1. Ablations on analyse

| Model | Silent recall eps0.1 | OOD | analyse | analyse seeds | analyse / eps0.1 | F1 analyse |
|---|---:|---:|---:|---|---:|---:|
| CMF (main) | 0.866 ± 0.015 | 0.776 ± 0.025 | 0.637 ± 0.041 | 0.663 / 0.659 / 0.590 | 0.74 | 0.8809 |
| full GRU | 0.858 ± 0.005 | 0.736 ± 0.005 | 0.706 ± 0.002 | 0.704 / 0.705 / 0.708 | 0.82 | 0.8621 |
| host_yok | 0.720 ± 0.012 | 0.591 ± 0.018 | 0.449 ± 0.008 | 0.455 / 0.440 / 0.452 | 0.62 | 0.8063 |
| duz78 | 0.860 ± 0.004 | 0.726 ± 0.009 | 0.711 ± 0.009 | 0.719 / 0.712 / 0.702 | 0.83 | 0.8631 |
| host_tam | 0.868 ± 0.008 | 0.774 ± 0.015 | 0.650 ± 0.017 | 0.645 / 0.669 / 0.636 | 0.75 | 0.8811 |
| host_gcn | 0.842 ± 0.009 | 0.730 ± 0.005 | 0.601 ± 0.011 | 0.613 / 0.590 / 0.599 | 0.71 | 0.8786 |
| hafiza_yok | 0.869 ± 0.002 | 0.772 ± 0.009 | 0.676 ± 0.005 | 0.674 / 0.681 / 0.672 | 0.78 | 0.8780 |
| yalniz_yas | 0.873 ± 0.012 | 0.789 ± 0.002 | 0.669 ± 0.022 | 0.647 / 0.692 / 0.667 | 0.77 | 0.8794 |
| kol_konv | 0.712 ± 0.010 | 0.562 ± 0.032 | 0.615 ± 0.006 | 0.616 / 0.620 / 0.608 | 0.86 | 0.8579 |
| kol_ssd | 0.870 ± 0.010 | 0.780 ± 0.024 | 0.625 ± 0.012 | 0.636 / 0.628 / 0.613 | 0.72 | 0.8794 |
| konum_yok | 0.857 ± 0.002 | 0.765 ± 0.003 | 0.633 ± 0.019 | 0.612 / 0.638 / 0.649 | 0.74 | 0.8807 |

## 2. Silent recall slices

Share: share of the set's silent host-steps. Recall is the seed mean. Host groups are subnets: ent = def and ent0–2 (def is always clean), op = ophost0–2 and opserv, user = user1–4 (user0 is not labelled). Last-alert type from the observation bits of that step: exploit activity, root flag without activity `[1,1]` (the form analyse rewrites), other; "then restore" = the host was restored since the last alert (slice-name suffix). A slice has at least 200 host-steps.

### ε=0.1 (training policy)

| Slice | Share | CMF (main) | full GRU | duz78 | host_yok |
|---|---:|---:|---:|---:|---:|
| bline × ent | 0.222 | 0.998 | 0.989 | 0.986 | 0.876 |
| bline × user | 0.147 | 0.998 | 0.990 | 0.974 | 0.911 |
| meander × ent | 0.197 | 0.778 | 0.747 | 0.779 | 0.516 |
| meander × user | 0.430 | 0.791 | 0.794 | 0.791 | 0.665 |
| age never | 0.254 | 0.921 | 0.918 | 0.916 | 0.632 |
| age 1–3 | 0.017 | 0.790 | 0.758 | 0.753 | 0.723 |
| age 4–7 | 0.041 | 0.650 | 0.667 | 0.660 | 0.556 |
| age 8–15 | 0.098 | 0.636 | 0.654 | 0.657 | 0.418 |
| age 16–31 | 0.200 | 0.797 | 0.787 | 0.795 | 0.593 |
| age 32–63 | 0.295 | 0.937 | 0.917 | 0.923 | 0.927 |
| age 64+ | 0.094 | 0.986 | 0.973 | 0.964 | 0.962 |
| last alert: no alert | 0.254 | 0.922 | 0.918 | 0.916 | 0.632 |
| last alert: exploit | 0.020 | 0.999 | 0.983 | 0.879 | 0.976 |
| last alert: exploit, then restore | 0.675 | 0.849 | 0.837 | 0.844 | 0.765 |
| last alert: other, then restore | 0.043 | 0.730 | 0.752 | 0.761 | 0.411 |

### ε=0.3 (OOD)

| Slice | Share | CMF (main) | full GRU | duz78 | host_yok |
|---|---:|---:|---:|---:|---:|
| bline × ent | 0.176 | 0.982 | 0.956 | 0.942 | 0.783 |
| bline × op | 0.036 | 1.000 | 0.995 | 1.000 | 0.983 |
| bline × user | 0.117 | 0.978 | 0.927 | 0.910 | 0.780 |
| meander × ent | 0.218 | 0.655 | 0.579 | 0.575 | 0.411 |
| meander × op | 0.015 | 0.980 | 0.954 | 0.690 | 0.960 |
| meander × user | 0.438 | 0.675 | 0.647 | 0.643 | 0.508 |
| age never | 0.301 | 0.830 | 0.789 | 0.773 | 0.454 |
| age 1–3 | 0.024 | 0.840 | 0.794 | 0.712 | 0.752 |
| age 4–7 | 0.043 | 0.660 | 0.667 | 0.606 | 0.579 |
| age 8–15 | 0.097 | 0.575 | 0.590 | 0.553 | 0.436 |
| age 16–31 | 0.194 | 0.659 | 0.635 | 0.631 | 0.475 |
| age 32–63 | 0.260 | 0.844 | 0.773 | 0.785 | 0.793 |
| age 64+ | 0.081 | 0.926 | 0.864 | 0.862 | 0.872 |
| last alert: no alert | 0.262 | 0.837 | 0.796 | 0.780 | 0.465 |
| last alert: no alert, then restore | 0.039 | 0.784 | 0.738 | 0.723 | 0.376 |
| last alert: exploit | 0.084 | 0.992 | 0.984 | 0.842 | 0.974 |
| last alert: exploit, then restore | 0.441 | 0.746 | 0.695 | 0.714 | 0.685 |
| last alert: root flag (no activity) | 0.003 | 0.998 | 0.881 | 0.767 | 0.998 |
| last alert: root flag (no activity), then restore | 0.011 | 0.465 | 0.507 | 0.543 | 0.105 |
| last alert: other | 0.035 | 1.000 | 0.996 | 0.836 | 1.000 |
| last alert: other, then restore | 0.126 | 0.569 | 0.535 | 0.561 | 0.252 |

### react_restore_analyse

| Slice | Share | CMF (main) | full GRU | duz78 | host_yok |
|---|---:|---:|---:|---:|---:|
| bline × ent | 0.198 | 0.991 | 0.971 | 0.951 | 0.831 |
| bline × user | 0.164 | 0.995 | 0.993 | 0.995 | 0.867 |
| meander × ent | 0.180 | 0.460 | 0.530 | 0.541 | 0.192 |
| meander × user | 0.458 | 0.426 | 0.557 | 0.573 | 0.235 |
| age never | 0.136 | 0.677 | 0.706 | 0.693 | 0.203 |
| age 1–3 | 0.092 | 0.798 | 0.768 | 0.750 | 0.764 |
| age 4–7 | 0.182 | 0.654 | 0.688 | 0.689 | 0.576 |
| age 8–15 | 0.297 | 0.580 | 0.673 | 0.677 | 0.351 |
| age 16–31 | 0.235 | 0.592 | 0.714 | 0.733 | 0.374 |
| age 32–63 | 0.056 | 0.710 | 0.791 | 0.843 | 0.927 |
| age 64+ | 0.002 | 0.905 | 0.914 | 0.961 | 0.969 |
| last alert: no alert | 0.136 | 0.677 | 0.706 | 0.693 | 0.203 |
| last alert: exploit, then restore | 0.787 | 0.651 | 0.707 | 0.720 | 0.525 |
| last alert: root flag (no activity), then restore | 0.063 | 0.419 | 0.704 | 0.656 | 0.081 |
| last alert: other, then restore | 0.013 | 0.438 | 0.634 | 0.655 | 0.185 |

## 3. Attention of the host mixer

400 random episodes per set; head mean; mean over 3 seeds (± std). Relation share: all queries; under uniform attention self 0.077, edge 0.059, subnet 0.260, other 0.604. Alerting: total share that silent-compromise queries give to other hosts alerting in the last 5 steps.

| Model | Set | Layer | Self | Edge | Subnet | Other | Alerting (uniform) | Caught / missed |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CMF (main) | eps0.1 | 0 | 0.042 | 0.100 | 0.401 | 0.457 | 0.389 (0.114) | 0.394 / 0.362 |
| CMF (main) | eps0.1 | 1 | 0.021 | 0.097 | 0.344 | 0.538 | 0.273 (0.114) | 0.275 / 0.259 |
| CMF (main) | eps0.1 | 2 | 0.045 | 0.080 | 0.321 | 0.554 | 0.099 (0.114) | 0.103 / 0.078 |
| CMF (main) | ood | 0 | 0.041 | 0.098 | 0.389 | 0.472 | 0.359 (0.124) | 0.361 / 0.355 |
| CMF (main) | ood | 1 | 0.020 | 0.094 | 0.343 | 0.544 | 0.272 (0.124) | 0.273 / 0.266 |
| CMF (main) | ood | 2 | 0.045 | 0.078 | 0.314 | 0.562 | 0.096 (0.124) | 0.105 / 0.074 |
| CMF (main) | analyse | 0 | 0.044 | 0.104 | 0.421 | 0.431 | 0.367 (0.110) | 0.367 / 0.367 |
| CMF (main) | analyse | 1 | 0.021 | 0.101 | 0.345 | 0.533 | 0.264 (0.110) | 0.255 / 0.270 |
| CMF (main) | analyse | 2 | 0.043 | 0.080 | 0.322 | 0.555 | 0.107 (0.110) | 0.108 / 0.106 |
| host_tam | eps0.1 | 0 | 0.070 | 0.098 | 0.283 | 0.548 | 0.399 (0.114) | 0.405 / 0.357 |
| host_tam | eps0.1 | 1 | 0.073 | 0.076 | 0.280 | 0.571 | 0.247 (0.114) | 0.245 / 0.262 |
| host_tam | eps0.1 | 2 | 0.131 | 0.061 | 0.292 | 0.516 | 0.092 (0.114) | 0.093 / 0.084 |
| host_tam | ood | 0 | 0.066 | 0.091 | 0.273 | 0.570 | 0.366 (0.124) | 0.369 / 0.356 |
| host_tam | ood | 1 | 0.068 | 0.072 | 0.279 | 0.581 | 0.249 (0.124) | 0.244 / 0.266 |
| host_tam | ood | 2 | 0.127 | 0.057 | 0.289 | 0.527 | 0.092 (0.124) | 0.096 / 0.083 |
| host_tam | analyse | 0 | 0.075 | 0.108 | 0.296 | 0.521 | 0.373 (0.110) | 0.376 / 0.367 |
| host_tam | analyse | 1 | 0.075 | 0.081 | 0.284 | 0.560 | 0.238 (0.110) | 0.223 / 0.261 |
| host_tam | analyse | 2 | 0.122 | 0.059 | 0.291 | 0.528 | 0.097 (0.110) | 0.094 / 0.103 |

**Attack-graph edge** (ε=0.1): attention that query hosts with an edge (ent2→opserv, user1→ent0, user2→ent0, user3→ent1, user4→ent1) give to their edge partner; uniform 1/13 = 0.077. Seed mean ± std.

| Model | Layer 0 | Layer 1 | Layer 2 |
|---|---:|---:|---:|
| CMF (main) | 0.180 ± 0.043 | 0.142 ± 0.075 | 0.103 ± 0.019 |
| host_tam | 0.194 ± 0.040 | 0.102 ± 0.011 | 0.086 ± 0.006 |

**Relation biases of CMF** (head mean; row = layer; mean over 3 seeds ± std; before softmax):

| Layer | self | edge | subnet | other |
|---:|---:|---:|---:|---:|
| 0 | -0.74 ± 0.21 | +0.24 ± 0.25 | +0.51 ± 0.21 | -0.44 ± 0.19 |
| 1 | -1.52 ± 0.09 | +0.23 ± 0.35 | +0.54 ± 0.12 | +0.08 ± 0.15 |
| 2 | -2.63 ± 0.26 | +0.65 ± 0.17 | +0.48 ± 0.10 | +0.41 ± 0.14 |

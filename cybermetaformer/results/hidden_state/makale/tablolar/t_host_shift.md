| Criterion | Model | Identity | P1 | P1s | P2 | P2s | P3 | Rewired simulator | Rewired, bias updated |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OOD host macro-F1 | CMF | 0.942 | 0.934 | 0.938 | 0.913 | 0.924 | 0.927 | 0.928 | 0.942 |
|  | CMF, no topology bias | 0.941 | 0.932 | 0.937 | 0.920 | 0.932 | 0.920 | 0.936 | — |
|  | Host-token GRU, topology bias | 0.942 | 0.935 | 0.938 | 0.906 | 0.921 | 0.926 | 0.924 | 0.942 |
|  | Host-token GRU, no topology bias | 0.941 | 0.933 | 0.937 | 0.921 | 0.935 | 0.921 | 0.938 | — |
|  | Full-history GRU | 0.926 | 0.925 | 0.925 | 0.877 | 0.900 | — | 0.901 | — |
| OOD silent recall | CMF | 0.772 | 0.712 | 0.761 | 0.585 | 0.668 | 0.670 | 0.679 | 0.772 |
|  | CMF, no topology bias | 0.766 | 0.695 | 0.752 | 0.628 | 0.720 | 0.628 | 0.734 | — |
|  | Host-token GRU, topology bias | 0.764 | 0.718 | 0.754 | 0.545 | 0.632 | 0.662 | 0.643 | 0.765 |
|  | Host-token GRU, no topology bias | 0.763 | 0.708 | 0.751 | 0.633 | 0.735 | 0.633 | 0.744 | — |
|  | Full-history GRU | 0.736 | 0.748 | 0.734 | 0.761 | 0.582 | — | 0.589 | — |
| OOD N_hist recall | CMF | 0.826 | 0.770 | 0.818 | 0.550 | 0.627 | 0.720 | 0.620 | 0.828 |
|  | CMF, no topology bias | 0.817 | 0.746 | 0.810 | 0.645 | 0.736 | 0.645 | 0.744 | — |
|  | Host-token GRU, topology bias | 0.826 | 0.786 | 0.820 | 0.511 | 0.596 | 0.718 | 0.589 | 0.826 |
|  | Host-token GRU, no topology bias | 0.814 | 0.747 | 0.806 | 0.631 | 0.762 | 0.631 | 0.764 | — |
|  | Full-history GRU | 0.791 | 0.796 | 0.790 | 0.735 | 0.533 | — | 0.539 | — |

Note. OOD test policy (ε = 0.3), mean over seeds 0–4; the largest seed SD in the table is 0.011 (F1), 0.057 (silent recall) and 0.090 (N_hist recall). All models are trained on the original host order and topology and evaluated without retraining. P1: permutation within attack-graph roles (ophost0 ← ophost1 ← ophost2 ← ophost0, user1 ↔ user2, user3 ↔ user4; edges and subnets preserved). P2: across roles (user1 ↔ user3, user2 ↔ user4; subnets preserved, user → enterprise edges misaligned). Observation columns and labels move together; in P1 and P2 the decoy-count column also moves, so the host-specific decoy capacity (user1/2: 4, user3: 2, user4: 1) travels with the host as an identity cue, while P1s and P2s keep the decoy-count columns in their slot. P3: only the host-embedding rows E_host are permuted as in P2, the data are unchanged (host-token models only); without topology bias P3 equals P2, because permuting the data is equivalent to permuting E_host and the relation matrix together. Original episodes: n = 4,800,000 host-steps, 321,008 silent, 96,486 N_hist. Rewired simulator: new episodes of the same policy with user1/2 → ent1 and user3/4 → ent0 (4,000 episodes; 326,693 silent, 97,795 N_hist host-steps); in the simulator the decoy capacity stays with the host slot, as in P2s. Bias updated: the relation matrix of the two topology-biased models is rebuilt from the rewired links, weights unchanged. Paired differences with 95% CIs: Table @tbl:host_kayma_fark.

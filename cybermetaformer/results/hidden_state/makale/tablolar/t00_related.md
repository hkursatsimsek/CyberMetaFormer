| Approach | Primary task | Output | Temporal / relational modelling | Key difference |
|---|---|---|---|---|
| BF-PPO \[3\] | Defence action | Approximate belief and policy | POMDP / particle filter | No separate three-class F1 target per host state |
| Nayeri–Rezvani \[36\] | Next victim / alert | Future target relation | Temporal graph and Transformer | Does not estimate the current hidden privilege state |
| NIDS-Mamba \[39\] | Network intrusion detection | Flow-level class | SSD and attention | Differs in multi-host privilege state and the silent-compromise subset |
| GTB-PPO \[44\] | Target prediction + action | Next-target belief / action | Graph + causal time + PPO | Different prediction horizon and outcome metrics |
| CyberMetaFormer | Hidden host-state estimation | Three-class probabilities per host | Causal host memory + cross-host attention | Silent (no local alert) compromise and OOD sub-analyses |

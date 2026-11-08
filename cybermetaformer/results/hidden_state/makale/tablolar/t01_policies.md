| Policy | Defender behaviour | Split | Episodes |
|---|---|---|---:|
| sleep | Never acts | Train / validation / test | 4,000 |
| react_restore | Restores a host after an exploit alert | Train / validation / test | 4,000 |
| restore_decoys | Restores after an alert; places decoys on idle steps | Train / validation / test | 4,000 |
| random | Uniformly random over the 53 actions | Train / validation / test | 4,000 |
| react_restore_eps0.1 | react_restore; random action with 10% probability | Train / validation / test | 4,000 |
| react_restore_eps0.3 | react_restore; random action with 30% probability | OOD test only | 4,000 |

Note. Each defender policy is paired with both attackers (B-line and Meander); 2,000 episodes per pair, 100 steps per episode. Episodes of the first five policies are split, stratified by attacker–defender pair, into training (16,000), validation (2,000) and test (2,000); the 4,000 episodes of the sixth policy form the OOD test set and enter no training, selection or calibration step.

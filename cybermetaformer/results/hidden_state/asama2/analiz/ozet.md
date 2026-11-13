# M2 analyses (attention, IG)

Test subset: 300 random episodes (seed 0), 30767 silent-compromise host-steps. Method: header of `hidden_state/explain.py`. All numbers in `analiz.json`.

## Rollout (Transformer)

Ratio = rollout share on a position / mean of that position over all windows (1 = position prior).

| k | seed | steps with alerts | steps without alerts | silent compromise: alert step (mean / median) | above prior | caught | missed | pairs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0 | 1.08 | 0.87 | 1.64 / 1.46 | 74% | 1.62 | 1.77 | 13126 |
| 32 | 1 | 1.07 | 0.90 | 1.62 / 1.44 | 73% | 1.60 | 1.71 | 13126 |
| 32 | 2 | 1.12 | 0.81 | 1.75 / 1.60 | 78% | 1.74 | 1.81 | 13126 |
| 64 | 0 | 1.15 | 0.77 | 1.73 / 1.53 | 74% | 1.70 | 2.02 | 20409 |
| 64 | 1 | 1.18 | 0.73 | 2.07 / 1.80 | 82% | 2.00 | 2.49 | 20409 |
| 64 | 2 | 1.17 | 0.73 | 2.00 / 1.75 | 79% | 1.94 | 2.33 | 20409 |

## iTransformer host attention

Uniform cell 0.077. Own host = mean of the diagonal.

| k | model | own host | query dependence (TV) | largest cell | strongest cross (query → key) |
|---:|---|---:|---:|---:|---|
| 32 | s0 | 0.075 | 0.004 | 0.095 | ent2→opserv 0.09, ophost0→opserv 0.09, ophost2→opserv 0.09 |
| 32 | s1 | 0.075 | 0.006 | 0.096 | ent2→opserv 0.10, user4→ent1 0.09, user3→ent1 0.09 |
| 32 | s2 | 0.074 | 0.004 | 0.090 | ent1→user4 0.09, user3→ent1 0.09, opserv→ent2 0.09 |
| 32 | untrained | 0.077 | 0.000 | 0.078 | ophost2→opserv 0.08, ophost0→opserv 0.08, ophost1→opserv 0.08 |
| 64 | s0 | 0.073 | 0.007 | 0.098 | ophost0→ent1 0.10, ophost1→ent1 0.10, ophost2→ent1 0.10 |
| 64 | s1 | 0.073 | 0.010 | 0.098 | ophost0→user3 0.10, ophost1→user3 0.10, ophost2→user3 0.10 |
| 64 | s2 | 0.072 | 0.008 | 0.096 | ophost0→opserv 0.10, ophost2→opserv 0.10, ophost1→opserv 0.10 |
| 64 | untrained | 0.077 | 0.000 | 0.078 | user4→def 0.08, user4→ophost2 0.08, user4→ophost0 0.08 |

## Integrated Gradients (k=32, seed 0, 1000 samples)

| Model | own host (alert in window / never) | bit | scan | decoy | last step | last 4 steps | alert step ratio | convergence (median relative; steps) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Transformer | 0.203 (0.256 / 0.149) | 0.297 | 0.330 | 0.374 | 0.127 | 0.363 | 2.21 | 0.019; 32 |
| PatchTST | 0.341 (0.395 / 0.287) | 0.265 | 0.455 | 0.280 | 0.182 | 0.312 | 2.58 | 0.000; 32 |
| iTransformer | 0.247 (0.282 / 0.212) | 0.325 | 0.409 | 0.266 | 0.115 | 0.336 | 2.12 | 0.005; 128 |
| TSMixer | 0.237 (0.314 / 0.161) | 0.333 | 0.335 | 0.332 | 0.075 | 0.188 | 2.19 | 0.020; 32 |
| TimesNet | 0.183 (0.214 / 0.151) | 0.312 | 0.373 | 0.314 | 0.117 | 0.378 | 1.47 | 0.000; 32 |
| TimeMixer | 0.186 (0.243 / 0.129) | 0.338 | 0.371 | 0.291 | 0.104 | 0.322 | 1.55 | 0.000; 32 |

## TimesNet: dependence on the evaluation batch

Host macro-F1, mean over 3 seeds (range). Official = the reported evaluation order (episode-major, batch 8192); shuffled = random order, batch 8192; one by one = bs=1 (online use), only every 19th window. In the subset columns the three modes use the same windows.

| k | set | official | shuffled | subset: official | subset: shuffled | subset: one by one |
|---:|---|---:|---:|---:|---:|---:|
| 8 | test | 0.936 (0.935–0.937) | 0.936 (0.935–0.937) | 0.936 (0.934–0.937) | 0.936 (0.934–0.936) | 0.934 (0.933–0.934) |
| 8 | OOD | 0.816 (0.812–0.821) | 0.821 (0.820–0.822) | 0.814 (0.811–0.818) | 0.820 (0.819–0.821) | 0.816 (0.815–0.818) |
| 32 | test | 0.925 (0.920–0.930) | 0.945 (0.944–0.946) | 0.924 (0.921–0.930) | 0.945 (0.940–0.947) | 0.918 (0.911–0.925) |
| 32 | OOD | 0.822 (0.816–0.826) | 0.822 (0.816–0.826) | 0.819 (0.813–0.823) | 0.819 (0.813–0.823) | 0.812 (0.807–0.816) |

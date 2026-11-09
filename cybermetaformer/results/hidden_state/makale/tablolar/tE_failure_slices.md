| Policy | Silent slice | n | Root share | CMF | Host-token GRU, topology bias | Full-history GRU | CMF − GRU [95% CI] | p (Holm) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Periodic analyse | Attacker × subnet: B-line × enterprise | 25,410 | 0.88 | 0.990 | 0.991 | 0.974 | +0.016 [+0.014, +0.017] | 0.0029 |
|  | Attacker × subnet: B-line × user | 21,040 | 0.88 | 0.995 | 0.995 | 0.993 | +0.002 [+0.001, +0.004] | 0.0029 |
|  | Attacker × subnet: Meander × enterprise | 23,163 | 0.88 | 0.435 | 0.420 | 0.530 | −0.095 [−0.104, −0.085] | 0.0029 |
|  | Attacker × subnet: Meander × user | 58,763 | 0.90 | 0.409 | 0.413 | 0.550 | −0.140 [−0.146, −0.134] | 0.0029 |
|  | Alert age: never alerted (N_hist) | 17,500 | 0.89 | 0.663 | 0.675 | 0.708 | −0.045 [−0.053, −0.036] | 0.0029 |
|  | Alert age: 1–3 steps since last alert | 11,819 | 0.47 | 0.798 | 0.796 | 0.773 | +0.026 [+0.023, +0.028] | 0.0029 |
|  | Alert age: 4–7 steps since last alert | 23,345 | 0.92 | 0.648 | 0.646 | 0.686 | −0.038 [−0.042, −0.034] | 0.0029 |
|  | Alert age: 8–15 steps since last alert | 38,151 | 0.94 | 0.567 | 0.558 | 0.668 | −0.101 [−0.108, −0.095] | 0.0029 |
|  | Alert age: 16–31 steps since last alert | 30,105 | 0.95 | 0.574 | 0.574 | 0.707 | −0.133 [−0.141, −0.126] | 0.0029 |
|  | Alert age: 32–63 steps since last alert | 7,211 | 0.95 | 0.686 | 0.696 | 0.788 | −0.102 [−0.117, −0.087] | 0.0029 |
|  | Alert age: 64+ steps since last alert | 245 | 0.96 | 0.896 | 0.894 | 0.920 | — | — |
|  | Last alert: exploit, restored since | 101,080 | 0.89 | 0.638 | 0.634 | 0.704 | −0.066 [−0.070, −0.062] | 0.0029 |
|  | Last alert: root flag (no activity), restored since | 8,086 | 0.92 | 0.424 | 0.444 | 0.693 | −0.269 [−0.290, −0.249] | 0.0029 |
|  | Last alert: other, restored since | 1,710 | 0.91 | 0.407 | 0.382 | 0.616 | −0.208 [−0.246, −0.171] | 0.0029 |
| Remove instead of restore | Attacker × subnet: B-line × enterprise | 366,847 | 1.00 | 0.977 | 0.975 | 0.974 | +0.003 [+0.002, +0.005] | 0.0068 |
|  | Attacker × subnet: B-line × operational | 173,023 | 1.00 | 1.000 | 1.000 | 1.000 | +0.000 [+0.000, +0.000] | 1.0000 |
|  | Attacker × subnet: B-line × user | 194,109 | 1.00 | 0.967 | 0.952 | 0.967 | −0.000 [−0.002, +0.002] | 1.0000 |
|  | Attacker × subnet: Meander × enterprise | 447,557 | 1.00 | 0.952 | 0.962 | 0.943 | +0.009 [+0.008, +0.011] | 0.0029 |
|  | Attacker × subnet: Meander × operational | 148,307 | 1.00 | 1.000 | 1.000 | 1.000 | +0.000 [+0.000, +0.000] | 0.6661 |
|  | Attacker × subnet: Meander × user | 627,539 | 1.00 | 0.930 | 0.954 | 0.895 | +0.036 [+0.034, +0.038] | 0.0029 |
|  | Alert age: never alerted (N_hist) | 164,396 | 0.99 | 0.516 | 0.609 | 0.440 | +0.076 [+0.068, +0.085] | 0.0029 |

Note. Silent recall within each slice, seed mean over seeds 0–4. n: silent host-steps of the slice; root share: share of true root among them. Subnets: enterprise = ent0–2 (and def, never compromised), user = user1–4, operational = ophost0–2 and opserv. Alert age: steps since the last alert on the host; never alerted = N_hist. Last alert: type of the last alert (exploit activity or a root flag without activity, the form re-emitted by analyse) and whether the host was restored after it. In the remove policy only the attacker × subnet and never-alerted slices are shown; the other slices have recall ≥ 0.99 for all models. CMF − GRU: paired episode bootstrap of the seed mean (100,000 resamples, Algorithm 2), Holm over all m = 146 rows of this analysis (smallest attainable adjusted p = 0.0029); slices with fewer than 1,000 host-steps are not tested (—).

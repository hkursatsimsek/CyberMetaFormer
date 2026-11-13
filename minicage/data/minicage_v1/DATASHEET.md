# MiniCAGE Autonomous Cyber Defence Dataset (minicage_v1)

> **In one paragraph:** This dataset consists of **24 000 simulated attack episodes** (**2 400 000 time
> steps** in total) in which an attacker (Red) moves through a small enterprise network while a defender (Blue) tries to
> stop it. At every step the defender's **partial and noisy observation** is recorded together with the **hidden true
> state** known to the simulator (which machine is compromised and at which privilege level). Perfect labels, which are
> very expensive in the real world, are thus obtained for free. From the same raw record, tasks are derived for
> supervised learning (attack detection, early warning, attacker modelling), sequence modelling, offline reinforcement
> learning and distribution-shift studies.

| At a glance | |
|---|---|
| Source | CybORG++ / MiniCAGE simulator (CAGE Challenge 2 scenario) |
| Size | 24 000 episodes × 100 steps = 2 400 000 transitions; 33 MB (compressed) |
| Learning types | multi-label classification, binary classification, sequence classification, offline RL / imitation, dynamics modelling |
| Label quality | exact true state from the simulator (no human labelling) |
| Splits | train 16 000, val 2 000, test 2 000, OOD test 4 000 episodes |
| Generation | ≈23 s on a single CPU core; fully reproducible (fixed seed) |
| Format | NumPy `.npz` (raw), `csv.gz` (tabular), Python loaders (`minicage/dataset/loaders.py`) |

---

## 1. Motivation

**Autonomous cyber defence** aims to develop learning agents that defend a network against attacks without (or with
reduced) human intervention. **CAGE Challenge 2** and its simulator **CybORG** are widely used as a common test bed in
this area. **CybORG++** (Emerson et al., 2024) provides a debugged version of this simulator and **MiniCAGE**, a
vectorized re-implementation that is ~1000 times faster.

Working with real intrusion detection (IDS) datasets has three basic problems:

1. **Labels are expensive and noisy.** Whether a machine was really compromised is often not known for certain.
2. **The attacker's progress is invisible.** Records contain individual events; there is no information on which stage
   the attack is in.
3. **The data are static.** The defender's actions do not affect future data, so decision making (RL) cannot be studied.

This simulation-generated dataset overcomes all three: the simulator knows the true state (exact labels), every step
of the attack is recorded, and the defender's actions change future states (interactive data). The price is
**realism**: the network is small and abstract, and the attackers are scripted (see §10). For a comparison with other
datasets and studies in the field, see §9.

**Formal framing.** The problem can be viewed as a partially observable Markov decision process (POMDP): at each step t
there is a hidden state `s_t` (the true compromise state of the 13 machines), the defender's observation `o_t`
(78 dimensions), a defender action `a_t` (53 options), an attacker action and a reward `r_t`. The distinguishing feature
of this dataset is that `s_t`, normally hidden, is recorded as well.

---

## 2. Simulation environment

### 2.1 Network

13 machines (hosts) are divided into three subnets:

| Subnet | Hosts | Role |
|---|---|---|
| User | `user0` … `user4` | Where the attack starts. `user0` is the attacker's permanent foothold; the defender cannot act on it. |
| Enterprise | `ent0`, `ent1`, `ent2`, `def` | Intermediate layer. `def` is the defender's own machine and is never compromised. |
| Operational | `ophost0` … `ophost2`, `opserv` | **`opserv` (operational server) is the critical asset**, the attacker's final target. |

Attack path (MiniCAGE links):

```
user0 (Red foothold) ─► user1/user2 ─► ent0 ─┐
                     └► user3/user4 ─► ent1 ─┴─► ent2 ─► opserv ─► IMPACT (service disruption)
```

### 2.2 Agents

| Agent | Role | In this dataset |
|---|---|---|
| **Red** (attacker) | Advances through a discovery → exploit → privilege escalation (escalate) chain and applies impact once it reaches `opserv`. | Two scripted agents: **B-line** (takes the shortest path to the target) and **Meander** (spreads slowly, exploring every subnet). |
| **Blue** (defender) | Analyses hosts, deploys decoy services, removes the attacker or restores the host. | Six behaviour policies (§3.1). |
| **Green** (normal user) | Scans a random host with probability 50% at every step. | Produces **false-alarm** noise in the observation; not recorded separately. |

Red and Blue move **simultaneously**: both decide on the same state. The defender is therefore always one step behind.
Conflicting moves follow priority rules (e.g. exploit and restore in the same step → exploit wins).

### 2.3 Actions

**Blue (53 actions):** `host = (a − 1) mod 13`

| Index | Action | Effect |
|---|---|---|
| 0 | `sleep` | Do nothing |
| 1–13 | `analyse_<host>` | Reflects in the observation whether the attacker has **root (administrator) privileges** on the host |
| 14–26 | `decoy_<host>` | Deploys a decoy service on the host; an attacker trying to exploit that service fails |
| 27–39 | `remove_<host>` | Removes user-level attacker access (no effect on root) |
| 40–52 | `restore_<host>` | Restores the host to a clean state (−1 reward cost per use) |

**Red (56 actions):** `sleep`, 3 subnet discoveries (`remote_sub*`) and, for each host, `network` (service scan),
`exploit`, `escalate`, `impact` (13 × 4). The names are in `meta.json → action_names`.

### 2.4 Defender observation (78 dimensions)

| Dimensions | Content | Values |
|---|---|---|
| 0–51 | 4 bits per host: **activity** [scan, exploit] + **safety** state | safety: `[0,0]` clean/unknown, `[0,1]` exploited, `[1,1]` root (seen only with `analyse`), `[1,0]` remove applied |
| 52–64 | Scan history per host | 0 never, 1 earlier, 2 this step |
| 65–77 | Remaining decoys per host | 0–4 |

**Noise sources** that separate the observation from the true state: exploits are visible only with probability 95%;
Green scans produce spurious scan activity; root privilege is invisible without `analyse`; and (§5.3) alert flags can
disappear while the attack is ongoing.

### 2.5 Reward (Blue perspective, per step)

| State | Reward |
|---|---|
| Each compromised user / operational host (except `opserv`; on user hosts only some exploit types) | −0.1 |
| Each compromised enterprise host and `opserv` | −1 |
| `opserv` under impact | −10 |
| Each use of `restore` | −1 |

The penalty is repeated **at every step** while the compromise lasts. Red's reward is the negative of Blue's (zero-sum
game).

### 2.6 An example episode

B-line attacker × React-Restore defender, first 15 steps (episode #2000). "True state" is the label the model cannot see; "Alerts" are the hosts with the exploit flag on in the defender's observation. ✓/✗ shows whether the attacker's action succeeded.

| Step | Red action | Blue action | Reward | True state (hidden) | Alerts (observation) |
|---|---|---|---|---|---|
| 0 | `remote_sub2` ✓ | `sleep` | 0.0 | — | — |
| 1 | `network_user1` ✓ | `sleep` | 0.0 | — | — |
| 2 | `exploit_user1` ✓ | `sleep` | 0.0 | user1(user) | user1 |
| 3 | `escalate_user1` ✓ | `restore_user1` | -1.0 | — | — |
| 4 | `exploit_user1` ✓ | `sleep` | -0.1 | user1(user) | — |
| 5 | `escalate_user1` ✓ | `sleep` | -0.1 | user1(root) | — |
| 6 | `network_ent0` ✓ | `sleep` | -0.1 | user1(root) | — |
| 7 | `exploit_ent0` ✓ | `sleep` | -0.1 | ent0(user), user1(root) | ent0 |
| 8 | `escalate_ent0` ✓ | `restore_ent0` | -1.1 | user1(root) | — |
| 9 | `exploit_ent0` ✓ | `sleep` | -0.1 | ent0(user), user1(root) | ent0 |
| 10 | `escalate_ent0` ✓ | `restore_ent0` | -1.1 | user1(root) | — |
| 11 | `exploit_ent0` ✓ | `sleep` | -0.1 | ent0(user), user1(root) | ent0 |
| 12 | `escalate_ent0` ✓ | `restore_ent0` | -1.1 | user1(root) | — |
| 13 | `exploit_ent0` ✓ | `sleep` | -0.1 | ent0(user), user1(root) | — |
| 14 | `escalate_ent0` ✓ | `sleep` | -1.1 | ent0(root), user1(root) | — |

Two basic properties of the dataset can be seen in this table: (1) The defender restores the host on which it sees an
alert, and the attacker exploits the same host again. This "cat-and-mouse" loop can last for the whole episode.
(2) At some steps the true state contains a compromise but the observation has no alert (e.g. hosts re-compromised after
a restore). This is the core difficulty detection models have to solve (§5.3).

---

## 3. Data generation

### 3.1 Behaviour policies

For data diversity, each attacker was paired with six defender policies of different quality. This approach resembles
the "random / medium / expert" data mixture of offline RL benchmarks (e.g. D4RL).

| Blue policy | Behaviour | Why it is included |
|---|---|---|
| `sleep` | Does nothing | Undefended reference; the whole attack chain is observed |
| `react_restore` | Restores the host on which it sees an exploit alert | Reactive, simple defence |
| `restore_decoys` | `react_restore` + deploys decoys on critical hosts when idle | Best defence in this set ("expert" role) |
| `random` | Chooses at random from the 53 actions | Covers the action space broadly (important for offline RL), low quality |
| `react_restore_eps0.1` | Random action with probability 10% | Noisy expert |
| `react_restore_eps0.3` | Random action with probability 30% | Noisier expert; **reserved for the out-of-distribution (OOD) test** |

Grid: 2 Red × 6 Blue × 2000 episodes, each episode 100 steps, `remove_bugs=True`
(the MiniCAGE mode that does not emulate the original CAGE 2 bugs). Each (Red, Blue) combination was generated in
1000-episode "shards", each shard with its own seed.

### 3.2 Splits (train / val / test)

The split is made at **episode level**: steps of the same episode never fall into different sets (no temporal leakage).
Train/val/test are stratified by (Red, Blue) combination (80%/10%/10%). All episodes of the `react_restore_eps0.3`
policy are only in the `ood_test` set, so that "generalization to a defender behaviour never seen in training" can be
measured.

| Set | Episodes | Steps |
|---|---|---|
| train | 16 000 | 1 600 000 |
| val | 2 000 | 200 000 |
| test | 2 000 | 200 000 |
| OOD test | 4 000 | 400 000 |

---

## 4. Data structure

### 4.1 Raw fields (episode-major: E episodes, T steps)

| Field | Shape | Description |
|---|---|---|
| `obs_blue` | (E, T+1, 78) | Blue observation (§2.4) |
| `obs_red` | (E, T+1, 40) | Red observation: success flag + 13 hosts × [scanned, user, root] |
| `true_state` | (E, T+1, 39) | **Hidden true state**: 13 hosts × [scanned, user access, root access] (−1 = Red does not know yet) |
| `act_blue` / `act_red` | (E, T) | Chosen actions (53 / 56) |
| `rew_blue` | (E, T) | Blue reward |
| `blue_success` / `red_success` | (E, T) | Whether the action succeeded (−1 = not applicable) |
| `impacted` | (E, T, 13) | Impact state (only `opserv` affects the reward) |
| `decoys_left` | (E, T+1, 13) | Remaining decoys per host |
| `red_policy`, `blue_policy`, `epsilon`, `remove_bugs`, `seed` | (E,) | Episode metadata |

**Time alignment:** `obs_*[:, t]` and `true_state[:, t]` are the state *after* step t (t = 0 is the start).
`act_*[:, t]` is the action chosen on seeing this state, and `rew_blue[:, t]` is its result.

### 4.2 Derived labels

- **Host state:** 3 classes per host from `true_state`: **0 clean**, **1 user access**, **2 root access**.
  `user0` (the attacker's permanent foothold) is excluded from the labels → 12 hosts.
- **Any compromise in the network:** 1 if at least one of the 12 hosts is not clean.
- **Red action type:** 56 actions → 6 types (sleep, remote, network, exploit, escalate, impact).
- **Early warning:** whether `opserv` will be impacted within the next h steps.

### 4.3 Files

| File | Content |
|---|---|
| `shard_###.npz` | Raw record, one (Red, Blue) combination × 1000 episodes |
| `meta.json` | Action names, host list, observation layout, shard list, versions |
| `splits.npz` | Episode indices: `train`, `val`, `test`, `ood_test` |
| `tabular.csv.gz` | One readable row per step (2 400 000 rows): action names, reward, compromised host counts, per-host alert and true label |
| `baselines.json` | Reference model results (§8) |

### 4.4 Usage

```python
from minicage.dataset import loaders as L

raw, meta = L.load_raw('minicage/data/minicage_v1')   # all fields, ~0.5 GB RAM
sp = L.load_splits('minicage/data/minicage_v1')

X, y, y_any = L.detection_view(raw, sp['train'], k=4)     # detection: (N, 312), (N, 12), (N,)
X, y = L.risk_view(raw, sp['train'], horizon=10)          # early warning
X, y = L.attacker_id_view(raw, sp['train'], n_steps=10)   # attacker recognition: (E, 10, 78)
D = L.d4rl_view(raw, sp['train'])                         # offline RL (D4RL dictionary)
D = L.dynamics_view(raw, sp['train'])                     # world model
```

For the tabular form (with pandas): `pd.read_csv('tabular.csv.gz')`.

---

## 5. Dataset statistics

### 5.1 Attacker × defender combinations

| Red | Blue | Episodes | Episode reward (mean ± SE) | Impact % | First impact (step) | Last compromise (host) |
|---|---|---|---|---|---|---|
| bline | sleep | 2 000 | -1134.6 ± 0.4 | 100.0 | 14 | 4.00 |
| bline | react_restore | 2 000 | -125.4 ± 1.2 | 0.0 | — | 2.68 |
| bline | restore_decoys | 2 000 | -54.6 ± 0.4 | 0.0 | — | 0.93 |
| bline | random | 2 000 | -828.1 ± 7.4 | 82.2 | 14 | 1.51 |
| bline | react_restore_eps0.1 | 2 000 | -124.1 ± 2.0 | 26.3 | 72 | 1.69 |
| bline | react_restore_eps0.3 | 2 000 | -178.7 ± 3.5 | 84.0 | 41 | 0.98 |
| meander | sleep | 2 000 | -1061.4 ± 0.6 | 100.0 | 28 | 7.32 |
| meander | react_restore | 2 000 | -117.1 ± 1.1 | 0.0 | — | 4.28 |
| meander | restore_decoys | 2 000 | -85.1 ± 0.8 | 0.0 | — | 4.15 |
| meander | random | 2 000 | -749.3 ± 2.7 | 99.9 | 33 | 4.35 |
| meander | react_restore_eps0.1 | 2 000 | -106.5 ± 1.2 | 5.1 | 71 | 3.38 |
| meander | react_restore_eps0.3 | 2 000 | -107.9 ± 1.6 | 31.8 | 66 | 2.48 |

*Impact %: share of episodes in which `opserv` is impacted at least once. First impact: median step. Last compromise:
number of hosts compromised at the end of the episode (mean, excluding `user0`).*

### 5.2 Label distribution (share of step × host)

| Host | clean | user | root |
|---|---|---|---|
| `def` | 1.000 | 0.000 | 0.000 |
| `ent0` | 0.729 | 0.039 | 0.232 |
| `ent1` | 0.705 | 0.031 | 0.264 |
| `ent2` | 0.679 | 0.028 | 0.293 |
| `ophost0` | 0.969 | 0.004 | 0.027 |
| `ophost1` | 0.969 | 0.004 | 0.027 |
| `ophost2` | 0.968 | 0.004 | 0.028 |
| `opserv` | 0.677 | 0.067 | 0.256 |
| `user1` | 0.708 | 0.031 | 0.261 |
| `user2` | 0.711 | 0.030 | 0.259 |
| `user3` | 0.708 | 0.032 | 0.260 |
| `user4` | 0.708 | 0.032 | 0.259 |

Share of steps with at least one compromise in the network: **0.851**. The classes are imbalanced: `ophost*`
are almost always clean, and the "user access" class is a short-lived intermediate state. Macro-F1 or class weighting is
recommended. `def` is never compromised (constant 0).

### 5.3 How well do the alerts reflect the true state?

Treating the "exploited" flag in the observation as a simple alert:

- Across the episodes a host was compromised **138 334** times. Of these, **94.2%** produced an alert at least once. The alert usually comes at the same step (median delay 0 steps, mean 0.59 steps).
- Alert rate at the step where a compromise **starts**: 91.7% on first compromise, 93.0% on re-compromise after a restore.
- **Per step** the picture differs: an alert means a certain compromise (precision 1.000), but an alert is on in only **0.576** of the compromised steps (recall).
- In `react_restore` episodes, in **79%** of the alert-free compromise steps the host had been restored before.

Change of per-step recall with the defender policy:

| Blue policy | Alert recall |
|---|---|
| `sleep` | 0.913 |
| `react_restore` | 0.194 |
| `restore_decoys` | 0.248 |
| `random` | 0.709 |
| `react_restore_eps0.1` | 0.264 |
| `react_restore_eps0.3` | 0.406 |

**Interpretation:** Alerts capture the *onset* of a compromise well, but when the defender is active (especially on hosts
re-compromised after a restore) the flag disappears while the compromise persists. So **the current observation is not
enough to determine the true state; models that remember the history are needed.** This relation also depends on the
defender policy, which makes the detection task sensitive to policy changes (§6.6). This fact is measured from the
data; its cause inside the simulator has not been examined in detail.

---

## 6. What can be done with this dataset?

All tasks below are derived from the same raw record. The "Ready" column shows whether a loader and a reference result
exist.

| # | Task | ML type | Difficulty | Ready |
|---|---|---|---|---|
| 6.1 | Attack detection / hidden-state estimation | Multi-label classification, sequence modelling | Medium | loader + result |
| 6.2 | Early warning for OpServer | Imbalanced binary classification | Medium–hard (hard under OOD) | loader + result |
| 6.3 | Attacker modelling | Sequence classification, next-action prediction | Currently easy | loader + result |
| 6.4 | Offline RL and imitation learning | Offline RL, behaviour cloning | Medium–hard | loader + BC result |
| 6.5 | World model / dynamics learning | Multi-output regression / classification | Medium | loader |
| 6.6 | Distribution shift and generalization | Domain generalization, OOD detection | Hard | split + results |
| 6.7 | Tabular data and interpretability | Gradient boosting, SHAP, EDA | Easy–medium | CSV |
| 6.8 | Label efficiency / semi-supervised learning | Learning with few labels, active learning | Medium | raw data |

### 6.1 Attack detection and hidden-state estimation

- **Problem:** From the defender's last k observations (`k × 78` features), predict the true state of every host
  {clean, user, root}. Binary version: "is there any compromise in the network?"
- **Why it is interesting:** It resembles the classic IDS problem, but the labels are exact. The observation is noisy and
  incomplete (§5.3), and root privilege is not directly visible. In this respect it is a **belief-state estimation**
  problem in a POMDP.
- **Suggested methods:** MLP (windowed input), GRU/LSTM/Transformer (full history), a graph neural network for
  cross-host relations (a natural candidate since the network topology is known), comparison with probabilistic
  filtering (HMM / particle filter). A particle-filter-based belief-state approach for CAGE-2 has been proposed by
  Le & Stadler (2025) (§9.3).
- **Metrics:** Per-host macro-F1, network-level F1, detection delay (steps from the compromise to the first correct
  prediction).
- **Reference result** (k = number of last observations seen by the model):

| Model | Host macro-F1 (test) | Host macro-F1 (OOD) | Compromise F1 (test) | Compromise F1 (OOD) |
|---|---|---|---|---|
| Majority class | 0.285 | 0.311 | — | — |
| Alert rule | — | — | 0.833 | 0.802 |
| MLP, k=1 | 0.910 | 0.808 | 0.940 | 0.889 |
| MLP, k=4 | 0.920 | 0.810 | 0.961 | 0.900 |
| MLP, k=8 | 0.929 | 0.816 | 0.969 | 0.899 |

  Compared with the simple alert rule (F1 0.833), the learned model is clearly better. Enlarging the history
  window (k = 1 → 8) raises the network-level F1 on the test set from 0.940 to 0.969; under the unseen policy
  the performance drops to 0.899.
- **Open questions:** Can sequence models using the long history solve the lost-flag problem? Can calibrated
  probabilities be obtained (important for the defender's decision making)?

### 6.2 Early warning: raising the alarm before the critical server falls

- **Problem:** From the observations so far, will `opserv` be impacted within the next 10 steps?
  (Steps already under impact are excluded; positive rate ≈ 0.078.)
- **Why it is interesting:** It is an imbalanced, forward-looking (prognostic) classification problem. In practice the
  most valuable question is not "what happened now" but "what will happen soon".
- **Metrics:** AUC, F1, recall at a given false-alarm rate, how far in advance the warning is given.
- **Reference result:**

| Model | F1 (test) | AUC (test) | F1 (OOD) | AUC (OOD) |
|---|---|---|---|---|
| Alert rule (`ent2`/`opserv` alert) | 0.360 | — | 0.385 | — |
| MLP, k=4 | 0.796 | 0.983 | 0.198 | 0.753 |

  On the training distribution the model is very good (AUC 0.983), but **under the defender policy unseen in
  training, AUC drops to 0.753 and F1 from 0.796 to 0.198**, falling behind even the
  simple rule. This shows that the data are a good test bed for distribution-shift studies (§6.6).

### 6.3 Attacker modelling

- **Problems:** (a) Recognize the attacker type (B-line / Meander) from the observation sequence of the first n steps.
  (b) Predict the attacker's next action type from the observation history.
- **Reference result:**

(a) Attacker recognition, accuracy:

| Model | test | OOD test |
|---|---|---|
| Chance | 0.500 | 0.500 |
| GRU, first 3 steps | 0.998 | 0.995 |
| GRU, first 5 steps | 1.000 | 1.000 |
| GRU, first 10 steps | 1.000 | 0.999 |

(b) Red action type (6 classes) prediction, accuracy:

| Model | test | OOD test |
|---|---|---|
| Majority class | 0.326 | 0.417 |
| MLP, k=4 | 0.969 | 0.949 |

- **Assessment:** Both tasks are **easy in their current form**, because both attackers are deterministic scripts and
  behave differently within the first few steps. To make the tasks meaningful, noisy (ε) attackers, mixed strategies or
  learning (PPO) attackers should be added to the generation (§11). As it stands, it can serve as a "sanity check" for
  sequence models. For the direct CAGE-2 counterpart of Red action prediction, see Samaddar et al. (2026), §9.3.

### 6.4 Offline reinforcement learning and imitation learning

- **Problem:** Learn a good defence policy only from recorded transitions (`observation, action, reward, next
  observation`), without interacting with the environment.
- **Why it is interesting:** Learning a defence by trial and error on real networks is unacceptable, so offline RL is
  important in practice. The dataset contains a mixture of policies of different quality (random → expert). Moreover,
  **every learned policy can be evaluated exactly in the simulator**. This is an advantage not possible with real
  datasets, and it makes it possible to compare the estimates of **off-policy evaluation (OPE)** methods with the true
  value.
- **Suggested methods:** Behaviour cloning (BC), CQL, IQL, BCQ, Decision Transformer; data-quality ablation (`random`
  only / expert only / mixed); OPE (FQE, importance sampling).
- **Reference result (BC, with `restore_decoys` data, 500 episodes in the environment):**

| Attacker | Teacher (`restore_decoys`) | BC policy |
|---|---|---|
| bline | -55.6 | -56.4 |
| meander | -85.0 | -87.8 |

  BC imitates its teacher almost exactly. Open question: can offline RL **surpass** the best policy in the dataset? This
  can be tested with mixed-quality data (e.g. `random` + `react_restore`).

### 6.5 World model / dynamics learning

- **Problem:** Predict (observation, true state, action) → (next observation, next true state, reward).
  Loader: `dynamics_view`.
- **Why it is interesting:** It is a basic component of model-based RL and planning. Since the true state is available,
  the "hidden-state transition" and the "observation model" can be learned separately. A learned model can replace the
  simulator (surrogate simulator).
- **Metrics:** One-step accuracy, error accumulation in multi-step rollouts, real-simulator performance of a policy
  trained in the learned model.

### 6.6 Distribution shift and generalization

- **Ready shifts:** (1) `ood_test`: defender policy unseen in training. (2) Attacker shift: train only with B-line and
  test on Meander. (3) Simulator version: a second dataset generated with `--remove-bugs false`. (4) Cross-simulator: a
  small test set that could be generated from the full CAGE 2 simulator for the same scenario (not generated yet).
- **Finding:** The OOD drop is moderate in the detection task but very pronounced in the early-warning task (§6.1,
  §6.2). The literature has also shown that CAGE agents struggle to generalize to new networks and attackers (Wolk et al.,
  2022; §9.3).
- **Methods that can be studied:** Domain generalization, policy-independent representation learning, uncertainty
  estimation and OOD detection (can the model know when it is wrong?).

### 6.7 Tabular data and interpretability

`tabular.csv.gz` gives every step with readable columns: policy names, action names (such as `exploit_ent1`), success
flags, reward and cumulative reward, compromised host counts, `opserv` impact state, per-host alert and true label. It
is suitable for detection with gradient boosting (XGBoost / LightGBM), for analysing with SHAP which observation
features drive the decision, and for exploratory data analysis (EDA). It is a good starting point for undergraduate
projects.

### 6.8 Label efficiency and semi-supervised learning

Labels are free here but expensive in the real world. Using only a small part of the labels (e.g. 1% of the episodes),
semi-supervised, self-supervised pre-training or active learning methods can be tried. Comparing with the fully labelled
result measures the question "how many labels are enough?".

### Suggested research questions

1. How much do sequence models (GRU/Transformer) gain over a windowed MLP in detection, especially on alerts lost after
   a restore?
2. Does a graph neural network that uses the network topology beat a flat MLP in per-host detection?
3. Which methods (data augmentation, policy randomization, domain generalization) can reduce the OOD collapse of the
   early-warning model?
4. Can offline RL extract, from mixed-quality data, a policy better than the best defender in the dataset?
5. How close are the estimates of OPE methods to the true policy value measured in the simulator?

---

## 7. Validation

Checks made to show that the dataset was generated correctly (`validate.py`):

**7.1 Reward bug in the source code (fixed).** MiniCAGE's `SimplifiedCAGE.step()` computes the reward twice, and because
the computation function modifies the reward dictionary in place, the state penalty is subtracted twice. For
Meander × Sleep the repository code gives **-2119.8** and the fixed version **-1059.9** (published value
−1067). The dataset was generated with the fixed version (`minicage/dataset/env.py::FixedCAGE`); the original repository
files were not modified.

**7.2 Attacker agent.** The B-line agent exported by the package (`test_agent.B_line_minimal`) gets stuck on the same
action after a few steps (B-line × Sleep ≈ −10, published value −1141). The working version in the repository
(`red_bline_agent.B_line_minimal`) was used instead.

**7.3 Consistency with published results** (`remove_bugs=False`, 500 episodes × 100 steps, MiniCAGE README table):

| Red | Blue | Published | This work (mean ± SE) | Difference |
|---|---|---|---|---|
| bline | react_restore | -156 | -89.6 ± 2.3 | 42.6% |
| bline | restore_decoys | -68 | -47.9 ± 0.3 | 29.5% |
| bline | sleep | -1141 | -1134.6 ± 0.8 | 0.6% |
| meander | react_restore | -68 | -93.5 ± 1.8 | 37.5% |
| meander | restore_decoys | -63 | -66.2 ± 1.3 | 5.1% |
| meander | sleep | -1067 | -1061.4 ± 1.1 | 0.5% |

In the undefended (`sleep`) scenarios and in Meander × Restore-Decoys, the results agree with the published values within
0.5–5%; the environment dynamics and the reward computation work correctly. The React-Restore rows, however, differ
clearly. The repository's `React_restore_minimal` implementation appears to differ from the one that produced the
published table. This affects only the score of that defence policy, not the environment or the labels.

**7.4 Label consistency:**

| Check | Expected | Measured |
|---|---|---|
| Share of B-line × Sleep episodes ending with a root label on `opserv` | 1.000 | 1.000 |
| Share of steps with reward ≤ −10 while `opserv` is under impact | 1.000 | 1.000 |
| Share of steps with a root label on `opserv` while `opserv` is under impact | 1.000 | 1.000 |
| Share of hosts labelled clean after a successful restore (745 483 restores) | 1.000 | 1.000 |
| `react_restore`: share of non-React-Restore actions (expected ε·39/53) | 0.000 | 0.000 |
| `react_restore_eps0.1`: share of non-React-Restore actions (expected ε·39/53) | 0.074 | 0.073 |
| `react_restore_eps0.3`: share of non-React-Restore actions (expected ε·39/53) | 0.221 | 0.220 |

**7.5 Reproducibility:** Shards regenerated with the same seed are identical in all fields: `shard_000.npz` ✓, `shard_023.npz` ✓.

---

## 8. Experimental details of the reference results

The reference models were trained only to show that the data are learnable and to provide a baseline for comparison; no
hyperparameter search was made. All of them run on the CPU in ≈1 minute in total (`baselines.py`).

| Task | Model | Training data | Training |
|---|---|---|---|
| Detection, early warning, Red action type | MLP (2 × 256, ReLU) | 4000 random episodes from train (6000 for early warning) | Adam, lr 1e-3, 5 epochs, batch 1024 |
| Attacker recognition | GRU (64 hidden units) | All of train (16 000 episodes) | Adam, lr 1e-3, 10 epochs, batch 256 |
| Behaviour cloning | MLP (2 × 256) | `restore_decoys` episodes in train | 10 epochs; evaluated with 500 episodes in the environment |

"Simple reference" columns: **majority class** (always predict the most frequent class), **alert rule** (say there is a
compromise if the observation has the exploit flag; for early warning, say an impact is coming if there is an alert on
`ent2` or `opserv`), **chance**.

All results are in raw form in `baselines.json`.

---

## 9. Related work and positioning

*This section is based on a literature search made in October 2026; it is not a comprehensive review.*

### 9.1 Classic intrusion detection (IDS) datasets

The datasets commonly used in machine-learning intrusion detection studies consist of network traffic or system logs:

| Dataset | Content | Difference from this dataset |
|---|---|---|
| NSL-KDD, UNSW-NB15, CIC-IDS2017, CSE-CIC-IDS2018 | Network flow features, attack label per flow | Static; no attack stage or hidden state; no defender action |
| DARPA TC / OpTC | System events in an enterprise network (OpTC: ~1000 Windows hosts), multi-stage APT scenarios [16] | Realistic but static; labelling is partial and requires interpretation |
| AIT Log Data Set | System logs from a test environment, attack-step label per line [15] | Has attack-step labels; no interaction |
| DAPT 2020, Unraveled | Network flows and system logs; APT stage labels (partial in DAPT 2020, semi-synthetic and comprehensive in Unraveled) [16, 18] | No interaction; defender decisions cannot be studied |

**Label quality** is a documented problem in the most widely used of these sets. Engelen et al. showed that more than 25%
of the flows in CIC-IDS2017 (up to 50% for some attack types) are meaningless artefacts and that there are labelling
errors [12]. Similar errors were reported for CSE-CIC-IDS2018, and these errors were shown to change detection
performance markedly [13]. Arp et al. documented that pitfalls such as sampling bias and data leakage are common in
machine-learning studies in security [14].

This dataset approaches these problems from a different angle: the labels come from the simulator's own state and are
therefore exact, and the data are interactive. But since it contains no packets or logs, **it cannot be compared directly
with these sets** and does not replace them.

### 9.2 Autonomous cyber defence: environments are shared, not datasets

In autonomous cyber defence, researchers share **simulation environments** (gyms) rather than datasets: CybORG and CAGE
Challenge 2/3/4, CybORG++ / MiniCAGE [1], Yawning Titan, PrimAITE, CyberBattleSim, NASim and Cyberwheel [2, 10].
Results are usually reported as the episode reward of an agent trained in these environments.

- **The common ground is CAGE-2.** Le & Stadler describe CAGE-2 as "an accepted benchmark for learning and evaluating
  defence strategies" [6]. Kiely et al. classified the methods of CAGE-2 participants and found hierarchical deep RL to
  be the most successful approach [9].
- **The field is fragmented.** Hicks et al. (2026) note that there is no single standard benchmark and that the
  environments require substantial customization. They also criticize existing environments on three points: "magic
  observations" that put information into the observation that could not be known in reality, attackers with fixed
  strategies and turn-based interaction. They mention CybORG++ / MiniCAGE in this context as well [2].

**Trajectory datasets.** In the search, the work closest to this dataset is **Trident** (Masukawa et al., 2026): more
than 13 000 attacker–defender trajectories from the CAGE 4 and Cyberwheel environments [4]. Its purpose is different,
however: the trajectories are in natural-language log form and are designed for training LLM-based attacker agents. At
the time of writing, its code is only in an anonymous reviewer repository. Training defence agents with offline RL has
also been studied in the CyberVAN environment [11]; no information was found that the data of that study are open.

**Not found:** The search found no **open** dataset generated from the CAGE-2 scenario, with per-step hidden-state labels,
prepared for detection, attacker modelling and offline RL.

### 9.3 Work related to the tasks in this dataset

Most tasks have been studied separately on CAGE-2, but these studies generate their own data and do not publish them:

| Task (in this document) | Related work |
|---|---|
| Red action prediction (§6.3) | Samaddar et al. (2026) predict Red's action on CAGE-2 from Blue observations and Blue actions with imitation learning. They also use an attacker that switches between B-line and Meander within an episode (RedSwitch) [5]. |
| Attacker recognition (§6.3) | The winning CAGE-2 solutions first recognize the attacker type and then choose a policy specific to it [3, 9]. This shows that the task is used in practice and is solved easily in the first steps. |
| Hidden-state estimation / detection (§6.1) | Le & Stadler (2025) model CAGE-2 as a POMDP, keep a belief state with a particle filter and surpass the leading CARDIFF solution [6]. Hammar & Li (2025) perform belief-based response planning on CAGE-2 with Bayesian learning [7]. |
| Distribution shift (§6.6) | Wolk et al. (2022) show that CAGE agents degrade markedly on unseen networks and against new attacker strategies [8]. The OOD drop in the early-warning task of this dataset (§6.2) points in the same direction. |
| Offline RL (§6.4) | Defence-agent training with offline RL in the CyberVAN environment [11]. |

### 9.4 Positioning

**What this dataset is:** A benchmark dataset derived from the CAGE-2 scenario (via MiniCAGE) and prepared for supervised
and offline learning tasks. Its contribution is to provide a **common, open, exactly labelled and reproducible** data
basis for tasks that the literature studies separately and with unpublished data (detection / belief estimation,
attacker modelling, offline RL). The document format follows the "Datasheets for Datasets" proposal of Gebru et al. [17].

**What this dataset is not:** It is not an IDS dataset containing real network traffic and cannot be compared directly
with the sets in §9.1. MiniCAGE is not CAGE-2 itself but a fast re-implementation of it (the equivalence claim belongs to
[1]; a difference was observed in the React-Restore scores in §7.3).

**What is needed to bring it to publication level:**

1. Attacker diversity: an attacker that switches strategy within an episode (RedSwitch [5]), noisy and learning (PPO)
   attackers.
2. A small test set from the full CybORG simulator (cross-simulator transfer).
3. Direct comparisons: with [5] in Red action prediction, and with the particle-filter-based approach [6] in hidden-state
   estimation.
4. Running the reference models with more than one seed and reporting the variance [2].
5. Publication with a DOI via Zenodo or Hugging Face; evaluation following the checklist in [14].

### References

1. H. Emerson, L. Bates, C. Hicks, V. Mavroudis. *CybORG++: An Enhanced Gym for the Development of Autonomous Cyber Agents.* arXiv:2410.16324, 2024. https://arxiv.org/abs/2410.16324
2. C. Hicks, E. Bates, S. McFadden, I. Symes Thompson et al. *Building Better Environments for Autonomous Cyber Defence.* AI4CNI Workshop @ AAMAS 2026. arXiv:2604.08805. https://arxiv.org/abs/2604.08805
3. C. Hicks, V. Mavroudis. *Autonomous cyber defence: Beyond games?* Turing Technical Report No. 2, 2024. https://doi.org/10.5281/zenodo.10974183
4. R. Masukawa et al. *Trident: How to Break Deep Reinforcement Learning Cyber Defenses (Agentic).* arXiv:2608.04317, 2026. https://arxiv.org/abs/2608.04317
5. A. Samaddar, S. Neema, D. Balasubramanian, X. Koutsoukos. *Learning Red Agent Policy from Observations for Neurosymbolic Autonomous Cyber Agents.* arXiv:2606.18223, 2026. https://arxiv.org/abs/2606.18223
6. D. H. Le, R. Stadler. *Learning Optimal Defender Strategies for CAGE-2 using a POMDP Model.* CNSM 2025. arXiv:2509.06539. https://arxiv.org/abs/2509.06539
7. K. Hammar, T. Li. *Online Incident Response Planning under Model Misspecification through Bayesian Learning and Belief Quantization.* ACM CCS AISec 2025. arXiv:2508.14385. https://arxiv.org/abs/2508.14385
8. M. Wolk et al. *Beyond CAGE: Investigating Generalization of Learned Autonomous Network Defense Policies.* NeurIPS 2022 Workshop on RL for Real Life. arXiv:2211.15557. https://arxiv.org/abs/2211.15557
9. M. Kiely, D. Bowman, M. Standen, C. Moir. *On Autonomous Agents in a Cyber Defence Environment.* 2nd International Workshop on Adaptive Cyber Defence, 2023. arXiv:2309.07388. https://arxiv.org/abs/2309.07388
10. G. Palmer et al. *Deep Reinforcement Learning for Autonomous Cyber Defence: A Survey.* arXiv:2310.07745. https://arxiv.org/abs/2310.07745
11. *Offline Reinforcement Learning for Autonomous Cyber Defense Agents.* IEEE Xplore, document 10838812. https://ieeexplore.ieee.org/document/10838812/
12. G. Engelen, V. Rimmer, W. Joosen. *Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study.* IEEE Security and Privacy Workshops, 2021. https://lirias.kuleuven.be/retrieve/713070
13. *Errors in the CICIDS2017 dataset and the significant differences in detection performances it makes.* 2022. https://hal.archives-ouvertes.fr/hal-03775466 ; *Error Prevalence in NIDS datasets: A Case Study on CIC-IDS-2017 and CSE-CIC-IDS-2018.* https://archive.cps-vo.org/node/97449
14. D. Arp et al. *Dos and Don'ts of Machine Learning in Computer Security.* USENIX Security 2022. arXiv:2010.09470. https://arxiv.org/abs/2010.09470
15. *AIT Log Data Set.* Austrian Institute of Technology, Zenodo. https://zenodo.org/records/5789064
16. *SAGA: Synthetic Audit Log Generation for APT Campaigns* (comparison of existing APT datasets). arXiv:2411.13138. https://arxiv.org/abs/2411.13138
17. T. Gebru et al. *Datasheets for Datasets.* Communications of the ACM, 64(12), 2021.
18. S. Myneni et al. *DAPT 2020 — Constructing a Benchmark Dataset for Advanced Persistent Threats*, 2020; *Unraveled — A semi-synthetic dataset for Advanced Persistent Threats*, Computer Networks, 2023.

---

## 10. Limitations and caveats

- **These are simulation data.** MiniCAGE is an abstract model of the CAGE 2 scenario: there are no packets, logs or
  network traffic; the results cannot be generalized directly to real networks. The criticisms directed at such
  environments (magic observations, fixed attackers, turn-based interaction) apply to this dataset as well (§9.2).
- **Fixed and small topology** (13 hosts, 3 subnets).
- **Low attacker diversity:** two deterministic scripts. Attacker-modelling tasks are therefore easy (§6.3).
- **The defenders are scripted or random;** there is no trained (e.g. PPO) expert policy yet.
- **Temporal dependence:** steps of the same episode are not independent of each other. The episode-level split
  (`splits.npz`) must be respected in evaluation; steps must not be split by random shuffling.
- **Class imbalance:** see §5.2.
- **Green agent scans** are present as noise in the observation but are not recorded as a separate field.
- **Two bugs in the source code** are fixed in this dataset (§7.1, §7.2). This must be taken into account when making
  numerical comparisons with studies that use the original repository directly.

**Ethics:** The data are fully synthetic; they contain no personal data, real systems or records of real attacks. The
purpose is defence research.

---

## 11. Extension possibilities

The generation pipeline is parametric; the following can be added with a few lines of code and a few minutes of
generation:

- **Noisy / mixed attackers** (ε-B-line, an attacker that switches strategy within an episode; see RedSwitch, §9.3) →
  makes the attacker-modelling tasks harder.
- **A trained PPO defender** → "expert" data for offline RL.
- **A `remove_bugs=False` version** → distribution shift between simulator versions.
- **A small test set from the full CAGE 2 simulator** → simulation-to-simulation transfer.
- **Different episode lengths** (30 / 50 / 100 steps), to match the evaluation protocol of the CAGE Challenge.

---

## 12. Reproduction

In the `<repo>` directory, with the CybORG++ repository and the `venv_cyborg` environment:

```bash
venv_cyborg/bin/python -m minicage.dataset.generate --episodes-per-config 2000   # ≈23 s
venv_cyborg/bin/python -c "from minicage.dataset import loaders as L; L.export_tabular('minicage/data/minicage_v1')"
venv_cyborg/bin/python -m minicage.dataset.baselines                              # ≈1 min
venv_cyborg/bin/python -m minicage.dataset.validate                               # checks + this document
```

Dependencies: Python 3.12.3, NumPy 2.5.3, PyTorch (only for the reference models).

---

## 13. Licence and citation

The CybORG++ simulator is distributed under the MIT licence. Work using the dataset should cite the simulator:

```bibtex
@article{emerson2024cyborg++,
  title   = {Cyborg++: An enhanced gym for the development of autonomous cyber agents},
  author  = {Emerson, Harry and Bates, Liz and Hicks, Chris and Mavroudis, Vasilios},
  journal = {arXiv preprint arXiv:2410.16324},
  year    = {2024}
}
```

---

## Appendix: Terms

| Term | Meaning |
|---|---|
| Episode | A single 100-step attack–defence game starting from an environment reset |
| Exploit | Gaining user-level access to a host by using a vulnerability in a service |
| Escalate | Raising user access to root (administrator) access |
| Impact | Causing a service disruption on the critical server |
| Decoy | A fake service that defeats the attacker's exploit attempt |
| Restore | Restoring a host from a clean backup |
| POMDP | Partially observable Markov decision process: the agent sees not the true state but an observation derived from it |
| OOD | Out-of-distribution: conditions not represented in the training data |
| D4RL | A widely used dataset format and benchmark suite for offline RL |
| OPE | Off-policy evaluation: estimating the value of a policy from data of other policies |

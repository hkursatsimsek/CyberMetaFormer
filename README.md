# CyberMetaFormer

Code, data and stored results for **CyberMetaFormer: host-structured temporal hidden-state estimation for
autonomous cyber defence under silent compromise** (manuscript in preparation).

A defender in autonomous cyber defence never observes the true compromise state of the network; it only sees partial,
noisy alerts. This repository studies *supervised hidden-state estimation*: from the causal history of the 78-dimensional
defender observation of the CAGE Challenge 2 scenario (MiniCAGE re-implementation), estimate the state of each of the
12 label hosts as **clean / user / root** at every step. A large share of compromised host-steps carries no current
alert (*silent compromise*), and the out-of-distribution (OOD) test uses a defender policy unseen in training.

The repository contains:

- **`minicage/`**: the data generator (on top of CybORG++ MiniCAGE) and the `minicage_v1` dataset: 24,000 episodes,
  2.4 M steps, episode-level splits, plus the derived OOD packages (simulator-bug shift, defender-policy shift,
  rewired topology, corrected remove-flag observation).
- **`cybermetaformer/hidden_state/`**: CyberMetaFormer (host tokens, causal convolution + selective state-space
  temporal mixer, topology-biased cross-host attention), the reference models (rules, XGBoost, MLP, 1D-CNN, LSTM/GRU,
  full-history recurrent models, Transformer, PatchTST, iTransformer, TSMixer, TimesNet, TimeMixer), the shared
  training and evaluation protocol, hyperparameter search, paired episode-level bootstrap, ablations, OOD suite,
  host-permutation and topology shifts, interventions on other hosts' observations, failure analysis, the
  corrected-observation evaluation, explanations, inference-cost measurement and the scripts that regenerate the
  paper's tables and figures.
- **`cybermetaformer/results/hidden_state/`**: per-run metrics, selected hyperparameters (`hpo/*/best.yaml`),
  summaries, and the paper's tables (`makale/tablolar/*.md`) and figures (`makale/sekiller/`).

> **Status (v0.2.0, pre-release).** This release adds the evaluation-only analyses of the second revision round:
> host permutations and a rewired simulator topology, interventions on the observations of other hosts, a failure
> analysis of the periodic-analyse and remove defenders, the evaluation on a copy of the data with the remove-flag
> observation bug corrected, and a single inference-cost measurement of ten models. They reuse the stored five-seed
> models and retrain nothing. Code comments, docstrings and some module and result-folder names are still in
> Turkish; the [glossary](#glossary) maps them. An English pass, the raw model outputs (logits and per-episode
> confusions, about 10 GB) and the trained weights will be published with `v1.0-submission` on Zenodo with a DOI.

## Installation

Python 3.12. The experiments were run with PyTorch 2.14.1 (Apple MPS for training, CPU for latency).

```bash
git clone --recurse-submodules https://github.com/hkursatsimsek/CyberMetaFormer.git
cd CyberMetaFormer
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock        # exact versions; requirements.txt lists the direct dependencies
```

All commands are run **from the repository root as modules** (`python -m ...`). On Apple silicon set
`PYTORCH_ENABLE_MPS_FALLBACK=1`. The `CybORG_plus_plus` submodule (Alan Turing Institute, MIT, pinned to commit
`c343476`) is needed only to generate or validate data; training and evaluation read the stored dataset.

```bash
python -m pytest cybermetaformer/hidden_state/tests -q
```

## Data

| Folder | Content |
|---|---|
| `minicage/data/minicage_v1/` | 24 shards (`shard_*.npz`), `meta.json`, `splits.npz`, `tabular.csv.gz`, `baselines.json`, `DATASHEET.md` |
| `minicage/data/minicage_v1/splits/*.csv` | Episode IDs of the train / val / test / ood_test splits with their attacker and defender policy |
| `minicage/data/minicage_v1_aux/` | Silent-compromise cause labels (`silent_cause.npz`) |
| `minicage/data/minicage_v1_bugged/` | Same 12 attacker–defender pairs generated without the simulator bug fixes (evaluation only) |
| `minicage/data/minicage_v1_shift/`, `_shift_aux/` | Unseen defender policies and mechanism changes (ε = 0.5, 0.7, periodic analyse, remove instead of restore) |
| `minicage/data/minicage_v1_topo/` | Rewired topology: user1/2 → ent1 and user3/4 → ent0, with the same hosts, subnets and observation layout; defender policies ε = 0.1 and 0.3, 2 attackers × 2,000 episodes (evaluation only) |
| `minicage/data/minicage_v1_fixedobs/`, `_shift_fixedobs/` | Paired copies of `minicage_v1` and `minicage_v1_shift`, re-simulated with the same seeds and the recorded defender actions with the remove-flag observation bug fixed: a failed remove no longer shows "removed". True states, attacker observations, actions and rewards are byte-identical (evaluation only) |

Splits are stratified by attacker–defender pair at the episode level; `ood_test` is the defender policy
`react_restore_eps0.3`, never seen in training. Verify the files and, optionally, regenerate the dataset
(generation is seeded and deterministic):

```bash
shasum -a 256 -c cybermetaformer/hidden_state/data_manifest.sha256
python -m minicage.dataset.generate --episodes-per-config 2000 --out /tmp/minicage_v1_regen
python -m minicage.dataset.generate --remove-bugs false --seed 1 --out /tmp/minicage_v1_bugged_regen
python -m minicage.dataset.generate --blue react_restore_eps0.5 react_restore_eps0.7 react_restore_analyse react_remove \
    --seed 2 --out /tmp/minicage_v1_shift_regen
python -m minicage.dataset.generate --topology rewired --blue react_restore_eps0.1 react_restore_eps0.3 \
    --seed 3 --out /tmp/minicage_v1_topo_regen
python -m minicage.dataset.resimulate --src minicage/data/minicage_v1 --out /tmp/minicage_v1_fixedobs_regen
python -m minicage.dataset.resimulate --src minicage/data/minicage_v1_shift --out /tmp/minicage_v1_shift_fixedobs_regen
python -m minicage.dataset.validate --data minicage/data/minicage_v1
```

The data are released under CC BY 4.0 (`minicage/data/LICENSE`).

## Reproducing the experiments

Every run writes to `cybermetaformer/results/hidden_state/<stage>/<run>/` (`metrics.json` is tracked; logits,
per-episode slices and weights are not). A run with an existing `metrics.json` is skipped, so interrupted grids resume.
The tuned settings are included (`<stage>/hpo/<study>/best.yaml`), so the grids below reuse them; rerun `hpo` only to
repeat the search. `--device` accepts `auto`, `cuda`, `mps` or `cpu`.

**Main pair (CyberMetaFormer and the full-history GRU, seeds 0–4):**

```bash
python -m cybermetaformer.hidden_state.run_grid --stage asama3 --models cybermetaformer --seeds 0 1 2 3 4
python -m cybermetaformer.hidden_state.run_grid --stage asama3 --models gru_full --ks 101 --seeds 0 1 2 3 4 --hpo-stage asama1
python -m cybermetaformer.hidden_state.compare --plan m3_5seed --n-boot 100000      # paired episode bootstrap, Holm
```

**Ablations and references (seeds 0–2 unless stated):**

```bash
python -m cybermetaformer.hidden_state.run_grid --stage asama3 --models cybermetaformer \
    --variants duz78 host_yok host_tam host_gcn kol_konv kol_ssd konv_dikkat konv_gru hafiza_yok b2 --seeds 0 1 2
python -m cybermetaformer.hidden_state.run_grid --stage asama3 --models cybermetaformer \
    --variants gru_host_topo gru_host_attn --seeds 0 1 2 3 4                       # host-token GRU references
python -m cybermetaformer.hidden_state.run_grid --stage asama3 --models cybermetaformer \
    --variants host_tam --seeds 3 4               # five seeds of the no-topology-bias variant for the shift analyses
python -m cybermetaformer.hidden_state.compare --plan m3        # ablations, RQ3, attacker transfer
python -m cybermetaformer.hidden_state.compare --plan m4 --n-boot 100000
```

**Stages 1 and 2 (windowed models, k ∈ {1, 4, 8, 16, 32}, extensions k = 64 and full history):**

```bash
python -m cybermetaformer.hidden_state.run_grid --models majority persist xgb_tf xgb
python -m cybermetaformer.hidden_state.run_grid --models mlp cnn1d lstm gru
python -m cybermetaformer.hidden_state.run_grid --models lstm_full gru_full --ks 101
python -m cybermetaformer.hidden_state.run_grid --stage asama2 \
    --models transformer patchtst itransformer tsmixer timesnet timemixer
python -m cybermetaformer.hidden_state.aggregate --stage asama2
```

**Paper analyses:**

```bash
python -m cybermetaformer.hidden_state.ood_paketi collect && python -m cybermetaformer.hidden_state.ood_paketi analyze
python -m cybermetaformer.hidden_state.hic_uyari           # no-alert subsets N_hist / N_ep
python -m cybermetaformer.hidden_state.sinif_duzeyi        # class-level metrics, confusion, worst host
python -m cybermetaformer.hidden_state.esdegerlik --settings cpu64 cpu32 mps32 --seeds 0 1 2 3 4   # step = episode mode
python -m cybermetaformer.hidden_state.gecikme             # online CPU step latency (first measurement)
python -m cybermetaformer.hidden_state.maliyet             # online inference cost of ten models (latency, memory)
python -m cybermetaformer.hidden_state.rapor.boyut_tablosu # tensor shapes and parameter counts
```

**Shift, intervention and failure analyses (stored five-seed models, no retraining):**

```bash
python -m cybermetaformer.hidden_state.ood_permutasyon collect          # host permutations P1, P1s, P2, P2s, P3
python -m cybermetaformer.hidden_state.ood_permutasyon analyze --n-boot 100000
python -m cybermetaformer.hidden_state.ood_topoloji collect             # rewired simulator topology
python -m cybermetaformer.hidden_state.ood_topoloji analyze --n-boot 100000
python -m cybermetaformer.hidden_state.basarisizlik --n-boot 100000     # error types and slices per defender policy
python -m cybermetaformer.hidden_state.mudahale collect                 # interventions on other hosts' observations
python -m cybermetaformer.hidden_state.mudahale analyze --n-boot 100000
python -m cybermetaformer.hidden_state.duzeltilmis_gozlem collect       # flawed vs corrected remove-flag observation
python -m cybermetaformer.hidden_state.duzeltilmis_gozlem analyze --n-boot 10000
```

**Paper tables and figures:**

```bash
python -m cybermetaformer.hidden_state.rapor.makale_tablolar --basis m4   # five-seed main pair, as in the paper
python -m cybermetaformer.hidden_state.rapor.makale_sekiller
```

`makale_tablolar` reads the per-episode outputs of the runs (`episode_slices.npz`) and, for parameter counts, the
trained weights (`model.pt`). Both are produced by the commands above; until the Zenodo archive is published, the
stored tables in `makale/tablolar/` are the reference (regenerating them from the original runs reproduces these
files byte for byte). The tables of the shift, intervention and failure analyses are built from the stored analysis
summaries (`makale/<analysis>/ozet.json`), which are included.

Hardware: training on Apple M3 Pro (MPS); some ablations and the host-token GRU references on Apple M4 Pro, always
compared with a copy of the main model trained on the same machine (`ana_m4`). The shift, intervention and failure
analyses ran on the M3 Pro (MPS); latency and cost on the M3 Pro CPU, batch size 1.

## Glossary

| Name | Meaning |
|---|---|
| `asama0` … `asama3` | Stage 0 (EDA, benchmarks) … Stage 3 (CyberMetaFormer) |
| `makale/`, `tablolar/`, `sekiller/` | paper, tables, figures |
| `ozet`, `karsilastirma`, `verimlilik`, `aciklama`, `belirsizlik` | summary, comparison, efficiency, explanations, uncertainty |
| `ood_paketi.py` | extended OOD suite (unseen defender, severity curve, mechanism changes) |
| `hic_uyari.py`, `sinif_duzeyi.py`, `esdegerlik.py`, `gecikme.py` | no-alert subsets, class-level report, step/episode equivalence, latency |
| `ood_permutasyon.py`, `ood_topoloji.py` | host-permutation shift; rewired simulator topology (optionally with the relation matrix rebuilt from the new links) |
| `basarisizlik.py`, `mudahale.py` | failure analysis by error type, attacker, subnet and alert age; interventions on other hosts' observations |
| `duzeltilmis_gozlem.py`, `maliyet.py` | flawed vs corrected observation; online inference cost (latency, state and working memory) |
| `rapor/makale_tablolar.py`, `rapor/makale_sekiller.py`, `rapor/boyut_tablosu.py` | paper tables, paper figures, shape table |
| `duz78`, `host_yok`, `host_tam`, `host_gcn` | no host tokens (one 78-d token); no host mixer; full attention, no topology bias; GCN-like host mixer |
| `kol_konv`, `kol_ssd`, `konv_dikkat`, `konv_gru` | short convolution only; SSD branch only; causal attention instead of SSD; GRU instead of SSD |
| `hafiza_yok`, `yalniz_yas`, `yalniz_ssd`, `konum_yok` | window 32 without alert age; window 32 with alert age; SSD without alert age; no positional encoding |
| `b2`, `ana_m4`, `aksiyon` | B2 scale (2.8 M parameters); main-model copy on the second machine; previous defender action as input |
| `dg_*`, `art_uyari`, `art_zaman` | domain generalization (GroupDRO, V-REx, CORAL, DANN); alert-dropout and time-mask augmentation |
| `saldirgan_bline`, `saldirgan_meander` | trained on one attacker (attacker transfer) |
| `gru_host_topo`, `gru_host_attn` | host-token GRU with topology-biased / full cross-host attention |
| `temiz`, `sessiz ihlal`, `bölüm`, `seed` | clean, silent compromise, episode, seed |

## Citation and license

If you use this code or data, please cite the repository (`CITATION.cff`); the article reference will be added on
publication. Code: MIT (`LICENSE`). Data: CC BY 4.0. `cybermetaformer/hidden_state/third_party/tslib/` is a pinned
subset of the [Time-Series-Library](https://github.com/thuml/Time-Series-Library) (MIT, THUML); CybORG++ is MIT
(The Alan Turing Institute).

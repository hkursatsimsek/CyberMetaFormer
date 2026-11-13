"""Paper (M4) tables, in English, from result files only.

Tables from the draft (phase 0; included in the md with `::: {#tbl:… .tablo include=…}`):
- t00_related.md           related-methods comparison of §2.4 (literature, not results; the content is fixed here)
- t01_policies.md          defender policies, episode counts and their role in the data split (dataset + splits.npz)
- t02_silent.md            prevalence of silent compromise (makale/no_alert/ozet.json)
- t03_protocol.md          training and evaluation protocol of the stages (ozet.json, hpo/*/best.yaml, configs/)
- t04_stage1.md            stage 1, best k on val (asama1/ozet/ozet.json; original MLP asama0/mlp_repro_mac.json)
- t05_stage2.md            stage 2, best k on val and the GRU references (asama2/ozet/ozet.json)
- t06_main.md              CMF vs full-history GRU: metrics, paired episode bootstrap, ECE (metrics.json + plan)
- t07_no_alert.md          silent recall with no alert up to t (N_hist) and paired differences (plan, RQ4 family)
- t08_ablations.md         selected ablations, Δ and 95% CI (plan)
- t09_policy_shift.md      defender policy shifts (asama3/ood_suite/ozet.json)
- t10_attacker.md          transfer across attackers (metrics.json of the saldirgan_* runs)
- t11_efficiency.md        old latency measurement (asama3/verimlilik), parameters and OOD F1
- tA1_ablations_all.md     all ablations (plan)
- tB1_hyperparameters.md   B1 settings (asama3/hpo/cybermetaformer_k101/best.yaml)
Items 4 and 5 (added to the text in phase 1): a4_reference.md, a4_pairs.md, a5_seeds.md, a5_two_sources.md.
Items 6–8 and 10 (phase 2; results in makale/<analysis>/ozet.json, seeds 0–4, no retraining):
- t_shift_types.md         evaluated shift types and episode counts (§3.2)
- t_host_shift.md          host permutation and rewired topology, OOD (ood_permutation, ood_topology)
- t_failure.md             error types under four policies (failure_analysis)
- t_intervention.md        intervention on the observations of other hosts (interventions)
- t_corrected.md           faulty and corrected observation, CMF − GRU (corrected_obs)
- t_cost.md                latency, memory and OOD F1 (inference_cost; replaces t11_efficiency and step_latency/table.md)
- tE_*.md                  Appendix E: shift differences, analyse/remove slices, representation type under analyse,
                           all intervention metrics and model differences, corrected − faulty differences
Combined tables of the JISA main text (advisor's revision; the detailed tables go to the supplement):
- t_compact_baselines.md   Table 4: selected baselines from stages 1–2 (t04_stage1 + t05_stage2)
- t_main_uncertainty.md    Table 5: main pair and two sources of uncertainty (t06_main + a5_two_sources)
- t_security.md            Table 6: root F1, root → clean, false alarm, detection of silent onsets (t_class_*)
- t_host_controls.md       Table 8: host-structured controls, performance and online cost (a4_reference + t_cost)
- t_robust_compact.md      Table 11: policy, host order, topology, corrected obs. (t09, t_host_shift, t_corrected)
t11_efficiency is the old measurement; it is produced only for the internal number check (basis m3), not the paper.

Basis (--basis): `m3` produces the draft's numbers (CMF seeds 0–2; M3 record, `compare --plan m3`, 10,000
resamples); `m4` uses the five-seed main pair (CMF seeds 0–4; `compare --plan m3_5seed`, 100,000 resamples).
Ablations and three-seed variants are compared with CMF seeds 0–2 under both bases. The internal number check calls
`build(basis='m3')`. Each cell carries both the formatted text and the raw values (`Cell`); the check compares the raw
values. The bootstrap point estimates are compared with the seed means; the script stops if they disagree.

    venv_ts/bin/python -m cybermetaformer.hidden_state.report.paper_tables [--basis m3]   # results/hidden_state/makale/tablolar/
"""
import argparse
import glob
import json
import os
from dataclasses import dataclass, field

import numpy as np
import yaml
from scipy import stats

from cybermetaformer.hidden_state.ood_suite import load_model

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, 'results', 'hidden_state')
CONFIGS = os.path.join(ROOT, 'hidden_state', 'configs')
CMP = os.path.join(RESULTS, 'asama3', 'karsilastirma_m4')
LATENCY = os.path.join(RESULTS, 'makale', 'inference_cost', 'olcum.json')   # item 10 (inference_cost.py; replaces step_latency.py)
OUT = os.path.join(RESULTS, 'makale', 'tablolar')
LAT_THREADS = '1'                                         # A4 column: one thread (default setting: latency table)
PLAN_FILE = {'m3': os.path.join(RESULTS, 'asama3', 'karsilastirma', 'ozet.json'),
             'm4': os.path.join(CMP, 'm3_5seed', 'ozet.json')}
CMF_SEEDS = {'m3': [0, 1, 2], 'm4': None}                 # None = all seeds

PM = '±\u00a0'        # no-break space between ± and SD: in a narrow column the cell breaks into "mean" / "± sd"
# key → (stage, run prefix, English name). Machine names stay out of the tables (user decision, 2026-11-08); cmf_m4
# is a second, independent five-seed CMF training on the same hardware as the host-token GRU references ("run 2").
RUNS = {
    'cmf': ('asama3', 'cybermetaformer_k101', 'CyberMetaFormer (conv + SSD, topology bias)'),
    'cmf_m4': ('asama3', 'cybermetaformer-ana_m4_k101', 'CyberMetaFormer, run 2'),
    'host_tam': ('asama3', 'cybermetaformer-host_tam_k101', 'CMF, no topology bias'),
    'konv_gru': ('asama3', 'cybermetaformer-konv_gru_k101', 'CMF, GRU instead of SSD'),
    'gru_host_topo': ('asama3', 'cybermetaformer-gru_host_topo_k101', 'Host-token GRU, topology bias'),
    'gru_host_attn': ('asama3', 'cybermetaformer-gru_host_attn_k101', 'Host-token GRU, no topology bias'),
    'gru_full': ('asama1', 'gru_full_k101', 'Full-history GRU (flat 78-d input)'),
    'host_yok': ('asama3', 'cybermetaformer-host_yok_k101', 'CMF, no host mixer'),
}
# ablations (configs/m3_variants.yaml) in the order of the draft's Table A1; all asama3, 3 seeds
ABLATIONS = {
    'duz78': 'No host tokens (single 78-d token)',
    'host_yok': 'No host mixer',
    'host_tam': 'Full attention instead of topology bias',
    'host_gcn': 'GCN-like host mixer',
    'konv_dikkat': 'Full-history causal attention instead of SSD',
    'hafiza_yok': 'No memory: window 32, no alert-age channel',
    'yalniz_yas': 'Window 32, alert-age channel kept',
    'yalniz_ssd': 'Full-history SSD without the alert-age channel',
    'konum_yok': 'No position embedding',
    'kol_konv': 'Short convolution only (no long branch)',
    'kol_ssd': 'SSD branch only',
    'konv_gru': 'GRU instead of SSD',
    'havuz_mlp': 'Pooling and MLP MetaFormer mixers',
    'b2': 'B2 scale',
}
ABL_SEEDS = [0, 1, 2]
SELECTED = ['duz78', 'host_yok', 'host_tam', 'host_gcn', 'kol_konv', 'kol_ssd', 'konv_dikkat', 'konv_gru',
            'hafiza_yok', 'b2']                                    # selected-ablations table, draft order
CRIT = {'test_host_macro_f1': 'Test host macro-F1', 'ood_host_macro_f1': 'OOD host macro-F1',
        'ood_rel_drop': 'Relative OOD drop', 'test_silent_recall': 'Test silent recall',
        'ood_silent_recall': 'OOD silent recall', 'test_never_recall': 'Test N_hist recall',
        'ood_never_recall': 'OOD N_hist recall'}
METRICS = ['test_f1', 'ood_f1', 'rel_drop', 'test_silent', 'ood_silent', 'ood_hist']
BETTER = {'test_f1': 1, 'ood_f1': 1, 'rel_drop': -1, 'test_silent': 1, 'ood_silent': 1, 'ood_hist': 1}
TO_CRIT = {'test_f1': 'test_host_macro_f1', 'ood_f1': 'ood_host_macro_f1', 'rel_drop': 'ood_rel_drop',
           'test_silent': 'test_silent_recall', 'ood_silent': 'ood_silent_recall', 'ood_hist': 'ood_never_recall'}


# ── table structure ──────────────────────────────────────────────────────

@dataclass
class Cell:
    text: str
    values: tuple = ()                 # raw values (fractions, even when shown as percent); the check uses them


@dataclass
class Table:
    name: str
    cols: list                         # [(key, header, alignment 'l'/'r')]
    rows: list = field(default_factory=list)   # [(row key, {column key: Cell})]
    note: str = ''
    group: tuple = ()                  # left blank in the md when equal to the previous row (display only)

    def add(self, key, **cells):
        self.rows.append((key, {k: (v if isinstance(v, Cell) else Cell(str(v))) for k, v in cells.items()}))

    def cell(self, row, col):
        return dict(self.rows)[row][col]

    def md(self):
        L = ['| ' + ' | '.join(h for _, h, _ in self.cols) + ' |',
             '|' + '|'.join('---:' if a == 'r' else '---' for _, _, a in self.cols) + '|']
        prev = None
        for _, cells in self.rows:
            g = tuple(cells[k].text for k in self.group)
            same = bool(self.group) and g == prev
            prev = g
            L.append('| ' + ' | '.join('' if same and k in self.group else cells[k].text if k in cells else '—'
                                       for k, _, _ in self.cols) + ' |')
        return '\n'.join(L) + ('\n\n' + self.note if self.note else '') + '\n'


def minus(s):
    return s.replace('-', '−')


def num(v, k):
    return Cell(minus(f'{v:.{k}f}'), (v,))


def sgn(v, k):
    return Cell(minus(f'{v:+.{k}f}'), (v,))


def pm_cell(m, s, k):
    return Cell(minus(f'{m:.{k}f}') + f' {PM}{s:.{k}f}', (m, s))


def ab(a, b, k):
    return Cell(f'{a:.{k}f} / {b:.{k}f}', (a, b))


def dci(d, lo, hi, k):
    return Cell(minus(f'{d:+.{k}f} [{lo:+.{k}f}, {hi:+.{k}f}]'), (d, lo, hi))


def intc(n):
    return Cell(f'{int(n):,}', (int(n),))


def sci(v, k=1):
    e = int(np.floor(np.log10(abs(v))))
    sup = str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹')
    return Cell(f'{v / 10 ** e:.{k}f} × 10' + str(e).translate(sup), (v,))


def txt(s, *values):
    return Cell(s, tuple(values))


def pfmt(p):
    return '< 0.0001' if p < 1e-4 else f'{p:.4f}'


def signed(v, k):
    return f'{v:+.{k}f}'.replace('-', '−')


def nd(metric):
    return 4 if metric in ('test_f1', 'ood_f1', 'rel_drop') else 3


def ms(d, metric, n):
    return f"{d['mean']:.{nd(metric)}f}" + (f" {PM}{d['sd']:.{nd(metric)}f}" if n > 1 else '')


# ── data ─────────────────────────────────────────────────────────────────

def run_seeds(stage, prefix):
    d = os.path.join(RESULTS, stage)
    return sorted(int(n.rsplit('_s', 1)[1]) for n in os.listdir(d)
                  if n.startswith(prefix + '_s') and n.rsplit('_s', 1)[1].isdigit()
                  and os.path.exists(os.path.join(d, n, 'metrics.json')))


def seed_metrics(stage, prefix, seed):
    """Metrics of one run: F1, silent recall and ECE from metrics.json; N_hist recall from the pooled 'never' slice."""
    d = os.path.join(RESULTS, stage, f'{prefix}_s{seed}')
    rep = json.load(open(os.path.join(d, 'metrics.json')))['reports']
    with np.load(os.path.join(d, 'episode_slices.npz')) as z:
        hist = z['ood_test'][:, list(z['names']).index('never')].sum(0)
    t, o = rep['test']['host_macro_f1'], rep['ood_test']['host_macro_f1']
    return {'seed': seed, 'test_f1': t, 'ood_f1': o, 'rel_drop': 1 - o / t, 'test_silent': rep['test']['silent_recall'],
            'ood_silent': rep['ood_test']['silent_recall'], 'ood_hist': float(hist[1] / hist[0]),
            'ood_ece': rep['ood_test']['calibration_T']['ece']}


def n_params(prefix):
    model, _ = load_model(prefix.rsplit('_k', 1)[0], 0, 'cpu')            # architecture does not depend on the seed
    return sum(p.numel() for p in model.parameters())


def stats_of(rows):
    return {m: {'mean': float(np.mean([r[m] for r in rows])),
                'sd': float(np.std([r[m] for r in rows], ddof=1)) if len(rows) > 1 else 0.0}
            for m in METRICS + ['ood_ece']}


def load_all():
    out = {}
    for key, (stage, prefix, label) in RUNS.items():
        rows = [seed_metrics(stage, prefix, s) for s in run_seeds(stage, prefix)]
        out[key] = {'stage': stage, 'prefix': prefix, 'label': label, 'params': n_params(prefix),
                    'seeds': [r['seed'] for r in rows], 'per_seed': rows, **stats_of(rows)}
    return out


def subset(run, seeds):
    """Only the given seeds of the same run (None = all)."""
    rows = [r for r in run['per_seed'] if seeds is None or r['seed'] in seeds]
    assert seeds is None or len(rows) == len(seeds), (run['prefix'], seeds)
    return {**run, 'seeds': [r['seed'] for r in rows], 'per_seed': rows, **stats_of(rows)}


def load_ablations():
    """Ablations use seeds 0–2 by protocol. Seeds added later to some variants (host_tam s3–s4, for the host
    permutation) are not counted here; those seeds enter their own tables through RUNS."""
    out = {}
    for v in ABLATIONS:
        prefix = f'cybermetaformer-{v}_k101'
        rows = [seed_metrics('asama3', prefix, s) for s in run_seeds('asama3', prefix) if s in ABL_SEEDS]
        out[v] = {'prefix': prefix, 'seeds': [r['seed'] for r in rows], 'per_seed': rows, **stats_of(rows)}
    return out


def load_plan(name):
    with open(os.path.join(CMP, name, 'ozet.json')) as f:
        return json.load(f)


def load_basis_plan(basis):
    with open(PLAN_FILE[basis]) as f:
        return json.load(f)


def find(plan, a, b, crit):
    rows = [r for r in plan['rows'] if r['model_a'].split('/')[-1] == a and r['model_b'].split('/')[-1] == b
            and r['criterion'] == crit]
    assert len(rows) == 1, (a, b, crit, len(rows))
    return rows[0]


def check_point(row, runs, ka, kb, metric):
    """The bootstrap point estimates of a and b must match the seed means of the tables on the same seed set."""
    for key, side in [(ka, 'a'), (kb, 'b')]:
        seeds = row[f'seeds_{side}']
        v = np.mean([r[metric] for r in runs[key]['per_seed'] if r['seed'] in seeds])
        assert abs(v - row[side]) < 1e-9, f'{key} {metric}: bootstrap {row[side]} ≠ seed mean {v}'


def stage_summary(stage):
    with open(os.path.join(RESULTS, stage, 'ozet', 'ozet.json')) as f:
        return json.load(f)


def best_k(summary, model):
    from cybermetaformer.hidden_state.aggregate import _best
    return _best(summary, model)


def gmean(g, col):
    return (g.get(col) or {}).get('mean')


def hpo(stage, model_k):
    with open(os.path.join(RESULTS, stage, 'hpo', model_k, 'best.yaml')) as f:
        return yaml.safe_load(f)


def config(model):
    with open(os.path.join(CONFIGS, f'{model}.yaml')) as f:
        return yaml.safe_load(f)


def n_boot_of(stage):
    """Bootstrap resample count of the stage 1/2 comparison JSONs (all must be equal)."""
    ns = {json.load(open(p))['n_boot'] for p in glob.glob(os.path.join(RESULTS, stage, 'karsilastirma', '*.json'))}
    assert len(ns) == 1, (stage, ns)
    return ns.pop()


def seed_text(seeds):
    return f'{seeds[0]}–{seeds[-1]}' if seeds == list(range(seeds[0], seeds[-1] + 1)) else ', '.join(map(str, seeds))


# ── §2.4: related methods (fixed content) ────────────────────────────────

RELATED = [  # (approach and reference no., primary task, output, temporal/relational approach, key difference)
    ('BF-PPO \\[3\\]', 'Defence action', 'Approximate belief and policy', 'POMDP / particle filter',
     'No separate three-class F1 target per host state'),
    ('Makki Nayeri and Rezvani \\[16\\]', 'Next victim / alert', 'Future target relation',
     'Temporal graph and Transformer', 'Does not estimate the current hidden privilege state'),
    ('NIDS-Mamba \\[17\\]', 'Network intrusion detection', 'Flow-level class', 'SSD and attention',
     'Differs in multi-host privilege state and the silent-compromise subset'),
    ('GTB-PPO \\[28\\]', 'Target prediction + action', 'Next-target belief / action', 'Graph + causal time + PPO',
     'Different prediction horizon and outcome metrics'),
    ('CyberMetaFormer', 'Hidden host-state estimation', 'Three-class probabilities per host',
     'Causal host memory + cross-host attention', 'Silent (no local alert) compromise and OOD sub-analyses'),
]


def t00_related():
    T = Table('t00_related', [('approach', 'Approach', 'l'), ('task', 'Primary task', 'l'), ('output', 'Output', 'l'),
                              ('model', 'Temporal / relational modelling', 'l'), ('diff', 'Key difference', 'l')])
    for row in RELATED:
        T.add(row[0], **{k: txt(v) for (k, _, _), v in zip(T.cols, row)})
    return T


# ── Tables 1–3: data and protocol ────────────────────────────────────────

POLICY_TEXT = {'sleep': 'Never acts', 'react_restore': 'Restores a host after an exploit alert',
               'restore_decoys': 'Restores after an alert; places decoys on idle steps'}


def t01_policies():
    from minicage.dataset import loaders as L
    from minicage.dataset.policies import NUM_BLUE_ACTIONS
    from cybermetaformer.hidden_state import data as D
    raw, meta = L.load_raw(D.DEFAULT_DATA, fields=['blue_policy', 'red_policy'])
    splits = L.load_splits(D.DEFAULT_DATA)
    blue = raw['blue_policy']
    split_name = {'train': 'train', 'val': 'validation', 'test': 'test', 'ood_test': 'OOD test'}
    T = Table('t01_policies', [('policy', 'Policy', 'l'), ('behaviour', 'Defender behaviour', 'l'),
                               ('split', 'Split', 'l'), ('episodes', 'Episodes', 'r')])
    for i, name in enumerate(meta['blue_policies']):
        idx = np.flatnonzero(blue == i)
        where = [s for s in ['train', 'val', 'test', 'ood_test'] if np.isin(splits[s], idx).any()]
        if 'eps' in name:
            eps = float(name.split('eps')[1])
            beh = txt(f'react_restore; random action with {100 * eps:.0f}% probability', eps)
        elif name == 'random':
            beh = txt(f'Uniformly random over the {NUM_BLUE_ACTIONS} actions', NUM_BLUE_ACTIONS)
        else:
            beh = txt(POLICY_TEXT[name])
        role = ('OOD test only' if where == ['ood_test'] else
                ' / '.join(split_name[s] for s in where).capitalize())
        T.add(name, policy=txt(name), behaviour=beh, split=txt(role), episodes=intc(len(idx)))
    red_name = {'bline': 'B-line', 'meander': 'Meander'}
    reds = ' and '.join(red_name[r] for r in meta['red_policies'])
    n_red = len(meta['red_policies'])
    assert n_red == 2
    T.note = (f"Note. Each defender policy is paired with both attackers ({reds}); "
              f"{len(blue) // (len(meta['blue_policies']) * n_red):,} episodes per pair, "
              f"{meta['steps_per_episode']} steps per episode. Episodes of the first five policies are split, stratified "
              f"by attacker–defender pair, into training ({len(splits['train']):,}), validation "
              f"({len(splits['val']):,}) and test ({len(splits['test']):,}); the "
              f"{len(splits['ood_test']):,} episodes of the sixth policy form the OOD test set and enter no training, "
              'selection or calibration step.')
    return T


def t02_silent():
    with open(os.path.join(RESULTS, 'makale', 'no_alert', 'ozet.json')) as f:
        C = json.load(f)['counts']
    cols = {'train': 'train', 'test': 'test', 'ood': 'ood_test'}
    T = Table('t02_silent', [('measure', 'Measure', 'l'), ('train', 'Train', 'r'), ('test', 'Test', 'r'),
                             ('ood', 'OOD', 'r')])
    spec = [('compromised', 'Compromised / all host-steps', 'compromised_share'),
            ('silent', 'Silent / compromised host-steps', 'silent_share_of_compromised'),
            ('root', 'Root share among silent host-steps', 'root_share_of_silent'),
            ('hist', 'No alert on the host up to t (N_hist) / silent', 'hist_share_of_silent'),
            ('prev32', 'Previous alert within the last 32 steps†', 'prev_alert_within_32_of_prev'),
            ('prev64', 'Previous alert within the last 64 steps†', 'prev_alert_within_64_of_prev')]
    spec[4:4] = [('ep', 'No alert on the host in the whole episode (N_ep) / silent', 'ep_share_of_silent'),
                 ('ep_hist', 'N_ep / N_hist', 'ep_share_of_hist')]
    for key, label, field_ in spec:
        T.add(key, measure=txt(label), **{c: num(C[s][field_], 3) for c, s in cols.items()})
    for key, label, field_ in [('n_silent', 'Silent host-steps (n)', 'silent'),
                               ('n_hist', 'N_hist host-steps (n)', 'hist'), ('n_ep', 'N_ep host-steps (n)', 'ep'),
                               ('pairs_hist', 'N_hist (episode, host) pairs (n)', 'hist_episode_host_pairs'),
                               ('pairs_ep', 'N_ep (episode, host) pairs (n)', 'ep_episode_host_pairs'),
                               ('episodes', 'Episodes', 'episodes')]:
        T.add(key, measure=txt(label), **{c: intc(C[s][field_]) for c, s in cols.items()})
    assert all(C[s]['ep'] <= C[s]['hist'] and not C[s]['obs0_has_alert'] for s in cols.values())
    T.note = ('Note. A host-step is silent when the host is compromised (user or root) and neither the exploit-activity '
              'bit nor the compromise flag is set in the current observation. N_hist(h, t): no alert on host h in '
              'obs[1..t] (online-applicable); N_ep(h): no alert on host h in obs[1..T] (retrospective; every N_ep '
              'host-step is also an N_hist host-step). Both subsets are restricted to silent host-steps (Section 3.3). '
              'obs[0] carries no alert in any split. † Among silent host-steps whose host had an earlier alert.')
    return T


def t03_protocol(basis):
    s1, s2 = stage_summary('asama1'), stage_summary('asama2')
    ks = lambda s, models: sorted({g['k'] for g in s.values() if g['model'] in models and 0 < g['k'] < 101})  # noqa: E731
    from cybermetaformer.hidden_state.aggregate import M1_LEARNED, M2_MODELS
    k1, k2 = ks(s1, M1_LEARNED), ks(s2, M2_MODELS)
    k1_main = [k for k in k1 if all(f'{m}_k{k}' in s1 for m in ['mlp', 'cnn1d', 'lstm', 'gru'])]
    full1 = sorted({g['model'] for g in s1.values() if g['k'] == 101 and g['model'] in M1_LEARNED + ['lstm_full',
                                                                                                    'gru_full']})
    single1 = {(g['model'], g['k']) for g in s1.values() if g['model'] in M1_LEARNED and len(g['seeds']) == 1}
    assert single1 == {('xgb', k) for k in ks(s1, ['xgb'])} | {('mlp', 101)}, single1
    seeds1 = sorted({len(g['seeds']) for g in s1.values() if g['model'] in M1_LEARNED + ['lstm_full', 'gru_full']})
    seeds2 = sorted({len(g['seeds']) for g in s2.values() if g['model'] in M2_MODELS})
    assert seeds1 == [1, 5] and len(seeds2) == 1, (seeds1, seeds2)
    narrow2 = [m for m in M2_MODELS if ks(s2, [m]) != k2]
    h1 = {hpo('asama1', d)['protocol']['n_trials'] for d in os.listdir(os.path.join(RESULTS, 'asama1', 'hpo'))}
    h2 = {hpo('asama2', d)['protocol']['n_trials'] for d in os.listdir(os.path.join(RESULTS, 'asama2', 'hpo'))}
    hk1 = sorted({hpo('asama1', d)['k'] for d in os.listdir(os.path.join(RESULTS, 'asama1', 'hpo'))})
    hk2 = sorted({hpo('asama2', d)['k'] for d in os.listdir(os.path.join(RESULTS, 'asama2', 'hpo'))})
    h3 = hpo('asama3', 'cybermetaformer_k101')['protocol']
    assert len(h1) == len(h2) == 1
    nn1 = ['mlp', 'cnn1d', 'lstm', 'gru', 'lstm_full', 'gru_full']
    ep = {config(m)['epochs'] for m in nn1 + M2_MODELS + ['cybermetaformer']}
    pat = {config(m)['early_stopping']['patience'] for m in nn1 + M2_MODELS + ['cybermetaformer']}
    xgb_rounds = {config(m)['early_stopping_rounds'] for m in ['xgb', 'xgb_tf']}
    assert len(ep) == len(pat) == len(xgb_rounds) == 1
    ep, pat, xgb_rounds = ep.pop(), pat.pop(), xgb_rounds.pop()
    plan = load_basis_plan(basis)
    cmf_seeds = run_seeds('asama3', 'cybermetaformer_k101') if CMF_SEEDS[basis] is None else CMF_SEEDS[basis]
    nb1, nb2, nb3 = n_boot_of('asama1'), n_boot_of('asama2'), plan['n_boot']
    lab = {'xgb': 'XGBoost', 'xgb_tf': 'XGB-TF', 'mlp': 'MLP', 'lstm_full': 'LSTM', 'gru_full': 'GRU',
           'patchtst': 'PatchTST', 'timesnet': 'TimesNet', 'timemixer': 'TimeMixer'}
    full_names = sorted({lab[m] for m in full1})
    T = Table('t03_protocol', [('setting', 'Setting', 'l'), ('s1', 'Stage 1', 'l'), ('s2', 'Stage 2', 'l'),
                               ('cmf', 'CyberMetaFormer', 'l')])
    T.add('history', setting=txt('Candidate history lengths k'),
          s1=txt(', '.join(map(str, k1_main)) + f"; full history: {', '.join(full_names)}"
                 + ''.join(f'; {lab[m]} also {k}' for m in ['xgb_tf'] for k in ks(s1, [m]) if k not in k1_main),
                 *k1_main),
          s2=txt(', '.join(map(str, k2)) + '‡', *k2), cmf=txt('Full history'))
    n_cmf = len(cmf_seeds)
    cmf_seed_txt = (f'{n_cmf} (seeds {seed_text(cmf_seeds)})' if basis == 'm3' else
                    f'{n_cmf} (main model, seeds {seed_text(cmf_seeds)}); 5 (host-token GRU references); '
                    '3 (ablations, seeds 0–2)')
    T.add('seeds', setting=txt('Random seeds'),
          s1=txt('5; 1 for XGBoost and the full-history MLP (marked †)', 5, 1),
          s2=txt(str(seeds2[0]), seeds2[0]),
          cmf=txt(cmf_seed_txt, n_cmf, *([5, 3] if basis == 'm4' else [])))
    T.add('hpo', setting=txt('Hyperparameter search trials'),
          s1=txt(f"{h1.pop()} per model at k = {' and '.join(map(str, hk1))}", 25, *hk1),
          s2=txt(f"{h2.pop()} per model at k = {' and '.join(map(str, hk2))}", 15, *hk2),
          cmf=txt(f"{h3['n_trials']} (main model only; variants reuse its setting)", h3['n_trials']))
    T.add('epochs', setting=txt('Maximum final epochs'), s1=txt(f'{ep} (neural networks)', ep), s2=txt(str(ep), ep),
          cmf=txt(str(ep), ep))
    T.add('stop', setting=txt('Early stopping'),
          s1=txt(f'Patience {pat} epochs; XGBoost {xgb_rounds} rounds', pat, xgb_rounds),
          s2=txt(f'Patience {pat} epochs', pat), cmf=txt(f'Patience {pat} epochs', pat))
    T.add('selection', setting=txt('Model selection'), s1=txt('Validation host macro-F1'),
          s2=txt('Validation host macro-F1'), cmf=txt('Validation host macro-F1'))
    bs = lambda n: txt(f'Paired episode bootstrap, {n:,} resamples; Holm', n)  # noqa: E731
    T.add('test', setting=txt('Paired statistical test'), s1=bs(nb1), s2=bs(nb2), cmf=bs(nb3))
    from cybermetaformer.hidden_state.models import NOT_APPLICABLE

    def gaps(m):
        miss = [k for k in k2 if k not in ks(s2, [m])]
        na = [k for k in miss if (m, k) in NOT_APPLICABLE]
        bud = [k for k in miss if k not in na]
        return (f"{lab[m]} not at k = {', '.join(map(str, miss))} ("
                + '; '.join(x for x in [f"{', '.join(map(str, na))} not applicable" if na else '',
                                        f"{', '.join(map(str, bud))} outside the compute budget" if bud else '']
                            if x) + ')')
    T.note = (f"Note. ‡ {'; '.join(gaps(m) for m in narrow2)}. Full history means "
              'obs[0..t] at every step t. The hyperparameter search used '
              f"{h3['train_episodes']:,} training episodes and {h3['epochs']} epochs per trial for CyberMetaFormer.")
    return T


# ── Tables 4–5: stages 1 and 2 ───────────────────────────────────────────

def k_label(k):
    return 'full' if k == 101 else ('—' if k == 0 else str(k))


def k_cell(k):
    return txt(k_label(k), *([k] if 0 < k < 101 else []))


def t04_stage1():
    from cybermetaformer.hidden_state.aggregate import ORIGINAL_MLP
    s = stage_summary('asama1')
    with open(os.path.join(RESULTS, 'asama0', 'mlp_repro_mac.json')) as f:
        ref = json.load(f)['reference']
    o_t, o_o = ref['8_test_host_macro_f1'], ref['8_ood_test_host_macro_f1']
    assert abs(round(o_t, 3) - ORIGINAL_MLP['test']) < 1e-9 and abs(round(o_o, 3) - ORIGINAL_MLP['ood_test']) < 1e-9
    T = Table('t04_stage1', [('model', 'Model', 'l'), ('k', 'History', 'r'), ('test', 'Test F1', 'r'),
                             ('ood', 'OOD F1', 'r'), ('drop', 'Relative OOD drop', 'r'),
                             ('silent', 'Silent recall, test / OOD', 'r')])
    label = {'majority': 'Majority class', 'persist': 'Alert-persistence rule', 'xgb': 'XGBoost', 'xgb_tf': 'XGB-TF',
             'mlp': 'MLP', 'cnn1d': '1D-CNN', 'lstm': 'LSTM', 'gru': 'GRU', 'lstm_full': 'LSTM (full history)',
             'gru_full': 'GRU (full history)'}
    for m in ['majority', 'persist', 'orig', 'xgb', 'xgb_tf', 'mlp', 'cnn1d', 'lstm', 'gru', 'lstm_full', 'gru_full']:
        if m == 'orig':
            T.add('orig_mlp', model=txt('Original MLP baseline'), k=k_cell(8), test=num(o_t, 3), ood=num(o_o, 3),
                  drop=num(1 - o_o / o_t, 3), silent=txt('—'))
            continue
        b = best_k(s, m)
        one = len(b['seeds']) == 1 and m not in ('majority', 'persist')
        T.add(m, model=txt(label[m] + ('†' if one else '')), k=k_cell(b['k']),
              test=num(gmean(b, 'test_host_macro_f1'), 3), ood=num(gmean(b, 'ood_host_macro_f1'), 3),
              drop=num(gmean(b, 'ood_rel_drop'), 3),
              silent=ab(gmean(b, 'test_silent_recall'), gmean(b, 'ood_silent_recall'), 3))
    T.note = ('Note. Each model at the history length k selected by validation host macro-F1; five-seed means unless '
              'marked († one seed). The two rules are deterministic. Relative OOD drop = (F1_test − F1_OOD) / F1_test; '
              'it is negative when the OOD score is higher. The original MLP baseline is the eight-observation MLP '
              'of the reference implementation, reproduced on this dataset; its silent recall was not recorded.')
    return T


def t05_stage2():
    from cybermetaformer.hidden_state.aggregate import M2_MODELS
    s1, s2 = stage_summary('asama1'), stage_summary('asama2')
    label = {'transformer': 'Transformer', 'itransformer': 'iTransformer', 'tsmixer': 'TSMixer',
             'timemixer': 'TimeMixer', 'timesnet': 'TimesNet', 'patchtst': 'PatchTST'}
    T = Table('t05_stage2', [('model', 'Architecture', 'l'), ('k', 'History', 'r'), ('test', 'Test F1', 'r'),
                             ('ood', 'OOD F1', 'r'), ('silent', 'Silent recall, test / OOD', 'r'),
                             ('params', 'Params', 'r')])
    rows = [(m, label[m], best_k(s2, m)) for m in ['transformer', 'itransformer', 'tsmixer', 'timemixer', 'timesnet',
                                                   'patchtst']]
    assert {m for m, _, _ in rows} == set(M2_MODELS)
    rows += [('gru_k32', 'GRU (windowed)', s1['gru_k32']), ('gru_full', 'GRU (full history)', s1['gru_full_k101'])]
    for key, lab, b in rows:
        T.add(key, model=txt(lab), k=k_cell(b['k']), test=num(gmean(b, 'test_host_macro_f1'), 3),
              ood=num(gmean(b, 'ood_host_macro_f1'), 3),
              silent=ab(gmean(b, 'test_silent_recall'), gmean(b, 'ood_silent_recall'), 3),
              params=intc(gmean(b, 'n_params')))
    T.note = ('Note. Stage 2 models: three-seed means at the k selected by validation host macro-F1. GRU references '
              'from Stage 1 (five seeds).')
    return T


# ── Tables 6–8, A1: main comparison and ablations ────────────────────────

def plan_note(plan):
    m, n = plan['n_rows'], plan['n_boot']
    return (f"Paired episode bootstrap with {n:,} resamples (Algorithm 2); p (Holm, plan) adjusts over all m = {m} "
            f"rows of the comparison plan, so its smallest attainable value is m × 2/(n + 1) = "
            f"{min(1.0, m * 2 / (n + 1)):.4f}.")


def t06_main(runs, plan, basis):
    cmf = subset(runs['cmf'], CMF_SEEDS[basis])
    gru = runs['gru_full']
    R = {'cmf': cmf, 'gru_full': gru}
    T = Table('t06_main', [('crit', 'Criterion', 'l'), ('gru', 'Full-history GRU', 'r'),
                           ('cmf', 'CyberMetaFormer', 'r'), ('diff', 'Paired CMF − GRU [95% CI]', 'r'),
                           ('p', 'p (Holm, plan)', 'r')])
    T.add('params', crit=txt('Parameters'), gru=intc(gru['params']), cmf=intc(cmf['params']),
          diff=txt('Similar budget (B1)'), p=txt('—'))
    for m, k in [('test_f1', 4), ('ood_f1', 4), ('rel_drop', 4), ('ood_silent', 3)]:
        r = find(plan, RUNS['cmf'][1], RUNS['gru_full'][1], TO_CRIT[m])
        check_point(r, R, 'cmf', 'gru_full', m)
        assert r['seeds_a'] == cmf['seeds'] and r['seeds_b'] == gru['seeds']
        T.add(m, crit=txt(CRIT[TO_CRIT[m]]), gru=num(gru[m]['mean'], 3 if k == 3 else 4),
              cmf=num(cmf[m]['mean'], 3 if k == 3 else 4), diff=dci(r['diff'], r['lo'], r['hi'], k),
              p=txt(pfmt(r['p_holm_plan']), r['p_holm_plan']))
    T.add('ece', crit=txt('OOD ECE (after temperature scaling)'), gru=num(gru['ood_ece']['mean'], 4),
          cmf=num(cmf['ood_ece']['mean'], 4), diff=txt('Not bootstrapped'), p=txt('—'))
    T.note = (f"Note. CyberMetaFormer: mean of seeds {seed_text(cmf['seeds'])}; full-history GRU: mean of seeds "
              f"{seed_text(gru['seeds'])}. {plan_note(plan)} Intervals use unrounded differences. Lower is better for "
              'the relative OOD drop and ECE. ECE uses 15 confidence bins after temperature scaling fitted on the '
              'validation set.')
    return T


def t07_no_alert(runs, abl, plan):
    """RQ4 family: silent-compromise recall with no alert up to t (N_hist); seeds 0–2."""
    P = 'cybermetaformer'
    T = Table('t07_no_alert', [('a', 'Model A', 'l'), ('b', 'Model B', 'l'), ('test', 'Test recall, A / B', 'r'),
                               ('ood', 'OOD recall, A / B', 'r'), ('diff', 'OOD A − B [95% CI]', 'r'),
                               ('p', 'p (Holm, plan)', 'r')])
    lab = {f'{P}_k101': 'CMF (topology-biased attention)', f'{P}-host_tam_k101': 'Full attention (no topology bias)',
           f'{P}-host_gcn_k101': 'GCN-like host mixer', f'{P}-host_yok_k101': 'No host mixer',
           f'{P}-duz78_k101': 'Single 78-d token'}
    pairs = [(f'{P}_k101', f'{P}-host_yok_k101'), (f'{P}-host_tam_k101', f'{P}-host_yok_k101'),
             (f'{P}-host_gcn_k101', f'{P}-host_yok_k101'), (f'{P}_k101', f'{P}-duz78_k101')]
    for a, b in pairs:
        rt, ro = find(plan, a, b, 'test_never_recall'), find(plan, a, b, 'ood_never_recall')
        assert rt['seeds_a'] == rt['seeds_b'] == [0, 1, 2]
        T.add(f'{a}__{b}', a=txt(lab[a]), b=txt(lab[b]), test=ab(rt['a'], rt['b'], 3), ood=ab(ro['a'], ro['b'], 3),
              diff=dci(ro['diff'], ro['lo'], ro['hi'], 3), p=txt(pfmt(ro['p_holm_plan']), ro['p_holm_plan']))
    # the point estimates are recomputed from the 'never' slice of the runs
    cmf3 = subset(runs['cmf'], [0, 1, 2])
    hy = find(plan, f'{P}_k101', f'{P}-host_yok_k101', 'ood_never_recall')
    assert abs(hy['a'] - cmf3['ood_hist']['mean']) < 1e-9 and abs(hy['b'] - abl['host_yok']['ood_hist']['mean']) < 1e-9
    C = hic()['counts']
    T.note = ('Note. Silent host-steps whose host has had no alert in obs[1..t] (N_hist; online-applicable, '
              f"Section 3.3): n = {C['test']['hist']:,} test and {C['ood_test']['hist']:,} OOD host-steps. All models: "
              'seeds 0–2; the retrospective subset N_ep is compared in Table @tbl:hic_kume. ' + plan_note(plan))
    return T


# ── item 1: the two no-alert sets (makale/no_alert, no_alert.py) ───────

HIC_LABEL = {'cybermetaformer': 'CMF (topology-biased attention)',
             'cybermetaformer-host_tam': 'CMF, full attention (no topology bias)',
             'cybermetaformer-host_gcn': 'CMF, GCN-like host mixer', 'cybermetaformer-host_yok': 'CMF, no host mixer',
             'cybermetaformer-duz78': 'CMF, single 78-d token',
             'cybermetaformer-gru_host_topo': 'Host-token GRU, topology-biased attention',
             'cybermetaformer-gru_host_attn': 'Host-token GRU, attention without bias',
             'gru_full': 'Full-history GRU (flat input)'}


def hic():
    with open(os.path.join(RESULTS, 'makale', 'no_alert', 'ozet.json')) as f:
        return json.load(f)


def t07b_no_alert_sets(runs):
    """Silent recall in the N_hist and N_ep sets (seed mean ± SD) and two paired differences (no_alert.py)."""
    H = hic()
    R, C = H['recall'], H['counts']
    cols = [('test_hist', 'Test N_hist'), ('ood_hist', 'OOD N_hist'), ('test_ep', 'Test N_ep'), ('ood_ep', 'OOD N_ep')]
    T = Table('t07b_no_alert_sets', [('model', 'Model', 'l'), ('seeds', 'Seeds', 'r')]
              + [(k, h, 'r') for k, h in cols])
    for name, lab in HIC_LABEL.items():
        r = R[name]
        T.add(name, model=txt(lab), seeds=intc(len(r['seeds'])),
              **{k: pm_cell(r[k]['mean'], r[k]['sd'], 3) for k, _ in cols})
    # N_hist recall must match the value from the 'never' slice of the runs (episode_slices.npz)
    for key, name in [('cmf', 'cybermetaformer'), ('gru_full', 'gru_full'), ('gru_host_topo', 'cybermetaformer-gru_host_topo'),
                      ('gru_host_attn', 'cybermetaformer-gru_host_attn'), ('host_yok', 'cybermetaformer-host_yok')]:
        assert R[name]['seeds'] == runs[key]['seeds'], name
        assert abs(R[name]['ood_hist']['mean'] - runs[key]['ood_hist']['mean']) < 2e-5, name
    tests = {fam: {(t['a'], t['b'], t['split']): t for t in rows} for fam, rows in H['paired_tests'].items()}
    shown = []
    for a, b, lab in [('cybermetaformer', 'cybermetaformer-host_yok', 'Δ CMF − no host mixer (seeds 0–2)'),
                      ('cybermetaformer', 'gru_full', 'Δ CMF − full-history GRU (seeds 0–4)')]:
        cells = {}
        for k, h in cols:
            split, fam = k.split('_')
            t = tests[fam][(a, b, split)]
            cells[k] = dci(t['diff'], t['lo'], t['hi'], 3)
            shown.append((lab.split(' (')[0][2:], h, t))
        T.add(f'{a}__{b}', model=txt(lab), seeds=txt('—'), **cells)
    n, fam = {t['n_boot'] for *_, t in shown}, {t['holm_family_size'] for *_, t in shown}
    assert len(n) == len(fam) == 1
    sig = max(t['p_holm'] for *_, t in shown if t['p_holm'] < 0.05)
    ns = [f"{who}, {h} (p = {t['p_holm']:.2f})" for who, h, t in shown if t['p_holm'] >= 0.05]
    T.note = ('Note. Recall = share of the silent host-steps of the subset predicted as compromised (user or root). '
              f"N_hist: no alert on the host in obs[1..t] (n = {C['test']['hist']:,} test, {C['ood_test']['hist']:,} OOD "
              f"host-steps); N_ep: no alert on the host in the whole episode (n = {C['test']['ep']:,} test, "
              f"{C['ood_test']['ep']:,} OOD). Seed mean ± SD over the seeds listed; the CMF row uses seeds 0–4, so its "
              'values differ slightly from the seed 0–2 values in Table @tbl:hic. Δ rows: paired episode bootstrap '
              f"({n.pop():,} resamples, Algorithm 2) with the seeds shared by both models, Holm over the {fam.pop()} "
              f"comparisons of each subset; p (Holm) ≤ {sig:.3f} except {'; '.join(ns)}.")
    return T


def abl_row(plan, abl, v, crit):
    b = RUNS['cmf_m4'][1] if v == 'b2' else RUNS['cmf'][1]
    r = find(plan, f'cybermetaformer-{v}_k101', b, crit)
    m = next(k for k, c in TO_CRIT.items() if c == crit)
    assert abs(r['a'] - abl[v][m]['mean']) < 1e-9 and r['seeds_a'] == abl[v]['seeds'] == r['seeds_b'] == [0, 1, 2]
    return r


def abl_label(v, abl_params):
    return ABLATIONS[v] + (f' ({abl_params:,} parameters)' if v == 'b2' else '')


def t08_ablations(abl, plan, b2_params):
    T = Table('t08_ablations', [('change', 'Model change', 'l'), ('ood', 'Δ OOD host macro-F1 [95% CI]', 'r'),
                                ('silent', 'Δ OOD silent recall [95% CI]', 'r')])
    for v in SELECTED:
        f, s = abl_row(plan, abl, v, 'ood_host_macro_f1'), abl_row(plan, abl, v, 'ood_silent_recall')
        T.add(v, change=txt(abl_label(v, b2_params)), ood=dci(f['diff'], f['lo'], f['hi'], 4),
              silent=dci(s['diff'], s['lo'], s['hi'], 3))
    T.note = ('Note. Δ = variant − main model, seeds 0–2 each. B2 is compared with seeds 0–2 of CyberMetaFormer run 2, '
              'which was trained on the same hardware (Table @tbl:host_kontrol); all other variants with seeds 0–2 of the '
              'main run. 95% CI from the paired episode bootstrap '
              f"({plan['n_boot']:,} resamples).")
    return T


def tA1_ablations_all(abl, plan, b2_params):
    T = Table('tA1_ablations_all', [('variant', 'Variant', 'l'), ('test', 'Δ test F1 [95% CI]', 'r'),
                                    ('ood', 'Δ OOD F1 [95% CI]', 'r'),
                                    ('silent', 'Δ OOD silent recall [95% CI]', 'r')])
    for v in ABLATIONS:
        t, f, s = (abl_row(plan, abl, v, c) for c in ['test_host_macro_f1', 'ood_host_macro_f1', 'ood_silent_recall'])
        T.add(v, variant=txt(abl_label(v, b2_params)), test=dci(t['diff'], t['lo'], t['hi'], 4),
              ood=dci(f['diff'], f['lo'], f['hi'], 4), silent=dci(s['diff'], s['lo'], s['hi'], 3))
    sd_f = max(abl[v]['ood_f1']['sd'] for v in ABLATIONS)
    sd_s = max(abl[v]['ood_silent']['sd'] for v in ABLATIONS)
    arg_s = max(ABLATIONS, key=lambda v: abl[v]['ood_silent']['sd'])
    T.note = (f"Note. Δ = variant − main model; seeds 0–2 each; reference as in Table @tbl:ablasyon. Across variants the "
              f"seed SD reaches {sd_f:.3f} in OOD F1 and {sd_s:.3f} in OOD silent recall ({ABLATIONS[arg_s].lower()}), so "
              'differences below these values are not interpreted. 95% CI from the paired episode bootstrap '
              f"({plan['n_boot']:,} resamples).")
    T.values = {'max_sd_ood_f1': sd_f, 'max_sd_ood_silent': sd_s}
    return T


# ── Tables 9–11 and B1 ───────────────────────────────────────────────────

def t09_policy_shift(basis='m3', runs=None):
    """m3: M3 record (seeds 0–2, ood_suite analyze); m4: 5 seeds (ood_suite members → ozet_5seed.json)."""
    name = 'ozet.json' if basis == 'm3' else 'ozet_5seed.json'
    with open(os.path.join(RESULTS, 'asama3', 'ood_suite', name)) as f:
        P = json.load(f)
    M = P['models']
    n_seed = 3 if basis == 'm3' else 5
    spec = [('eps0.1', 'ε = 0.1 (training policy, test episodes)'), ('ood', 'ε = 0.3 (main OOD)'),
            ('eps0.5', 'ε = 0.5'), ('eps0.7', 'ε = 0.7'), ('analyse', 'Periodic analyse on idle steps'),
            ('remove', 'Remove instead of restore'), ('bug_ood', 'Earlier simulator version, OOD')]
    T = Table('t09_policy_shift', [('cond', 'Policy / condition', 'l'), ('n', 'Episodes', 'r'),
                                   ('cmf_f1', 'CMF F1', 'r'), ('gru_f1', 'GRU F1', 'r'),
                                   ('cmf_s', 'CMF silent recall', 'r'), ('gru_s', 'GRU silent recall', 'r')])
    for key, lab in spec:
        c, g = M['cybermetaformer']['uye'][key], M['gru_full']['uye'][key]
        assert len(c['seeds']['host_macro_f1']) == len(g['seeds']['host_macro_f1']) == n_seed
        eps = [float(key[3:])] if key.startswith('eps') else ([0.3] if key == 'ood' else [])
        T.add(key, cond=txt(lab, *eps), n=intc(P['sets'][key]['n_episodes']), cmf_f1=num(c['host_macro_f1'], 3),
              gru_f1=num(g['host_macro_f1'], 3), cmf_s=num(c['silent_recall'], 3), gru_s=num(g['silent_recall'], 3))
    if basis == 'm4':
        # the ε = 0.3 row must match the 5-seed values of the main table (metrics.json); the package computes from
        # float16 logits, so the difference is ~3e-5 (same 1e-4 tolerance as the check in ood_suite.collect)
        for k, key in [('cybermetaformer', 'cmf'), ('gru_full', 'gru_full')]:
            for m, crit in [('ood_f1', 'host_macro_f1'), ('ood_silent', 'silent_recall')]:
                assert abs(M[k]['uye']['ood'][crit] - runs[key][m]['mean']) < 1e-4, (k, m)
    T.note = (f"Note. Host macro-F1 and silent recall of each seed ({seed_text(list(range(n_seed)))}), averaged; each "
              'seed uses its own validation temperature. ε = 0.1: the test episodes of the training policy '
              'react_restore_eps0.1. ε = 0.5 and 0.7, '
              'periodic analyse (react_restore_analyse) and remove (react_remove): new episodes, both attackers, no '
              'training on them. Earlier simulator version: OOD-policy episodes generated with the simulator\'s '
              'documented bugs left in place (separate generation seed).')
    return T


def t10_attacker():
    def ms3(run, split):
        v = [json.load(open(os.path.join(RESULTS, 'asama3', f'{run}_s{s}', 'metrics.json')))['reports'][split]
             ['host_macro_f1'] for s in run_seeds('asama3', run)]
        assert len(v) == 3, run
        return pm_cell(float(np.mean(v)), float(np.std(v, ddof=1)), 3)
    T = Table('t10_attacker', [('setting', 'Training → test attacker', 'l'), ('cmf', 'CyberMetaFormer', 'r'),
                               ('gru', 'Full-history GRU', 'r'), ('tm', 'TimeMixer (k = 64)', 'r')])
    for tr, te in [('bline', 'Meander'), ('meander', 'B-line')]:
        name = {'bline': 'B-line', 'meander': 'Meander'}[tr]
        for split, sl in [('test_shift', 'test'), ('ood_test_shift', 'OOD')]:
            T.add(f'{tr}_{split}', setting=txt(f'{name} → {te}, {sl}'),
                  cmf=ms3(f'cybermetaformer-saldirgan_{tr}_k101', split),
                  gru=ms3(f'gru_full-saldirgan_{tr}_k101', split), tm=ms3(f'timemixer-saldirgan_{tr}_k64', split))
    T.note = ('Note. Host macro-F1, mean ± SD over seeds 0–2. Each model is trained on one attacker and evaluated on '
              'the test and OOD episodes of the other attacker.')
    return T


def t11_efficiency(runs, abl, basis):
    with open(os.path.join(RESULTS, 'asama3', 'verimlilik', 'olcum_apple-m3-pro.json')) as f:
        E = json.load(f)
    cell = {(c['stage'], c['model'], c['k']): c for c in E['cells']}
    s2 = stage_summary('asama2')
    cmf = subset(runs['cmf'], CMF_SEEDS[basis])
    rows = [('b1', 'CyberMetaFormer (B1)', ('asama3', 'cybermetaformer', 101), cmf['ood_f1']['mean'], cmf['params']),
            ('b2', 'CyberMetaFormer (B2)', ('asama3', 'cybermetaformer-b2', 101), abl['b2']['ood_f1']['mean'], None),
            ('gru_full', 'GRU (full history)', ('asama1', 'gru_full', 101), runs['gru_full']['ood_f1']['mean'],
             runs['gru_full']['params']),
            ('timemixer', 'TimeMixer (k = 64)', ('asama2', 'timemixer', 64),
             gmean(s2['timemixer_k64'], 'ood_host_macro_f1'), None),
            ('transformer', 'Transformer (k = 64)', ('asama2', 'transformer', 64),
             gmean(s2['transformer_k64'], 'ood_host_macro_f1'), None)]
    T = Table('t11_efficiency', [('model', 'Model', 'l'), ('params', 'Params', 'r'),
                                 ('lat', 'CPU latency (ms / step)', 'r'), ('ood', 'OOD host macro-F1', 'r')])
    for key, lab, ck, ood, par in rows:
        c = cell[ck]
        assert par is None or par == c['params'], (key, par, c['params'])
        T.add(key, model=txt(lab), params=intc(c['params']), lat=num(c['lat_cpu_ms'], 2), ood=num(ood, 3))
    T.note = (f"Note. Earlier measurement (Stage 3 efficiency run): batch size 1 on an Apple M3 Pro CPU, "
              f"{E['cpu_threads']} threads, PyTorch {E['torch']}; median of 50 timed calls after 10 warm-up calls "
              '(CyberMetaFormer: 91 consecutive step-mode calls). Full-history GRU and CyberMetaFormer run in step '
              f"mode. OOD F1: CyberMetaFormer seeds {seed_text(cmf['seeds'])}, B2 seeds 0–2, GRU seeds "
              f"{seed_text(runs['gru_full']['seeds'])}, Stage 2 models seeds 0–2.")
    return T


def tB1_hyperparameters(cmf_params):
    B = hpo('asama3', 'cybermetaformer_k101')
    a = B['overrides']['model_args']
    T = Table('tB1_hyperparameters', [('name', 'Hyperparameter', 'l'), ('value', 'Value', 'r')])
    for key, lab, cell in [('d', 'Token dimension d', intc(a['d'])), ('layers', 'MetaFormer blocks', intc(a['layers'])),
                           ('heads', 'Host-attention heads', intc(a['heads'])),
                           ('mlp_ratio', 'Channel-MLP expansion ratio', intc(a['mlp_ratio'])),
                           ('state', 'SSD state size N', intc(a['state'])),
                           ('kernel', 'Causal depthwise kernel (steps)', intc(a['kernel'])),
                           ('dropout', 'Dropout', num(a['dropout'], 3)), ('lr', 'Learning rate', sci(B['overrides']['lr'])),
                           ('wd', 'Weight decay', sci(B['overrides']['weight_decay'])),
                           ('params', 'Model parameters', intc(cmf_params)),
                           ('val', 'Validation host macro-F1 (best trial)', num(B['value'], 4))]:
        T.add(key, name=txt(lab), value=cell)
    p = B['protocol']
    T.note = (f"Note. Best of {p['n_trials']} trials ({B['n_complete']} completed, {B['n_pruned']} pruned; trial "
              f"{B['trial']}); each trial trained on {p['train_episodes']:,} episodes for {p['epochs']} epochs.")
    return T


# ── item 3: class level (makale/class_level, class_level.py) ───────────

CLASS_MODELS = [('CyberMetaFormer', 'CMF', 'cmf'), ('Full-history GRU', 'GRU', 'gru_full')]
SPLIT_LABEL = {'test': 'Test', 'ood': 'OOD'}
HOSTS11 = ['user1', 'user2', 'user3', 'user4', 'ent0', 'ent1', 'ent2', 'ophost0', 'ophost1', 'ophost2', 'opserv']


def sinif():
    with open(os.path.join(RESULTS, 'makale', 'class_level', 'ozet.json')) as f:
        return json.load(f)


def pct(d, nd=2, i=None):
    m, s = (d['mean'], d['sd']) if i is None else (d['mean'][i], d['sd'][i])
    return Cell(f'{100 * m:.{nd}f} {PM}{100 * s:.{nd}f}', (m, s))


def check_class(S, runs):
    """Host macro-F1 and seeds of class_level must match the seed means of the tables."""
    for label, _, key in CLASS_MODELS:
        assert S[label]['seeds'] == runs[key]['seeds'], label
        for split, m in [('test', 'test_f1'), ('ood', 'ood_f1')]:
            assert abs(S[label][split]['host_macro_f1']['mean'] - runs[key][m]['mean']) < 1e-9, (label, split)
            assert S[label][split]['prediction_seeds'] == runs[key]['seeds'], (label, split)


def t_class_level(runs):
    S = sinif()
    check_class(S, runs)
    T = Table('t_class_level', [('split', 'Split', 'l'), ('model', 'Model', 'l'), ('cls', 'Class', 'l'),
                                ('prec', 'Precision', 'r'), ('rec', 'Recall', 'r'), ('f1', 'F1', 'r'),
                                ('hf1', 'Host-averaged F1', 'r')])
    for split in ['test', 'ood']:
        for label, short_, _ in CLASS_MODELS:
            r = S[label][split]
            for i, c in enumerate(S['classes']):
                T.add(f'{split}_{short_}_{c}', split=txt(SPLIT_LABEL[split]), model=txt(short_), cls=txt(c),
                      prec=pm_cell(r['precision']['mean'][i], r['precision']['sd'][i], 3),
                      rec=pm_cell(r['recall']['mean'][i], r['recall']['sd'][i], 3),
                      f1=pm_cell(r['f1']['mean'][i], r['f1']['sd'][i], 3),
                      hf1=pm_cell(r['host_avg_f1']['mean'][i], r['host_avg_f1']['sd'][i], 3))
    T.note = ('Note. Seed mean ± SD over seeds 0–4 of each model. Precision, recall and F1 of each class from the '
              'confusion pooled over the 11 evaluated hosts and all host-steps of the split; host-averaged F1 averages '
              'the per-host F1 of the class over the hosts in which it occurs. The 11 evaluated hosts are the label '
              'hosts with at least one compromised step (def is never compromised; user0, the attacker\'s foothold, '
              'is not a label host). Computed from the stored per-episode confusion matrices without new inference.')
    return T


def t_class_errors(runs):
    S = sinif()
    check_class(S, runs)
    T = Table('t_class_errors', [('split', 'Split', 'l'), ('model', 'Model', 'l'),
                                 ('rc', 'Root → clean (%)', 'r'), ('ru', 'Root → user (%)', 'r'),
                                 ('uc', 'User → clean (%)', 'r'), ('fa', 'False alarm on clean (%)', 'r'),
                                 ('worst', 'Worst host (macro-F1)', 'l'), ('det', 'Silent-onset spells detected (%)', 'r'),
                                 ('lat', 'Silent-onset latency, median / p90 (steps)', 'r')])
    n_spells, alert_det = {}, []
    for split in ['test', 'ood']:
        for label, short_, _ in CLASS_MODELS:
            r = S[label][split]
            w = r['worst_host']
            L = [x['latency']['silent_onset'] for x in r['prediction_per_seed']]
            det = [x['breach']['detected_fraction'] for x in L]
            A = [x['latency']['alert_onset']['breach'] for x in r['prediction_per_seed']]
            assert all(a['median'] == 0 for a in A)
            alert_det += [a['detected_fraction'] for a in A]
            n_spells[split] = {x['n_spells'] for x in L}
            assert len(n_spells[split]) == 1
            med, p90 = np.mean([x['breach']['median'] for x in L]), np.mean([x['breach']['p90'] for x in L])
            T.add(f'{split}_{short_}', split=txt(SPLIT_LABEL[split]), model=txt(short_),
                  rc=pct(r['root_to_clean']), ru=pct(r['root_to_user']), uc=pct(r['user_to_clean']),
                  fa=pct(r['clean_false_alarm']),
                  worst=txt(f"{w} ({r['per_host_macro_f1'][w]['mean']:.3f})", r['per_host_macro_f1'][w]['mean']),
                  det=Cell(f'{100 * np.mean(det):.1f} {PM}{100 * np.std(det, ddof=1):.1f}', (np.mean(det),)),
                  lat=Cell(f'{med:.1f} / {p90:.1f}', (med, p90)))
    T.note = ('Note. Seed mean ± SD over seeds 0–4. Error rates are rows of the pooled, row-normalized confusion: '
              'root → clean = P(ŷ = clean | y = root); false alarm on clean = P(ŷ ≠ clean | y = clean). Worst host: '
              'the evaluated host with the lowest seed-mean macro-F1 (all hosts in Table @tbl:host). A spell is a '
              'maximal run of consecutive compromised steps (y > 0) on one host; it is silent-onset when its first step '
              f"carries no alert (n = {n_spells['test'].pop():,} test and {n_spells['ood'].pop():,} OOD spells; spells "
              f'with an alert at onset are detected in at least {100 * min(alert_det):.2f}% of cases by both models, '
              'with median latency 0). '
              'Detected: the model predicts a compromise (ŷ > 0) at some step of the spell; latency: steps from the '
              'onset to that first prediction, over detected spells; median and 90th percentile per seed, averaged '
              'over seeds. Latencies use the argmax of the stored float16 logits.')
    return T


def tA_host_level(runs):
    S = sinif()
    check_class(S, runs)
    T = Table('tA_host_level', [('host', 'Host', 'l')] +
              [(f'{sp}_{s}', f'{SPLIT_LABEL[sp]}, {s}', 'r') for sp in ['test', 'ood'] for _, s, _ in CLASS_MODELS])
    for h in HOSTS11:
        cells = {}
        for sp in ['test', 'ood']:
            for label, s, _ in CLASS_MODELS:
                d = S[label][sp]['per_host_macro_f1'][h]
                cells[f'{sp}_{s}'] = pm_cell(d['mean'], d['sd'], 3)
        T.add(h, host=txt(h), **cells)
    cells = {}
    for sp in ['test', 'ood']:
        for label, s, _ in CLASS_MODELS:
            vals = [S[label][sp]['per_host_macro_f1'][h]['mean'] for h in HOSTS11]
            assert abs(np.mean(vals) - S[label][sp]['host_macro_f1']['mean']) < 1e-9
            d = S[label][sp]['host_macro_f1']
            cells[f'{sp}_{s}'] = pm_cell(d['mean'], d['sd'], 3)
    T.add('mean', host=txt('Host macro-F1 (mean of the 11 hosts)'), **cells)
    T.note = ('Note. Per-host macro-F1 (Section 3.6), seed mean ± SD over seeds 0–4 of CyberMetaFormer (CMF) and the '
              'full-history GRU. The macro average of each host covers the classes that occur on that host; the last '
              'row is the host macro-F1 of Table @tbl:ana.')
    return T


def tA_silent_confusion(runs):
    S = sinif()
    check_class(S, runs)
    T = Table('tA_silent_confusion', [('split', 'Split', 'l'), ('model', 'Model', 'l')] +
              [(f'{t}_{p}', f'{t} → {p} (%)', 'r') for t in ['user', 'root'] for p in S['classes']])
    for split in ['test', 'ood']:
        for label, s, _ in CLASS_MODELS:
            d = S[label][split]['silent_row_normalized']
            cells = {f'{t}_{p}': Cell(f"{100 * d['mean'][ti][pi]:.1f} {PM}{100 * d['sd'][ti][pi]:.1f}",
                                      (d['mean'][ti][pi],))
                     for ti, t in [(1, 'user'), (2, 'root')] for pi, p in enumerate(S['classes'])}
            T.add(f'{split}_{s}', split=txt(SPLIT_LABEL[split]), model=txt(s), **cells)
    T.note = ('Note. Confusion restricted to silent host-steps (compromised, no alert in the current observation), '
              'row-normalized; seed mean ± SD over seeds 0–4, from the stored logits. About 98% of silent host-steps '
              'are root compromises (Table @tbl:sessiz), so the root rows dominate silent recall.')
    return T


def tA_item4_pairs(runs, m4):
    """Item 4 family (35 rows) as one table: the version in the paper's appendix; the first block of a4_pairs.md."""
    allc = list(CRIT)
    T = Table('tA_item4_pairs', [('a', 'Model A', 'l'), ('b', 'Model B', 'l'), ('crit', 'Criterion', 'l'),
                                 ('va', 'A', 'r'), ('vb', 'B', 'r'), ('d', 'A − B [95% CI]', 'r'),
                                 ('pp', 'p (Holm, pair)', 'r'), ('pf', 'p (Holm, family)', 'r')],
              group=('a', 'b'))
    for ka, kb in [('gru_host_topo', 'cmf_m4'), ('gru_host_attn', 'cmf_m4'), ('gru_host_topo', 'gru_host_attn'),
                   ('gru_host_topo', 'gru_full'), ('gru_host_attn', 'gru_full')]:
        for c in allc:
            r = find(m4, RUNS[ka][1], RUNS[kb][1], c)
            m = next((k for k, v in TO_CRIT.items() if v == c), None)
            if m is not None:
                check_point(r, runs, ka, kb, m)
            assert len(r['seeds_a']) == len(r['seeds_b']) == 5
            k = 4 if 'f1' in c or 'drop' in c else 3
            T.add(f'{ka}__{kb}__{c}', a=txt(short(ka)), b=txt(short(kb)), crit=txt(CRIT[c]), va=num(r['a'], k),
                  vb=num(r['b'], k), d=dci(r['diff'], r['lo'], r['hi'], k),
                  pp=txt(pfmt(r['p_holm']), r['p_holm']), pf=txt(pfmt(r['p_holm_plan']), r['p_holm_plan']))
    T.note = (f"Note. All models seeds 0–4. Paired episode bootstrap ({m4['n_boot']:,} resamples, Algorithm 2); Holm "
              f"(pair) over the {len(allc)} criteria of a pair, Holm (family) over the m = {m4['n_rows']} rows of this "
              'table. The host-token GRU references are compared with CyberMetaFormer run 2, trained on the same hardware '
              '(Table @tbl:hostgru). Lower is better for the relative OOD drop.')
    return T


# ── items 4 and 5 (phase 1) ──────────────────────────────────────────────

def latency(prefix, lat):
    """Run prefix → (median, p95) ms; ana_m4 has the same architecture as CMF, so the CMF measurement is used."""
    name = prefix.rsplit('_k', 1)[0].replace('-ana_m4', '')
    m = lat['models'][name]
    return m['latency'][LAT_THREADS]['median_ms'], m['latency'][LAT_THREADS]['p95_ms'], m['params']


A4_LABEL = {'cmf': 'CyberMetaFormer', 'cmf_m4': 'CyberMetaFormer, run 2', 'host_tam': 'CMF, no topology bias',
            'konv_gru': 'CMF, GRU instead of SSD', 'gru_host_topo': 'Host-token GRU, topology bias',
            'gru_host_attn': 'Host-token GRU, no topology bias', 'gru_full': 'Full-history GRU',
            'host_yok': 'CMF, no host mixer'}


def a4_reference(runs, lat):
    """Item 4 references. Latency is in the cost table (same measurement); only the parameter count is checked here."""
    L = ['| Model | Params | Test host macro-F1 | OOD host macro-F1 | Test silent recall | OOD silent recall | '
         'OOD N_hist recall |', '|---|---:|---:|---:|---:|---:|---:|']
    assert set(A4_LABEL) == set(RUNS)
    for key in RUNS:
        r = runs[key]
        n = len(r['seeds'])
        assert n in (3, 5), key
        _, _, n_par = latency(r['prefix'], lat)
        assert n_par == r['params'], f'{key}: parameter count differs from the latency measurement'
        L.append(f"| {A4_LABEL[key]}{'†' if n == 3 else ''} | {r['params']:,} | " +
                 ' | '.join(ms(r[m], m, n) for m in ['test_f1', 'ood_f1', 'test_silent', 'ood_silent', 'ood_hist'])
                 + ' |')
    note = ('Note. Seed mean ± SD over seeds 0–4; † seeds 0–2. Host macro-F1 as defined in Section 3.6; silent recall '
            'over compromised host-steps without a current alert; N_hist recall over silent host-steps whose host has '
            'had no alert up to t (online-applicable; Section 3.3). CyberMetaFormer is the main B1 model (conv + SSD, '
            'topology bias); run 2 is an independent five-seed training of the same configuration on the same hardware '
            'as the host-token GRU references, which are compared with it (Table @tbl:hostgru_cift). The host-token '
            'GRU variants keep every CyberMetaFormer component except the temporal mixer, which is a per-host GRU over '
            'the same host tokens. The full-history GRU takes the flat 78-d observation. Online CPU step latency and '
            'carried state of these models are given in Table @tbl:inference_cost.')
    return '\n'.join(L) + '\n\n' + note + '\n'


def pair_block(plan, pairs, runs, title):
    L = [f"**{title}** (paired episode bootstrap, {plan['n_boot']:,} resamples; plan-wide Holm over "
         f"m = {plan['n_rows']} rows)", '', '| A | B | Seeds A / B | Criterion | A | B | A − B | 95% CI | '
         'p (Holm, pair) | p (Holm, plan) |', '|---|---|---|---|---:|---:|---:|---|---:|---:|']
    for ka, kb, crits in pairs:
        for c in crits:
            r = find(plan, RUNS[ka][1], RUNS[kb][1], c)
            m = next((k for k, v in TO_CRIT.items() if v == c), None)
            if m is not None:
                check_point(r, runs, ka, kb, m)
            k = 4 if 'f1' in c or 'drop' in c else 3
            L.append(f"| {short(ka)} | {short(kb)} | {len(r['seeds_a'])} / {len(r['seeds_b'])} | {CRIT[c]} | "
                     f"{r['a']:.{k}f} | {r['b']:.{k}f} | {signed(r['diff'], k)} | "
                     f"[{signed(r['lo'], k)}, {signed(r['hi'], k)}] | {pfmt(r['p_holm'])} | {pfmt(r['p_holm_plan'])} |")
    return L


SHORT = {'cmf': 'CMF', 'cmf_m4': 'CMF, run 2', 'host_tam': 'CMF, no topology bias', 'konv_gru': 'CMF, GRU for SSD',
         'gru_host_topo': 'Host-token GRU, topology bias', 'gru_host_attn': 'Host-token GRU, no bias',
         'gru_full': 'Full-history GRU', 'host_yok': 'CMF, no host mixer'}


def short(k):
    return SHORT[k]


def a4_pairs(runs, m4, m35):
    allc = list(CRIT)
    L = pair_block(m4, [('gru_host_topo', 'cmf_m4', allc), ('gru_host_attn', 'cmf_m4', allc),
                        ('gru_host_topo', 'gru_host_attn', allc), ('gru_host_topo', 'gru_full', allc),
                        ('gru_host_attn', 'gru_full', allc)], runs, 'Item 4 family')
    abl = ['test_host_macro_f1', 'ood_host_macro_f1', 'ood_silent_recall']
    L += [''] + pair_block(m35, [('host_tam', 'cmf', abl), ('konv_gru', 'cmf', abl), ('host_yok', 'cmf', abl)],
                           runs, 'Related ablations from the main comparison plan (5-seed rule)')
    L += [''] + pair_block(m35, [('cmf', 'host_yok', ['test_never_recall', 'ood_never_recall']),
                                 ('host_tam', 'host_yok', ['test_never_recall', 'ood_never_recall'])],
                           runs, 'No-alert subset, host mixing (same plan)')
    note = ('Note. A − B uses each model\'s seed mean over the seeds listed; ablations are compared with '
            'CyberMetaFormer seeds 0–2. Both models see the same episode weights in every resample; test and OOD '
            'episodes are resampled independently (Algorithm 2). Two-sided p = 2(min(#Δ ≤ 0, #Δ ≥ 0) + 1)/(n + 1); '
            'the smallest attainable raw p is 2/(n + 1). Lower is better for the relative OOD drop.')
    return '\n'.join(L) + '\n\n' + note + '\n'


def seed_diffs(runs, a='cmf', b='gru_full'):
    """Differences at the same seed index (a − b): mean, SD, min–max, number of seeds favouring a, t₄ interval."""
    A = {r['seed']: r for r in runs[a]['per_seed']}
    B = {r['seed']: r for r in runs[b]['per_seed']}
    seeds = sorted(set(A) & set(B))
    out = {'seeds': seeds}
    for m in METRICS:
        d = np.array([A[s][m] - B[s][m] for s in seeds])
        half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
        out[m] = {'per_seed': d.tolist(), 'mean': float(d.mean()), 'sd': float(d.std(ddof=1)), 'min': float(d.min()),
                  'max': float(d.max()), 'favouring_a': int((np.sign(d) == BETTER[m]).sum()),
                  't_lo': float(d.mean() - half), 't_hi': float(d.mean() + half), 'df': len(d) - 1}
    return out


def a5_seeds(runs, diffs):
    cols = ['test_f1', 'ood_f1', 'rel_drop', 'ood_silent', 'ood_hist']
    L = ['| Seed | Model | Test host macro-F1 | OOD host macro-F1 | Relative OOD drop | OOD silent recall | '
         'OOD N_hist recall |', '|---|---|---:|---:|---:|---:|---:|']
    A = {r['seed']: r for r in runs['cmf']['per_seed']}
    B = {r['seed']: r for r in runs['gru_full']['per_seed']}
    for s in diffs['seeds']:
        for name, R in [('CMF', A), ('GRU', B)]:
            L.append(f'| {s} | {name} | ' + ' | '.join(f'{R[s][m]:.{nd(m)}f}' for m in cols) + ' |')
    n = len(diffs['seeds'])
    for name, k in [('CMF', 'cmf'), ('GRU', 'gru_full')]:
        L.append(f'| Mean ± sd | {name} | ' + ' | '.join(ms(runs[k][m], m, n) for m in cols) + ' |')
    D = diffs
    L.append('| CMF − GRU | Mean ± sd | ' + ' | '.join(
        f"{signed(D[m]['mean'], nd(m))} {PM}{D[m]['sd']:.{nd(m)}f}" for m in cols) + ' |')
    L.append('| CMF − GRU | Min – max | ' + ' | '.join(
        f"{signed(D[m]['min'], nd(m))} – {signed(D[m]['max'], nd(m))}" for m in cols) + ' |')
    L.append('| CMF − GRU | Seeds favouring CMF | ' + ' | '.join(f"{D[m]['favouring_a']} / {n}" for m in cols) + ' |')
    L.append(f"| CMF − GRU | Seed-level 95% t-interval (df = {n - 1}) | " + ' | '.join(
        f"[{signed(D[m]['t_lo'], nd(m))}, {signed(D[m]['t_hi'], nd(m))}]" for m in cols) + ' |')
    note = ('Note. Both models trained with seeds 0–4 (main CyberMetaFormer run). Differences pair runs by seed '
            'index; the index '
            'carries no shared randomness between the two architectures, so the pairing is a bookkeeping convention. '
            'The t-interval treats the five seeds as the sample and describes retraining variability only; it is a '
            'seed-level interval, not an episode-level one, and with five seeds it is wide. Lower is better for the '
            'relative OOD drop.')
    return '\n'.join(L) + '\n\n' + note + '\n'


def a5_two_sources(runs, diffs, m35):
    L = ['| Criterion | CMF (mean ± sd) | GRU (mean ± sd) | Δ (CMF − GRU) | Episode bootstrap 95% CI | '
         'p (Holm, pair) | p (Holm, plan) | Seed-level 95% t-interval | Seeds favouring CMF |',
         '|---|---:|---:|---:|---|---:|---:|---|---:|']
    n = len(diffs['seeds'])
    for m in ['test_f1', 'ood_f1', 'rel_drop', 'ood_silent']:
        r = find(m35, RUNS['cmf'][1], RUNS['gru_full'][1], TO_CRIT[m])
        check_point(r, runs, 'cmf', 'gru_full', m)
        assert r['seeds_a'] == r['seeds_b'] == diffs['seeds']
        assert abs(r['diff'] - diffs[m]['mean']) < 1e-9
        k, D = nd(m), diffs[m]
        L.append(f"| {CRIT[TO_CRIT[m]]} | {ms(runs['cmf'][m], m, n)} | {ms(runs['gru_full'][m], m, n)} | "
                 f"{signed(r['diff'], k)} | [{signed(r['lo'], k)}, {signed(r['hi'], k)}] | {pfmt(r['p_holm'])} | "
                 f"{pfmt(r['p_holm_plan'])} | [{signed(D['t_lo'], k)}, {signed(D['t_hi'], k)}] | "
                 f"{D['favouring_a']} / {n} |")
    note = (f"Note. Five seeds per model (0–4). The episode bootstrap ({m35['n_boot']:,} resamples, "
            'Algorithm 2) holds the trained models fixed and resamples test and OOD episodes; it does not include '
            'retraining variability. The seed-level interval resamples nothing and uses the spread of the five '
            'same-index differences; it does not include episode sampling. Neither interval alone is the total '
            f"uncertainty. Holm (pair) adjusts over the four criteria of this pair; Holm (plan) over all "
            f"m = {m35['n_rows']} rows of the main comparison plan, so the smallest attainable plan-wide p is "
            f"{min(1.0, m35['n_rows'] * 2 / (m35['n_boot'] + 1)):.4f}.")
    return '\n'.join(L) + '\n\n' + note + '\n'


# ── items 6–8 and 10 (phase 2): shift types, host order and topology, failure, intervention, corrected observation,
#    cost. Source: makale/<analysis>/ozet.json; all seeds 0–4, without retraining. ───────────────────────────────

F2_LABEL = {'cybermetaformer': 'CMF', 'cybermetaformer-host_tam': 'CMF, no topology bias',
            'cybermetaformer-gru_host_topo': 'Host-token GRU, topology bias',
            'cybermetaformer-gru_host_attn': 'Host-token GRU, no topology bias', 'gru_full': 'Full-history GRU'}
F2_RUN = {'cybermetaformer': 'cmf', 'cybermetaformer-host_tam': 'host_tam', 'cybermetaformer-gru_host_topo': 'gru_host_topo',
          'cybermetaformer-gru_host_attn': 'gru_host_attn', 'gru_full': 'gru_full'}
TOPO_BIASED = ('cybermetaformer', 'cybermetaformer-gru_host_topo')
F2_CRIT = [('host_macro_f1', 'OOD host macro-F1', 'ood_f1'), ('silent_recall', 'OOD silent recall', 'ood_silent'),
           ('never_recall', 'OOD N_hist recall', 'ood_hist')]
F2_SEEDS = [0, 1, 2, 3, 4]


def analysis(name):
    with open(os.path.join(RESULTS, 'makale', name, 'ozet.json')) as f:
        A = json.load(f)
    assert A['seeds'] == F2_SEEDS, name
    return A


def get_row(A, **kw):
    rows = [r for r in A['rows'] if all(r.get(k) == v for k, v in kw.items())]
    assert len(rows) == 1, (kw, len(rows))
    return rows[0]


def check_main(value, runs, model, metric, tol=1e-4):
    """The value an analysis computes from the saved logits must match the 5-seed mean of the main tables
    (metrics.json and episode_slices.npz); logits are stored in float16, so the tolerance is 1e-4 (as in ood_suite)."""
    key = F2_RUN[model]
    assert runs[key]['seeds'] == F2_SEEDS, key
    assert abs(value - runs[key][metric]['mean']) < tol, (model, metric, value, runs[key][metric]['mean'])


def dpm(d, k=3):
    """Δ mean ± SD cell."""
    return Cell(minus(f"{d['mean']:+.{k}f}") + f" {PM}{d['sd']:.{k}f}", (d['mean'], d['sd']))


def holm_note(A):
    n, m = A['n_boot'], len(A['rows'])
    return (f"paired episode bootstrap of the seed mean ({n:,} resamples, Algorithm 2), Holm over all m = {m} rows of "
            f"this analysis (smallest attainable adjusted p = {min(1.0, m * 2 / (n + 1)):.4f})")


def t_shift_types():
    """§3.2: types and episode counts of the evaluated distribution shifts."""
    P = json.load(open(os.path.join(RESULTS, 'asama3', 'ood_suite', 'ozet_5seed.json')))['sets']
    perm, topo, corr = analysis('ood_permutation'), analysis('ood_topology'), analysis('corrected_obs')
    att = {}
    for tr in ['bline', 'meander']:
        rep = json.load(open(os.path.join(RESULTS, 'asama3', f'cybermetaformer-saldirgan_{tr}_k101_s0',
                                          'metrics.json')))['reports']
        att[tr] = (rep['test_shift']['n_episodes'], rep['ood_test_shift']['n_episodes'])
    assert att['bline'] == att['meander']
    assert P['eps0.5']['n_episodes'] == P['eps0.7']['n_episodes']
    assert perm['sets']['test']['n_episodes'] == corr['sets']['test']['n_episodes']
    assert len({corr['sets'][s]['n_episodes'] for s in ['ood', 'eps0.5', 'eps0.7', 'remove']}) == 1
    assert topo['sets']['eps0.1']['rewired']['n_episodes'] == topo['sets']['ood']['rewired']['n_episodes']
    T = Table('t_shift_types', [('type', 'Shift type', 'l'), ('cond', 'Condition', 'l'), ('change', 'What changes', 'l'),
                                ('n', 'Episodes', 'r'), ('retrain', 'Retraining', 'l'), ('where', 'Results', 'l')],
              group=('type',))
    rows = [
        ('Policy parameter', 'ε = 0.3 (main OOD test)',
         'Random-action probability of the restore-based defender raised from 0.1 to 0.3',
         intc(P['ood']['n_episodes']), 'No', 'Tables @tbl:ana, @tbl:politika'),
        ('Policy parameter', 'ε = 0.5 and 0.7', 'Same parameter, larger values',
         txt(f"{P['eps0.5']['n_episodes']:,} each", P['eps0.5']['n_episodes']), 'No', 'Table @tbl:politika'),
        ('Observation mechanism', 'Periodic analyse on idle steps',
         'Idle steps are used to analyse hosts, which re-emits the root flag of compromised hosts (alert refresh)',
         intc(P['analyse']['n_episodes']), 'No', 'Tables @tbl:politika, @tbl:failure_analysis'),
        ('Observation mechanism', 'Remove instead of restore',
         'Removal replaces restore; a failed remove keeps root access but shows the "removed" flag',
         intc(P['remove']['n_episodes']), 'No', 'Tables @tbl:politika, @tbl:failure_analysis'),
        ('Observation mechanism', 'Corrected remove flag',
         'Same episodes and defender actions re-simulated; a failed remove no longer shows "removed"',
         txt(f"test {corr['sets']['test']['n_episodes']:,}; four shift sets "
             f"{corr['sets']['ood']['n_episodes']:,} each"), 'No', 'Table @tbl:duzeltilmis'),
        ('Simulator version', 'Earlier simulator version',
         'Documented simulator bugs left in place; OOD policy', intc(P['bug_ood']['n_episodes']), 'No',
         'Table @tbl:politika'),
        ('Attacker', 'B-line ↔ Meander transfer', 'Model trained on one attacker and tested on the other',
         txt(f"test {att['bline'][0]:,}; OOD {att['bline'][1]:,}"), 'Yes: one model per training attacker',
         'Table @tbl:saldirgan'),
        ('Host identity', 'Host permutations P1, P1s, P2, P2s, P3',
         'Observation columns and labels of hosts permuted within or across attack-graph roles; P3 permutes only '
         'the host embeddings', txt(f"test {perm['sets']['test']['n_episodes']:,}; OOD "
                                    f"{perm['sets']['ood']['n_episodes']:,}"), 'No', 'Table @tbl:host_kayma'),
        ('Topology', 'Rewired user → enterprise links',
         'Simulator links user1/2 → ent1 and user3/4 → ent0; host count, subnets and observation layout unchanged',
         txt(f"{topo['sets']['ood']['rewired']['n_episodes']:,} per policy (ε = 0.1, 0.3)"),
         'No; relation matrix rebuilt in one condition', 'Table @tbl:host_kayma'),
    ]
    for i, (typ, cond, change, n, retrain, where) in enumerate(rows):
        T.add(f'r{i}', type=txt(typ), cond=txt(cond), change=txt(change), n=n, retrain=txt(retrain), where=txt(where))
    T.note = ('Note. Shifts relative to the training data, which contain the five training defender policies '
              '(Table @tbl:politikalar), both attackers and the original MiniCAGE topology. Policy and mechanism shifts '
              'and the earlier simulator version use new episodes of both attackers; the corrected copy re-simulates '
              'the episodes of the test, OOD and shift sets with the recorded defender actions. Not evaluated: the full '
              'CybORG engine and networks with a different number of hosts (Section 5.3).')
    return T


PERM_COND = [('id', 'Identity'), ('P1', 'P1'), ('P1s', 'P1s'), ('P2', 'P2'), ('P2s', 'P2s'), ('P3', 'P3')]


def t_host_shift(runs):
    """§4.8: host permutation and rewired topology, OOD (ε = 0.3); seed mean."""
    P, TP = analysis('ood_permutation'), analysis('ood_topology')
    T = Table('t_host_shift', [('crit', 'Criterion', 'l'), ('model', 'Model', 'l')] +
              [(k, h, 'r') for k, h in PERM_COND] +
              [('rew', 'Rewired simulator', 'r'), ('upd', 'Rewired, bias updated', 'r')], group=('crit',))
    max_sd = {}
    for crit, clab, metric in F2_CRIT:
        sds = []
        for m, lab in F2_LABEL.items():
            pm, tm = P['models'][m]['ood'], TP['models'][m]['ood']
            # original topology = evaluation without permutation: same episodes, same logits
            assert abs(tm['original'][crit]['mean'] - pm['id'][crit]['mean']) < 1e-9, (m, crit)
            check_main(pm['id'][crit]['mean'], runs, m, metric)
            cells = {}
            for k, _ in PERM_COND:
                if k in pm:
                    cells[k] = num(pm[k][crit]['mean'], 3)
                    sds.append(pm[k][crit]['sd'])
                else:
                    cells[k] = txt('—')
            cells['rew'] = num(tm['rewired'][crit]['mean'], 3)
            sds.append(tm['rewired'][crit]['sd'])
            if 'rewired_prior' in tm:
                assert m in TOPO_BIASED
                cells['upd'] = num(tm['rewired_prior'][crit]['mean'], 3)
                sds.append(tm['rewired_prior'][crit]['sd'])
            else:
                cells['upd'] = txt('—')
            T.add(f'{crit}__{m}', crit=txt(clab), model=txt(lab), **cells)
        max_sd[crit] = max(sds)
    S, R = P['sets']['ood'], TP['sets']['ood']['rewired']
    T.note = ('Note. OOD test policy (ε = 0.3), mean over seeds 0–4; the largest seed SD in the table is '
              f"{max_sd['host_macro_f1']:.3f} (F1), {max_sd['silent_recall']:.3f} (silent recall) and "
              f"{max_sd['never_recall']:.3f} (N_hist recall). All models are trained on the original host order and "
              'topology and evaluated without retraining. P1: permutation within attack-graph roles (ophost0 ← ophost1 '
              '← ophost2 ← ophost0, user1 ↔ user2, user3 ↔ user4; edges and subnets preserved). P2: across roles (user1 ↔ '
              'user3, user2 ↔ user4; subnets preserved, user → enterprise edges misaligned). Observation columns and '
              'labels move together; in P1 and P2 the decoy-count column also moves, so the host-specific decoy capacity '
              '(user1/2: 4, user3: 2, user4: 1) travels with the host as an identity cue, while P1s and P2s keep the '
              'decoy-count columns in their slot. P3: only the host-embedding rows E_host are permuted as in P2, the data '
              'are unchanged (host-token models only); without topology bias P3 equals P2, because permuting the data is '
              'equivalent to permuting E_host and the relation matrix together. Original episodes: '
              f"n = {S['n_host_steps']:,} host-steps, {S['n_silent']:,} silent, {S['n_never']:,} N_hist. Rewired "
              f"simulator: new episodes of the same policy with user1/2 → ent1 and user3/4 → ent0 ({R['n_episodes']:,} "
              f"episodes; {R['n_silent']:,} silent, {R['n_never']:,} N_hist host-steps); in the simulator the decoy "
              'capacity stays with the host slot, as in P2s. Bias updated: the relation matrix of the two '
              'topology-biased models is rebuilt from the rewired links, weights unchanged. Paired differences with '
              '95% CIs: Table @tbl:host_kayma_fark.')
    return T


def tE_host_shift_diffs():
    """Appendix E: permutation and rewiring differences (same model) and model differences per condition, OOD."""
    P, TP = analysis('ood_permutation'), analysis('ood_topology')
    T = Table('tE_host_shift_diffs', [('model', 'Model', 'l'), ('comp', 'Comparison', 'l'),
                                      ('f1', 'Δ host macro-F1 [95% CI]', 'r'), ('s', 'Δ silent recall [95% CI]', 'r'),
                                      ('h', 'Δ N_hist recall [95% CI]', 'r'), ('p', 'p (Holm), silent recall', 'r')],
              group=('model',))

    def add(key, mlab, clab, A, kw):
        cells = {}
        for col, crit, k in [('f1', 'host_macro_f1', 4), ('s', 'silent_recall', 3), ('h', 'never_recall', 3)]:
            r = get_row(A, criterion=crit, set='ood', **kw)
            cells[col] = dci(r['diff'], r['lo'], r['hi'], k)
            if col == 's':
                cells['p'] = txt(pfmt(r['p_holm']), r['p_holm'])
        T.add(key, model=txt(mlab), comp=txt(clab), **cells)

    for m, lab in F2_LABEL.items():
        for c in ['P1', 'P1s', 'P2', 'P2s', 'P3']:
            if c in P['models'][m]['ood']:
                add(f'{m}__{c}', lab, f'{c} − identity', P, dict(kind='perm_minus_id', model=m, condition=c))
        add(f'{m}__rew', lab, 'Rewired − original', TP, dict(kind='rewired_minus_original', model=m))
        if m in TOPO_BIASED:
            add(f'{m}__upd', lab, 'Rewired, bias updated − original', TP, dict(kind='prior_minus_original', model=m))
    for a, b in [('cybermetaformer', 'gru_full'), ('cybermetaformer', 'cybermetaformer-host_tam'),
                 ('cybermetaformer-gru_host_topo', 'cybermetaformer-gru_host_attn')]:
        mlab = f'{F2_LABEL[a]} − {F2_LABEL[b]}'
        for cond, clab, kind in [('original', 'Original topology', 'pair'), ('P2s', 'P2s', 'pair'),
                                 ('rewired', 'Rewired simulator', 'pair'),
                                 ('rewired_prior', 'Rewired, bias of A updated', 'pair_prior')]:
            add(f'{a}__{b}__{cond}', mlab, clab, TP, dict(kind=kind, a=a, b=b, condition=cond))
    T.note = ('Note. OOD test policy (ε = 0.3), seeds 0–4, no retraining; conditions as in Table @tbl:host_kayma. '
              'Permutations reuse the original episodes (paired bootstrap); the rewired simulator has its own episodes, '
              'so rewired − original differences use a two-sample bootstrap with both sets resampled independently. '
              f"Permutation rows: {holm_note(P)}. Topology rows (rewired, bias updated and model pairs): "
              f"{holm_note(TP)}; both analyses also cover the test set (permutations) and the training policy ε = 0.1 "
              '(topology), which are counted in m but not shown.')
    return T


FAIL_SETS = [('eps0.1', 'ε = 0.1 (training policy)'), ('ood', 'ε = 0.3 (main OOD)'),
             ('analyse', 'Periodic analyse on idle steps'), ('remove', 'Remove instead of restore')]
FAIL_MODELS = ['cybermetaformer', 'cybermetaformer-gru_host_topo', 'cybermetaformer-gru_host_attn', 'gru_full']


def t_failure(runs):
    """§4.7: error types, four policies × four models (failure_analysis.py)."""
    B = analysis('failure_analysis')
    T = Table('t_failure', [('set', 'Policy', 'l'), ('n', 'Silent host-steps, user / root', 'r'),
                            ('model', 'Model', 'l'), ('f1', 'Host macro-F1', 'r'), ('s', 'Silent recall', 'r'),
                            ('src', 'Silent root → clean', 'r'), ('suc', 'Silent user → clean', 'r'),
                            ('arc', 'Root → clean, all root steps', 'r')], group=('set', 'n'))
    sd_err = []
    for s, slab in FAIL_SETS:
        S = B['sets'][s]
        for m in FAIL_MODELS:
            M = B['models'][m][s]
            if s == 'ood':
                check_main(M['host_macro_f1']['mean'], runs, m, 'ood_f1')
                check_main(M['silent']['recall']['mean'], runs, m, 'ood_silent')
            sd_err += [M['silent']['root_to_clean']['sd'], M['silent']['user_to_clean']['sd'],
                       M['all_errors']['root_to_clean']['sd']]
            T.add(f'{s}__{m}', set=txt(slab),
                  n=txt(f"{S['silent_by_class']['user']:,} / {S['silent_by_class']['root']:,}"), model=txt(F2_LABEL[m]),
                  f1=pm_cell(M['host_macro_f1']['mean'], M['host_macro_f1']['sd'], 3),
                  s=pm_cell(M['silent']['recall']['mean'], M['silent']['recall']['sd'], 3),
                  src=num(M['silent']['root_to_clean']['mean'], 3), suc=num(M['silent']['user_to_clean']['mean'], 3),
                  arc=num(M['all_errors']['root_to_clean']['mean'], 3))
    T.note = ('Note. Seed mean (± SD for F1 and silent recall) over seeds 0–4, from the stored logits (argmax), no '
              f"retraining; the error-rate columns have seed SD ≤ {max(sd_err):.3f}. Error rates are within the true "
              'class: silent root → clean is the share of silent root host-steps predicted clean; the last column uses '
              'all root host-steps. ε = 0.1: the test episodes of the training policy '
              f"({B['sets']['eps0.1']['n_episodes']:,}); the other policies: {B['sets']['ood']['n_episodes']:,} episodes "
              'each, both attackers. Slices by attacker, subnet and alert age: Table @tbl:analyse_dilim.')
    return T


SLICE_GROUP = {'attacker_group': 'Attacker × subnet', 'age': 'Alert age', 'last_alert': 'Last alert'}


def tE_failure_slices():
    """Appendix E: silent slices under analyse and remove (attacker × subnet, alert age, last alert)."""
    B = analysis('failure_analysis')
    T = Table('tE_failure_slices', [('set', 'Policy', 'l'), ('slice', 'Silent slice', 'l'), ('n', 'n', 'r'),
                                    ('root', 'Root share', 'r'), ('cmf', 'CMF', 'r'),
                                    ('ht', 'Host-token GRU, topology bias', 'r'), ('gru', 'Full-history GRU', 'r'),
                                    ('d', 'CMF − GRU [95% CI]', 'r'), ('p', 'p (Holm)', 'r')], group=('set',))
    for s, slab in [('analyse', 'Periodic analyse'), ('remove', 'Remove instead of restore')]:
        for key, sl in B['slices'][s].items():
            kind = key.split('|', 1)[0]
            if kind not in SLICE_GROUP or key == 'last_alert|no alert so far':     # this slice equals N_hist
                continue
            if s == 'remove' and kind != 'attacker_group' and key != 'age|never alerted (N_hist)':
                continue                                                         # recall ≈ 1 in the other remove slices
            M = sl['models']
            rows = [r for r in B['rows'] if r['set'] == s and r['slice'] == key and r['b'] == 'gru_full']
            assert len(rows) <= 1 and all(r['a'] == 'cybermetaformer' for r in rows), key
            d = dci(rows[0]['diff'], rows[0]['lo'], rows[0]['hi'], 3) if rows else txt('—')
            p = txt(pfmt(rows[0]['p_holm']), rows[0]['p_holm']) if rows else txt('—')
            T.add(f'{s}__{key}', set=txt(slab), slice=txt(f"{SLICE_GROUP[kind]}: {sl['name']}"), n=intc(sl['n']),
                  root=num(sl['root_share'], 2), cmf=num(M['cybermetaformer']['recall']['mean'], 3),
                  ht=num(M['cybermetaformer-gru_host_topo']['recall']['mean'], 3),
                  gru=num(M['gru_full']['recall']['mean'], 3), d=d, p=p)
    T.note = ('Note. Silent recall within each slice, seed mean over seeds 0–4. n: silent host-steps of the slice; root '
              'share: share of true root among them. Subnets: enterprise = ent0–2 (and def, never compromised), user = '
              'user1–4, operational = ophost0–2 and opserv. Alert age: steps since the last alert on the host; never '
              'alerted = N_hist. Last alert: type of the last alert (exploit activity or a root flag without activity, '
              'the form re-emitted by analyse) and whether the host was restored after it. In the remove policy only the '
              'attacker × subnet and never-alerted slices are shown; the other slices have recall ≥ 0.99 for all models. '
              f"CMF − GRU: {holm_note(B)}; slices with fewer than {B['min_boot']:,} host-steps are not tested (—).")
    return T


ANALYSE_REPR = [('cybermetaformer', 'CyberMetaFormer (main)'), ('cybermetaformer-host_tam', None),
                ('cybermetaformer-host_gcn', None), ('cybermetaformer-host_yok', None),
                ('cybermetaformer-hafiza_yok', None), ('cybermetaformer-yalniz_yas', None),
                ('cybermetaformer-kol_konv', None), ('cybermetaformer-kol_ssd', None),
                ('cybermetaformer-konum_yok', None), ('cybermetaformer-duz78', None), ('gru_full', 'Full-history GRU')]


def tE_analyse_repr():
    """Appendix E: silent recall under analyse, by representation type (seeds 0–2; asama3/aciklama, explain_m3.py)."""
    with open(os.path.join(RESULTS, 'asama3', 'aciklama', 'ozet.json')) as f:
        A = json.load(f)['ablasyon']
    B = analysis('failure_analysis')
    T = Table('tE_analyse_repr', [('model', 'Model', 'l'), ('tok', 'Input', 'l'),
                                  ('e1', 'Silent recall, ε = 0.1', 'r'), ('ood', 'Silent recall, OOD', 'r'),
                                  ('an', 'Silent recall, periodic analyse', 'r'), ('ratio', 'Analyse / ε = 0.1', 'r'),
                                  ('f1', 'Host macro-F1, periodic analyse', 'r')])
    for m, lab in ANALYSE_REPR:
        r = A[m]
        v = {s: np.asarray(r[s]['silent_recall']) for s in ['eps0.1', 'ood', 'analyse']}
        assert all(len(x) == 3 for x in v.values()), m
        if m in B['models']:      # seeds 0–2 must match the first three seeds of the 5-seed failure analysis
            assert abs(np.mean(B['models'][m]['analyse']['silent']['recall']['seeds'][:3]) - v['analyse'].mean()) < 1e-4
        f1 = np.asarray(r['analyse']['host_macro_f1'])
        T.add(m, model=txt(lab or ABLATIONS[m.split('-', 1)[1]]),
              tok=txt('Flat 78-d' if m in ('gru_full', 'cybermetaformer-duz78') else 'Host tokens'),
              **{k: pm_cell(v[s].mean(), v[s].std(ddof=1), 3) for k, s in [('e1', 'eps0.1'), ('ood', 'ood'),
                                                                            ('an', 'analyse')]},
              ratio=num(v['analyse'].mean() / v['eps0.1'].mean(), 2), f1=pm_cell(f1.mean(), f1.std(ddof=1), 3))
    T.note = ('Note. Seed mean ± SD over seeds 0–2 (the ablation seeds), from the stored logits (argmax). ε = 0.1: test '
              'episodes of the training policy; OOD: ε = 0.3; periodic analyse: react_restore_analyse. With seeds 0–4 '
              'the periodic-analyse silent recall is 0.625 for CyberMetaFormer and 0.703 for the full-history GRU '
              '(Table @tbl:failure_analysis).')
    T.values = {m: float(np.mean(A[m]['analyse']['silent_recall'])) for m, _ in ANALYSE_REPR}
    return T


IV = [('empty', 'Empty (t = 0 observation)'), ('swap', 'Swap (other episode, same policies)'),
      ('shuffle', 'Shuffle (in time)')]
IV_HOSTS = [('related', 'Related hosts'), ('random', 'Random unrelated hosts')]
IV_MODELS = [('cybermetaformer', 'cmf', 'CMF'), ('cybermetaformer-host_tam', 'tam', 'CMF, no topology bias'),
             ('gru_full', 'gru', 'Full-history GRU')]


def iv_note_common(M):
    n_sil = sum(t['silent'] for t in M['targets'].values())
    n_hist = sum(t['never'] for t in M['targets'].values())
    return (f"Test episodes of the training policies ({M['n_episodes']:,}). The target host's own observation columns "
            'are kept; the columns of a host set S are replaced at every step. Empty: S\'s own t = 0 observation; swap: '
            'S\'s columns from a random other episode with the same Red and Blue policy (keeps the marginal and the alert '
            'count, breaks the joint distribution); shuffle: S\'s columns permuted in time within the episode (t ≥ 1). '
            'S = related hosts (attack-graph neighbours and same subnet in the training topology) or three random '
            f"equal-size sets of unrelated hosts (pooled). Targets: the {len(M['targets'])} label hosts "
            f"{', '.join(M['targets'])}; host-steps pooled over targets ({n_sil:,} silent, {n_hist:,} N_hist). Silent "
            'and N_hist are defined by the target\'s own observation and are unchanged by the intervention.')


def t_intervention():
    """§4.6: intervention on the observations of other hosts (interventions.py)."""
    M = analysis('interventions')
    sh = M['shift']
    T = Table('t_intervention', [('iv', 'Intervention on S', 'l'), ('hosts', 'Host set S', 'l'),
                                 ('novel', 'Novel observation rows', 'r'), ('trans', 'Novel per-host transitions', 'r')] +
              [(k, lab, 'r') for _, k, lab in IV_MODELS], group=('iv',))
    T.add('base', iv=txt('None (silent recall)'), hosts=txt('—'), novel=num(sh['original']['novel_rows'], 3),
          trans=num(sh['original']['novel_transitions'], 3),
          **{k: pm_cell(M['models'][m]['base']['silent']['mean'], M['models'][m]['base']['silent']['sd'], 3)
             for m, k, _ in IV_MODELS})
    ps = []
    for iv, lab in IV:
        for hs, hlab in IV_HOSTS:
            key = f'{iv}|{hs}'
            T.add(key, iv=txt(lab), hosts=txt(hlab), novel=num(sh[key]['novel_rows'], 3),
                  trans=num(sh[key]['novel_transitions'], 3),
                  **{k: dpm(M['models'][m][key]['silent']) for m, k, _ in IV_MODELS})
        cells = {}
        for m, k, _ in IV_MODELS:
            r = get_row(M, kind='related_minus_random', intervention=iv, model=m, criterion='silent')
            cells[k] = dci(r['diff'], r['lo'], r['hi'], 3)
            ps.append(r['p_holm'])
        T.add(f'{iv}|diff', iv=txt(lab), hosts=txt('Related − random [95% CI]'), novel=txt('—'), trans=txt('—'),
              **cells)
    T.note = (f"Note. {iv_note_common(M)} First row: silent recall of the original episodes; intervention rows: Δ silent "
              'recall = intervened − original, mean ± SD over seeds 0–4; related − random rows: '
              f"{holm_note(M)}; all p (Holm) ≤ {max(ps):.4f}. Novel observation rows: share of the 78-d observations "
              'never seen in the training split (original test episodes: '
              f"{sh['original']['novel_rows']:.3f}); novel transitions: share of consecutive per-host token pairs of "
              'the intervened hosts never seen in training. The interventions act on observations, not on the '
              'simulator; they measure how much the estimate depends on other hosts\' observations and are not causal '
              'effects in the network. N_hist recall, decision changes, alerted breaches and false alarms: '
              'Table @tbl:interventions_tum; differences between models: Table @tbl:interventions_cift.')
    return T


def tE_intervention():
    M = analysis('interventions')
    T = Table('tE_intervention', [('iv', 'Intervention', 'l'), ('hosts', 'Host set S', 'l'), ('model', 'Model', 'l'),
                                  ('s', 'Δ silent recall', 'r'), ('h', 'Δ N_hist recall', 'r'),
                                  ('dec', 'Decision change, silent', 'r'), ('al', 'Δ recall, alerted breaches', 'r'),
                                  ('fa', 'Δ false alarm, clean', 'r')], group=('iv', 'hosts'))
    for iv, lab in IV:
        for hs, hlab in IV_HOSTS:
            for m, _, mlab in IV_MODELS:
                d = M['models'][m][f'{iv}|{hs}']
                T.add(f'{iv}|{hs}|{m}', iv=txt(lab), hosts=txt(hlab), model=txt(mlab), s=dpm(d['silent']),
                      h=dpm(d['never']), dec=pm_cell(d['decision_change_silent']['mean'],
                                                     d['decision_change_silent']['sd'], 3),
                      al=dpm(d['alerted']), fa=dpm(d['clean_fa']))
    base = {m: M['models'][m]['base'] for m, _, _ in IV_MODELS}
    T.note = (f"Note. {iv_note_common(M)} Δ = intervened − original, mean ± SD over seeds 0–4. Decision change: share "
              'of the silent target host-steps whose predicted class changes. Alerted breaches: compromised target '
              'host-steps with an alert in the current observation (original recall '
              + ', '.join(f"{lab} {base[m]['alerted']['mean']:.3f}" for m, _, lab in IV_MODELS)
              + '); false alarm: P(ŷ ≠ clean | y = clean) on the targets (original '
              + ', '.join(f"{lab} {base[m]['clean_fa']['mean']:.3f}" for m, _, lab in IV_MODELS) + ').')
    return T


def tE_intervention_pairs():
    M = analysis('interventions')
    T = Table('tE_intervention_pairs', [('iv', 'Intervention', 'l'), ('hosts', 'Host set S', 'l'),
                                        ('comp', 'Δ_A − Δ_B', 'l'), ('crit', 'Criterion', 'l'),
                                        ('d', 'Difference [95% CI]', 'r'), ('p', 'p (Holm)', 'r')],
              group=('iv', 'hosts'))
    lab = {m: l for m, _, l in IV_MODELS}
    crit = {'silent': 'Silent recall', 'never': 'N_hist recall'}
    for iv, ilab in IV:
        for hs, hlab in IV_HOSTS:
            for r in [r for r in M['rows'] if r['kind'] == 'pair_delta' and r['intervention'] == iv
                      and r['set_kind'] == hs]:
                T.add(f"{iv}|{hs}|{r['b']}|{r['criterion']}", iv=txt(ilab), hosts=txt(hlab),
                      comp=txt(f"{lab[r['a']]} − {lab[r['b']]}"), crit=txt(crit[r['criterion']]),
                      d=dci(r['diff'], r['lo'], r['hi'], 3), p=txt(pfmt(r['p_holm']), r['p_holm']))
    assert len(T.rows) == sum(r['kind'] == 'pair_delta' for r in M['rows'])
    T.note = ('Note. Difference between two models of the intervention effect Δ = intervened − original (Table '
              '@tbl:interventions); a positive value means that A loses less than B. ' + holm_note(M)[0].upper()
              + holm_note(M)[1:] + '.')
    return T


CORR_SETS = [('test', 'Test (training policies)'), ('ood', 'ε = 0.3 (main OOD)'), ('eps0.5', 'ε = 0.5'),
             ('eps0.7', 'ε = 0.7'), ('remove', 'Remove instead of restore')]
CORR_COPY = [('flawed', 'Flawed (as in training)'), ('corrected', 'Corrected')]


def t_corrected(runs):
    """§4.9: paired copies with faulty and corrected observations (corrected_obs.py)."""
    D = analysis('corrected_obs')
    T = Table('t_corrected', [('set', 'Set', 'l'), ('copy', 'Observation', 'l'), ('cf', 'CMF host macro-F1', 'r'),
                              ('gf', 'GRU host macro-F1', 'r'), ('cs', 'CMF silent recall', 'r'),
                              ('gs', 'GRU silent recall', 'r'), ('d', 'CMF − GRU silent recall [95% CI]', 'r'),
                              ('ch', 'CMF N_hist recall', 'r'), ('gh', 'GRU N_hist recall', 'r')], group=('set',))
    for s, slab in CORR_SETS:
        for v, vlab in CORR_COPY:
            c, g = D['models']['cybermetaformer'][s][v], D['models']['gru_full'][s][v]
            if v == 'flawed' and s in ('test', 'ood'):
                for m, x in [('cybermetaformer', c), ('gru_full', g)]:
                    check_main(x['host_macro_f1']['mean'], runs, m, f'{s}_f1')
                    check_main(x['silent_recall']['mean'], runs, m, f'{s}_silent')
                    if s == 'ood':
                        check_main(x['never_recall']['mean'], runs, m, 'ood_hist')
            r = get_row(D, kind='cmf_minus_gru', set=s, version=v, criterion='silent_recall')
            T.add(f'{s}__{v}', set=txt(slab), copy=txt(vlab), cf=num(c['host_macro_f1']['mean'], 4),
                  gf=num(g['host_macro_f1']['mean'], 4), cs=num(c['silent_recall']['mean'], 3),
                  gs=num(g['silent_recall']['mean'], 3), d=dci(r['diff'], r['lo'], r['hi'], 3),
                  ch=num(c['never_recall']['mean'], 3), gh=num(g['never_recall']['mean'], 3))
    T.note = ('Note. Both models are trained on the flawed data (seeds 0–4) and evaluated without retraining on two '
              'copies of the same episodes: the flawed observation of the training data and a copy re-simulated with the '
              'same seeds and the recorded defender actions in which a failed remove no longer shows the "removed" '
              'flag (and an invalid restore of user0 no longer clears its flags); true states, attacker observations, '
              'actions and rewards are byte-identical. Silent and N_hist host-steps are defined on the flawed '
              'observation, so both copies are scored on the same host-steps. Mean over seeds 0–4, each with its own '
              f"validation temperature; CMF − GRU: {holm_note(D)}. Per-model corrected − flawed differences, the "
              'remove-mask slice and the share of host-steps that the correction turns into alerted ones: '
              'Table @tbl:duzeltilmis_fark.')
    return T


def tE_corrected_diffs():
    D = analysis('corrected_obs')
    T = Table('tE_corrected_diffs', [('set', 'Set', 'l'), ('chg', 'Episodes changed', 'r'),
                                     ('rma', 'Remove-mask steps alerted when corrected', 'r'), ('model', 'Model', 'l'),
                                     ('f1', 'Δ host macro-F1', 'r'), ('s', 'Δ silent recall', 'r'),
                                     ('h', 'Δ N_hist recall', 'r'), ('rm', 'Δ remove-mask recall', 'r'),
                                     ('own', 'Δ silent recall, own definition', 'r')], group=('set', 'chg', 'rma'))
    for s, slab in CORR_SETS:
        S = D['sets'][s]
        for m, mlab in [('cybermetaformer', 'CMF'), ('gru_full', 'Full-history GRU')]:
            cells = {}
            for col, crit, k in [('f1', 'host_macro_f1', 4), ('s', 'silent_recall', 3), ('h', 'never_recall', 3),
                                 ('rm', 'remove_mask_recall', 3), ('own', 'silent_recall_own', 3)]:
                r = get_row(D, kind='corrected_minus_flawed', set=s, model=m, criterion=crit)
                cells[col] = dci(r['diff'], r['lo'], r['hi'], k)
            T.add(f'{s}__{m}', set=txt(slab),
                  chg=txt(f"{S['n_episodes_changed']:,} / {S['n_episodes']:,}", S['n_episodes_changed']),
                  rma=Cell(f"{S['remove_mask_alerted_when_corrected']:.3f} of {S['n_remove_mask']:,}",
                           (S['remove_mask_alerted_when_corrected'],)), model=txt(mlab), **cells)
    T.note = ('Note. Δ = corrected − flawed for the same model, seed mean over seeds 0–4, 95% CI from the '
              f"{holm_note(D)}. Episodes changed: episodes whose defender observation differs between the copies. "
              'Remove-mask host-steps: silent host-steps whose root access is masked by a failed remove in the flawed '
              'observation; in the corrected copy the given share of them carries an alert (the root flag), so it is '
              'no longer silent. Own definition: silent host-steps recomputed from each copy\'s own observation, which '
              'removes those host-steps from the silent set of the corrected copy.')
    return T


COST_MODELS = [('cybermetaformer', 'CyberMetaFormer (B1)', 'run', 'cmf'),
               ('cybermetaformer-b2', 'CyberMetaFormer (B2)', 'abl', 'b2'),
               ('cybermetaformer-host_tam', 'CMF, no topology bias', 'run', 'host_tam'),
               ('cybermetaformer-konv_gru', 'CMF, GRU instead of SSD', 'run', 'konv_gru'),
               ('cybermetaformer-host_yok', 'CMF, no host mixer', 'run', 'host_yok'),
               ('cybermetaformer-gru_host_topo', 'Host-token GRU, topology bias', 'run', 'gru_host_topo'),
               ('cybermetaformer-gru_host_attn', 'Host-token GRU, no topology bias', 'run', 'gru_host_attn'),
               ('gru_full', 'Full-history GRU', 'run', 'gru_full'),
               ('transformer_k64', 'Transformer (k = 64)', 's2', 'transformer_k64'),
               ('timemixer_k64', 'TimeMixer (k = 64)', 's2', 'timemixer_k64')]


def t_cost(runs, abl):
    """§4.11: online step latency, memory and OOD F1 (inference_cost.py; replaces the two earlier latency tables)."""
    with open(LATENCY) as f:
        L = json.load(f)
    E, Pr = L['environment'], L['protocol']
    th = [str(t) for t in Pr['threads']]
    assert th == ['1', str(E['torch_default_threads'])], th
    s2 = stage_summary('asama2')
    T = Table('t_cost', [('model', 'Model', 'l'), ('params', 'Trainable params', 'r'), ('w', 'Weights (KiB)', 'r'),
                         ('state', 'Online state (bytes)', 'r'), ('wm', 'Working memory (MiB)', 'r'),
                         ('t1', '1 thread: median / p95 (ms)', 'r'),
                         ('tn', f'{th[1]} threads: median / p95 (ms)', 'r'), ('ood', 'OOD host macro-F1', 'r')])
    three = []
    for name, lab, src, key in COST_MODELS:
        m = L['models'][name]
        if src == 'run':
            r = runs[key]
            assert m['params'] == r['params'], name
            ood, n = r['ood_f1']['mean'], len(r['seeds'])
        elif src == 'abl':
            ood, n = abl[key]['ood_f1']['mean'], len(abl[key]['seeds'])
        else:
            ood, n = gmean(s2[key], 'ood_host_macro_f1'), s2[key]['ood_host_macro_f1']['n']
        assert n in (3, 5), (name, n)
        if n == 3:
            three.append(lab)
        lat = {t: m['latency'][t] for t in th}
        assert all(lat[t]['n_steps'] == Pr['rounds'] * Pr['steps_per_round'] for t in th)
        T.add(name, model=txt(lab + ('†' if n == 3 else '')), params=intc(m['params']),
              w=intc(round(m['param_bytes'] / 1024)), state=intc(m['state_bytes']),
              wm=num(m['memory']['working_memory_bytes'] / 2 ** 20, 1),
              t1=Cell(f"{lat['1']['median_ms']:.3f} / {lat['1']['p95_ms']:.3f}", (lat['1']['median_ms'], lat['1']['p95_ms'])),
              tn=Cell(f"{lat[th[1]]['median_ms']:.3f} / {lat[th[1]]['p95_ms']:.3f}",
                      (lat[th[1]]['median_ms'], lat[th[1]]['p95_ms'])), ood=num(ood, 3))
    mem = float(E['memory_bytes']) / 2 ** 30
    dtype = {'float32': 'FP32'}[Pr['dtype']]
    T.note = (f"Note. One online step for one network (batch size {Pr['batch_size']}, {dtype}, PyTorch "
              f"{E['torch']} eager, NumPy {E['numpy']}, Python {E['python']}) on the CPU of an {E['cpu']} "
              f"({E['performance_cores']} performance + {E['efficiency_cores']} efficiency cores, {mem:.0f} GiB), macOS "
              f"{E['macos']}; the PyTorch default is {E['torch_default_threads']} threads. Each model was warmed up for "
              f"{Pr['warmup_steps']} steps and then timed for {Pr['rounds']} rounds × {Pr['steps_per_round']} steps, "
              'models interleaved within each round; the online state is reset at every episode start and the reset is '
              'not timed. CyberMetaFormer variants use the step mode (Algorithm 1); the GRU models carry their hidden '
              'state; window models shift a k-observation buffer and rerun the network at every step. Before timing, '
              'the step mode of every model was checked against its evaluation mode (largest absolute logit difference '
              f"{max(L['models'][n]['step_check_max_abs'] for n, *_ in COST_MODELS):.1e}). Online state: tensors carried "
              'between steps. Working memory: increase of the peak resident set size of a separate process after loading '
              'the model and running the warm-up and one timed round with one thread (includes allocator caches; for '
              'comparison between models). Latency excludes observation encoding and I/O. Only one network size per '
              'architecture was measured, so no scaling claim is made. OOD F1: seed mean over seeds 0–4; † seeds 0–2. '
              'Load average before / after: ' + ', '.join(f'{x:.2f}' for x in E['load_average_before']) + ' / '
              + ', '.join(f'{x:.2f}' for x in E['load_average_after']) + '.')
    return T


# ── combined tables of the JISA main text (advisor's revision). The detailed tables go to the supplement;
#    each cell here is recomputed from the same source and checked against the source table. ───────────────────────

def t_compact_baselines():
    """Table 4: selected baselines from stages 1 and 2 (k selected on val); details in t04_stage1 and t05_stage2."""
    from cybermetaformer.hidden_state.aggregate import ORIGINAL_MLP
    s1, s2 = stage_summary('asama1'), stage_summary('asama2')
    with open(os.path.join(RESULTS, 'asama0', 'mlp_repro_mac.json')) as f:
        ref = json.load(f)['reference']
    o_t, o_o = ref['8_test_host_macro_f1'], ref['8_ood_test_host_macro_f1']
    assert abs(round(o_t, 3) - ORIGINAL_MLP['test']) < 1e-9 and abs(round(o_o, 3) - ORIGINAL_MLP['ood_test']) < 1e-9
    T = Table('t_compact_baselines', [('model', 'Model', 'l'), ('k', 'History', 'r'), ('test', 'Test F1', 'r'),
                                      ('ood', 'OOD F1', 'r'), ('silent', 'OOD silent recall', 'r'),
                                      ('seeds', 'Seeds', 'r')])
    T.add('orig_mlp', model=txt('Original MLP baseline'), k=k_cell(8), test=num(o_t, 3), ood=num(o_o, 3),
          silent=txt('—'), seeds=intc(1))
    rows = [(s1, 'xgb', 'XGBoost'), (s1, 'lstm_full', 'LSTM'), (s1, 'gru_full', 'GRU'), (s2, 'transformer', 'Transformer'),
            (s2, 'itransformer', 'iTransformer'), (s2, 'timemixer', 'TimeMixer'), (s2, 'patchtst', 'PatchTST')]
    for s, m, lab in rows:
        b = best_k(s, m)
        T.add(m, model=txt(lab), k=k_cell(b['k']), test=num(gmean(b, 'test_host_macro_f1'), 3),
              ood=num(gmean(b, 'ood_host_macro_f1'), 3), silent=num(gmean(b, 'ood_silent_recall'), 3),
              seeds=intc(len(b['seeds'])))
    T.note = ('Note. Host macro-F1 and OOD silent recall, seed means; each model at the history length k selected by '
              'validation host macro-F1 (full: the whole episode so far). Seed counts and tuning budgets differ between '
              'the families (Section 3.6). The original MLP baseline is the eight-observation MLP of the reference '
              'implementation reproduced on this dataset; its silent recall was not recorded. All Stage 1 and Stage 2 '
              'models, including the rules, the windowed recurrent models, TSMixer and TimesNet: Tables @tbl:asama1 '
              'and @tbl:asama2.')
    return T


def t_main_uncertainty(runs, diffs, m35):
    """Table 5: main pair and two sources of uncertainty (t06_main + a5_two_sources)."""
    n = len(diffs['seeds'])
    T = Table('t_main_uncertainty', [('crit', 'Criterion', 'l'), ('gru', 'Full-history GRU', 'r'),
                                     ('cmf', 'CyberMetaFormer', 'r'), ('d', 'Δ CMF − GRU', 'r'),
                                     ('boot', 'Episode bootstrap 95% CI', 'r'), ('seed', 'Seed-level 95% CI', 'r')])
    p_plan = set()
    for m in ['test_f1', 'ood_f1', 'rel_drop', 'ood_silent']:
        r = find(m35, RUNS['cmf'][1], RUNS['gru_full'][1], TO_CRIT[m])
        check_point(r, runs, 'cmf', 'gru_full', m)
        assert r['seeds_a'] == r['seeds_b'] == diffs['seeds'] and abs(r['diff'] - diffs[m]['mean']) < 1e-9
        k, D = nd(m), diffs[m]
        p_plan.add(round(r['p_holm_plan'], 6))
        T.add(m, crit=txt(CRIT[TO_CRIT[m]]),
              gru=pm_cell(runs['gru_full'][m]['mean'], runs['gru_full'][m]['sd'], k),
              cmf=pm_cell(runs['cmf'][m]['mean'], runs['cmf'][m]['sd'], k), d=sgn(r['diff'], k),
              boot=Cell(f"[{signed(r['lo'], k)}, {signed(r['hi'], k)}]", (r['lo'], r['hi'])),
              seed=Cell(f"[{signed(D['t_lo'], k)}, {signed(D['t_hi'], k)}]", (D['t_lo'], D['t_hi'])))
    assert len(p_plan) == 1
    floor = m35['n_rows'] * 2 / (m35['n_boot'] + 1)
    p = p_plan.pop()
    assert abs(p - floor) < 1e-6, (p, floor)
    T.note = (f"Note. Seed mean ± SD over seeds {seed_text(diffs['seeds'])} of each model. Episode bootstrap: paired, "
              f"{m35['n_boot']:,} resamples (Algorithm 2), trained models held fixed. Seed-level interval: 95% "
              f"t-interval (df = {n - 1}) of the {n} same-index differences, without episode resampling. The two "
              'intervals describe different sources of uncertainty and neither alone is the total uncertainty. '
              f"Plan-wide Holm-adjusted p = {p:.4f} in all four rows, the smallest attainable value over the "
              f"m = {m35['n_rows']} rows of the comparison plan. Lower is better for the relative OOD drop. OOD ECE "
              f"after temperature scaling: CMF {runs['cmf']['ood_ece']['mean']:.4f}, GRU "
              f"{runs['gru_full']['ood_ece']['mean']:.4f}. Full tables: Tables @tbl:ana, @tbl:iki_kaynak and "
              '@tbl:seedler.')
    return T


def t_security(runs):
    """Table 6: security-critical errors (root F1 of t_class_level + t_class_errors)."""
    S = sinif()
    check_class(S, runs)
    root = S['classes'].index('root')
    T = Table('t_security', [('split', 'Split', 'l'), ('model', 'Model', 'l'), ('f1', 'Root F1', 'r'),
                             ('rc', 'Root → clean (%)', 'r'), ('fa', 'False alarm on clean (%)', 'r'),
                             ('det', 'Silent-onset spells detected (%)', 'r'),
                             ('lat', 'Latency, median / p90 (steps)', 'r')])
    for split in ['test', 'ood']:
        for label, short_, _ in CLASS_MODELS:
            r = S[label][split]
            L = [x['latency']['silent_onset'] for x in r['prediction_per_seed']]
            det = float(np.mean([x['breach']['detected_fraction'] for x in L]))
            med, p90 = np.mean([x['breach']['median'] for x in L]), np.mean([x['breach']['p90'] for x in L])
            T.add(f'{split}_{short_}', split=txt(SPLIT_LABEL[split]), model=txt(short_),
                  f1=num(r['f1']['mean'][root], 3), rc=Cell(f"{100 * r['root_to_clean']['mean']:.2f}",
                                                           (r['root_to_clean']['mean'],)),
                  fa=Cell(f"{100 * r['clean_false_alarm']['mean']:.2f}", (r['clean_false_alarm']['mean'],)),
                  det=Cell(f'{100 * det:.1f}', (det,)), lat=Cell(f'{med:.1f} / {p90:.1f}', (med, p90)))
    T.note = ('Note. Seed mean over seeds 0–4 of CyberMetaFormer (CMF) and the full-history GRU. Root F1 from the '
              'confusion pooled over the 11 evaluated hosts. Root → clean = P(ŷ = clean | y = root); false alarm on '
              'clean = P(ŷ ≠ clean | y = clean). A silent-onset spell is a maximal run of compromised steps on one host '
              'whose first step carries no alert; detected: the model predicts a compromise at some step of the spell; '
              'latency: steps from the onset to that prediction over detected spells (median and 90th percentile per '
              'seed, averaged). Seed SDs, all classes and error types, and per-host results: Tables @tbl:sinif, '
              '@tbl:sinif_hata, @tbl:host and @tbl:sessiz_karisiklik.')
    return T


HOST_CONTROLS = [('cmf', 'cybermetaformer', 'CyberMetaFormer (B1)'),
                 ('cmf_m4', 'cybermetaformer', 'CyberMetaFormer (B1), run 2'),
                 ('konv_gru', 'cybermetaformer-konv_gru', 'CMF, GRU instead of SSD'),
                 ('gru_host_topo', 'cybermetaformer-gru_host_topo', 'Host-token GRU, topology bias'),
                 ('gru_host_attn', 'cybermetaformer-gru_host_attn', 'Host-token GRU, no topology bias'),
                 ('gru_full', 'gru_full', 'Full-history GRU (flat input)')]


def t_host_controls(runs):
    """Table 8: host-structured controls, performance and online cost (a4_reference + t_cost). CMF run 2 has the
    same architecture; its cost is that of the main model."""
    with open(LATENCY) as f:
        L = json.load(f)
    T = Table('t_host_controls', [('model', 'Model', 'l'), ('params', 'Params', 'r'), ('ood', 'OOD F1', 'r'),
                                  ('silent', 'OOD silent recall', 'r'), ('hist', 'OOD N_hist recall', 'r'),
                                  ('lat', 'CPU step (ms)', 'r'), ('state', 'Online state (bytes)', 'r')])
    for key, name, lab in HOST_CONTROLS:
        r, m = runs[key], L['models'][name]
        assert m['params'] == r['params'], key
        n = len(r['seeds'])
        assert n in (3, 5), key
        T.add(key, model=txt(lab + ('†' if n == 3 else '')), params=intc(r['params']), ood=num(r['ood_f1']['mean'], 4),
              silent=num(r['ood_silent']['mean'], 3), hist=num(r['ood_hist']['mean'], 3),
              lat=num(m['latency'][LAT_THREADS]['median_ms'], 3), state=intc(m['state_bytes']))
    T.note = ('Note. Accuracy and recall: seed mean over seeds 0–4; † seeds 0–2. Run 2 is an independent five-seed '
              'training of the main configuration on the same hardware as the host-token GRU references, which are '
              'compared with it (Table @tbl:hostgru_cift); its architecture and cost are those of the main model. The '
              'host-token GRU references keep every CyberMetaFormer component except the temporal mixer, which is a '
              'per-host GRU over the same host tokens. CPU step: median single-thread latency of one online step '
              f"({L['environment']['cpu']} CPU, {dict(float32='FP32')[L['protocol']['dtype']]}, batch size "
              f"{L['protocol']['batch_size']}), excluding observation encoding and I/O; online state: tensors carried "
              'between steps. Seed SDs and test results: Table @tbl:hostgru; measurement protocol, six-thread latency '
              'and memory: Table @tbl:inference_cost.')
    return T


def t_robust_compact(runs):
    """Table 11: summary robustness under defender, topology and corrected-observation conditions (t09 m4,
    t_host_shift, t_corrected); all seeds 0–4, without retraining."""
    with open(os.path.join(RESULTS, 'asama3', 'ood_suite', 'ozet_5seed.json')) as f:
        Pk = json.load(f)['models']
    P, TP, D = analysis('ood_permutation'), analysis('ood_topology'), analysis('corrected_obs')
    T = Table('t_robust_compact', [('cond', 'Evaluation condition', 'l'), ('cf', 'CMF F1', 'r'), ('gf', 'GRU F1', 'r'),
                                   ('cs', 'CMF silent recall', 'r'), ('gs', 'GRU silent recall', 'r')])
    C, G = 'cybermetaformer', 'gru_full'

    def row(key, lab, c, g, eps=()):
        T.add(key, cond=txt(lab, *eps), cf=num(c['host_macro_f1'], 3), gs=num(g['silent_recall'], 3) if g else txt('—'),
              gf=num(g['host_macro_f1'], 3) if g else txt('—'), cs=num(c['silent_recall'], 3))

    for k, key in [(C, 'cmf'), (G, 'gru_full')]:
        for m, crit in [('ood_f1', 'host_macro_f1'), ('ood_silent', 'silent_recall')]:
            assert abs(Pk[k]['uye']['ood'][crit] - runs[key][m]['mean']) < 1e-4, (k, m)
    mean = lambda d: {c: d[c]['mean'] for c in ('host_macro_f1', 'silent_recall')}     # noqa: E731
    row('ood', 'OOD defender policy (ε = 0.3)', Pk[C]['uye']['ood'], Pk[G]['uye']['ood'], (0.3,))
    row('analyse', 'Periodic analyse on idle steps', Pk[C]['uye']['analyse'], Pk[G]['uye']['analyse'])
    row('p2s', 'Host permutation P2s', mean(P['models'][C]['ood']['P2s']), mean(P['models'][G]['ood']['P2s']))
    row('rewired', 'Rewired simulator topology', mean(TP['models'][C]['ood']['rewired']),
        mean(TP['models'][G]['ood']['rewired']))
    assert 'rewired_prior' not in TP['models'][G]['ood']
    row('rewired_prior', 'Rewired, relation matrix updated', mean(TP['models'][C]['ood']['rewired_prior']), None)
    row('corrected', 'Corrected remove flag (ε = 0.3)', mean(D['models'][C]['ood']['corrected']),
        mean(D['models'][G]['ood']['corrected']), (0.3,))
    T.note = ('Note. Seed mean over seeds 0–4; all models trained on the original data and evaluated without '
              'retraining. Periodic analyse: the defender analyses hosts on idle steps (react_restore_analyse). P2s: '
              'host permutation across attack-graph roles (user1 ↔ user3, user2 ↔ user4) with the decoy-count columns '
              'kept in their slot. Rewired: new OOD-policy episodes with user1/2 → ent1 and user3/4 → ent0; relation '
              'matrix updated: the topology bias of CMF rebuilt from the rewired links, weights unchanged (the flat GRU '
              'has no relation matrix). Corrected remove flag: the OOD episodes re-simulated so that a failed remove '
              'no longer shows the "removed" flag; silent host-steps are defined on the flawed observation. All '
              'policies and paired differences: Tables @tbl:politika, @tbl:failure_analysis, @tbl:host_kayma, '
              '@tbl:host_kayma_fark, @tbl:duzeltilmis and @tbl:duzeltilmis_fark.')
    return T


def build_jisa(runs, m35, diffs):
    """The five combined tables of the JISA main text: {name: Table}."""
    tables = [t_compact_baselines(), t_main_uncertainty(runs, diffs, m35), t_security(runs), t_host_controls(runs),
              t_robust_compact(runs)]
    return {t.name: t for t in tables}


def build_faz2(runs, abl):
    """Tables added in phase 2 (items 6–8, 10): {name: Table}."""
    tables = [t_shift_types(), t_host_shift(runs), tE_host_shift_diffs(), t_failure(runs), tE_failure_slices(),
              tE_analyse_repr(), t_intervention(), tE_intervention(), tE_intervention_pairs(), t_corrected(runs),
              tE_corrected_diffs(), t_cost(runs, abl)]
    return {t.name: t for t in tables}


# ── batch generation ─────────────────────────────────────────────────────

def build(basis='m3', runs=None, abl=None):
    """The 14 tables from the draft, on the given basis: {name: Table}."""
    runs = runs or load_all()
    abl = abl or load_ablations()
    plan = load_basis_plan(basis)
    b2_params = n_params('cybermetaformer-b2_k101')
    tables = [t00_related(), t01_policies(), t02_silent(), t03_protocol(basis), t04_stage1(), t05_stage2(),
              t06_main(runs, plan, basis), t07_no_alert(runs, abl, plan), t08_ablations(abl, plan, b2_params),
              t09_policy_shift(basis, runs), t10_attacker(), t11_efficiency(runs, abl, basis),
              tA1_ablations_all(abl, plan, b2_params), tB1_hyperparameters(runs['cmf']['params'])]
    return {t.name: t for t in tables}


def build_faz1(runs, m4):
    """Tables added in phase 1 (items 1, 3, 4): {name: Table}. All use the five-seed main pair or their own seeds."""
    tables = [t07b_no_alert_sets(runs), t_class_level(runs), t_class_errors(runs), tA_host_level(runs),
              tA_silent_confusion(runs), tA_item4_pairs(runs, m4)]
    return {t.name: t for t in tables}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--basis', choices=list(PLAN_FILE), default='m3',
                   help="m3: the draft's numbers (CMF seeds 0–2); m4: five-seed main pair (phase 1)")
    args = p.parse_args()
    runs, abl = load_all(), load_ablations()
    m4, m35 = load_plan('m4'), load_plan('m3_5seed')
    with open(LATENCY) as f:
        lat = json.load(f)
    diffs = seed_diffs(runs)
    texts = {name: t.md() for name, t in build(args.basis, runs, abl).items()}
    texts |= {name: t.md() for name, t in build_faz1(runs, m4).items()}
    texts |= {name: t.md() for name, t in build_faz2(runs, abl).items()}
    texts |= {name: t.md() for name, t in build_jisa(runs, m35, diffs).items()}
    texts |= {'a4_reference': a4_reference(runs, lat), 'a4_pairs': a4_pairs(runs, m4, m35),
              'a5_seeds': a5_seeds(runs, diffs), 'a5_two_sources': a5_two_sources(runs, diffs, m35)}
    os.makedirs(OUT, exist_ok=True)
    for name, text in texts.items():
        with open(os.path.join(OUT, f'{name}.md'), 'w') as f:
            f.write(text)
    with open(os.path.join(OUT, 'summary.json'), 'w') as f:
        json.dump({'basis': args.basis, 'runs': runs, 'ablations': abl, 'cmf_minus_gru_same_seed': diffs}, f,
                  indent=1)
    print('→', OUT, f'(basis {args.basis}; ' + ', '.join(texts) + ')')


if __name__ == '__main__':
    main()

"""Makale (M4) tabloları, İngilizce, yalnız sonuç dosyalarından.

Taslaktan gelen tablolar (Faz 0; md'de `::: {#tbl:… .tablo include=…}` ile):
- t00_related.md           §2.4'teki ilgili yöntemler karşılaştırması (sonuç değil, literatür; içerik burada sabit)
- t01_policies.md          savunmacı politikaları, bölüm sayısı ve veri bölmesindeki rolü (veri seti + splits.npz)
- t02_silent.md            sessiz ihlalin yaygınlığı (makale/hic_uyari/ozet.json)
- t03_protocol.md          aşamaların eğitim ve değerlendirme protokolü (ozet.json, hpo/*/best.yaml, configs/)
- t04_stage1.md            aşama 1, val'e göre en iyi k (asama1/ozet/ozet.json; orijinal MLP asama0/mlp_repro_mac.json)
- t05_stage2.md            aşama 2, val'e göre en iyi k ve GRU referansları (asama2/ozet/ozet.json)
- t06_main.md              CMF ile tam geçmiş GRU: ölçütler, eşli bölüm bootstrap'ı, ECE (metrics.json + plan)
- t07_no_alert.md          N_geçmiş (t'ye kadar uyarısız) sessiz ihlal recall'u ve eşli farklar (plan, RQ4 ailesi)
- t08_ablations.md         seçili ablasyonlar, Δ ve %95 GA (plan)
- t09_policy_shift.md      savunmacı politika kaymaları (asama3/ood_paketi/ozet.json)
- t10_attacker.md          saldırganlar arası aktarım (saldirgan_* koşularının metrics.json'ı)
- t11_efficiency.md        eski gecikme ölçümü (asama3/verimlilik), parametre ve OOD F1
- tA1_ablations_all.md     bütün ablasyonlar (plan)
- tB1_hyperparameters.md   B1 ayarları (asama3/hpo/cybermetaformer_k101/best.yaml)
Madde 4 ve 5 (Faz 1'de metne girer): a4_reference.md, a4_pairs.md, a5_seeds.md, a5_two_sources.md.

Temel (--basis): `m3` taslağın sayılarını üretir (CMF seed 0–2; M3 kaydı, `compare --plan m3`, 10.000 örnek);
`m4` 5 seed'li ana çifti kullanır (CMF seed 0–4; `compare --plan m3_5seed`, 100.000 örnek). Ablasyonlar ve 3 seed'li
varyantlar her iki temelde CMF seed 0–2'ye karşıdır. İç sayı denetimi `build(basis='m3')`'ü çağırır.
Her hücre biçimli metni ve ham değerleri birlikte taşır (`Cell`); denetim ham değerlerle karşılaştırır.
Bootstrap'ın nokta tahminleri seed ortalamalarıyla karşılaştırılır; tutmazsa betik durur.

    venv_ts/bin/python -m cybermetaformer.hidden_state.rapor.makale_tablolar [--basis m3]   # results/hidden_state/makale/tablolar/
"""
import argparse
import glob
import json
import os
from dataclasses import dataclass, field

import numpy as np
import yaml
from scipy import stats

from cybermetaformer.hidden_state.ood_paketi import load_model

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, 'results', 'hidden_state')
CONFIGS = os.path.join(ROOT, 'hidden_state', 'configs')
CMP = os.path.join(RESULTS, 'asama3', 'karsilastirma_m4')
LATENCY = os.path.join(RESULTS, 'makale', 'gecikme', 'olcum.json')
OUT = os.path.join(RESULTS, 'makale', 'tablolar')
LAT_THREADS = '1'                                         # A4 sütunu: tek iş parçacığı (varsayılan ayar gecikme tablosunda)
PLAN_FILE = {'m3': os.path.join(RESULTS, 'asama3', 'karsilastirma', 'ozet.json'),
             'm4': os.path.join(CMP, 'm3_5seed', 'ozet.json')}
CMF_SEEDS = {'m3': [0, 1, 2], 'm4': None}                 # None = bütün seed'ler

PM = '±\u00a0'        # ± ile sapma arasında bölünmez boşluk: dar sütunda hücre "ortalama" / "± sd" diye iki satıra bölünür
# anahtar → (aşama, koşu öneki, İngilizce ad). Makine adı tablolara girmez (kullanıcı kararı, 2026-11-08); cmf_m4,
# host-token GRU referanslarıyla aynı donanımda yapılan ikinci, bağımsız beş seed'li CMF eğitimidir ("run 2").
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
# ablasyonlar (configs/m3_varyantlar.yaml), taslağın Tablo A1 sırasıyla; hepsi asama3, 3 seed
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
SELECTED = ['duz78', 'host_yok', 'host_tam', 'host_gcn', 'kol_konv', 'kol_ssd', 'konv_dikkat', 'konv_gru',
            'hafiza_yok', 'b2']                                    # Tablo 8, taslağın sırası
CRIT = {'test_host_macro_f1': 'Test host macro-F1', 'ood_host_macro_f1': 'OOD host macro-F1',
        'ood_rel_drop': 'Relative OOD drop', 'test_silent_recall': 'Test silent recall',
        'ood_silent_recall': 'OOD silent recall', 'test_never_recall': 'Test N_hist recall',
        'ood_never_recall': 'OOD N_hist recall'}
METRICS = ['test_f1', 'ood_f1', 'rel_drop', 'test_silent', 'ood_silent', 'ood_hist']
BETTER = {'test_f1': 1, 'ood_f1': 1, 'rel_drop': -1, 'test_silent': 1, 'ood_silent': 1, 'ood_hist': 1}
TO_CRIT = {'test_f1': 'test_host_macro_f1', 'ood_f1': 'ood_host_macro_f1', 'rel_drop': 'ood_rel_drop',
           'test_silent': 'test_silent_recall', 'ood_silent': 'ood_silent_recall', 'ood_hist': 'ood_never_recall'}


# ── tablo yapısı ─────────────────────────────────────────────────────────

@dataclass
class Cell:
    text: str
    values: tuple = ()                 # ham değerler (oran olarak; yüzde gösterilse de); denetim bunları kullanır


@dataclass
class Table:
    name: str
    cols: list                         # [(anahtar, başlık, hizalama 'l'/'r')]
    rows: list = field(default_factory=list)   # [(satır anahtarı, {sütun anahtarı: Cell})]
    note: str = ''
    group: tuple = ()                  # bu sütunlar önceki satırla aynıysa md'de boş yazılır (yalnız görünüm)

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


# ── veri ─────────────────────────────────────────────────────────────────

def run_seeds(stage, prefix):
    d = os.path.join(RESULTS, stage)
    return sorted(int(n.rsplit('_s', 1)[1]) for n in os.listdir(d)
                  if n.startswith(prefix + '_s') and n.rsplit('_s', 1)[1].isdigit()
                  and os.path.exists(os.path.join(d, n, 'metrics.json')))


def seed_metrics(stage, prefix, seed):
    """Bir koşunun ölçütleri: metrics.json'dan F1, sessiz recall ve ECE; N_geçmiş recall'u 'hiç' diliminin havuzundan."""
    d = os.path.join(RESULTS, stage, f'{prefix}_s{seed}')
    rep = json.load(open(os.path.join(d, 'metrics.json')))['reports']
    with np.load(os.path.join(d, 'episode_slices.npz')) as z:
        hist = z['ood_test'][:, list(z['names']).index('hiç')].sum(0)
    t, o = rep['test']['host_macro_f1'], rep['ood_test']['host_macro_f1']
    return {'seed': seed, 'test_f1': t, 'ood_f1': o, 'rel_drop': 1 - o / t, 'test_silent': rep['test']['silent_recall'],
            'ood_silent': rep['ood_test']['silent_recall'], 'ood_hist': float(hist[1] / hist[0]),
            'ood_ece': rep['ood_test']['calibration_T']['ece']}


def n_params(prefix):
    model, _ = load_model(prefix.rsplit('_k', 1)[0], 0, 'cpu')            # mimari seed'den bağımsız
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
    """Aynı koşunun yalnız verilen seed'leri (None = hepsi)."""
    rows = [r for r in run['per_seed'] if seeds is None or r['seed'] in seeds]
    assert seeds is None or len(rows) == len(seeds), (run['prefix'], seeds)
    return {**run, 'seeds': [r['seed'] for r in rows], 'per_seed': rows, **stats_of(rows)}


def load_ablations():
    out = {}
    for v in ABLATIONS:
        prefix = f'cybermetaformer-{v}_k101'
        rows = [seed_metrics('asama3', prefix, s) for s in run_seeds('asama3', prefix)]
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
    """Bootstrap'ın a ve b nokta tahminleri, tabloların seed ortalamalarıyla aynı seed kümesinde tutmalı."""
    for key, side in [(ka, 'a'), (kb, 'b')]:
        seeds = row[f'seeds_{side}']
        v = np.mean([r[metric] for r in runs[key]['per_seed'] if r['seed'] in seeds])
        assert abs(v - row[side]) < 1e-9, f'{key} {metric}: bootstrap {row[side]} ≠ seed ortalaması {v}'


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
    """Aşama 1/2 karşılaştırma JSON'larının bootstrap örnek sayısı (hepsi aynı olmalı)."""
    ns = {json.load(open(p))['n_boot'] for p in glob.glob(os.path.join(RESULTS, stage, 'karsilastirma', '*.json'))}
    assert len(ns) == 1, (stage, ns)
    return ns.pop()


def seed_text(seeds):
    return f'{seeds[0]}–{seeds[-1]}' if seeds == list(range(seeds[0], seeds[-1] + 1)) else ', '.join(map(str, seeds))


# ── §2.4: ilgili yöntemler (sabit içerik) ────────────────────────────────

RELATED = [  # (yaklaşım ve kaynak no, birincil görev, çıktı, zamansal/ilişkisel yaklaşım, temel fark)
    ('BF-PPO \\[3\\]', 'Defence action', 'Approximate belief and policy', 'POMDP / particle filter',
     'No separate three-class F1 target per host state'),
    ('Nayeri–Rezvani \\[36\\]', 'Next victim / alert', 'Future target relation', 'Temporal graph and Transformer',
     'Does not estimate the current hidden privilege state'),
    ('NIDS-Mamba \\[39\\]', 'Network intrusion detection', 'Flow-level class', 'SSD and attention',
     'Differs in multi-host privilege state and the silent-compromise subset'),
    ('GTB-PPO \\[44\\]', 'Target prediction + action', 'Next-target belief / action', 'Graph + causal time + PPO',
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


# ── Tablo 1–3: veri ve protokol ──────────────────────────────────────────

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
    with open(os.path.join(RESULTS, 'makale', 'hic_uyari', 'ozet.json')) as f:
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


# ── Tablo 4–5: aşama 1 ve 2 ──────────────────────────────────────────────

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


# ── Tablo 6–8, A1: ana karşılaştırma ve ablasyonlar ──────────────────────

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
    """RQ4 ailesi: t'ye kadar uyarısız (N_geçmiş) sessiz ihlal recall'u; seed 0–2."""
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
    # nokta tahminleri koşuların 'hiç' diliminden yeniden hesaplanır
    cmf3 = subset(runs['cmf'], [0, 1, 2])
    hy = find(plan, f'{P}_k101', f'{P}-host_yok_k101', 'ood_never_recall')
    assert abs(hy['a'] - cmf3['ood_hist']['mean']) < 1e-9 and abs(hy['b'] - abl['host_yok']['ood_hist']['mean']) < 1e-9
    C = hic()['counts']
    T.note = ('Note. Silent host-steps whose host has had no alert in obs[1..t] (N_hist; online-applicable, '
              f"Section 3.3): n = {C['test']['hist']:,} test and {C['ood_test']['hist']:,} OOD host-steps. All models: "
              'seeds 0–2; the retrospective subset N_ep is compared in Table @tbl:hic_kume. ' + plan_note(plan))
    return T


# ── madde 1: iki hiç-uyarı kümesi (makale/hic_uyari, hic_uyari.py) ───────

HIC_LABEL = {'cybermetaformer': 'CMF (topology-biased attention)',
             'cybermetaformer-host_tam': 'CMF, full attention (no topology bias)',
             'cybermetaformer-host_gcn': 'CMF, GCN-like host mixer', 'cybermetaformer-host_yok': 'CMF, no host mixer',
             'cybermetaformer-duz78': 'CMF, single 78-d token',
             'cybermetaformer-gru_host_topo': 'Host-token GRU, topology-biased attention',
             'cybermetaformer-gru_host_attn': 'Host-token GRU, attention without bias',
             'gru_full': 'Full-history GRU (flat input)'}


def hic():
    with open(os.path.join(RESULTS, 'makale', 'hic_uyari', 'ozet.json')) as f:
        return json.load(f)


def t07b_no_alert_sets(runs):
    """N_geçmiş ve N_bölüm kümelerinde sessiz ihlal recall'u (seed ort. ± sd) ve iki eşli fark (hic_uyari.py)."""
    H = hic()
    R, C = H['recall'], H['counts']
    cols = [('test_hist', 'Test N_hist'), ('ood_hist', 'OOD N_hist'), ('test_ep', 'Test N_ep'), ('ood_ep', 'OOD N_ep')]
    T = Table('t07b_no_alert_sets', [('model', 'Model', 'l'), ('seeds', 'Seeds', 'r')]
              + [(k, h, 'r') for k, h in cols])
    for name, lab in HIC_LABEL.items():
        r = R[name]
        T.add(name, model=txt(lab), seeds=intc(len(r['seeds'])),
              **{k: pm_cell(r[k]['mean'], r[k]['sd'], 3) for k, _ in cols})
    # N_geçmiş recall'u koşuların 'hiç' diliminden (episode_slices.npz) gelen değerle tutmalı
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
              'which was trained on the same hardware (Table @tbl:hostgru); all other variants with seeds 0–2 of the '
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


# ── Tablo 9–11 ve B1 ─────────────────────────────────────────────────────

def t09_policy_shift(basis='m3', runs=None):
    """m3: M3 kaydı (seed 0–2, ood_paketi analyze); m4: 5 seed (ood_paketi members → ozet_5seed.json)."""
    name = 'ozet.json' if basis == 'm3' else 'ozet_5seed.json'
    with open(os.path.join(RESULTS, 'asama3', 'ood_paketi', name)) as f:
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
        # ε = 0.3 satırı ana tablonun (metrics.json) 5 seed'li değerleriyle tutmalı; paket float16 logitlerden
        # hesaplandığı için fark ~3e-5 (ood_paketi.collect'in denetimiyle aynı 1e-4 toleransı)
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


# ── madde 3: sınıf düzeyi (makale/sinif_duzeyi, sinif_duzeyi.py) ─────────

CLASS_MODELS = [('CyberMetaFormer', 'CMF', 'cmf'), ('Full-history GRU', 'GRU', 'gru_full')]
SPLIT_LABEL = {'test': 'Test', 'ood': 'OOD'}
HOSTS11 = ['user1', 'user2', 'user3', 'user4', 'ent0', 'ent1', 'ent2', 'ophost0', 'ophost1', 'ophost2', 'opserv']


def sinif():
    with open(os.path.join(RESULTS, 'makale', 'sinif_duzeyi', 'ozet.json')) as f:
        return json.load(f)


def pct(d, nd=2, i=None):
    m, s = (d['mean'], d['sd']) if i is None else (d['mean'][i], d['sd'][i])
    return Cell(f'{100 * m:.{nd}f} {PM}{100 * s:.{nd}f}', (m, s))


def check_class(S, runs):
    """sinif_duzeyi'nin host makro-F1'i ve seed'leri, tabloların seed ortalamalarıyla tutmalı."""
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
    """Madde 4 ailesi (35 satır) tek tablo: makalenin ekindeki sürüm; a4_pairs.md'nin ilk bloğu."""
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


# ── madde 4 ve 5 (Faz 1) ─────────────────────────────────────────────────

def latency(prefix, lat):
    """Koşu öneki → (medyan, p95) ms; ana_m4 CMF ile aynı mimari olduğu için CMF ölçümü kullanılır."""
    name = prefix.rsplit('_k', 1)[0].replace('-ana_m4', '')
    m = lat['models'][name]
    return m['latency'][LAT_THREADS]['median_ms'], m['latency'][LAT_THREADS]['p95_ms'], m['params']


A4_LABEL = {'cmf': 'CyberMetaFormer', 'cmf_m4': 'CyberMetaFormer, run 2', 'host_tam': 'CMF, no topology bias',
            'konv_gru': 'CMF, GRU instead of SSD', 'gru_host_topo': 'Host-token GRU, topology bias',
            'gru_host_attn': 'Host-token GRU, no topology bias', 'gru_full': 'Full-history GRU',
            'host_yok': 'CMF, no host mixer'}


def a4_reference(runs, lat):
    """Tablo 14: madde 4 referansları. Gecikme Tablo 19'da (aynı ölçüm); burada yalnız parametre sayısı denetlenir."""
    L = ['| Model | Params | Test host macro-F1 | OOD host macro-F1 | Test silent recall | OOD silent recall | '
         'OOD N_hist recall |', '|---|---:|---:|---:|---:|---:|---:|']
    assert set(A4_LABEL) == set(RUNS)
    for key in RUNS:
        r = runs[key]
        n = len(r['seeds'])
        assert n in (3, 5), key
        _, _, n_par = latency(r['prefix'], lat)
        assert n_par == r['params'], f'{key}: gecikme ölçümündeki parametre sayısı farklı'
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
            'carried state of these models are given in Table @tbl:gecikme.')
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
    """Aynı seed indeksindeki farklar (a − b): ortalama, sd, min–maks, a lehine seed sayısı, t₄ aralığı."""
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


# ── toplu üretim ─────────────────────────────────────────────────────────

def build(basis='m3', runs=None, abl=None):
    """Taslaktan gelen 14 tablo, verilen temelde: {ad: Table}."""
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
    """Faz 1'de eklenen tablolar (madde 1, 3, 4): {ad: Table}. Hepsi 5 seed'li ana çift ya da kendi seed'leri."""
    tables = [t07b_no_alert_sets(runs), t_class_level(runs), t_class_errors(runs), tA_host_level(runs),
              tA_silent_confusion(runs), tA_item4_pairs(runs, m4)]
    return {t.name: t for t in tables}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--basis', choices=list(PLAN_FILE), default='m3',
                   help="m3: taslağın sayıları (CMF seed 0–2); m4: 5 seed'li ana çift (Faz 1)")
    args = p.parse_args()
    runs, abl = load_all(), load_ablations()
    m4, m35 = load_plan('m4'), load_plan('m3_5seed')
    with open(LATENCY) as f:
        lat = json.load(f)
    diffs = seed_diffs(runs)
    texts = {name: t.md() for name, t in build(args.basis, runs, abl).items()}
    texts |= {name: t.md() for name, t in build_faz1(runs, m4).items()}
    texts |= {'a4_reference': a4_reference(runs, lat), 'a4_pairs': a4_pairs(runs, m4, m35),
              'a5_seeds': a5_seeds(runs, diffs), 'a5_two_sources': a5_two_sources(runs, diffs, m35)}
    os.makedirs(OUT, exist_ok=True)
    for name, text in texts.items():
        with open(os.path.join(OUT, f'{name}.md'), 'w') as f:
            f.write(text)
    with open(os.path.join(OUT, 'summary.json'), 'w') as f:
        json.dump({'basis': args.basis, 'runs': runs, 'ablations': abl, 'cmf_minus_gru_same_seed': diffs}, f,
                  indent=1)
    print('→', OUT, f'(temel {args.basis}; ' + ', '.join(texts) + ')')


if __name__ == '__main__':
    main()

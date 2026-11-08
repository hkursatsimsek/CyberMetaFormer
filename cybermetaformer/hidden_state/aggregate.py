"""Aşama 1 ve 2 sonuçlarının toplanması: tablolar ve figürler.

Yalnızca koşu klasörlerindeki metrics.json + config.yaml okunur (git'teki dosyalar); torch/CybORG gerekmez.

    python -m cybermetaformer.hidden_state.aggregate                 # Mac venv_ts
    python -m cybermetaformer.hidden_state.aggregate --stage asama2  # M2; M1 referansları asama1'den okunur
    venv_mini/bin/python -m cybermetaformer.hidden_state.aggregate   # VM (matplotlib venv_mini'de)

Aşama 2'de tablolar ve figürler M1 referanslarını da verir (README Kararlar 2026-10-18, üç çıta): orijinal MLP,
aynı k'daki ayarlanmış MLP, aynı k'daki en iyi M1 modeli (val host makro-F1 ile) ve tam geçmiş GRU.

Çıktılar `results/hidden_state/<asama>/ozet/`: kosular.csv (koşu başına), ozet.json (model × k ortalama ± std,
uyarı-yaşı ve neden dilimleri, politika kırılımı), tablolar.md; figürler `<asama>/figures/`.
Renkler: dataviz varsayılan paletinin kategorik slotları sabit sırayla (model → slot), ayrıca her modelin kendi
işaretçisi var (renge ek ikinci kodlama); kural gri kesikli referans çizgisidir.
Tek seed'li öğrenen satırlar (kısaltılmış plan, Kararlar 2026-10-17) tablolarda ve ısı haritalarında † ile, hafıza
eğrisinde içi boş işaretçiyle gösterilir; bunlarda std yoktur.
"""
import argparse
import csv
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt                     # noqa: E402
import numpy as np                                  # noqa: E402
import yaml                                         # noqa: E402
from matplotlib.colors import LinearSegmentedColormap   # noqa: E402
from matplotlib.lines import Line2D                 # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), 'results', 'hidden_state')
SKIP_DIRS = {'hpo', 'ozet', 'figures'}

M1_LEARNED = ['xgb', 'xgb_tf', 'mlp', 'cnn1d', 'lstm', 'gru']
M2_MODELS = ['transformer', 'patchtst', 'itransformer', 'tsmixer', 'timesnet', 'timemixer']
MODELS = ['majority', 'persist', *M1_LEARNED, 'lstm_full', 'gru_full', *M2_MODELS]
RULES = {'majority', 'persist'}                      # deterministik, seed'siz tek koşu
LABEL = {'majority': 'Çoğunluk', 'persist': 'Uyarı-kalıcılık kuralı', 'xgb': 'XGBoost', 'xgb_tf': 'XGB-TF',
         'mlp': 'MLP', 'cnn1d': '1D-CNN', 'lstm': 'LSTM', 'gru': 'GRU', 'lstm_full': 'LSTM (tam geçmiş)',
         'gru_full': 'GRU (tam geçmiş)', 'transformer': 'Transformer', 'patchtst': 'PatchTST',
         'itransformer': 'iTransformer', 'tsmixer': 'TSMixer', 'timesnet': 'TimesNet', 'timemixer': 'TimeMixer'}
LEARNED = M1_LEARNED + M2_MODELS
# Her aşamanın altı öğrenen modeli kategorik 1–6 slotlarını aynı sırayla kullanır (iki aşama aynı figürde buluşmaz;
# aşama 2'de M1 referansları nötr renktedir).
_SLOTS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']                 # kategorik 1–6
_MARKERS = ['o', 's', 'D', '^', 'v', 'P']
SLOT = {**dict(zip(M1_LEARNED, _SLOTS)), **dict(zip(M2_MODELS, _SLOTS))}
MARKER = {**dict(zip(M1_LEARNED, _MARKERS)), **dict(zip(M2_MODELS, _MARKERS))}
M2_GAPS = ('`—`: PatchTST ve TimesNet k ∈ {4, 16, 64} bütçe (paket B, README Kararlar 2026-10-19); k=1 N/A '
           '(PatchTST, TimesNet, TimeMixer; `models.NOT_APPLICABLE`).')
FULL_OF = {'lstm_full': 'lstm', 'gru_full': 'gru'}
SURFACE, INK, INK2, MUTED, GRID, BASE = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
BLUES = ['#f0efec', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']   # sıralı mavi
ORIGINAL_MLP = {'test': 0.929, 'ood_test': 0.816}      # baselines.json, MLP k=8 (geçilecek çıta)
AGE_BINS = ['hiç', '1–3', '4–7', '8–15', '16–31', '32–63', '64+']
CAUSE_LABEL = {'kacak_5': '%5 tespit kaçağı', 'green_maske': 'Green maskesi', 'remove_maske': 'remove maskesi'}

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.family': 'sans-serif', 'font.size': 10, 'text.color': INK, 'axes.labelcolor': INK2,
    'axes.edgecolor': BASE, 'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.grid': False,
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
})

SCALARS = {                                  # sütun adı → rapor içindeki yol
    'host_macro_f1': ('host_macro_f1',), 'breach_f1': ('breach_f1',), 'silent_recall': ('silent_recall',),
    'silent_exact': ('silent_exact',), 'noalert_macro_f1': ('noalert_macro_f1',), 'alert_recall': ('alert_recall',),
    'latency': ('latency', 'mean_latency'), 'detected_spells': ('latency', 'detected_spells'),
    'held_fraction': ('persistence', 'held_fraction'), 'ece': ('calibration', 'ece'),
    'ece_T': ('calibration_T', 'ece'), 'nll_T': ('calibration_T', 'nll'),
    'ci_lo': ('bootstrap', 'lo'), 'ci_hi': ('bootstrap', 'hi'),
}


def _get(d, path):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return None
        d = d[p]
    return d


def load_runs(stage_dir):
    runs = []
    for name in sorted(os.listdir(stage_dir)):
        path = os.path.join(stage_dir, name)
        if name in SKIP_DIRS or not os.path.exists(os.path.join(path, 'metrics.json')):
            continue
        with open(os.path.join(path, 'metrics.json')) as f:
            m = json.load(f)
        with open(os.path.join(path, 'config.yaml')) as f:
            cfg = yaml.safe_load(f)
        runs.append({'name': name, 'model': cfg['model'], 'k': int(cfg.get('k', 0) if cfg['model'] != 'majority' else 0),
                     'seed': cfg.get('seed'), 'info': m['info'], 'reports': m['reports'], 'cfg': cfg})
    return runs


def run_rows(runs):
    rows = []
    for r in runs:
        row = {'kosu': r['name'], 'model': r['model'], 'k': r['k'], 'seed': r['seed'],
               'temperature': r['reports'].get('temperature'),
               'val_host_macro_f1': _get(r['reports'], ('val', 'host_macro_f1')),
               'train_seconds': r['info'].get('train_seconds'), 'epochs': r['info'].get('epochs_run'),
               'n_params': r['info'].get('n_params'), 'n_trees': r['info'].get('n_trees')}
        for split, tag in [('test', 'test'), ('ood_test', 'ood')]:
            for col, path in SCALARS.items():
                row[f'{tag}_{col}'] = _get(r['reports'].get(split, {}), path)
        if row['test_host_macro_f1'] is not None and row['ood_host_macro_f1'] is not None:
            row['ood_rel_drop'] = (row['test_host_macro_f1'] - row['ood_host_macro_f1']) / row['test_host_macro_f1']
        rows.append(row)
    return rows


def _mean_std(vals):
    v = np.array([x for x in vals if x is not None], float)
    if not len(v):
        return None
    return {'mean': float(v.mean()), 'std': float(v.std(ddof=1)) if len(v) > 1 else 0.0, 'n': int(len(v))}


def _mean_dicts(ds, leaf):
    """Aynı anahtarlı sözlüklerin (ör. alert_age_recall) leaf alanının seed ortalaması."""
    keys = list(dict.fromkeys(k for d in ds if d for k in d))
    return {k: _mean_std([_get(d, (k, leaf)) if leaf else d.get(k) for d in ds if d]) for k in keys}


def summarize(runs, rows):
    groups = {}
    for r, row in zip(runs, rows):
        groups.setdefault((r['model'], r['k']), []).append((r, row))
    out = {}
    for (model, k), items in sorted(groups.items(), key=lambda kv: (MODELS.index(kv[0][0]) if kv[0][0] in MODELS
                                                                      else 99, kv[0][1])):
        cols = [c for c in items[0][1] if c not in ('kosu', 'model', 'k', 'seed')]
        g = {c: _mean_std([row[c] for _, row in items]) for c in cols}
        for split in ['test', 'ood_test']:
            reps = [r['reports'].get(split) for r, _ in items]
            g[f'{split}_alert_age_recall'] = _mean_dicts([_get(x, ('alert_age_recall',)) for x in reps], 'recall')
            g[f'{split}_cause_recall'] = _mean_dicts([_get(x, ('cause_recall',)) for x in reps], 'recall')
            g[f'{split}_policy'] = _mean_dicts([_get(x, ('policy',)) for x in reps], None)
        g['seeds'] = [r['seed'] for r, _ in items]
        out[f'{model}_k{k}'] = {'model': model, 'k': k, **g}
    return out


# ── tablolar ────────────────────────────────────────────────────────────

def _cell(s, digits=3):
    if not s or not np.isfinite(s['mean']):
        return '—'
    return f"{s['mean']:.{digits}f}" + (f" ± {s['std']:.{digits}f}" if s['n'] > 1 else '')


def _single(g):
    """Tek seed'le koşulmuş öğrenen model (kurallar seed'siz olduğu için sayılmaz)."""
    return g['model'] not in RULES and len(g['seeds']) == 1


def _mark(g, c):
    return c + ' †' if c != '—' and g and _single(g) else c


def _gcell(g, col, digits=3):
    return _mark(g, _cell(g.get(col), digits) if g else '—')


def _k_label(k):
    return 'tam' if k == 101 else ('—' if k == 0 else str(k))


def _n_seeds(summary):
    return max(len(g['seeds']) for g in summary.values() if g['model'] not in RULES)


def _single_note(summary):
    by_model = {}
    for g in summary.values():
        if _single(g):
            by_model.setdefault(g['model'], []).append(g['k'])
    return ', '.join(f"{LABEL[m]} k={'/'.join(_k_label(k) for k in sorted(ks))}"
                     for m, ks in sorted(by_model.items(), key=lambda kv: MODELS.index(kv[0])))


def _best(summary, m):
    """Modelin val host makro-F1 ortalamasına göre en iyi k'si (test/OOD seçime girmez)."""
    cands = [g for g in summary.values() if g['model'] == m]
    return max(cands, key=lambda g: (g.get('val_host_macro_f1') or {}).get('mean', -1))


def m1_refs(ref):
    """Aşama 2'nin M1 referansları (README Kararlar 2026-10-18, üç çıta): k → ayarlanmış MLP ve val'e göre aynı k'daki
    en iyi M1 modeli; en iyi pencereli M1 modeli ve k'dan bağımsız tam geçmiş GRU."""
    learned = [g for g in ref.values() if g['model'] in M1_LEARNED]
    val = lambda g: (g.get('val_host_macro_f1') or {}).get('mean', -1)       # noqa: E731
    best = {}
    for g in learned:
        if g['k'] not in best or val(g) > val(best[g['k']]):
            best[g['k']] = g
    return {'mlp': {g['k']: g for g in learned if g['model'] == 'mlp'}, 'best': best,
            'best_window': max([g for g in learned if g['k'] != 101], key=val), 'gru_full': ref.get('gru_full_k101')}


def _ref_cell(g, col, digits=3):
    return f"{_gcell(g, col, digits)} ({LABEL[g['model']]})" if g else '—'


def _best_row(label, b, k_label=None):
    return (f"| {label} | {k_label or _k_label(b['k'])} | {_gcell(b, 'test_host_macro_f1')} | "
            f"{_gcell(b, 'ood_host_macro_f1')} | {_gcell(b, 'test_silent_recall')} / "
            f"{_gcell(b, 'ood_silent_recall')} | {_gcell(b, 'test_latency', 2)} | "
            f"{_gcell(b, 'test_held_fraction')} | {_gcell(b, 'test_ece_T')} |")


def tables_md(summary, title='Aşama 1 özet tabloları', ref=None):
    ks = sorted({g['k'] for g in summary.values() if g['k'] > 0})
    models = [m for m in MODELS if any(g['model'] == m for g in summary.values())]
    R = m1_refs(ref) if ref else None
    if R:
        intro = (f'Hücreler: {_n_seeds(summary)} seed ortalaması ± std. {M2_GAPS} M1 referansları (`asama1`, '
                 "5 seed): ayarlanmış MLP ve aynı k'daki en iyi M1 modeli (val host makro-F1 ile seçilir; adı hücrede, "
                 "† = tek seed); tam geçmiş GRU k'dan bağımsızdır, tablo altında. "
                 f"Orijinal MLP (k=8): test {ORIGINAL_MLP['test']} / OOD {ORIGINAL_MLP['ood_test']}.")
    else:
        intro = (f'Hücreler: {_n_seeds(summary)} seed ortalaması ± std. **† = tek seed (s0), std yok:** '
                 f'{_single_note(summary)} (kısaltılmış plan, Kararlar 2026-10-17). Kurallar deterministik, tek koşu. '
                 '`tam` = tam geçmiş (k=101: obs[0..t]). '
                 f"Geçilecek çıta (orijinal MLP, k=8): test {ORIGINAL_MLP['test']} / OOD {ORIGINAL_MLP['ood_test']}.")
    lines = [f'# {title}', '', intro, '']
    specs = [('Test host makro-F1', 'test_host_macro_f1'), ('OOD host makro-F1', 'ood_host_macro_f1'),
             ('Test sessiz recall', 'test_silent_recall'), ('OOD sessiz recall', 'ood_silent_recall'),
             ('Test uyarısız host-adım makro-F1', 'test_noalert_macro_f1'), ('OOD göreli düşüş', 'ood_rel_drop'),
             ('Test ECE (sıcaklık sonrası)', 'test_ece_T')]
    for title_, col in specs:
        lines += [f'## {title_}', '', '| Model | ' + ' | '.join(f'k={_k_label(k)}' for k in ks) + ' |',
                  '|---|' + '---:|' * len(ks)]
        for m in models:
            if m == 'majority':
                g = summary.get('majority_k0', {})
                lines.append(f'| {LABEL[m]} | ' + ' | '.join(_gcell(g, col) for _ in ks) + ' |')
                continue
            lines.append(f'| {LABEL[m]} | ' + ' | '.join(_gcell(summary.get(f'{m}_k{k}'), col) for k in ks) + ' |')
        if R:
            lines.append('| M1: MLP | ' + ' | '.join(_gcell(R['mlp'].get(k), col) for k in ks) + ' |')
            lines.append('| M1: en iyi (val) | ' + ' | '.join(_ref_cell(R['best'].get(k), col) for k in ks) + ' |')
            lines += ['', f"M1: GRU (tam geçmiş) {_gcell(R['gru_full'], col)}."]
        lines.append('')
    # her modelin val'e göre en iyi k'sı: özet, maliyet ve politika kırılımı
    best = {m: _best(summary, m) for m in models}
    lines += ['## Model başına en iyi k (val host makro-F1 ile seçilir)', '',
              '| Model | k | Test F1 | OOD F1 | Sessiz recall (test / OOD) | Gecikme (adım) | Kalıcılık payı | ECE_T |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for m in models:
        lines.append(_best_row(LABEL[m], best[m]))
    if R:
        lines += [f"| Orijinal MLP | 8 | {ORIGINAL_MLP['test']} | {ORIGINAL_MLP['ood_test']} | — | — | — | — |",
                  _best_row('M1: MLP', _best(ref, 'mlp')),
                  _best_row(f"M1: {LABEL[R['best_window']['model']]} (en iyi pencereli)", R['best_window']),
                  _best_row('M1: GRU (tam geçmiş)', R['gru_full'])]
    cost_note = ('Eğitim süresi: MPS; iki Mac (M3 Pro, M4 Pro) ve eş zamanlı şeritler, iki koşuda kapak uykusu. '
                 "Süreler kabaca karşılaştırma içindir; verimlilik bench'ten." if R else
                 'Eğitim süresi: Mac, sinir ağları MPS, XGBoost CPU; iki kuyruk paralel koştuğu için süreler birbirini '
                 'etkilemiş olabilir.')
    lines += ['', '## Maliyet (en iyi k)', '', cost_note, '',
              '| Model | k | Parametre / ağaç | Epoch | Eğitim (dk) | Val F1 |', '|---|---:|---:|---:|---:|---:|']
    for m in [m for m in models if m not in RULES]:
        b = best[m]
        if b.get('n_trees'):
            size = f"{b['n_trees']['mean']:,.0f} ağaç"
        else:
            size = f"{b['n_params']['mean']:,.0f}" if b.get('n_params') else '—'
        secs = b.get('train_seconds')
        mins = {**secs, 'mean': secs['mean'] / 60, 'std': secs['std'] / 60} if secs else None
        lines.append(f"| {LABEL[m]} | {_k_label(b['k'])} | {size.replace(',', ' ')} | {_gcell(b, 'epochs', 1)} | "
                     f"{_mark(b, _cell(mins, 1))} | {_gcell(b, 'val_host_macro_f1')} |")
    shown = [(f"{LABEL[m]} k={_k_label(best[m]['k'])}", best[m]) for m in models if m != 'majority']
    if R:
        shown += [(f"M1: {LABEL[R['best_window']['model']]} k={_k_label(R['best_window']['k'])}", R['best_window']),
                  ('M1: GRU tam', R['gru_full'])]
    pols = list(dict.fromkeys(p for _, g in shown for split in ['test', 'ood_test']
                              for p in (g.get(f'{split}_policy') or {})))
    lines += ['', '## Politika kırılımı (host makro-F1, Red × Blue; model başına en iyi k)', '',
              f"Son {sum('eps0.3' in p for p in pols)} satır OOD (`react_restore_eps0.3`), diğerleri test.", '',
              '| Red × Blue | ' + ' | '.join(h for h, _ in shown) + ' |', '|---|' + '---:|' * len(shown)]
    for p in pols:
        split = 'ood_test' if 'eps0.3' in p else 'test'
        cells = [_mark(g, _cell((g.get(f'{split}_policy') or {}).get(p))) for _, g in shown]
        lines.append(f"| {p.replace('×', ' × ')} | " + ' | '.join(cells) + ' |')
    lines.append('')
    return '\n'.join(lines)


# ── figürler ────────────────────────────────────────────────────────────

def _grid(ax):
    ax.grid(axis='y', color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def fig_memory_curve(summary, fig_dir, R=None):
    panels = [('test_host_macro_f1', 'Test host makro-F1', 'test'), ('ood_host_macro_f1', 'OOD host makro-F1', 'ood_test'),
              ('test_silent_recall', 'Test sessiz recall', None)]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4))
    allk = sorted({g['k'] for g in summary.values() if g['k'] > 0})
    for ax, (col, title, orig) in zip(axes, panels):
        if R:                                         # M1 referansları nötr renkte (aşama 2)
            bk = [k for k in allk if R['best'].get(k) and R['best'][k].get(col)]
            ax.plot(bk, [R['best'][k][col]['mean'] for k in bk], color=INK2, lw=1.2, marker='x', ms=5, zorder=1)
            if R['gru_full'] and R['gru_full'].get(col):
                ax.axhline(R['gru_full'][col]['mean'], color=MUTED, lw=1.2, ls='--', zorder=1)
        rule = sorted([g for g in summary.values() if g['model'] == 'persist' and g.get(col)], key=lambda g: g['k'])
        if rule:
            ax.plot([g['k'] for g in rule], [g[col]['mean'] for g in rule], color=MUTED, lw=1.5, ls='--',
                    label=LABEL['persist'])
        for m in LEARNED:
            gs = sorted([g for g in summary.values() if g['model'] == m and g.get(col)], key=lambda g: g['k'])
            if not gs:
                continue
            ax.plot([g['k'] for g in gs], [g[col]['mean'] for g in gs], color=SLOT[m], lw=2, zorder=2)
            multi = [g for g in gs if not _single(g)]
            if multi:
                ax.errorbar([g['k'] for g in multi], [g[col]['mean'] for g in multi],
                            yerr=[g[col]['std'] for g in multi], color=SLOT[m], marker=MARKER[m], ms=6, ls='none',
                            capsize=0, elinewidth=1, markeredgecolor=SURFACE, markeredgewidth=1, zorder=3)
            single = [g for g in gs if _single(g)]       # tek seed: içi boş işaretçi, std yok
            if single:
                ax.plot([g['k'] for g in single], [g[col]['mean'] for g in single], marker=MARKER[m], ms=6,
                        mfc=SURFACE, mec=SLOT[m], mew=1.5, ls='none', zorder=3)
        for full, base in FULL_OF.items():
            g = summary.get(f'{full}_k101')
            if g and g.get(col):
                if _single(g):
                    ax.plot([101], [g[col]['mean']], marker=MARKER[base], ms=9, mfc=SURFACE, mec=SLOT[base], mew=2,
                            ls='none', zorder=3)
                else:
                    ax.errorbar([101], [g[col]['mean']], yerr=[g[col]['std']], color=SLOT[base], marker=MARKER[base],
                                ms=9, ls='none', capsize=0, elinewidth=1, markeredgecolor=SURFACE,
                                markeredgewidth=1, zorder=3)
        if orig:
            ax.axhline(ORIGINAL_MLP[orig], color=INK2, lw=1, ls=':')
            ax.text(1, ORIGINAL_MLP[orig], ' orijinal MLP (k=8)', va='bottom', fontsize=8, color=INK2)
        ax.set_xscale('log', base=2)
        ax.set_xticks(allk, [_k_label(k) for k in allk])
        ax.minorticks_off()
        ax.set_xlabel('geçmiş uzunluğu k (gözlem)')
        ax.set_title(title, loc='left', fontsize=10, color=INK)
        _grid(ax)
    present = {g['model'] for g in summary.values()}
    handles = []
    for m in [m for m in LEARNED if m in present]:
        filled = any(not _single(g) for g in summary.values() if g['model'] == m)
        handles.append(Line2D([], [], color=SLOT[m], lw=2, marker=MARKER[m], ms=6,
                              mfc=SLOT[m] if filled else SURFACE, mec=SURFACE if filled else SLOT[m],
                              mew=1 if filled else 1.5, label=LABEL[m] + ('' if filled else ' (tek seed)')))
    for full, base in FULL_OF.items():
        if full in present:
            single = _single(summary[f'{full}_k101'])
            handles.append(Line2D([], [], ls='none', marker=MARKER[base], ms=9, mfc=SURFACE if single else SLOT[base],
                                  mec=SLOT[base] if single else SURFACE, mew=2 if single else 1,
                                  label=LABEL[full] + (' (tek seed)' if single else '')))
    if 'persist' in present:
        handles.append(Line2D([], [], color=MUTED, lw=1.5, ls='--', label=LABEL['persist']))
    if R:
        handles += [Line2D([], [], color=INK2, lw=1.2, marker='x', ms=5, label="M1: aynı k'da en iyi (val)"),
                    Line2D([], [], color=MUTED, lw=1.2, ls='--', label='M1: GRU (tam geçmiş)')]
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('Hafıza eğrisi: geçmiş uzadıkça kestirim nasıl değişiyor?', x=0.01, ha='left', fontsize=11,
                 fontweight='bold')
    singles = any(_single(g) for g in summary.values())
    fig.text(0.01, 0.905, f'Dolu işaretçi: {_n_seeds(summary)} seed ortalaması, çubuk = std.'
             + (' İçi boş işaretçi: tek seed (s0), std yok.' if singles else ''), ha='left', fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.1, 1, 0.9))
    fig.savefig(os.path.join(fig_dir, 'fig_hafiza_egrisi.png'), dpi=180)
    plt.close(fig)


def _heatmap(ax, M, rows, cols, title):
    cmap = LinearSegmentedColormap.from_list('mavi', BLUES)
    ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=0, vmax=1, aspect='auto')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=7.5,
                        color=SURFACE if M[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(cols)), cols, fontsize=8.5)
    ax.set_yticks(range(len(rows)), rows, fontsize=8.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc='left', fontsize=10)


def _heat_label(g, prefix=''):
    return f"{prefix}{LABEL[g['model']]}, k={_k_label(g['k'])}" + (' †' if _single(g) else '')


def _heat_rows(summary, ks_pick, R=None):
    """(etiket, grup) satırları; aşama 2'de sonda M1 referansları: k=32'de en iyi M1 modeli ve tam geçmiş GRU."""
    rows = []
    for m in MODELS[1:]:
        for g in sorted([g for g in summary.values() if g['model'] == m], key=lambda g: g['k']):
            if g['k'] in ks_pick or m in FULL_OF:
                rows.append((_heat_label(g), g))
    if R:
        rows += [(_heat_label(g, 'M1: '), g) for g in (R['best'].get(32), R['gru_full']) if g]
    return rows


def fig_alert_age(summary, fig_dir, ks_pick=(8, 32, 101), R=None):
    rows = _heat_rows(summary, ks_pick, R)
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.32 * len(rows) + 1.6), sharey=True)
    for ax, split, title in [(axes[0], 'test', 'Test'), (axes[1], 'ood_test', 'OOD')]:
        M = np.array([[(g[f'{split}_alert_age_recall'].get(b) or {}).get('mean', np.nan) for b in AGE_BINS]
                      for _, g in rows], float)
        _heatmap(ax, M, [lab for lab, _ in rows], AGE_BINS, title)
        ax.set_xlabel('son uyarıdan bu yana geçen adım (sessiz ihlaller)')
    fig.suptitle(f'Sessiz ihlallerde recall, uyarı yaşına göre ({_n_seeds(summary)} seed ortalaması; † = tek seed)',
                 x=0.01, ha='left', fontsize=11, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(fig_dir, 'fig_uyari_yasi.png'), dpi=180)
    plt.close(fig)


def fig_cause(summary, fig_dir, ks_pick=(8, 32, 101), R=None):
    rows = _heat_rows(summary, ks_pick, R)
    causes = [c for c in CAUSE_LABEL if any(c in (g.get('test_cause_recall') or {}) for _, g in rows)]
    if not rows or not causes:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 0.32 * len(rows) + 1.6), sharey=True)
    for ax, split, title in [(axes[0], 'test', 'Test'), (axes[1], 'ood_test', 'OOD')]:
        M = np.array([[(g[f'{split}_cause_recall'].get(c) or {}).get('mean', np.nan) for c in causes]
                      for _, g in rows], float)
        _heatmap(ax, M, [lab for lab, _ in rows], [CAUSE_LABEL[c] for c in causes], title)
    fig.suptitle(f'Sessiz ihlallerde recall, nedene göre ({_n_seeds(summary)} seed ortalaması, † = tek seed)',
                 x=0.01, ha='left', fontsize=11, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(fig_dir, 'fig_neden_recall.png'), dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', default='asama1')
    p.add_argument('--ref', help="M1 referanslarının aşaması (varsayılan: aşama 2'de asama1)")
    args = p.parse_args()
    stage_dir = os.path.join(RESULTS, args.stage)
    ref_stage = args.ref or ('asama1' if args.stage.startswith('asama2') else None)
    if ref_stage:
        ref_runs = load_runs(os.path.join(RESULTS, ref_stage))
        ref = summarize(ref_runs, run_rows(ref_runs))
    else:
        ref = None
    R = m1_refs(ref) if ref else None
    ks_pick = (8, 32, 64) if R else (8, 32, 101)
    out_dir, fig_dir = os.path.join(stage_dir, 'ozet'), os.path.join(stage_dir, 'figures')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    runs = load_runs(stage_dir)
    rows = run_rows(runs)
    with open(os.path.join(out_dir, 'kosular.csv'), 'w', newline='') as f:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    summary = summarize(runs, rows)
    with open(os.path.join(out_dir, 'ozet.json'), 'w') as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    with open(os.path.join(out_dir, 'tablolar.md'), 'w') as f:
        f.write(tables_md(summary, f"Aşama {args.stage.removeprefix('asama')} özet tabloları", ref))
    fig_memory_curve(summary, fig_dir, R)
    fig_alert_age(summary, fig_dir, ks_pick, R)
    fig_cause(summary, fig_dir, ks_pick, R)
    print(f'{len(runs)} koşu toplandı → {out_dir}, {fig_dir}')


if __name__ == '__main__':
    main()

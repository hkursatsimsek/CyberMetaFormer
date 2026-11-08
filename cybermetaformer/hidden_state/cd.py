"""Kritik fark (CD) diyagramları: Friedman + Nemenyi, `autorank`.

Bloklar bölümlerdir (sınama birimi bölüm, README Kararlar 2026-10-18): her model her bölümde bir host makro-F1 alır
(seed'lerinin ortalaması), sıralar bölümler üzerinden toplanır. Pozitif etiketi olmayan bölümlerde metrik tanımsızdır;
bu bölümler bütün modellerde aynıdır ve düşülür. Binlerce blokla CD küçük çıkar, yani küçük sıra farkları da anlamlıdır;
yorum ortalama sıranın büyüklüğüne dayanır.

Düzenler (README Kararlar 2026-10-19: ağır ikili yalnız k ∈ {8, 32}'de):
- `k8`, `k32`: M1'in altı öğrenen modeli ve M2'nin altı modeli aynı k'da;
- `enIyiK`: her model val'e göre en iyi k'sında, tam geçmiş LSTM/GRU dahil.

Girdi: koşu klasörlerindeki `episode_confusion.npz` (git dışı, yalnızca Mac'te) ve `asama*/ozet/ozet.json`.

    python -m cybermetaformer.hidden_state.cd
"""
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt                     # noqa: E402
import numpy as np                                  # noqa: E402
import pandas as pd                                 # noqa: E402
from autorank import autorank, plot_stats           # noqa: E402

from cybermetaformer.hidden_state import metrics as M                                 # noqa: E402
from cybermetaformer.hidden_state.aggregate import LABEL, M1_LEARNED, M2_MODELS, _best, _k_label  # noqa: E402
from cybermetaformer.hidden_state.compare import load_runs                            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), 'results', 'hidden_state')
SPLITS = {'test': 'Test', 'ood_test': 'OOD'}


def episode_f1(runs, split):
    """(E,) bölüm başına host makro-F1, seed ortalaması (seed'lerde tanımsızlık aynı bölümlerde)."""
    return np.mean([M.host_macro_f1_batch(r['conf'][split]) for r in runs], 0)


def layouts():
    """Düzen adı → [(etiket, aşama, koşu öneki)]."""
    s1, s2 = (json.load(open(os.path.join(RESULTS, st, 'ozet', 'ozet.json'))) for st in ['asama1', 'asama2'])
    out = {f'k{k}': [(LABEL[m], 'asama1', f'{m}_k{k}') for m in M1_LEARNED]
           + [(LABEL[m], 'asama2', f'{m}_k{k}') for m in M2_MODELS] for k in (8, 32)}
    best = [(s1, 'asama1', m) for m in M1_LEARNED + ['lstm_full', 'gru_full']] + [(s2, 'asama2', m) for m in M2_MODELS]
    out['enIyiK'] = []
    for s, st, m in best:
        g = _best(s, m)
        name = LABEL[m] if m.endswith('_full') else f"{LABEL[m]} k={_k_label(g['k'])}"
        out['enIyiK'].append((name, st, f"{m}_k{g['k']}"))
    return out


def run_layout(entries, cache):
    """Düzen → {split: (bölüm × model DataFrame, autorank sonucu)}."""
    out = {}
    for split in SPLITS:
        cols, hashes = {}, set()
        for label, st, name in entries:
            if (st, name) not in cache:
                cache[st, name] = load_runs(os.path.join(RESULTS, st), name)
            runs = cache[st, name]
            hashes |= {r['hash'][split] for r in runs}
            cols[label] = episode_f1(runs, split)
        assert len(hashes) == 1, f'{split}: modeller aynı bölümlerde değerlendirilmemiş'
        df = pd.DataFrame(cols).dropna()
        res = autorank(df, alpha=0.05, verbose=False, order='descending', force_mode='nonparametric')
        out[split] = (df, res)
    return out


def summary_rows(df, res):
    """Ortalama sıraya göre artan (1 = en iyi; autorank `rankdf`'i ters sırada verir)."""
    r = res.rankdf.sort_values('meanrank')
    return [{'model': m, 'meanrank': float(r.loc[m, 'meanrank']), 'mean': float(df[m].mean()),
             'median': float(r.loc[m, 'median'])} for m in r.index]


def main():
    out_dir, fig_dir = (os.path.join(RESULTS, 'asama2', d) for d in ('cd', 'figures'))
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    cache, lines = {}, ['# Kritik fark diyagramları (Friedman + Nemenyi, bloklar = bölümler)', '',
                        "Her model her bölümde seed'lerinin ortalamasıyla girer; sıra 1 = en iyi. Pozitif etiketi "
                        'olmayan bölümler düşüldü. `python -m cybermetaformer.hidden_state.cd`.', '']
    for name, entries in layouts().items():
        print(f'{name}: {len(entries)} model', flush=True)
        res = run_layout(entries, cache)
        fig, axes = plt.subplots(2, 1, figsize=(10, 2.2 + 0.3 * len(entries)))
        record = {'entries': entries, 'splits': {}}
        lines += [f'## {name}', '']
        for ax, (split, (df, r)) in zip(axes, res.items()):
            plot_stats(r, ax=ax)
            for ln in ax.lines:                         # autorank eksen sınırı için beyaz çizgi çiziyor
                if ln.get_color() in ('w', 'white', '#ffffff'):
                    ln.set_visible(False)
            ax.set_title(f'{SPLITS[split]} ({len(df)} bölüm)', loc='left', fontsize=10)
            record['splits'][split] = {'n_blocks': len(df), 'pvalue': float(r.pvalue), 'cd': float(r.cd),
                                       'omnibus': r.omnibus, 'posthoc': r.posthoc, 'ranks': summary_rows(df, r)}
            lines += [f"**{SPLITS[split]}:** {len(df)} bölüm, Friedman p = {r.pvalue:.2g}, CD = {r.cd:.3f}.", '',
                      '| Sıra | Model | Ortalama sıra | Bölüm F1 ortalaması | Bölüm F1 medyanı |',
                      '|---:|---|---:|---:|---:|']
            lines += [f"| {i} | {row['model']} | {row['meanrank']:.2f} | {row['mean']:.3f} | {row['median']:.3f} |"
                      for i, row in enumerate(summary_rows(df, r), 1)]
            lines.append('')
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f'fig_cd_{name}.png'), dpi=180)
        plt.close(fig)
        with open(os.path.join(out_dir, f'cd_{name}.json'), 'w') as f:
            json.dump(record, f, indent=1, ensure_ascii=False)
    with open(os.path.join(out_dir, 'ozet.md'), 'w') as f:
        f.write('\n'.join(lines))
    print('→', out_dir)


if __name__ == '__main__':
    main()

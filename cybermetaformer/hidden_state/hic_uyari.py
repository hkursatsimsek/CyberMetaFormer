"""M4 revizyonu, madde 1: "hiç uyarı görülmemiş" kümelerinin kesin tanımı, sayıları ve recall'u.

Tanımlar (kodla birebir):
    A(h, τ)        = features.alerts(obs[τ], 'export') = istismar etkinliği biti b1 ∨ güvenlik bayrağı g1 (b3), τ = 1..T
    sessiz(h, t)   = y(h, t) > 0 ∧ ¬A(h, t)                                  (metrics.silent_metrics)
    N_geçmiş(h, t) = 1[Σ_{τ=1..t} A(h, τ) = 0]  = features.alert_age == NEVER   (çevrimiçi; metrics 'hiç' dilimi)
    N_bölüm(h)     = 1[Σ_{τ=1..T} A(h, τ) = 0]                                (geriye dönük; yalnız analizde)
Her iki küme sessiz ihlallerle kesiştirilerek kullanılır; N_bölüm ⊆ N_geçmiş. obs[0] (başlangıç gözlemi) uyarı
tanımına girmez; burada uyarı içermediği de denetlenir.

Yeniden eğitim ve çıkarım yok. N_geçmiş sayımları koşuların `episode_slices.npz` 'hiç' diliminden (eğitim sonunda
float32 tahminlerle yazılmış; kesin). N_bölüm kayıtlı float16 logitlerden: `asama3/ood_paketi/logits` (CMF, GRU tam,
gru_host_*; 5 seed) ve `asama3/aciklama/logits` (host ablasyonları; 3 seed). Denetim: aynı logitlerle N_geçmiş yeniden sayıldığında n'ler birebir,
isabetler 2.1 milyonda 61 farklı (2.9e-5; float16 yuvarlamasındaki argmax eşitlikleri; 2026-11-08). Eşli bölüm
bootstrap'ı compare.py ile aynı: her model seed ortalamasıyla, test ve OOD ayrı; Holm her küme ailesinde ayrı. Bir
çiftte iki model de yalnız ortak seed'leriyle girer: ablasyon çiftlerinde CMF s0–s2, CMF–GRU'da s0–s4.

    python -m cybermetaformer.hidden_state.hic_uyari     # results/hidden_state/makale/hic_uyari/{ozet.json, tablolar.md}
"""
import argparse
import json
import os

import numpy as np

from cybermetaformer.hidden_state import data as D
from cybermetaformer.hidden_state import features as F
from cybermetaformer.hidden_state import metrics as M

RESULTS = os.path.join(D.ROOT, 'results', 'hidden_state')
STAGE = os.path.join(RESULTS, 'asama3')
OUT_DIR = os.path.join(RESULTS, 'makale', 'hic_uyari')
LOGIT_DIRS = [os.path.join(STAGE, 'ood_paketi', 'logits'), os.path.join(STAGE, 'aciklama', 'logits')]
SPLITS = {'test': 'test', 'ood': 'ood_test'}          # logit anahtarı → store split'i
MODELS = {
    'cybermetaformer': 'CMF (topology-biased host attention)',
    'cybermetaformer-host_tam': 'CMF, full host attention (no topology bias)',
    'cybermetaformer-host_gcn': 'CMF, GCN-like host mixer',
    'cybermetaformer-host_yok': 'CMF, no host mixer',
    'cybermetaformer-duz78': 'CMF, single 78-d token',
    'gru_full': 'Full-history GRU',
    'cybermetaformer-gru_host_topo': 'Host-token GRU, topology-biased host attention',
    'cybermetaformer-gru_host_attn': 'Host-token GRU, full host attention',
}
PAIRS = [('cybermetaformer', 'cybermetaformer-host_yok'), ('cybermetaformer-host_tam', 'cybermetaformer-host_yok'),
         ('cybermetaformer-host_gcn', 'cybermetaformer-host_yok'), ('cybermetaformer', 'cybermetaformer-duz78'),
         ('cybermetaformer', 'gru_full')]
SUBSETS = {'hist': 'N_hist(h,t): no alert on h up to t (online)', 'ep': 'N_ep(h): no alert on h in the whole episode'}


def subsets(store, idx):
    """(E, T, 12) bool maskeleri: sessiz, sessiz ∩ N_geçmiş, sessiz ∩ N_bölüm; ayrıca y ve uyarılar."""
    y = store.labels[idx]
    alert = F.episode_alerts(store.obs[idx])
    age = F.alert_age(alert)
    silent = (y > 0) & ~alert
    hist = silent & (age == F.NEVER)
    ep = silent & ~alert.any(axis=1, keepdims=True)
    assert not (ep & ~hist).any(), 'N_bölüm ⊄ N_geçmiş'
    return {'silent': silent, 'hist': hist, 'ep': ep}, y, alert, age


def split_counts(store, idx):
    m, y, alert, age = subsets(store, idx)
    comp = y > 0
    g = age[m['silent']]
    prev = g[g != F.NEVER]
    pairs = lambda mask: int(mask.any(axis=1).sum())          # noqa: E731  (bölüm, host) çiftleri
    return {
        'episodes': int(len(idx)), 'host_steps': int(y.size),
        'compromised': int(comp.sum()), 'silent': int(m['silent'].sum()),
        'hist': int(m['hist'].sum()), 'ep': int(m['ep'].sum()),
        'hist_episode_host_pairs': pairs(m['hist']), 'ep_episode_host_pairs': pairs(m['ep']),
        'hist_episodes': int(m['hist'].any(axis=(1, 2)).sum()), 'ep_episodes': int(m['ep'].any(axis=(1, 2)).sum()),
        'compromised_share': float(comp.mean()),
        'silent_share_of_compromised': float(m['silent'].sum() / comp.sum()),
        'root_share_of_silent': float((y[m['silent']] == 2).mean()),
        'hist_share_of_silent': float(m['hist'].sum() / m['silent'].sum()),
        'ep_share_of_silent': float(m['ep'].sum() / m['silent'].sum()),
        'ep_share_of_hist': float(m['ep'].sum() / m['hist'].sum()),
        'prev_alert_within_32_of_prev': float((prev < 32).mean()),
        'prev_alert_within_64_of_prev': float((prev < 64).mean()),
        'obs0_has_alert': bool(F.alerts(store.obs[idx][:, 0]).any()),
    }


def logit_file(name, seed):
    for d in LOGIT_DIRS:
        p = os.path.join(d, f'{name}_k101_s{seed}.npz')
        if os.path.exists(p):
            return p
    return None


def seeds_of(name):
    return [s for s in range(10) if logit_file(name, s)]


def episode_counts(pred, mask):
    """(E, 2): bölüm başına [n, isabet (tahmin > 0)]."""
    return np.stack([mask.sum((1, 2)), (mask & (pred > 0)).sum((1, 2))], -1).astype(np.float64)


def slice_counts(name, seed, split):
    """Koşunun episode_slices.npz 'hiç' dilimi: (E, 2) [n, isabet]; dosya yoksa None."""
    path = os.path.join(STAGE, f'{name}_k101_s{seed}', 'episode_slices.npz')
    if not os.path.exists(path):
        return None
    with np.load(path) as z:
        return z[split][:, list(z['names']).index('hiç')].astype(np.float64)


def slice_check(name, seed, split, hist_counts):
    """episode_slices 'hiç' sayımları ile float16 logitlerden hesaplananların en büyük farkı (bölüm başına host-adım)."""
    ref = slice_counts(name, seed, split)
    return None if ref is None else float(np.abs(ref - hist_counts).max())


def recall(counts, W):
    n_hit = W @ counts
    return n_hit[:, 1] / np.maximum(n_hit[:, 0], 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n-boot', type=int, default=10_000)
    args = p.parse_args()
    store = D.EpisodeStore()
    with store.final_evaluation():
        idx = {s: store.split(s) for s in ['train', 'val', 'test', 'ood_test']}
    out = {'definitions': {
        'alert': "A(h,τ) = exploit-activity bit (b1) OR security flag g1 (b3) in obs[τ], τ = 1..T "
                 "(features.alerts(kind='export'))",
        'silent': 'y(h,t) > 0 and not A(h,t)', **SUBSETS}}
    out['counts'] = {s: split_counts(store, i) for s, i in idx.items()}

    masks = {k: subsets(store, idx[s])[0] for k, s in SPLITS.items()}
    recalls, ep_counts, checks = {}, {}, []
    for name in MODELS:
        seeds = seeds_of(name)
        recalls[name] = {'seeds': seeds}
        for k, s in SPLITS.items():
            per = {sub: [] for sub in SUBSETS}
            for seed in seeds:
                with np.load(logit_file(name, seed)) as z:
                    pred = z[k].astype(np.float32).argmax(-1)
                for sub in SUBSETS:
                    ep_counts[name, k, sub, seed] = episode_counts(pred, masks[k][sub])
                checks.append({'model': name, 'seed': seed, 'split': s,
                               'max_abs_diff_vs_episode_slices': slice_check(name, seed, s,
                                                                             ep_counts[name, k, 'hist', seed])})
                ref = slice_counts(name, seed, s)          # N_geçmiş: eğitimdeki float32 tahminlerin kesin sayımı
                if ref is not None:
                    assert (ref[:, 0] == ep_counts[name, k, 'hist', seed][:, 0]).all()
                    ep_counts[name, k, 'hist', seed] = ref
                for sub in SUBSETS:
                    c = ep_counts[name, k, sub, seed]
                    per[sub].append(float(c[:, 1].sum() / c[:, 0].sum()))
            for sub in SUBSETS:
                recalls[name][f'{k}_{sub}'] = {'mean': float(np.mean(per[sub])), 'sd': float(np.std(per[sub], ddof=1))
                                               if len(per[sub]) > 1 else None, 'per_seed': per[sub]}
    out['recall'] = recalls
    out['slice_check'] = checks

    tests = {sub: [] for sub in SUBSETS}
    for a, b in PAIRS:
        common = sorted(set(recalls[a]['seeds']) & set(recalls[b]['seeds']))
        for k in SPLITS:
            for sub in SUBSETS:
                E = len(masks[k][sub])
                sa = lambda W, n=a: np.mean([recall(ep_counts[n, k, sub, s], W[k]) for s in common], 0)  # noqa: E731
                sb = lambda W, n=b: np.mean([recall(ep_counts[n, k, sub, s], W[k]) for s in common], 0)  # noqa: E731
                res = M.paired_bootstrap(sa, sb, {k: E}, n_boot=args.n_boot, seed=0)
                tests[sub].append({'a': a, 'b': b, 'split': k, 'seeds': common, **res})
    for sub, rows in tests.items():
        for row, ph in zip(rows, M.holm([r['p'] for r in rows])):
            row['p_holm'] = ph
            row['holm_family_size'] = len(rows)
    out['paired_tests'] = tests
    out['n_boot'] = args.n_boot

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, 'ozet.json'), 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    with open(os.path.join(OUT_DIR, 'tablolar.md'), 'w') as f:
        f.write(tables_md(out))
    print('→', OUT_DIR)


def tables_md(out):
    c = out['counts']
    sp = [('train', 'Train'), ('test', 'Test'), ('ood_test', 'OOD')]
    L = ['# No-alert subsets (M4 item 1)', '', f"- Alert: {out['definitions']['alert']}",
         f"- Silent: {out['definitions']['silent']}", f"- {SUBSETS['hist']}", f"- {SUBSETS['ep']}", '',
         '## Prevalence', '', '| Measure | ' + ' | '.join(n for _, n in sp) + ' |', '|---|' + '---:|' * len(sp)]
    rows = [('Compromised / all host-steps', 'compromised_share', '{:.3f}'),
            ('Silent / compromised host-steps', 'silent_share_of_compromised', '{:.3f}'),
            ('Root share among silent', 'root_share_of_silent', '{:.3f}'),
            ('N_hist / silent', 'hist_share_of_silent', '{:.3f}'),
            ('N_ep / silent', 'ep_share_of_silent', '{:.3f}'),
            ('N_ep / N_hist', 'ep_share_of_hist', '{:.3f}'),
            ('Previous alert within 32 steps*', 'prev_alert_within_32_of_prev', '{:.3f}'),
            ('Previous alert within 64 steps*', 'prev_alert_within_64_of_prev', '{:.3f}'),
            ('Silent host-steps (n)', 'silent', '{:,}'), ('N_hist host-steps (n)', 'hist', '{:,}'),
            ('N_ep host-steps (n)', 'ep', '{:,}'), ('N_hist (episode, host) pairs', 'hist_episode_host_pairs', '{:,}'),
            ('N_ep (episode, host) pairs', 'ep_episode_host_pairs', '{:,}'), ('Episodes', 'episodes', '{:,}')]
    for label, key, fmt in rows:
        L.append(f'| {label} | ' + ' | '.join(fmt.format(c[s][key]) for s, _ in sp) + ' |')
    L += ['', '*Among silent compromises whose host had an earlier alert.', '', '## Recall (seed mean ± sd)', '',
          '| Model | Seeds | Test N_hist | OOD N_hist | Test N_ep | OOD N_ep |', '|---|---:|---:|---:|---:|---:|']
    fmt = lambda v: f"{v['mean']:.3f}" + (f" ± {v['sd']:.3f}" if v['sd'] is not None else '')  # noqa: E731
    for name, label in MODELS.items():
        r = out['recall'][name]
        L.append(f"| {label} | {len(r['seeds'])} | {fmt(r['test_hist'])} | {fmt(r['ood_hist'])} | "
                 f"{fmt(r['test_ep'])} | {fmt(r['ood_ep'])} |")
    for sub in SUBSETS:
        L += ['', f'## Paired episode bootstrap, {sub} ({out["n_boot"]} resamples; Holm within this table)', '',
              'Both models of a pair enter with the seeds they share (seed mean).', '',
              '| A | B | Seeds | Split | A − B | 95% CI | p (Holm) |', '|---|---|---|---|---:|---|---:|']
        for t in out['paired_tests'][sub]:
            L.append(f"| {MODELS[t['a']]} | {MODELS[t['b']]} | {', '.join(map(str, t['seeds']))} | {t['split']} | "
                     f"{t['diff']:+.3f} | [{t['lo']:+.3f}, {t['hi']:+.3f}] | {t['p_holm']:.4f} |")
    return '\n'.join(L) + '\n'


if __name__ == '__main__':
    main()

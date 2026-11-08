"""Keşifsel veri analizi istatistikleri ve sessiz ihlal ön ölçümü (numpy; figürler figures.py'de).

    venv_cyborg/bin/python -m cybermetaformer.hidden_state.eda
Çıktı: cybermetaformer/results/hidden_state/asama0/eda.json
Sessiz ihlal neden payları için önce silent_cause.py koşmuş olmalı (minicage/data/minicage_v1_aux/).
"""
import json
import os

import numpy as np

from minicage.dataset import loaders as L
from cybermetaformer.hidden_state import data as D
from cybermetaformer.hidden_state import features as F

OUT_DIR = os.path.join(D.ROOT, 'results', 'hidden_state', 'asama0')
CAUSE_FILE = os.path.join(D.DATA_ROOT, 'minicage_v1_aux', 'silent_cause.npz')
GAP_KS = [1, 4, 8, 16, 32, 64]


def silent_table(y, alert):
    """Sessiz ihlal ön ölçüm tablosu (makaledeki sessiz ihlal tablosuyla aynı tanımlar)."""
    comp = y > 0
    silent = comp & ~alert
    age = F.alert_age(alert)
    g = age[silent]
    prev = g[g != F.NEVER]
    return {
        'ele_gecirilmis_orani': float(comp.mean()),
        'sessiz_orani': float(silent.sum() / comp.sum()),
        'sessizlerde_root_payi': float((y[silent] == 2).mean()),
        'hic_uyari_gorulmemis': float((g == F.NEVER).mean()),
        **{f'son_uyari_lt_{k}': float((prev < k).mean()) for k in GAP_KS},
        # tüm sessizler içinde son uyarının k'lık pencereye (t-k+1..t) düştüğü oran: gap ≤ k-1
        **{f'pencere_kapsami_k{k}': float(((g != F.NEVER) & (g <= k - 1)).mean()) for k in GAP_KS + [100]},
    }


def main():
    store = D.EpisodeStore()
    meta, hosts = store.meta, [store.meta['hosts'][h] for h in D.LABEL_HOSTS]
    res = {'hosts': hosts, 'blue_policies': meta['blue_policies'], 'red_policies': meta['red_policies']}
    with store.final_evaluation():
        sp = {s: store.split(s) for s in ['train', 'val', 'test', 'ood_test']}
    y_all, a_all = store.labels, F.episode_alerts(store.obs)

    res['silent_table'] = {s: silent_table(y_all[i], a_all[i]) for s, i in sp.items()}

    tr = sp['train']
    res['label_dist_train'] = {h: np.bincount(y_all[tr][..., j].reshape(-1), minlength=3).tolist()
                               for j, h in enumerate(hosts)}

    comp, silent = y_all > 0, (y_all > 0) & ~a_all
    res['silent_by_policy'] = {}
    for r, rn in enumerate(meta['red_policies']):
        for b, bn in enumerate(meta['blue_policies']):
            m = (store.red_policy == r) & (store.blue_policy == b)
            res['silent_by_policy'][f'{rn}×{bn}'] = {
                'sessiz_orani': float(silent[m].sum() / max(comp[m].sum(), 1)),
                'ele_gecirilmis_orani': float(comp[m].mean())}

    age = F.alert_age(a_all)
    g = age[silent]
    res['alert_age_hist'] = {'never': int((g == F.NEVER).sum()),
                             'counts': np.bincount(g[g != F.NEVER], minlength=100)[:100].tolist()}
    res['window_coverage'] = [float(((g != F.NEVER) & (g <= k - 1)).mean()) for k in range(1, 101)]
    res['window_coverage_split'] = {}
    for s in ['test', 'ood_test']:
        gs = age[sp[s]][silent[sp[s]]]
        res['window_coverage_split'][s] = {
            'coverage': [float(((gs != F.NEVER) & (gs <= k - 1)).mean()) for k in range(1, 101)],
            'never': float((gs == F.NEVER).mean())}

    if os.path.exists(CAUSE_FILE):
        with np.load(CAUSE_FILE) as z:
            cause, names = z['cause'], z['causes'].tolist()
        res['cause_names'] = names
        res['cause_by_blue'] = {
            bn: np.bincount(cause[store.blue_policy == b].reshape(-1), minlength=len(names)).tolist()
            for b, bn in enumerate(meta['blue_policies'])}
        res['cause_age'] = {}      # neden × uyarı yaşı ('hiç' ya da pencere k=8/32 dışında mı)
        c = cause[silent]
        for i, nm in enumerate(names[1:], start=1):
            m = c == i
            if m.any():
                res['cause_age'][nm] = {'hic_uyari': float((g[m] == F.NEVER).mean()),
                                        'k8_disinda': float(((g[m] == F.NEVER) | (g[m] > 7)).mean()),
                                        'k32_disinda': float(((g[m] == F.NEVER) | (g[m] > 31)).mean())}

    # örnek bölüm: test, bline × react_restore, en uzun sessiz root'a sahip bölüm
    te = sp['test']
    rr = te[(store.red_policy[te] == 0) & (store.blue_policy[te] == meta['blue_policies'].index('react_restore'))]
    score = (silent[rr] & (y_all[rr] == 2)).sum((1, 2))
    e = int(rr[np.argsort(score)[len(score) // 2]])      # medyan bölüm (uç değil, tipik)
    acts = L.load_raw(D.DEFAULT_DATA, fields=['act_red', 'act_blue'])[0]
    res['example_episode'] = {
        'episode': e, 'red': meta['red_policies'][store.red_policy[e]], 'blue': meta['blue_policies'][store.blue_policy[e]],
        'labels': y_all[e].T.tolist(), 'alerts': a_all[e].T.astype(int).tolist(),
        'cause': cause[e].T.tolist() if os.path.exists(CAUSE_FILE) else None,
        'act_red': [meta['action_names']['Red'][a] for a in acts['act_red'][e]],
        'act_blue': [meta['action_names']['Blue'][a] for a in acts['act_blue'][e]]}

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, 'eda.json'), 'w') as f:
        json.dump(res, f, ensure_ascii=False)
    for s, row in res['silent_table'].items():
        print(s, '  '.join(f'{k}={v:.3f}' for k, v in row.items()))
    for k, v in res.get('cause_age', {}).items():
        print('neden', k, v)


if __name__ == '__main__':
    main()

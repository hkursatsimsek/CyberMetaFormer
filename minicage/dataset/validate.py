"""Veri seti doğrulaması + DATASHEET.md üretimi (DATASHEET_template.md doldurulur).

    venv_cyborg/bin/python -m minicage.dataset.validate [--data minicage/data/minicage_v1]
"""
import argparse
import json
import os
import re

import numpy as np

from CybORG_plus_plus.mini_CAGE import SimplifiedCAGE, Meander_minimal, Blue_sleep
from minicage.dataset import loaders as L
from minicage.dataset.generate import ROOT, run_shard

TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DATASHEET_template.md')

README_SCORES = {  # mini_CAGE/README.md "Performance" (Mini CAGE sütunu, 500 bölüm x 100 adım)
    ('bline', 'react_restore'): -156, ('bline', 'restore_decoys'): -68, ('bline', 'sleep'): -1141,
    ('meander', 'react_restore'): -68, ('meander', 'restore_decoys'): -63, ('meander', 'sleep'): -1067,
}
SPLIT_NAMES = {'train': 'train', 'val': 'val', 'test': 'test', 'ood_test': 'OOD test'}


def se(x):
    return x.std() / np.sqrt(len(x))


def num(n):
    return f'{n:,}'.replace(',', ' ')


def md_table(header, rows):
    return '\n'.join(['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
                     + ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows])


# ── kontroller ──────────────────────────────────────────────────────────

def readme_consistency():
    rows = []
    for (red, blue), ref in README_SCORES.items():
        d, _ = run_shard(red, blue, 500, 100, remove_bugs=False, seed=12345)
        s = d['rew_blue'].sum(1)
        rows.append((red, blue, ref, s.mean(), se(s)))
    return rows


def repo_reward_bug():
    """Depodaki SimplifiedCAGE.step ile FixedCAGE'i Meander x Sleep'te karşılaştırır."""
    np.random.seed(0)
    env = SimplifiedCAGE(num_envs=500)
    s, _ = env.reset()
    red, blue, tot = Meander_minimal(), Blue_sleep(), np.zeros(500)
    for _ in range(100):
        s, r, _, _ = env.step(red.get_action(s['Red']), blue.get_action(s['Blue']))
        tot += r['Blue'][:, 0]
    fixed, _ = run_shard('meander', 'sleep', 500, 100, remove_bugs=False, seed=0)
    return tot.mean(), fixed['rew_blue'].sum(1).mean()


def label_checks(raw, meta):
    out = []
    bp, rp = meta['blue_policies'], meta['red_policies']
    labels = L.host_labels(raw['true_state'])                         # (E, T+1, 13)
    opserv = meta['hosts'].index('opserv')

    m = (raw['red_policy'] == rp.index('bline')) & (raw['blue_policy'] == bp.index('sleep'))
    out.append(('B-line × Sleep bölümlerinin sonunda `opserv`\'te root etiketi olan oran', 1.0,
                (labels[m, -1, opserv] == 2).mean()))
    imp = raw['impacted'][:, :, opserv] == 1
    out.append(('`opserv` impact altındayken ödülün ≤ −10 olduğu adım oranı', 1.0, (raw['rew_blue'][imp] <= -10).mean()))
    out.append(('`opserv` impact altındayken `opserv` etiketinin root olduğu adım oranı', 1.0,
                (labels[:, 1:, opserv][imp] == 2).mean()))
    a, ok = raw['act_blue'], raw['blue_success'] == 1
    e, t = np.where((a >= 40) & ok)
    out.append((f'Başarılı restore sonrası hostun temiz etiketlendiği oran ({num(len(e))} restore)', 1.0,
                (labels[e, t + 1, a[e, t] - 40] == 0).mean()))
    react_only = np.isin(a, [0] + list(range(40, 53)))               # react_restore aksiyon kümesi
    for name, eps in [('react_restore', 0.0), ('react_restore_eps0.1', 0.1), ('react_restore_eps0.3', 0.3)]:
        sel = raw['blue_policy'] == bp.index(name)
        out.append((f'`{name}`: React-Restore dışı aksiyon oranı (beklenen ε·39/53)', eps * 39 / 53,
                    1 - react_only[sel].mean()))
    return out


def reproducibility(data_dir, meta):
    res = []
    for shard in meta['shards'][:1] + meta['shards'][-1:]:
        regen, _ = run_shard(shard['red'], shard['blue'], shard['episodes'],
                             meta['steps_per_episode'], meta['remove_bugs'], shard['seed'])
        with np.load(os.path.join(data_dir, shard['file'])) as z:
            res.append((shard['file'], all(np.array_equal(regen[k], z[k]) for k in z.files)))
    return res


# ── istatistikler ───────────────────────────────────────────────────────

def combo_stats(raw, meta):
    labels = L.host_labels(raw['true_state'][:, 1:])[..., L.LABEL_HOSTS]
    imp = raw['impacted'][:, :, meta['hosts'].index('opserv')] == 1
    rows = []
    for ri, r in enumerate(meta['red_policies']):
        for bi, b in enumerate(meta['blue_policies']):
            m = (raw['red_policy'] == ri) & (raw['blue_policy'] == bi)
            if not m.any():
                continue
            s = raw['rew_blue'][m].sum(1)
            any_imp = imp[m].any(1)
            first = np.median(imp[m].argmax(1)[any_imp]) if any_imp.any() else None
            rows.append((r, b, num(int(m.sum())), f'{s.mean():.1f} ± {se(s):.1f}', f'{100 * any_imp.mean():.1f}',
                         '—' if first is None else f'{first:.0f}', f'{(labels[m][:, -1] > 0).sum(-1).mean():.2f}'))
    return md_table(['Red', 'Blue', 'Bölüm', 'Bölüm ödülü (ort. ± SE)', 'Impact %', 'İlk impact (adım)', 'Son ihlal (host)'], rows)


def label_stats(raw, meta):
    labels = L.host_labels(raw['true_state'][:, 1:])
    rows = []
    for h in L.LABEL_HOSTS:
        d = np.bincount(labels[..., h].ravel(), minlength=3) / labels[..., h].size
        rows.append((f'`{meta["hosts"][h]}`', f'{d[0]:.3f}', f'{d[1]:.3f}', f'{d[2]:.3f}'))
    any_breach = (labels[..., L.LABEL_HOSTS] > 0).any(-1).mean()
    return md_table(['Host', 'temiz', 'kullanıcı', 'root'], rows), f'{any_breach:.3f}'


def visibility(raw, meta):
    H = np.array(L.LABEL_HOSTS)
    lab = L.host_labels(raw['true_state'][:, 1:])[..., H]
    alert = raw['obs_blue'][:, 1:, :52].reshape(*raw['act_blue'].shape, 13, 4)[..., H, 3] == 1
    comp = lab > 0
    E, T, _ = comp.shape

    has = comp.any(1)
    tc = comp.argmax(1)
    after = alert & (np.arange(T)[None, :, None] >= tc[:, None, :])
    alerted = after.any(1)
    delay = (after.argmax(1) - tc)[has & alerted]

    a, ok = raw['act_blue'], raw['blue_success'] == 1
    rest = np.zeros_like(comp)
    e, t = np.where((a >= 40) & ok & (a != 40 + L.USER0))
    rest[e, t, np.searchsorted(H, a[e, t] - 40)] = True
    prev_rest = np.maximum.accumulate(rest.astype(np.int8), axis=1).astype(bool)
    onset = comp[:, 1:] & ~comp[:, :-1]
    first_on, re_on = onset & ~prev_rest[:, :-1], onset & prev_rest[:, :-1]

    rr = raw['blue_policy'] == meta['blue_policies'].index('react_restore')
    miss_rr = (comp & ~alert)[rr]
    tp = (alert & comp).sum()

    lines = [
        f'- Bölümler boyunca **{num(int(has.sum()))}** kez bir host ele geçirildi. Bunların '
        f'**%{100 * alerted[has].mean():.1f}**\'i en az bir kez uyarı üretti. Uyarı genellikle aynı adımda geliyor '
        f'(medyan gecikme {np.median(delay):.0f} adım, ortalama {delay.mean():.2f} adım).',
        f'- İhlalin **başladığı** adımda uyarı oranı: ilk ele geçirmede %{100 * (alert[:, 1:] & first_on).sum() / first_on.sum():.1f}, '
        f'restore sonrası yeniden ele geçirmede %{100 * (alert[:, 1:] & re_on).sum() / re_on.sum():.1f}.',
        f'- **Adım bazında** ise tablo farklı: uyarı varsa ihlal kesin (kesinlik {tp / alert.sum():.3f}), ama ihlal '
        f'süren adımların yalnızca **{tp / comp.sum():.3f}**\'ünde uyarı açık (duyarlılık).',
        f'- `react_restore` bölümlerinde uyarısız ihlal adımlarının **%{100 * (miss_rr & prev_rest[rr]).sum() / miss_rr.sum():.0f}**\'inde '
        f'o host daha önce restore edilmiş.', '',
        'Adım bazında duyarlılığın savunmacı politikasına göre değişimi:', '',
    ]
    rows = []
    for bi, b in enumerate(meta['blue_policies']):
        s = raw['blue_policy'] == bi
        rows.append((f'`{b}`', f'{(alert & comp)[s].sum() / comp[s].sum():.3f}'))
    return '\n'.join(lines) + '\n' + md_table(['Blue politikası', 'Uyarı duyarlılığı'], rows)


def trace(raw, meta, red='bline', blue='react_restore', steps=15):
    e = int(np.where((raw['red_policy'] == meta['red_policies'].index(red))
                     & (raw['blue_policy'] == meta['blue_policies'].index(blue)))[0][0])
    names_r, names_b, hosts = meta['action_names']['Red'], meta['action_names']['Blue'], meta['hosts']
    lab = L.host_labels(raw['true_state'][e, 1:])
    alert = raw['obs_blue'][e, 1:, :52].reshape(-1, 13, 4)[..., 3] == 1
    tag = {1: 'user', 2: 'root'}
    rows = []
    for t in range(steps):
        comp = ', '.join(f'{hosts[h]}({tag[lab[t, h]]})' for h in L.LABEL_HOSTS if lab[t, h] > 0) or '—'
        alr = ', '.join(hosts[h] for h in L.LABEL_HOSTS if alert[t, h]) or '—'
        ok = {1: '✓', 0: '✗'}.get(int(raw['red_success'][e, t]), '')
        rows.append((t, f'`{names_r[raw["act_red"][e, t]]}` {ok}', f'`{names_b[raw["act_blue"][e, t]]}`',
                     f'{raw["rew_blue"][e, t]:.1f}', comp, alr))
    caption = (f'B-line saldırganı × React-Restore savunmacısı, ilk {steps} adım (bölüm #{e}). '
               '"Gerçek durum" modelin göremediği etikettir; "Uyarılar" savunmacının gözleminde istismar bayrağı '
               'açık olan hostlardır. ✓/✗ saldırganın aksiyonunun başarılı olup olmadığını gösterir.')
    return caption, md_table(['Adım', 'Red aksiyonu', 'Blue aksiyonu', 'Ödül', 'Gerçek durum (gizli)', 'Uyarılar (gözlem)'], rows)


# ── referans sonuçlar ───────────────────────────────────────────────────

def ref_value(r):
    return float(re.findall(r'[-\d.]+$', r['reference'])[0])


def baseline_values(data_dir):
    with open(os.path.join(data_dir, 'baselines.json')) as f:
        res = json.load(f)
    for r in res:
        r['task'] = r['task'].split(' [')[0]

    def get(task, split, model):
        return [r for r in res if r['task'].startswith(task) and r['split'] == split and r['model'] == model][0]

    f3 = lambda x: f'{x:.3f}'
    host, net = 'tespit (host', 'tespit (ağda'
    det_models = [r['model'] for r in res if r['task'].startswith(host) and r['split'] == 'test']
    m0 = det_models[0]
    det_rows = [('Çoğunluk sınıfı', f3(ref_value(get(host, 'test', m0))), f3(ref_value(get(host, 'ood_test', m0))), '—', '—'),
                ('Uyarı kuralı', '—', '—', f3(ref_value(get(net, 'test', m0))), f3(ref_value(get(net, 'ood_test', m0))))]
    det_rows += [(m, f3(get(host, 'test', m)['value']), f3(get(host, 'ood_test', m)['value']),
                  f3(get(net, 'test', m)['value']), f3(get(net, 'ood_test', m)['value'])) for m in det_models]

    risk_t = [r for r in res if r['task'].startswith('OpServer') and r['split'] == 'test'][0]
    risk_o = [r for r in res if r['task'].startswith('OpServer') and r['split'] == 'ood_test'][0]
    risk_rows = [('Uyarı kuralı (`ent2`/`opserv` uyarısı)', f3(ref_value(risk_t)), '—', f3(ref_value(risk_o)), '—'),
                 (risk_t['model'], f3(risk_t['value']), f3(risk_t['auc']), f3(risk_o['value']), f3(risk_o['auc']))]

    att_rows = [('Şans', '0.500', '0.500')]
    att_rows += [(f"{r['model']}", f3(r['value']), f3(get('saldırgan', 'ood_test', r['model'])['value']))
                 for r in res if r['task'].startswith('saldırgan') and r['split'] == 'test']
    nr_t = [r for r in res if r['task'].startswith('Red aksiyon') and r['split'] == 'test'][0]
    nr_o = get('Red aksiyon', 'ood_test', nr_t['model'])
    nr_rows = [('Çoğunluk sınıfı', f3(ref_value(nr_t)), f3(ref_value(nr_o))),
               (nr_t['model'], f3(nr_t['value']), f3(nr_o['value']))]

    bc = [r for r in res if r['task'].startswith('davranış')]
    bc_rows = [(r['metric'].replace('bölüm ödülü vs ', ''), f"{ref_value(r):.1f}", f"{r['value']:.1f}") for r in bc]

    return {
        'res_detection': md_table(['Model', 'Host makro-F1 (test)', 'Host makro-F1 (OOD)',
                                   'İhlal F1 (test)', 'İhlal F1 (OOD)'], det_rows),
        'res_risk': md_table(['Model', 'F1 (test)', 'AUC (test)', 'F1 (OOD)', 'AUC (OOD)'], risk_rows),
        'res_attacker': '(a) Saldırgan tanıma, doğruluk:\n\n' + md_table(['Model', 'test', 'OOD test'], att_rows)
                        + '\n\n(b) Red aksiyon tipi (6 sınıf) tahmini, doğruluk:\n\n'
                        + md_table(['Model', 'test', 'OOD test'], nr_rows),
        'res_bc': md_table(['Saldırgan', 'Öğretmen (`restore_decoys`)', 'BC politikası'], bc_rows),
        'det_rule_f1': f3(ref_value(get(net, 'test', m0))),
        'det_k1': f3(get(net, 'test', det_models[0])['value']),
        'det_k8': f3(get(net, 'test', det_models[-1])['value']),
        'det_ood_k8': f3(get(net, 'ood_test', det_models[-1])['value']),
        'risk_pos_rate': f3(risk_t['pos_rate']),
        'risk_auc_test': f3(risk_t['auc']), 'risk_auc_ood': f3(risk_o['auc']),
        'risk_f1_test': f3(risk_t['value']), 'risk_f1_ood': f3(risk_o['value']),
        'bc_episodes': '500',
    }


def write_datasheet(data_dir, meta, raw, splits, checks, readme_rows, bug, repro):
    E, T = raw['act_blue'].shape
    size_mb = sum(os.path.getsize(os.path.join(data_dir, f)) for f in os.listdir(data_dir)) / 2**20
    per_combo = {}
    for s in meta['shards']:
        per_combo[(s['red'], s['blue'])] = per_combo.get((s['red'], s['blue']), 0) + s['episodes']
    caption, trace_tbl = trace(raw, meta)
    label_tbl, any_breach = label_stats(raw, meta)

    v = {
        'name': meta['name'], 'n_episodes': num(E), 'n_steps_total': num(E * T), 'T': str(T),
        'size_mb': f'{size_mb:.0f}', 'gen_seconds': f"{meta['generation_seconds']:.0f}",
        'episodes_per_config': str(max(per_combo.values())), 'remove_bugs': str(meta['remove_bugs']),
        'split_summary': ', '.join(f'{SPLIT_NAMES[k]} {num(len(x))}' for k, x in splits.items()) + ' bölüm',
        'split_rows': '\n'.join(f'| {SPLIT_NAMES[k]} | {num(len(x))} | {num(len(x) * T)} |' for k, x in splits.items()),
        'combo_table': combo_stats(raw, meta), 'label_table': label_tbl, 'any_breach': any_breach,
        'visibility_block': visibility(raw, meta), 'trace_caption': caption, 'trace_table': trace_tbl,
        'bug_repo': f'{bug[0]:.1f}', 'bug_fixed': f'{bug[1]:.1f}',
        'readme_table': md_table(['Red', 'Blue', 'Yayımlanmış', 'Bu çalışma (ort. ± SE)', 'Fark'],
                                 [(r, b, ref, f'{m:.1f} ± {s:.1f}', f'%{100 * abs(m - ref) / abs(ref):.1f}')
                                  for r, b, ref, m, s in readme_rows]),
        'checks_list': md_table(['Kontrol', 'Beklenen', 'Ölçülen'],
                                [(n, f'{exp:.3f}', f'{got:.3f}') for n, exp, got in checks]),
        'repro_list': ', '.join(f"`{f}` {'✓' if ok else '✗'}" for f, ok in repro),
        'python_version': meta['versions']['python'], 'numpy_version': meta['versions']['numpy'],
    }
    v.update(baseline_values(data_dir))

    with open(TEMPLATE) as f:
        text = f.read()
    for k, val in v.items():
        text = text.replace('{{' + k + '}}', val)
    left = re.findall(r'\{\{(\w+)\}\}', text)
    assert not left, f'Doldurulmamış alanlar: {left}'
    with open(os.path.join(data_dir, 'DATASHEET.md'), 'w') as f:
        f.write(text)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=os.path.join(ROOT, 'data', 'minicage_v1'))
    args = p.parse_args()

    raw, meta = L.load_raw(args.data)
    splits = L.load_splits(args.data)
    readme_rows = readme_consistency()
    for r in readme_rows:
        print('README  %-8s %-15s ref %6d  bizim %8.1f ± %.1f' % r)
    bug = repo_reward_bug()
    print(f'Ödül hatası: depo step() {bug[0]:.1f}  FixedCAGE {bug[1]:.1f}')
    checks = label_checks(raw, meta)
    for n, exp, got in checks:
        print(f'{n}: beklenen {exp:.3f} ölçülen {got:.3f}')
    repro = reproducibility(args.data, meta)
    print('Yeniden üretilebilirlik:', repro)
    write_datasheet(args.data, meta, raw, splits, checks, readme_rows, bug, repro)
    print('DATASHEET.md yazıldı')


if __name__ == '__main__':
    main()

"""M4 revizyonu, madde 3: güvenlik açısından kritik sınıfların ayrı raporu.

Karışıklık tabanlı ölçütler koşuların `episode_confusion.npz` dosyalarından (bölüm × 12 host × 3 × 3; eğitim sonundaki
float32 tahminler; bütün seed'ler; çıkarım yok). Değerlendirilen hostlar host makro-F1 kuralıyla aynı: kümede en az bir
pozitif (user/root) gerçek etiketi olan hostlar (pratikte def hariç 11 host; user0 etiket dışı, Red'in dayanağı).

- havuzlanmış 3×3 karışıklık (11 host, bütün host-adımlar): sayım ve satır normalize; seed ortalaması
- sınıf başına precision / recall / F1: havuzlanmış ve host ortalamalı (sınıfın bulunduğu hostlar üzerinden)
- root → user, root → clean, user → clean oranları; temiz host-adımlarda yanlış alarm P(ŷ > 0 | y = 0)
- host başına makro-F1 (metrics.host_macro_f1 ile aynı tanım), en kötü ve en iyi host

Tahmin gerektirenler (kayıtlı float16 logitlerden: `asama3/ood_paketi/logits`; bulunan seed'ler):
- sessiz host-adımlarda karışıklık (gerçek user/root × tahmin)
- tespit gecikmesi: ihlal dönemi (y > 0'ın kesintisiz koşusu) başlangıcından ilk ŷ > 0'a ve ilk ŷ = y'ye kadar geçen
  adım; dönem başlangıcı uyarılı (A(h, t₀)) ve sessiz olarak ayrı; medyan, p90, ortalama, tespit edilen dönem oranı.

    python -m cybermetaformer.hidden_state.sinif_duzeyi     # results/hidden_state/makale/sinif_duzeyi/
"""
import json
import os

import numpy as np

from cybermetaformer.hidden_state import data as D
from cybermetaformer.hidden_state import features as F
from cybermetaformer.hidden_state import metrics as M

RESULTS = os.path.join(D.ROOT, 'results', 'hidden_state')
OUT_DIR = os.path.join(RESULTS, 'makale', 'sinif_duzeyi')
LOGIT_DIR = os.path.join(RESULTS, 'asama3', 'ood_paketi', 'logits')
CLASSES = ['clean', 'user', 'root']
SPLITS = {'test': 'test', 'ood': 'ood_test'}
# ad → (aşama klasörü, koşu öneki, logit öneki); GRU'nun 5 seed'i asama1'de (compare.py'deki RQ3 referansı; logitleri
# asama3'teki yeniden eğitimden, metrikleri asama1 ile birebir aynı). gru_host_*: M4 madde 4 referansları.
MODELS = {'CyberMetaFormer': ('asama3', 'cybermetaformer_k101', 'cybermetaformer_k101'),
          'Full-history GRU': ('asama1', 'gru_full_k101', 'gru_full_k101'),
          'Host-token GRU + topo attention': ('asama3', 'cybermetaformer-gru_host_topo_k101',
                                              'cybermetaformer-gru_host_topo_k101'),
          'Host-token GRU + attention': ('asama3', 'cybermetaformer-gru_host_attn_k101',
                                         'cybermetaformer-gru_host_attn_k101')}


def run_seeds(stage, prefix):
    d = os.path.join(RESULTS, stage)
    return sorted(int(n.rsplit('_s', 1)[1]) for n in os.listdir(d)
                  if n.startswith(prefix + '_s') and os.path.exists(os.path.join(d, n, 'metrics.json')))


def class_metrics(C):
    """C (3, 3) [gerçek, tahmin] → sınıf başına precision, recall, F1 ve oranlar."""
    C = C.astype(np.float64)
    tp = np.diag(C)
    prec = tp / np.maximum(C.sum(0), 1)
    rec = tp / np.maximum(C.sum(1), 1)
    f1 = 2 * tp / np.maximum(C.sum(0) + C.sum(1), 1)
    row = C / np.maximum(C.sum(1, keepdims=True), 1)
    return {'precision': prec.tolist(), 'recall': rec.tolist(), 'f1': f1.tolist(), 'row_normalized': row.tolist(),
            'root_to_user': row[2, 1], 'root_to_clean': row[2, 0], 'user_to_clean': row[1, 0],
            'user_to_root': row[1, 2], 'clean_false_alarm': (C[0, 1] + C[0, 2]) / max(C[0].sum(), 1),
            'counts': C.astype(np.int64).tolist()}


def host_class_f1(Ch):
    """(H, 3, 3) → (3,) sınıf başına F1'in hostlar üzerinden ortalaması (sınıfın bulunduğu hostlar)."""
    out = []
    for c in range(3):
        vals = [2 * Ch[h, c, c] / max(Ch[h, c].sum() + Ch[h, :, c].sum(), 1) for h in range(len(Ch)) if Ch[h, c].sum() > 0]
        out.append(float(np.mean(vals)))
    return out


def confusion_part(stage, prefix, seeds, split, hosts):
    rows = []
    for s in seeds:
        with np.load(os.path.join(RESULTS, stage, f'{prefix}_s{s}', 'episode_confusion.npz')) as z:
            Ch = z[split].astype(np.int64).sum(0)                        # (12, 3, 3)
        ev = [h for h in range(Ch.shape[0]) if Ch[h, 1:].sum() > 0]
        cm = class_metrics(Ch[ev].sum(0))
        per_host = {hosts[h]: M.macro_f1_from_confusion(Ch[h]) for h in ev}
        rows.append({'seed': s, 'evaluated_hosts': [hosts[h] for h in ev], **cm,
                     'host_avg_f1': host_class_f1(Ch[ev]), 'per_host_macro_f1': per_host,
                     'host_macro_f1': float(np.mean(list(per_host.values())))})
    return rows


def spells(y, p, alert):
    """(E, T, H) → dönem kayıtları: başlangıçta uyarı var mı, ilk ŷ>0 gecikmesi, ilk ŷ=y gecikmesi (yoksa -1)."""
    E, T, H = y.shape
    pos = y > 0
    out = []
    for e in range(E):
        for h in range(H):
            col = pos[e, :, h]
            if not col.any():
                continue
            t = 0
            while t < T:
                if not col[t]:
                    t += 1
                    continue
                t0 = t
                while t < T and col[t]:
                    t += 1
                seg = slice(t0, t)
                det = np.flatnonzero(p[e, seg, h] > 0)
                exact = np.flatnonzero(p[e, seg, h] == y[e, seg, h])
                out.append((bool(alert[e, t0, h]), det[0] if len(det) else -1, exact[0] if len(exact) else -1))
    return np.array(out, dtype=np.int64).reshape(-1, 3)


def latency_summary(rec):
    res = {}
    for name, m in [('all', np.ones(len(rec), bool)), ('alert_onset', rec[:, 0] == 1), ('silent_onset', rec[:, 0] == 0)]:
        r = rec[m]
        d = {'n_spells': int(len(r))}
        for col, key in [(1, 'breach'), (2, 'exact')]:
            lat = r[r[:, col] >= 0, col]
            d[key] = {'detected_fraction': float(len(lat) / max(len(r), 1)),
                      'median': float(np.median(lat)) if len(lat) else None,
                      'p90': float(np.quantile(lat, 0.9)) if len(lat) else None,
                      'mean': float(lat.mean()) if len(lat) else None}
        res[name] = d
    return res


def prediction_part(store, idx, logit_prefix, split_key):
    """Kayıtlı logitlerin bulunduğu seed'ler: sessiz karışıklık ve gecikme."""
    y = store.labels[idx].astype(np.int64)
    alert = F.episode_alerts(store.obs[idx])
    silent = (y > 0) & ~alert
    rows = []
    for s in range(10):
        path = os.path.join(LOGIT_DIR, f'{logit_prefix}_s{s}.npz')
        if not os.path.exists(path):
            continue
        with np.load(path) as z:
            p = z[split_key].astype(np.float32).argmax(-1)
        Cs = np.zeros((3, 3), np.int64)
        np.add.at(Cs, (y[silent], p[silent]), 1)
        rows.append({'seed': s, 'silent_confusion': Cs.tolist(),
                     'silent_row_normalized': (Cs / np.maximum(Cs.sum(1, keepdims=True), 1)).tolist(),
                     'latency': latency_summary(spells(y, p, alert))})
    return rows


def mean_sd(vals):
    a = np.asarray(vals, np.float64)
    return {'mean': a.mean(0).tolist(), 'sd': (a.std(0, ddof=1) if len(a) > 1 else np.zeros_like(a[0])).tolist(),
            'n': int(len(a))}


def summarize(rows, keys):
    return {k: mean_sd([r[k] for r in rows]) for k in keys}


def main():
    store = D.EpisodeStore()
    hosts = [store.meta['hosts'][h] for h in D.LABEL_HOSTS]
    with store.final_evaluation():
        idx = {k: store.split(s) for k, s in SPLITS.items()}
    out = {'classes': CLASSES, 'definitions': {
        'evaluated_hosts': 'hosts with at least one compromised (user/root) label in the split; def is never '
                           'compromised and user0 (Red foothold) is not a label host, so 11 hosts are evaluated',
        'pooled': 'confusion summed over evaluated hosts and all host-steps',
        'host_avg_f1': 'per-class F1 averaged over the evaluated hosts in which the class occurs',
        'clean_false_alarm': 'P(pred > 0 | y = 0) over evaluated hosts',
        'spell': 'maximal run of consecutive steps with y > 0 on one host; latency = steps from spell onset to the '
                 'first prediction > 0 (breach) or to the first prediction equal to the true class (exact)'}}
    keys = ['precision', 'recall', 'f1', 'host_avg_f1', 'row_normalized', 'root_to_user', 'root_to_clean',
            'user_to_clean', 'user_to_root', 'clean_false_alarm', 'host_macro_f1']
    for label, (stage, prefix, lprefix) in MODELS.items():
        seeds = run_seeds(stage, prefix)
        out[label] = {'seeds': seeds}
        for k, s in SPLITS.items():
            rows = confusion_part(stage, prefix, seeds, s, hosts)
            ref = [json.load(open(os.path.join(RESULTS, stage, f'{prefix}_s{r["seed"]}', 'metrics.json')))
                   ['reports'][s]['host_macro_f1'] for r in rows]
            assert np.allclose([r['host_macro_f1'] for r in rows], ref, atol=1e-9), f'{label} {s}: host makro-F1 tutmuyor'
            ph = summarize([{'v': list(r['per_host_macro_f1'].values())} for r in rows], ['v'])['v']
            names = list(rows[0]['per_host_macro_f1'])
            pred_rows = prediction_part(store, idx[k], lprefix, k)
            out[label][k] = {
                'per_seed': rows, **summarize(rows, keys),
                'per_host_macro_f1': {n: {'mean': m, 'sd': sd} for n, m, sd in zip(names, ph['mean'], ph['sd'])},
                'worst_host': names[int(np.argmin(ph['mean']))], 'best_host': names[int(np.argmax(ph['mean']))],
                'prediction_seeds': [r['seed'] for r in pred_rows], 'prediction_per_seed': pred_rows,
                'silent_row_normalized': mean_sd([r['silent_row_normalized'] for r in pred_rows]),
            }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, 'ozet.json'), 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=float)
    with open(os.path.join(OUT_DIR, 'tablolar.md'), 'w') as f:
        f.write(tables_md(out))
    print('→', OUT_DIR)


def _ms(d, i=None, nd=3):
    m, s = (d['mean'], d['sd']) if i is None else (d['mean'][i], d['sd'][i])
    return f'{m:.{nd}f} ± {s:.{nd}f}'


def tables_md(out):
    L = ['# Class-level performance (M4 item 3)', '']
    for k, title in [('test', 'Standard test'), ('ood', 'OOD test')]:
        L += [f'## {title}: per-class metrics (pooled over evaluated hosts; seed mean ± sd)', '',
              '| Model | Seeds | Class | Precision | Recall | F1 | Host-avg F1 |', '|---|---:|---|---:|---:|---:|---:|']
        for label in MODELS:
            r = out[label][k]
            for c, cn in enumerate(CLASSES):
                L.append(f"| {label} | {r['precision']['n']} | {cn} | {_ms(r['precision'], c)} | {_ms(r['recall'], c)} | "
                         f"{_ms(r['f1'], c)} | {_ms(r['host_avg_f1'], c)} |")
        L += ['', f'## {title}: security-relevant error rates', '',
              '| Model | Root→user | Root→clean | User→clean | User→root | Clean false alarm | Worst host (macro-F1) | '
              'Best host (macro-F1) |', '|---|---:|---:|---:|---:|---:|---|---|']
        for label in MODELS:
            r = out[label][k]
            w, b = r['worst_host'], r['best_host']
            L.append(f"| {label} | {_ms(r['root_to_user'], nd=4)} | {_ms(r['root_to_clean'], nd=4)} | "
                     f"{_ms(r['user_to_clean'], nd=4)} | {_ms(r['user_to_root'], nd=4)} | "
                     f"{_ms(r['clean_false_alarm'], nd=4)} | {w} ({r['per_host_macro_f1'][w]['mean']:.3f}) | "
                     f"{b} ({r['per_host_macro_f1'][b]['mean']:.3f}) |")
        L += ['', f'## {title}: detection latency (steps; seeds with stored predictions)', '',
              '| Model | Seeds | Onset | Spells | Detected (breach) | Median | p90 | Detected (exact class) | Median | p90 |',
              '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
        for label in MODELS:
            r = out[label][k]
            for part in ['all', 'alert_onset', 'silent_onset']:
                vals = [x['latency'][part] for x in r['prediction_per_seed']]
                agg = lambda key, f: np.mean([v[key][f] for v in vals if v[key][f] is not None])  # noqa: E731
                L.append(f"| {label} | {len(vals)} | {part} | {np.mean([v['n_spells'] for v in vals]):,.0f} | "
                         f"{agg('breach', 'detected_fraction'):.4f} | {agg('breach', 'median'):.1f} | "
                         f"{agg('breach', 'p90'):.1f} | {agg('exact', 'detected_fraction'):.4f} | "
                         f"{agg('exact', 'median'):.1f} | {agg('exact', 'p90'):.1f} |")
        L += ['', f'## {title}: pooled confusion (row-normalized, seed mean)', '']
        for label in MODELS:
            m = np.asarray(out[label][k]['row_normalized']['mean'])
            L += [f'**{label}**', '', '| True \\ Pred | ' + ' | '.join(CLASSES) + ' |', '|---|---:|---:|---:|']
            L += [f'| {CLASSES[i]} | ' + ' | '.join(f'{v:.4f}' for v in m[i]) + ' |' for i in range(3)]
            L.append('')
    return '\n'.join(L) + '\n'


if __name__ == '__main__':
    main()

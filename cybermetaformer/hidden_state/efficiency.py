"""Verimlilik ve Pareto figürü.

Final modeller HPO'nun seçtiği boyutlarla ölçülür (bench.py temel konfigürasyonları ölçüyordu): tek örnek gecikmesi
(bs=1; MPS ve CPU) ve çıkarım verimi (MPS, bs=8192). Seed 0'ın config'i kullanılır (seed'ler aynı ayarı paylaşır);
ağırlıklar rastgele, çünkü süre ağırlıktan bağımsız. Değerlendirme eager (train.py'deki gibi).
- Tam geçmiş RNN çevrimiçi kullanımda adım başına tek hücre güncellemesi yapar; gecikmesi aynı ayarla k=1 penceresi
  üzerinden ölçülür.
- XGBoost ölçülmez (ağırlıkları saklanmadı); kurallar ve çoğunluk öğrenen model değil.
Ölçüm başka iş koşmazken yapılmalı. Pareto: düşük gecikme ve yüksek host makro-F1 (seed ortalaması, `ozet.json`).

`--stage asama3`: aynı hücreler + CyberMetaFormer (B1 ana, B2), aynı oturumda yeniden ölçülür. CMF eğitilmiş
ağırlıklarla yüklenir; gecikmesi çevrimiçi adım kipiyle (`CyberMetaFormer.step`: konv tamponu + SSD durumu, maliyeti
t'den bağımsız; tam geçmiş GRU'nun tek hücre güncellemesinin karşılığı) ölçülür ve adım kipi önce gerçek bölümlerde
forward_episode ile karşılaştırılır. Ayrıca her adımda tam yeniden hesap (S=101, bs=1; adım kipi olmasaydı) ve bölüm toplu
verimi (MPS). Seed ortalamaları koşuların `metrics.json`'undan; 3 seed'lik topluluk satırları `ood_paketi/ozet.json`'dan,
gecikme üye sayısıyla çarpılarak.

    python -m cybermetaformer.hidden_state.efficiency --measure     # asama2/verimlilik/olcum_<makine>.json
    python -m cybermetaformer.hidden_state.efficiency               # figür ve tablo (ölçüm json'u + ozet.json)
    python -m cybermetaformer.hidden_state.efficiency --stage asama3 [--measure]   # asama3/verimlilik/
"""
import argparse
import glob
import json
import os
import platform
import socket
import statistics
import time

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt                     # noqa: E402
import numpy as np                                  # noqa: E402
import torch                                        # noqa: E402
import yaml                                         # noqa: E402

from cybermetaformer.hidden_state import data as D                                   # noqa: E402
from cybermetaformer.hidden_state.aggregate import (INK, INK2, LABEL, M1_LEARNED, M2_MODELS, MARKER, MUTED,  # noqa: E402
                                                  SLOT, SURFACE, _best, _grid, _k_label)
from cybermetaformer.hidden_state.bench import _sync                                  # noqa: E402
from cybermetaformer.hidden_state.models import build_model                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), 'results', 'hidden_state')
OUT_DIRS = {st: os.path.join(RESULTS, st, 'verimlilik') for st in ('asama2', 'asama3')}
M3_RUNS = ['cybermetaformer', 'cybermetaformer-b2']   # asama3 koşu adları (B1 ana, B2 ölçekleme kontrolü)
M3_LABEL = {'cybermetaformer': 'CyberMetaFormer', 'cybermetaformer-b2': 'CyberMetaFormer B2'}
M3_STYLE = {'cybermetaformer': dict(color='#b3261e', marker='*', ms=13, mec=SURFACE, mew=0.8),
            'cybermetaformer-b2': dict(color='#b3261e', marker='*', ms=11, mfc=SURFACE, mew=1.2)}
ENSEMBLE = {'cybermetaformer': 'cybermetaformer', 'gru_full': 'gru_full'}   # ood_paketi adı → hücre modeli
NEURAL_M1 = [m for m in M1_LEARNED if not m.startswith('xgb')] + ['lstm_full', 'gru_full']
STEP_OF = {'lstm_full': 'lstm', 'gru_full': 'gru'}   # çevrimiçi adım: aynı ayarlı RNN, k=1


def label(c):
    return M3_LABEL[c['model']] if c['stage'] == 'asama3' else ('M1: ' if c['stage'] == 'asama1' else '') + LABEL[c['model']]


def cells(stage='asama2'):
    """[(aşama, model, k, config)]: seed 0 koşuları, öğrenen sinir ağları; asama3'te CMF koşuları da (model = koşu adı)."""
    out = []
    for st, models in [('asama1', NEURAL_M1), ('asama2', M2_MODELS)]:
        for path in sorted(glob.glob(os.path.join(RESULTS, st, '*_s0', 'config.yaml'))):
            cfg = yaml.safe_load(open(path))
            if cfg['model'] in models:
                out.append((st, cfg['model'], int(cfg['k']), cfg))
    if stage == 'asama3':
        for run in M3_RUNS:
            cfg = yaml.safe_load(open(os.path.join(RESULTS, 'asama3', f'{run}_k101_s0', 'config.yaml')))
            out.append(('asama3', run, int(cfg['k']), cfg))
    return out


def latency_ms(model, x1, m1, device, reps=60, warm=10):
    lat = []
    for i in range(reps):
        _sync(device)
        t0 = time.perf_counter()
        model(x1, m1)
        _sync(device)
        if i >= warm:
            lat.append(time.perf_counter() - t0)
    return 1000 * statistics.median(lat)


def measure_one(name, k, args, wds, bs=8192):
    """{params, lat_mps_ms, lat_cpu_ms, us_per_sample_mps}."""
    name, k = (STEP_OF[name], 1) if name in STEP_OF else (name, k)
    res = {}
    for device, wd in wds.items():
        torch.manual_seed(0)
        model = build_model(name, k, **args).to(device).eval()
        res['params'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        with torch.no_grad():
            ep, t = next(D.eval_indices(wd.n, wd.T, bs))
            x, m = wd.windows(ep, t, k)
            x = D.scale_obs(x)
            res[f'lat_{device}_ms'] = latency_ms(model, x[:1], m[:1], device)
            if device == 'mps':
                for _ in range(2):
                    model(x, m)
                _sync(device)
                t0 = time.perf_counter()
                for _ in range(3):
                    model(x, m)
                _sync(device)
                res['us_per_sample_mps'] = 1e6 * (time.perf_counter() - t0) / (3 * bs)
        del model
        if device == 'mps':
            torch.mps.empty_cache()
    return res


def measure_episode(run, cfg, wds, n_ep=82, reps=60, warm=10):
    """CMF (bölüm kipi, eğitilmiş ağırlıklar): {params, lat_<cihaz>_ms (çevrimiçi adım, bs=1), lat_full_<cihaz>_ms (tam
    yeniden hesap, S=101, bs=1), us_per_sample_mps (n_ep bölüm × T hedef, tek geçiş), step_max_abs_<cihaz>}."""
    res = {}
    for device, wd in wds.items():
        model = build_model(cfg['model'], cfg['k'], **cfg['model_args'])
        model.load_state_dict(torch.load(os.path.join(RESULTS, 'asama3', f'{run}_k101_s0', 'model.pt'),
                                         map_location='cpu'))
        model = model.to(device).eval()
        res['params'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        x = D.scale_obs(wd.obs[:n_ep])
        S = x.shape[1]
        with torch.no_grad():
            ref = model.forward_episode(x[:4])
            state = model.init_state(4, device)
            out = torch.stack([model.step(x[:4, t], state) for t in range(S)], 1)
            err = float((out - ref).abs().max())
            if err > 1e-3:
                raise RuntimeError(f'{run} {device}: adım kipi forward_episode\'dan {err:.2e} sapıyor')
            res[f'step_max_abs_{device}'] = err
            lat = []
            for e in range((reps + warm) // S + 1):
                state = model.init_state(1, device)
                for t in range(S):
                    _sync(device)
                    t0 = time.perf_counter()
                    model.step(x[e:e + 1, t], state)
                    _sync(device)
                    lat.append(time.perf_counter() - t0)
            res[f'lat_{device}_ms'] = 1000 * statistics.median(lat[warm:])
            res[f'lat_full_{device}_ms'] = latency_ms(model, x[:1], None, device, reps=reps, warm=warm)
            if device == 'mps':
                for _ in range(2):
                    model.forward_episode(x)
                _sync(device)
                t0 = time.perf_counter()
                for _ in range(3):
                    model.forward_episode(x)
                _sync(device)
                res['us_per_sample_mps'] = 1e6 * (time.perf_counter() - t0) / (3 * n_ep * (S - 1))
        del model
        if device == 'mps':
            torch.mps.empty_cache()
    return res


def measure(stage='asama2'):
    store = D.EpisodeStore()
    idx = store.split('test')[:100]
    wds = {d: D.WindowData(store, idx, device=d) for d in ('mps', 'cpu')}
    res = {'machine': socket.gethostname(), 'platform': platform.platform(), 'torch': torch.__version__,
           'cpu_threads': torch.get_num_threads(), 'cells': []}
    todo = cells(stage)
    for i, (st, name, k, cfg) in enumerate(todo, 1):
        if st == 'asama3':
            r = measure_episode(name, cfg, wds)
        else:
            r = measure_one(name, k, cfg.get('model_args') or {}, wds)
        res['cells'].append({'stage': st, 'model': name, 'k': k, **r})
        extra = f"  tam yeniden hesap CPU {r['lat_full_cpu_ms']:6.2f} ms" if st == 'asama3' else ''
        print(f"[{i}/{len(todo)}] {name:12s} k={_k_label(k):>3s} param {r['params']:>9,d}  "
              f"MPS {r['lat_mps_ms']:6.2f} ms  CPU {r['lat_cpu_ms']:6.2f} ms  verim {r['us_per_sample_mps']:7.2f} µs/örnek"
              + extra, flush=True)
    os.makedirs(OUT_DIRS[stage], exist_ok=True)
    path = os.path.join(OUT_DIRS[stage], f"olcum_{res['machine']}.json")
    with open(path, 'w') as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print('→', path)


def pareto_front(points):
    """points: [(x, y)]; x küçük, y büyük daha iyi → baskın olmayanların indeksleri (x'e göre sıralı)."""
    order = sorted(range(len(points)), key=lambda i: (points[i][0], -points[i][1]))
    front, best = [], -np.inf
    for i in order:
        if points[i][1] > best:
            front.append(i)
            best = points[i][1]
    return front


def m3_summary(run, seeds=(0, 1, 2)):
    """asama3 koşusunun seed ortalamaları, ozet.json biçiminde ({ölçüt: {'mean'}})."""
    reps = [json.load(open(os.path.join(RESULTS, 'asama3', f'{run}_k101_s{s}', 'metrics.json')))['reports']
            for s in seeds]
    out = {}
    for key, split, col in [('test_host_macro_f1', 'test', 'host_macro_f1'),
                            ('ood_host_macro_f1', 'ood_test', 'host_macro_f1'),
                            ('ood_silent_recall', 'ood_test', 'silent_recall')]:
        out[key] = {'mean': float(np.mean([r[split][col] for r in reps]))}
    return out


def load_rows(meas):
    s = {st: json.load(open(os.path.join(RESULTS, st, 'ozet', 'ozet.json'))) for st in ('asama1', 'asama2')}
    s['asama3'] = {f'{run}_k101': m3_summary(run) for run in M3_RUNS
                   if any(c['model'] == run for c in meas['cells'])}
    rows = []
    for c in meas['cells']:
        g = s[c['stage']].get(f"{c['model']}_k{c['k']}")
        if g:
            best_k = c['stage'] == 'asama3' or _best(s[c['stage']], c['model'])['k'] == c['k']
            rows.append({**c, 'test': g['test_host_macro_f1']['mean'], 'ood': g['ood_host_macro_f1']['mean'],
                         'ood_silent': g['ood_silent_recall']['mean'], 'best_k': best_k})
    return rows


def _style(c):
    if c['stage'] == 'asama3':
        return M3_STYLE[c['model']]
    if c['stage'] == 'asama2':
        return dict(color=SLOT[c['model']], marker=MARKER[c['model']], ms=6, mec=SURFACE, mew=0.8)
    base = STEP_OF.get(c['model'], c['model'])
    full = c['model'] in STEP_OF
    return dict(color=INK if full else MUTED, marker=MARKER[base], ms=9 if full else 5, mfc=SURFACE, mew=1.2)


def plot(rows, path, stage='asama2'):
    panels = [('lat_cpu_ms', 'CPU gecikmesi, bs=1 (ms)'), ('params', 'eğitilebilir parametre')]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    for ax, (xcol, xlabel) in zip(axes, panels):
        for c in rows:
            ax.plot([c[xcol]], [c['ood']], ls='none', zorder=3, **_style(c))
        front = [rows[i] for i in pareto_front([(c[xcol], c['ood']) for c in rows])]
        ax.step([c[xcol] for c in front], [c['ood'] for c in front], where='post', color=INK2, lw=1, ls='--', zorder=2)
        for c in front:
            ax.annotate(M3_LABEL[c['model']] if c['stage'] == 'asama3' else f"{LABEL[c['model']]} k={_k_label(c['k'])}",
                        (c[xcol], c['ood']), xytext=(4, -10),
                        textcoords='offset points', fontsize=7.5, color=INK)
        ax.set_xscale('log')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('OOD host makro-F1')
        _grid(ax)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], ls='none', **_style({'stage': 'asama2', 'model': m}), label=LABEL[m]) for m in M2_MODELS]
    handles += [Line2D([], [], ls='none', **_style({'stage': 'asama1', 'model': m}), label=f'M1: {LABEL[m]}')
                for m in NEURAL_M1]
    if stage == 'asama3':
        handles = [Line2D([], [], ls='none', **M3_STYLE[m], label=M3_LABEL[m]) for m in M3_RUNS] + handles
    handles.append(Line2D([], [], color=INK2, lw=1, ls='--', label='Pareto sınırı'))
    fig.legend(handles=handles, loc='lower center', ncol=6, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle('Verimlilik–doğruluk: her nokta bir model × k hücresi (seed ortalaması)', x=0.01, ha='left',
                 fontsize=11, fontweight='bold')
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def ensemble_rows(rows):
    """3 seed'lik topluluk satırları (ood_paketi, topluluk_T): gecikme ve verim üye sayısı × üye."""
    path = os.path.join(RESULTS, 'asama3', 'ood_paketi', 'ozet.json')
    if not os.path.exists(path):
        return []
    models = json.load(open(path))['models']
    out = []
    for name, model in ENSEMBLE.items():
        c = next(c for c in rows if c['model'] == model and c['best_k'])
        e = models[name]['topluluk_T']
        n = len(models[name]['temperature']['uye'])
        out.append({**c, 'ensemble': n, 'params': n * c['params'], 'lat_cpu_ms': n * c['lat_cpu_ms'],
                    'lat_mps_ms': n * c['lat_mps_ms'], 'us_per_sample_mps': n * c['us_per_sample_mps'],
                    'test': e['test']['host_macro_f1'], 'ood': e['ood']['host_macro_f1'],
                    'ood_silent': e['ood']['silent_recall']})
    return out


def table_md(rows, meas, stage='asama2'):
    if stage == 'asama3':
        return table_m3_md(rows, meas)
    lines = ['# Verimlilik (final modeller, HPO boyutları)', '',
             f"Ölçüm: {meas['machine']}, {meas['platform']}, torch {meas['torch']}, CPU {meas['cpu_threads']} iş "
             'parçacığı. Gecikme bs=1 medyanı (50 tekrar); verim MPS bs=8192. Tam geçmiş RNN: adım başına tek hücre '
             "güncellemesi (aynı ayarla k=1). ★ = Pareto sınırında (CPU gecikmesi × OOD F1). Model başına val'e göre "
             'en iyi k; bütün hücreler `olcum_*.json`.', '',
             '| Model | k | Parametre | Gecikme CPU (ms) | Gecikme MPS (ms) | Verim MPS (µs/örnek) | Test F1 '
             '| OOD F1 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    front = {id(rows[i]) for i in pareto_front([(c['lat_cpu_ms'], c['ood']) for c in rows])}
    for c in sorted([c for c in rows if c['best_k']], key=lambda c: -c['ood']):
        name = label(c)
        lines.append(f"| {name}{' ★' if id(c) in front else ''} | {_k_label(c['k'])} | {c['params']:,} | "
                     f"{c['lat_cpu_ms']:.2f} | {c['lat_mps_ms']:.2f} | {c['us_per_sample_mps']:.2f} | "
                     f"{c['test']:.3f} | {c['ood']:.3f} |".replace(',', ' '))
    lines += ['', '## Pareto sınırı (bütün hücreler)', '', '| Model | k | Gecikme CPU (ms) | OOD F1 |',
              '|---|---:|---:|---:|']
    for i in pareto_front([(c['lat_cpu_ms'], c['ood']) for c in rows]):
        c = rows[i]
        lines.append(f"| {label(c)} | {_k_label(c['k'])} | {c['lat_cpu_ms']:.2f} | {c['ood']:.3f} |")
    return '\n'.join(lines) + '\n'


def table_m3_md(rows, meas):
    front = {id(rows[i]) for i in pareto_front([(c['lat_cpu_ms'], c['ood']) for c in rows])}
    lines = ['# Verimlilik, Aşama 3 (CyberMetaFormer + M1/M2 final modelleri, aynı oturumda)', '',
             f"Ölçüm: {meas['machine']}, {meas['platform']}, torch {meas['torch']}, CPU {meas['cpu_threads']} iş "
             'parçacığı. Gecikme bs=1 medyanı (50 tekrar); verim MPS bs≈8192. Tam geçmiş modeller (GRU/LSTM tam, CMF) '
             'çevrimiçi adım kipinde: RNN tek hücre güncellemesi, CMF `CyberMetaFormer.step` (konv tamponu + SSD durumu; '
             'maliyet t\'den bağımsız). CMF verimi: 82 bölüm × 100 hedef tek geçişte. ★ = Pareto sınırında (CPU '
             "gecikmesi × OOD F1). Model başına val'e göre en iyi k; F1'ler seed ortalaması (CMF 3, M1/M2 ozet.json). "
             'Topluluk satırları (ood_paketi, topluluk_T, 3 seed): maliyet üye × 3, Pareto\'ya girmez.', '',
             '| Model | k | Parametre | Gecikme CPU (ms) | Gecikme MPS (ms) | Verim MPS (µs/örnek) | Test F1 '
             '| OOD F1 | OOD sessiz recall |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    best = [c for c in rows if c['best_k']] + ensemble_rows(rows)
    for c in sorted(best, key=lambda c: -c['ood']):
        name = label(c) + (f" topluluk ({c['ensemble']})" if c.get('ensemble') else '')
        lines.append(f"| {name}{' ★' if id(c) in front else ''} | {_k_label(c['k'])} | {c['params']:,} | "
                     f"{c['lat_cpu_ms']:.2f} | {c['lat_mps_ms']:.2f} | {c['us_per_sample_mps']:.2f} | "
                     f"{c['test']:.3f} | {c['ood']:.3f} | {c['ood_silent']:.3f} |".replace(',', ' '))
    lines += ['', '## CMF çevrimiçi çıkarım: adım kipi ve tam yeniden hesap', '',
              'Adım kipi olmasaydı her adımda bütün geçmiş (S=101) yeniden hesaplanırdı; "tam yeniden hesap" bunun en kötü '
              'durumu (son adım). Sapma: adım kipinin forward_episode\'a göre en büyük mutlak logit farkı (4 test bölümü).', '',
              '| Model | Adım CPU (ms) | Adım MPS (ms) | Tam yeniden hesap CPU (ms) | Tam yeniden hesap MPS (ms) | Sapma CPU '
              '| Sapma MPS |', '|---|---:|---:|---:|---:|---:|---:|']
    for c in rows:
        if c['stage'] == 'asama3':
            lines.append(f"| {label(c)} | {c['lat_cpu_ms']:.2f} | {c['lat_mps_ms']:.2f} | {c['lat_full_cpu_ms']:.2f} | "
                         f"{c['lat_full_mps_ms']:.2f} | {c['step_max_abs_cpu']:.1e} | {c['step_max_abs_mps']:.1e} |")
    lines += ['', '## Pareto sınırı (bütün hücreler)', '', '| Model | k | Gecikme CPU (ms) | OOD F1 |',
              '|---|---:|---:|---:|']
    for i in pareto_front([(c['lat_cpu_ms'], c['ood']) for c in rows]):
        c = rows[i]
        lines.append(f"| {label(c)} | {_k_label(c['k'])} | {c['lat_cpu_ms']:.2f} | {c['ood']:.3f} |")
    return '\n'.join(lines) + '\n'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--measure', action='store_true', help='gecikme ve verimi bu makinede ölç')
    p.add_argument('--stage', choices=list(OUT_DIRS), default='asama2')
    args = p.parse_args()
    if args.measure:
        return measure(args.stage)
    out_dir = OUT_DIRS[args.stage]
    paths = sorted(glob.glob(os.path.join(out_dir, 'olcum_*.json')))
    if not paths:
        raise SystemExit(f'Ölçüm yok: önce --measure ({out_dir})')
    meas = json.load(open(paths[0]))
    rows = load_rows(meas)
    fig = (os.path.join(RESULTS, 'asama2', 'figures', 'fig_pareto.png') if args.stage == 'asama2'
           else os.path.join(out_dir, 'fig_pareto.png'))
    plot(rows, fig, args.stage)
    with open(os.path.join(out_dir, 'tablo.md'), 'w') as f:
        f.write(table_md(rows, meas, args.stage))
    print('→', out_dir)


if __name__ == '__main__':
    main()

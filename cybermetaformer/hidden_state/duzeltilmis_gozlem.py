"""Remove bayrağı kusuru: aynı modellerin kusurlu (v1) ve düzeltilmiş gözlemli eşli kopyalarda karşılaştırılması.

Yeniden eğitim yok: v1'de eğitilmiş `model.pt` (CMF ve GRU tam, asama3, s0–s4).

Veri: `minicage_v1_fixedobs` ve `minicage_v1_shift_fixedobs` (minicage/dataset/resimulate.py, FixedObsCAGE). Gerçek durum,
Red gözlemi, aksiyonlar ve bölüm kimlikleri v1 ile aynıdır, yalnız Blue gözlemi farklıdır: başarısız remove "kaldırıldı"
basmaz, geçersiz user0 restore'u user0'ın bayraklarını silmez. Etiketlerin ve politikaların aynı olduğu denetlenir.

Kümeler (ood_paketi ile aynı): test, ood (ε = 0.3), eps0.5, eps0.7, remove (react_remove).
Dilimler kusurlu veriden tanımlanır (birincil), böylece iki sürümde birebir aynı host-adımları karşılaştırılır:
    sessiz        y > 0 ∧ kusurlu gözlemde uyarı yok
    N_geçmiş      sessiz ∧ kusurlu gözlemde bölüm başından beri o hostta hiç uyarı yok (uyarı yaşı 'hiç')
    remove maskesi  sessiz ihlal nedeni 'remove_maske' (*_aux/silent_cause.npz): uyarıyı başarısız remove'un "kaldırıldı"
                  bayrağı silmiş
İkincil: sessiz ve N_geçmiş, her sürümün kendi gözleminden (düzeltilmiş sürümde başka host-adımları).
Üye başına kendi v1 val sıcaklığı (ood_paketi 'uye'), seed ortalaması. Eşli bölüm bootstrap'ı compare.py ile aynı (her
model seed ortalamasıyla, kümenin bölümleri örneklenir): (i) her model için düzeltilmiş − kusurlu, (ii) her sürümde
CMF − GRU tam. Holm bütün satırlar üzerinden.

collect, değişmeyen bölümlerde (gözlemi iki sürümde aynı olanlar) yeni logitlerin kayıtlı kusurlu logitlerle aynı tahmini
verdiğini denetler.

    python -m cybermetaformer.hidden_state.duzeltilmis_gozlem collect     # asama3/duzeltilmis_gozlem/logits/ (git dışı)
    python -m cybermetaformer.hidden_state.duzeltilmis_gozlem analyze     # makale/duzeltilmis_gozlem/{ozet.json, tablolar.md}
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from cybermetaformer.hidden_state import data as D
from cybermetaformer.hidden_state import evaluate as E
from cybermetaformer.hidden_state import features as F
from cybermetaformer.hidden_state import metrics as M
from cybermetaformer.hidden_state import ood_paketi as O
from cybermetaformer.hidden_state import uncertainty as U

RESULTS = os.path.join(D.ROOT, 'results', 'hidden_state')
LOGIT_DIR = os.path.join(RESULTS, 'asama3', 'duzeltilmis_gozlem', 'logits')
OUT_DIR = os.path.join(RESULTS, 'makale', 'duzeltilmis_gozlem')
FIXED = {'v1': os.path.join(D.DATA_ROOT, 'minicage_v1_fixedobs'),
         'shift': os.path.join(D.DATA_ROOT, 'minicage_v1_shift_fixedobs')}
BASE = {b: O.BASE[b] for b in ('test', 'ood', 'shift')}          # temel küme → (veri, split)
SETS = {s: O.SETS[s] for s in ('test', 'ood', 'eps0.5', 'eps0.7', 'remove')}
SET_LABEL = {'test': 'Test', 'ood': 'OOD (ε = 0.3)', 'eps0.5': 'ε = 0.5', 'eps0.7': 'ε = 0.7', 'remove': 'react_remove'}
MODELS = {'cybermetaformer': 'CyberMetaFormer', 'gru_full': 'GRU (full history)'}
SEEDS = (0, 1, 2, 3, 4)
VERSIONS = ('flawed', 'corrected')
VERSION_LABEL = {'flawed': 'flawed', 'corrected': 'corrected'}
# ölçüt → (kayıt anahtarı, etiket); 'conf' host makro-F1'dir, diğerleri (n, isabet) sayımları
CRITERIA = {'host_macro_f1': ('conf', 'Host macro-F1'),
            'silent_recall': ('silent', 'Silent recall'),
            'never_recall': ('never', 'N_hist recall'),
            'remove_mask_recall': ('remove_mask', 'Remove-mask recall'),
            'silent_recall_own': ('silent_own', 'Silent recall (own def.)'),
            'never_recall_own': ('never_own', 'N_hist recall (own def.)')}
PRIMARY = ('host_macro_f1', 'silent_recall', 'never_recall', 'remove_mask_recall')


def open_stores():
    return {'flawed': {'v1': D.EpisodeStore(D.DEFAULT_DATA), 'shift': D.EpisodeStore(O.DATA['shift'])},
            'corrected': {src: D.EpisodeStore(path) for src, path in FIXED.items()}}


def check_pairing(stores):
    """İki sürüm aynı bölümleri taşımalı: etiketler, politikalar ve split'ler aynı."""
    for src in FIXED:
        a, b = stores['flawed'][src], stores['corrected'][src]
        for attr in ('labels', 'red_policy', 'blue_policy', 'prev_action'):
            assert np.array_equal(getattr(a, attr), getattr(b, attr)), f'{src}: {attr} farklı'
        assert all(np.array_equal(a._splits[k], b._splits[k]) for k in a._splits), f'{src}: split farklı'


def base_indices(stores):
    out = {}
    for b, (src, split) in BASE.items():
        st = stores['flawed'][src]
        with st.final_evaluation():
            out[b] = np.arange(st.E) if split is None else st.split(split)
    return out


def changed_episodes(stores, idx):
    return {b: (stores['flawed'][src].obs[idx[b]] != stores['corrected'][src].obs[idx[b]]).any((1, 2))
            for b, (src, _) in BASE.items()}


def logit_path(name, seed):
    return os.path.join(LOGIT_DIR, f'{name}_k{O.K}_s{seed}.npz')


# ── toplama ──────────────────────────────────────────────────────────────

def collect(device, models=tuple(MODELS), seeds=SEEDS):
    """Düzeltilmiş kopyalardaki logitler (float16). Var olan dosya atlanır."""
    stores = open_stores()
    check_pairing(stores)
    idx = base_indices(stores)
    changed = changed_episodes(stores, idx)
    os.makedirs(LOGIT_DIR, exist_ok=True)
    for name in models:
        for seed in seeds:
            path = logit_path(name, seed)
            if os.path.exists(path):
                continue
            model, cfg = O.load_model(name, seed, device)
            t0, out = time.time(), {}
            for b, (src, _) in BASE.items():
                wd = D.WindowData(stores['corrected'][src], idx[b], device=device)
                out[b] = E.predict_logits(model, wd, O.K, cfg['scale'], cfg['eval_batch_size']).astype(np.float16)
                del wd
            ref = O.load_member(name, seed)
            notes = []
            for b in BASE:
                same = ~changed[b]
                agree = float((out[b][same].argmax(-1) == ref[b][same].argmax(-1)).mean())
                diff = float(np.abs(out[b][same].astype(np.float32) - ref[b][same].astype(np.float32)).max())
                assert agree > 0.9999, f'{name} s{seed} {b}: değişmeyen bölümlerde tahmin uyumu {agree}'
                notes.append(f'{b} uyum {agree:.6f} en büyük fark {diff:.4f} ({int(same.sum())} bölüm)')
            np.savez_compressed(path, **out)
            print(f'{name} s{seed}: {time.time() - t0:.0f} s; değişmeyen bölümler: ' + '; '.join(notes), flush=True)
            del model
            if device == 'mps':
                torch.mps.empty_cache()


# ── analiz ───────────────────────────────────────────────────────────────

def _member_records(lg, t, y, alert_f, age_f, alert_v, age_v, cause, cause_names):
    pred = U.probs(lg, t).argmax(-1)
    cnt, names = M.episode_slice_counts(y, pred, alert_f, age_f, cause, cause_names)
    own, own_names = M.episode_slice_counts(y, pred, alert_v, age_v)
    return {'conf': M.episode_confusion(y, pred),
            'silent': cnt[:, names.index('sessiz')], 'never': cnt[:, names.index('hiç')],
            'remove_mask': cnt[:, names.index('remove_maske')],
            'silent_own': own[:, own_names.index('sessiz')], 'never_own': own[:, own_names.index('hiç')]}


def _value(rec, key):
    if key == 'conf':
        return float(M.host_macro_f1_batch(rec['conf'].sum(0)[None])[0])
    n = rec[key].sum(0)
    return float(n[1] / n[0]) if n[0] else None


def _stat(runs, key, s):
    if key == 'conf':
        return lambda W: np.mean([M.host_macro_f1_batch(np.tensordot(W[s], r['conf'], axes=1)) for r in runs], 0)
    return lambda W: np.mean([(W[s] @ r[key][:, 1].astype(np.float64)) / np.maximum(W[s] @ r[key][:, 0].astype(
        np.float64), 1) for r in runs], 0)


def analyze(n_boot=10_000, models=tuple(MODELS), seeds=SEEDS, log=print):
    stores = open_stores()
    check_pairing(stores)
    idx = base_indices(stores)
    changed = changed_episodes(stores, idx)
    causes = {}
    for src, path in O.CAUSES.items():
        with np.load(path) as z:
            causes[src] = (z['cause'], [str(c) for c in z['causes']])
    v1 = stores['flawed']['v1']
    val_y = v1.labels[v1.split('val')]
    members = {name: {seed: {'flawed': O.load_member(name, seed), 'corrected': dict(np.load(logit_path(name, seed)))}
                      for seed in seeds} for name in models}
    temps = {name: {seed: E.fit_temperature(m['flawed']['val'].astype(np.float32), val_y)
                    for seed, m in ms.items()} for name, ms in members.items()}

    res = {'seeds': list(seeds), 'n_boot': n_boot, 'temperature': temps, 'sets': {}, 'models': {}, 'rows': []}
    runs = {}                                   # (küme, model, sürüm) → üye kayıtları
    for s, (b, pol) in SETS.items():
        src = BASE[b][0]
        st_f, st_c = stores['flawed'][src], stores['corrected'][src]
        base = idx[b]
        pos = np.arange(len(base)) if pol is None else \
            np.flatnonzero(st_f.blue_policy[base] == st_f.meta['blue_policies'].index(pol))
        ep = base[pos]
        y = st_f.labels[ep].astype(np.int64)
        alert = {'flawed': F.episode_alerts(st_f.obs[ep]), 'corrected': F.episode_alerts(st_c.obs[ep])}
        age = {v: F.alert_age(a) for v, a in alert.items()}
        cause, cause_names = causes[src][0][ep], causes[src][1]
        silent = (y > 0) & ~alert['flawed']
        never = silent & (age['flawed'] == -1)
        rmask = cause == cause_names.index('remove_maske')
        res['sets'][s] = {
            'source': [b, pol], 'n_episodes': len(ep), 'n_episodes_changed': int(changed[b][pos].sum()),
            'n_host_steps': int(y.size), 'n_silent': int(silent.sum()), 'n_never': int(never.sum()),
            'n_remove_mask': int(rmask.sum()),
            # kusurlu tanımla sessiz olup düzeltilmiş gözlemde uyarı gören host-adımlarının payı
            'silent_alerted_when_corrected': float((silent & alert['corrected']).sum() / max(silent.sum(), 1)),
            'never_alerted_when_corrected': float((never & alert['corrected']).sum() / max(never.sum(), 1)),
            'remove_mask_alerted_when_corrected': float((rmask & alert['corrected']).sum() / max(rmask.sum(), 1)),
            'n_silent_own_corrected': int(((y > 0) & ~alert['corrected']).sum()),
        }
        for name in models:
            for v in VERSIONS:
                runs[s, name, v] = [_member_records(members[name][seed][v][b][pos], temps[name][seed], y,
                                                    alert['flawed'], age['flawed'], alert[v], age[v],
                                                    cause, cause_names) for seed in seeds]
                vals = {c: [_value(r, key) for r in runs[s, name, v]] for c, (key, _) in CRITERIA.items()}
                res['models'].setdefault(name, {}).setdefault(s, {})[v] = {
                    c: {'mean': float(np.mean(x)), 'sd': float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
                        'seeds': x} for c, x in vals.items() if all(e is not None for e in x)}
            m = res['models'][name][s]
            log(f"{name:16s} {s:7s} " + '  '.join(
                f"{c} {m['flawed'][c]['mean']:.4f}→{m['corrected'][c]['mean']:.4f}" for c in PRIMARY if c in m['flawed']))

    for s in SETS:
        E_ = res['sets'][s]['n_episodes']
        for name in models:                                  # (i) düzeltilmiş − kusurlu
            for c, (key, _) in CRITERIA.items():
                if c not in res['models'][name][s]['flawed']:
                    continue
                r = M.paired_bootstrap(_stat(runs[s, name, 'corrected'], key, s), _stat(runs[s, name, 'flawed'], key, s),
                                       {s: E_}, n_boot=n_boot, seed=0)
                res['rows'].append({'kind': 'corrected_minus_flawed', 'set': s, 'model': name, 'criterion': c, **r})
        if {'cybermetaformer', 'gru_full'} <= set(models):   # (ii) CMF − GRU tam, sürüm başına
            for v in VERSIONS:
                for c in PRIMARY:
                    if c not in res['models']['gru_full'][s][v]:
                        continue
                    key = CRITERIA[c][0]
                    r = M.paired_bootstrap(_stat(runs[s, 'cybermetaformer', v], key, s),
                                           _stat(runs[s, 'gru_full', v], key, s), {s: E_}, n_boot=n_boot, seed=0)
                    res['rows'].append({'kind': 'cmf_minus_gru', 'set': s, 'version': v, 'criterion': c, **r})
    for r, p in zip(res['rows'], M.holm([r['p'] for r in res['rows']])):
        r['p_holm'] = p
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, 'ozet.json'), 'w') as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    with open(os.path.join(OUT_DIR, 'tablolar.md'), 'w') as f:
        f.write(tables_md(res))
    log('→', OUT_DIR)
    return res


# ── çıktılar ─────────────────────────────────────────────────────────────

def _ms(d):
    return f"{d['mean']:.3f} ± {d['sd']:.3f}" if d else ''


def _ci(r):
    return f"{r['diff']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]"


def _p(p):
    return '< 0.001' if p < 0.001 else f'{p:.3f}'


def tables_md(res):
    n = len(res['seeds'])
    out = ['# Flawed vs. corrected observation (remove flag)', '',
           f"Models trained on the flawed data; evaluated on both copies. Mean ± sd over {n} seeds; per-member "
           f"temperature from v1 validation. Silent, N_hist and remove-mask slices are defined on the flawed observation "
           f"(identical host-steps in both copies); \"own def.\" recomputes silent and N_hist from each copy's "
           f"observation. Differences: paired episode bootstrap ({res['n_boot']:,} resamples), Holm over all "
           f"{len(res['rows'])} rows.", '',
           '## Sets', '',
           '| Set | Episodes | Changed episodes | Silent host-steps | N_hist | Remove-mask | Silent steps alerted when '
           'corrected | Remove-mask steps alerted when corrected |',
           '|---|---|---|---|---|---|---|---|']
    for s, d in res['sets'].items():
        out.append(f"| {SET_LABEL[s]} | {d['n_episodes']:,} | {d['n_episodes_changed']:,} | {d['n_silent']:,} | "
                   f"{d['n_never']:,} | {d['n_remove_mask']:,} | {d['silent_alerted_when_corrected']:.3f} | "
                   f"{d['remove_mask_alerted_when_corrected']:.3f} |")
    out += ['', '## Metrics', '', '| Set | Model | Copy | ' + ' | '.join(lab for _, lab in CRITERIA.values()) + ' |',
            '|---|---|---|' + '---|' * len(CRITERIA)]
    for s in res['sets']:
        for name, lab in MODELS.items():
            if name not in res['models']:
                continue
            for v in VERSIONS:
                m = res['models'][name][s][v]
                out.append(f"| {SET_LABEL[s]} | {lab} | {VERSION_LABEL[v]} | "
                           + ' | '.join(_ms(m.get(c)) for c in CRITERIA) + ' |')
    out += ['', '## Corrected − flawed (same model)', '', '| Set | Model | Criterion | Δ [95% CI] | p (Holm) |',
            '|---|---|---|---|---|']
    for r in res['rows']:
        if r['kind'] == 'corrected_minus_flawed':
            out.append(f"| {SET_LABEL[r['set']]} | {MODELS[r['model']]} | {CRITERIA[r['criterion']][1]} | {_ci(r)} | "
                       f"{_p(r['p_holm'])} |")
    rows = [r for r in res['rows'] if r['kind'] == 'cmf_minus_gru']
    if rows:
        out += ['', '## CyberMetaFormer − GRU (full history), per copy', '',
                '| Set | Copy | Criterion | Δ [95% CI] | p (Holm) |', '|---|---|---|---|---|']
        for r in rows:
            out.append(f"| {SET_LABEL[r['set']]} | {VERSION_LABEL[r['version']]} | {CRITERIA[r['criterion']][1]} | "
                       f"{_ci(r)} | {_p(r['p_holm'])} |")
    return '\n'.join(out) + '\n'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['collect', 'analyze'])
    p.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    p.add_argument('--n-boot', type=int, default=10_000)
    p.add_argument('--models', nargs='+', default=list(MODELS))
    p.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    args = p.parse_args()
    if args.cmd == 'collect':
        collect(args.device, args.models, tuple(args.seeds))
    else:
        analyze(args.n_boot, tuple(args.models), tuple(args.seeds))


if __name__ == '__main__':
    main()

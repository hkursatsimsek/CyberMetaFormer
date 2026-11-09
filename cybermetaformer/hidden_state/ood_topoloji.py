"""Topoloji OOD'si (revizyon madde 6 (b)): ağ bağlantısı değişince modellerin tahmini.

Yeniden eğitim yok: v1'de (MiniCAGE topolojisi: user1/2 → ent0, user3/4 → ent1) eğitilmiş `model.pt` (asama3, s0–s4).
Değerlendirme verisi `minicage_v1_topo`: RewiredCAGE ile user1/2 → ent1, user3/4 → ent0. Alt ağlar, host sayısı ve
gözlem düzeni aynı olduğu için modeller değişmeden çalışır. Host sayısı ya da gözlem düzeni değişseydi girdi boyutu,
host gömmesi ve ilişki matrisi değişeceği için model de değişmek zorunda kalırdı. B-line, ent hedefini bağlantı
tablosundan okuyan LinkedBline'dır. CMF ve gru_host_topo'nun topoloji önseli (ilişki matrisi) burada yanlış
tanımlıdır; host_tam ve gru_host_attn'in önseli yoktur.

Koşullar (her biri aynı Blue politikasıyla iki kümede):
    original   v1: eps0.1 = v1 test'in react_restore_eps0.1 bölümleri, ood = v1 ood_test (ε = 0.3)
    P2         aynı v1 bölümleri, ood_permutasyon'un P2 permütasyonuyla (user1↔user3, user2↔user4): yeniden
               bağlamanın veri içindeki karşılığı; decoy kapasitesi hostla birlikte taşınır
    P2s        P2, ama decoy sayısı slotta kalır. Gerçek yeniden bağlamada da kapasite slotun (hostun) özelliğidir;
               en yakın veri içi karşılık budur
    rewired    minicage_v1_topo: eps0.1 = react_restore_eps0.1'in bütün bölümleri (hiçbiri eğitimde kullanılmadı),
               ood = react_restore_eps0.3'ün bütün bölümleri
    rewired_prior  aynı bölümler; topoloji sapmalı modellerde (CMF, gru_host_topo) ilişki matrisi yeni bağlantı
               tablosundan yeniden kurulur (meta.json → topology). Yeniden eğitim yok, öğrenilen ağırlıklar aynı:
               ortam topolojisi biliniyorsa yalnız önselin güncellenmesi yetiyor mu?
original, P2 ve P2s aynı bölümlerdir (eşli). rewired ayrı bölümlerdir: rewired − original ve rewired − P2s farkları eşsiz
iki örneklemli bölüm bootstrap'ıyla hesaplanır (iki küme bağımsız örneklenir). Model çiftleri her koşulda eşlidir.
"(a_rewired − a_original) − (b_rewired − b_original)" farkı da iki örneklemlidir.

Model çiftlerinde ayrıca "a_rewired_prior − b_rewired" (eşli) verilir. Ölçütler ve dökümler ood_permutasyon ile aynıdır: host makro-F1, sessiz recall, N_geçmiş recall, hata türleri, host
grubu (slot) başına sessiz recall. Ek olarak saldırgan (B-line / Meander) başına döküm verilir. Holm bütün satırlar
üzerinden uygulanır.

    python -m cybermetaformer.hidden_state.ood_topoloji collect     # asama3/ood_topoloji/preds/ (git dışı)
    python -m cybermetaformer.hidden_state.ood_topoloji analyze     # ood_permutasyon collect'i de ister;
                                                                  # makale/ood_topoloji/{ozet.json, tablolar.md}
"""
import argparse
import contextlib
import json
import os
import time

import numpy as np
import torch

from cybermetaformer.hidden_state import data as D
from cybermetaformer.hidden_state import features as F
from cybermetaformer.hidden_state import metrics as M
from cybermetaformer.hidden_state import ood_paketi as O
from cybermetaformer.hidden_state import ood_permutasyon as P
from cybermetaformer.hidden_state.models.cybermetaformer import HOSTS, REL_OTHER, host_relations
from cybermetaformer.hidden_state.ood_permutasyon import (CRITERIA, ERROR_LABEL, ERRORS, GROUPS, MODELS, PAIRS, SEEDS,
                                                        _ci, _ms, _p)

RESULTS = os.path.join(D.ROOT, 'results', 'hidden_state')
PRED_DIR = os.path.join(RESULTS, 'asama3', 'ood_topoloji', 'preds')
OUT_DIR = os.path.join(RESULTS, 'makale', 'ood_topoloji')
TOPO = os.path.join(D.DATA_ROOT, 'minicage_v1_topo')
SETS = {'eps0.1': ('test', 'react_restore_eps0.1'), 'ood': ('ood', 'react_restore_eps0.3')}   # → (perm kümesi, politika)
SET_LABEL = {'eps0.1': 'ε = 0.1 (training policy)', 'ood': 'ε = 0.3 (OOD policy)'}
CONDITIONS = ('original', 'P2', 'P2s', 'rewired', 'rewired_prior')
COND_LABEL = {'original': 'original topology', 'P2': 'P2 (rewiring by permutation)',
              'P2s': 'P2, decoy count stays in slot', 'rewired': 'rewired simulator',
              'rewired_prior': 'rewired simulator, topology bias updated'}
TOPO_BIASED = ('cybermetaformer', 'cybermetaformer-gru_host_topo')     # host_mixer: topo
TOPO_SIDE = ('rewired', 'rewired_prior')
REDS = ('bline', 'meander')
RED_LABEL = {'bline': 'B-line', 'meander': 'Meander'}


def _policy(st, name):
    return st.meta['blue_policies'].index(name)


def open_sets():
    """v1 ve topo store'ları; her küme için v1 bölümleri (perm kümesindeki konumlarıyla) ve topo bölümleri."""
    v1, topo = D.EpisodeStore(D.DEFAULT_DATA), D.EpisodeStore(TOPO)
    assert v1.meta['red_policies'] == topo.meta['red_policies'] == list(REDS)
    _, base = P.open_sets()
    sets = {}
    for s, (b, pol) in SETS.items():
        pos = np.flatnonzero(v1.blue_policy[base[b]] == _policy(v1, pol))
        sets[s] = {'perm_set': b, 'pos': pos, 'v1': base[b][pos],
                   'topo': np.flatnonzero(topo.blue_policy == _policy(topo, pol))}
    return v1, topo, sets


def pred_path(name, seed):
    return P.pred_path(name, seed, PRED_DIR)


def links_of(store):
    return [store.meta['topology']['connected_hosts'][h] for h in HOSTS]


@contextlib.contextmanager
def updated_prior(model, links):
    """Topoloji sapmalı HostMixer'ların ilişki matrisini (ve varsa maskesini) verilen bağlantı tablosundan kurar."""
    saved = []
    for blk in model.blocks:
        hm = blk.host
        if hm is None or hm.kind != 'topo':
            continue
        saved.append((hm, hm.rel, hm.mask))
        rel = host_relations(links).to(hm.rel.device)
        masked = bool(torch.isinf(hm.mask).any())
        hm.rel = rel
        hm.mask = torch.where(rel == REL_OTHER, float('-inf'), 0.0).to(hm.mask.device) if masked else hm.mask
    assert saved, 'modelde topoloji sapmalı host karıştırıcısı yok'
    try:
        yield model
    finally:
        for hm, rel, mask in saved:
            hm.rel, hm.mask = rel, mask


# ── toplama ──────────────────────────────────────────────────────────────

def collect(device, models=tuple(MODELS), seeds=SEEDS):
    """Topo verisinin bütün bölümlerinde tahmin (E, T, 12) int8: 'all' ve topoloji sapmalı modellerde 'prior'
    (güncellenmiş önsel). Var olan dosyaya yalnız eksik anahtarlar eklenir."""
    topo = D.EpisodeStore(TOPO)
    assert topo.meta['topology']['name'] == 'rewired'
    links = links_of(topo)
    os.makedirs(PRED_DIR, exist_ok=True)
    for name in models:
        for seed in seeds:
            path = pred_path(name, seed)
            out = dict(np.load(path)) if os.path.exists(path) else {}
            todo = [k for k in ('all', 'prior') if k not in out and (k == 'all' or name in TOPO_BIASED)]
            if not todo:
                continue
            model, cfg = O.load_model(name, seed, device)
            t0 = time.time()
            for k in todo:
                if k == 'all':
                    out[k] = P.predict(model, cfg, topo.obs, topo.labels, topo.prev_action, device)
                else:
                    with updated_prior(model, links):
                        out[k] = P.predict(model, cfg, topo.obs, topo.labels, topo.prev_action, device)
            np.savez_compressed(path, **out)
            f1 = {k: M.host_macro_f1(topo.labels.reshape(-1, 12), out[k].reshape(-1, 12)) for k in out}
            print(f'{name} s{seed}: {time.time() - t0:.0f} s; topo F1 (bütün bölümler) '
                  + ', '.join(f'{k} {v:.4f}' for k, v in f1.items()), flush=True)
            del model
            if device == 'mps':
                torch.mps.empty_cache()


# ── analiz ───────────────────────────────────────────────────────────────

def analyze(n_boot=10_000, models=tuple(MODELS), seeds=SEEDS, log=print):
    v1, topo, sets = open_sets()
    perm = {(n, s): dict(np.load(P.pred_path(n, s))) for n in models for s in seeds}
    rew = {(n, s): dict(np.load(pred_path(n, s))) for n in models for s in seeds}
    res = {'seeds': list(seeds), 'n_boot': n_boot, 'topology': topo.meta['topology'],
           'p2_permutation': {P.HOSTS[i]: P.HOSTS[j] for i, j in enumerate(P.PERMS['P2']) if i != j},
           'sets': {}, 'models': {}, 'by_red': {}, 'rows': []}
    for s, d in sets.items():
        sources = {c: (topo, d['topo']) if c in TOPO_SIDE else (v1, d['v1']) for c in CONDITIONS}
        runs, red = {}, {}
        for c, (st, ep) in sources.items():
            obs, y = st.obs[ep], st.labels[ep].astype(np.int64)
            if c in P.DATA_CONDITIONS:
                obs, y = P.permute(obs, y, *P.DATA_CONDITIONS[c])
            alert = F.episode_alerts(obs)
            age = F.alert_age(alert)
            red[c] = st.red_policy[ep]
            res['sets'].setdefault(s, {})[c] = P.set_stats(y, alert, age, red[c], REDS)
            for name in models:
                if c == 'rewired_prior' and name not in TOPO_BIASED:
                    continue
                preds = [rew[name, seed]['all' if c == 'rewired' else 'prior'][ep] if c in TOPO_SIDE else
                         perm[name, seed][f"{d['perm_set']}__{'id' if c == 'original' else c}"][d['pos']] for seed in seeds]
                runs[c, name] = [P.records(y, p, alert, age) for p in preds]
                res['models'].setdefault(name, {}).setdefault(s, {})[c] = P.summarize(runs[c, name])
                for r, rname in enumerate(REDS):
                    res['by_red'].setdefault(name, {}).setdefault(s, {}).setdefault(c, {})[rname] = \
                        P.summarize(runs[c, name], red[c] == r)
        for name in models:
            m = res['models'][name][s]
            log(f'{name:30s} {s:6s} ' + '  '.join(
                f"{c}: F1 {m[c]['host_macro_f1']['mean']:.4f} sessiz {m[c]['silent_recall']['mean']:.4f} "
                f"N {m[c]['never_recall']['mean']:.4f}" for c in CONDITIONS if c in m))

        Er, Et = len(d['v1']), len(d['topo'])
        side = {(c, n): 'topo' if c in TOPO_SIDE else 'ref' for c in CONDITIONS for n in models}
        two = {'ref': Er, 'topo': Et}
        for name in models:
            for a, b, kind in (('rewired', 'original', 'rewired_minus_original'), ('P2', 'original', 'p2_minus_original'),
                               ('P2s', 'original', 'p2s_minus_original'), ('rewired', 'P2s', 'rewired_minus_p2s'),
                               ('rewired_prior', 'rewired', 'prior_minus_rewired'),
                               ('rewired_prior', 'original', 'prior_minus_original')):
                ka, kb = (a, name), (b, name)
                if ka not in runs or kb not in runs:
                    continue
                sizes = {side[ka]: two[side[ka]]} if side[ka] == side[kb] else two
                res['rows'] += P.bootstrap_rows(runs, sizes, n_boot, ka, kb, side={ka: side[ka], kb: side[kb]},
                                                kind=kind, set=s, model=name)
        for a, b in PAIRS:
            if a not in models or b not in models:
                continue
            for c in CONDITIONS:
                ka, kb = (c, a), (c, b)
                if ka not in runs or kb not in runs:
                    continue
                res['rows'] += P.bootstrap_rows(runs, {side[ka]: two[side[ka]]}, n_boot, ka, kb,
                                                side={ka: side[ka], kb: side[kb]}, kind='pair', set=s, a=a, b=b,
                                                condition=c)
            if ('rewired_prior', a) in runs and ('rewired_prior', b) not in runs:
                ka, kb = ('rewired_prior', a), ('rewired', b)
                res['rows'] += P.bootstrap_rows(runs, {'topo': Et}, n_boot, ka, kb, side={ka: 'topo', kb: 'topo'},
                                                kind='pair_prior', set=s, a=a, b=b, condition='rewired_prior')
            ka, kb, ba, bb = ('rewired', a), ('rewired', b), ('original', a), ('original', b)
            res['rows'] += P.bootstrap_rows(runs, two, n_boot, ka, kb, base=(ba, bb),
                                            side={k: side[k] for k in (ka, kb, ba, bb)}, kind='pair_drop', set=s,
                                            a=a, b=b, condition='rewired')
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

KIND_LABEL = {'rewired_minus_original': 'rewired − original (two-sample)',
              'p2_minus_original': 'P2 − original (paired)',
              'p2s_minus_original': 'P2s − original (paired)',
              'rewired_minus_p2s': 'rewired − P2s (two-sample)',
              'prior_minus_rewired': 'rewired, bias updated − rewired (paired)',
              'prior_minus_original': 'rewired, bias updated − original (two-sample)'}


def tables_md(res):
    n = len(res['seeds'])
    ch = {h: t for h, t in res['topology']['connected_hosts'].items() if t != res['topology']['minicage_connected_hosts'][h]}
    out = ['# Topology OOD (rewired user → enterprise links)', '',
           f"Models trained on the MiniCAGE topology (user1/2 → ent0, user3/4 → ent1); evaluated on a rewired simulator "
           f"({', '.join(f'{h} → {t}' for h, t in ch.items())}; subnets, host count and observation layout unchanged, so "
           f"the models run unmodified). B-line reads its enterprise target from the link table. The topology bias of "
           f"CyberMetaFormer and the host-token GRU with topology bias is therefore misspecified; the other two "
           f"host-token models have no topology bias. \"Topology bias updated\" rebuilds the relation matrix of the two "
           f"topology-biased models from the rewired link table (same weights, no retraining). P2 applies the equivalent host permutation to the original "
           f"episodes (decoy capacity moves with the host); P2s keeps the decoy-count columns in their slot, as in the "
           f"rewired simulator, where decoy capacity stays with the host slot. Original, P2 and P2s share episodes (paired "
           f"bootstrap); the rewired simulator has its own episodes (two-sample bootstrap, sets resampled "
           f"independently). Mean ± sd over {n} seeds (argmax). {res['n_boot']:,} resamples; Holm over all "
           f"{len(res['rows'])} rows.", '',
           '## Sets', '', '| Set | Condition | Episodes (B-line / Meander) | Host-steps | Clean | User | Root | Silent | N_hist |',
           '|---|---|---|---|---|---|---|---|---|']
    for s, ds in res['sets'].items():
        for c, d in ds.items():
            if c == 'rewired_prior':                       # rewired ile aynı bölümler
                continue
            cc, er = d['class_counts'], d['episodes_by_red']
            out.append(f"| {SET_LABEL[s]} | {COND_LABEL[c]} | {d['n_episodes']:,} ({er['bline']:,} / {er['meander']:,}) | "
                       f"{d['n_host_steps']:,} | {cc['clean']:,} | {cc['user']:,} | {cc['root']:,} | {d['n_silent']:,} | "
                       f"{d['n_never']:,} |")
    out += ['', '## Metrics', '', '| Set | Model | Condition | ' + ' | '.join(lab for _, lab in CRITERIA.values()) + ' |',
            '|---|---|---|' + '---|' * len(CRITERIA)]
    for s in res['sets']:
        for name, lab in MODELS.items():
            for c, m in res['models'].get(name, {}).get(s, {}).items():
                out.append(f"| {SET_LABEL[s]} | {lab} | {COND_LABEL[c]} | " + ' | '.join(_ms(m[k]) for k in CRITERIA) + ' |')
    out += ['', '## Condition differences (same model)', '', '| Set | Model | Difference | Criterion | Δ [95% CI] | p (Holm) |',
            '|---|---|---|---|---|---|']
    for r in res['rows']:
        if r['kind'] in KIND_LABEL:
            out.append(f"| {SET_LABEL[r['set']]} | {MODELS[r['model']]} | {KIND_LABEL[r['kind']]} | "
                       f"{CRITERIA[r['criterion']][1]} | {_ci(r)} | {_p(r['p_holm'])} |")
    for kind, title in (('pair', 'Model pairs per condition (a − b, paired)'),
                        ('pair_prior', 'Topology-biased model with updated bias vs. the other model on the rewired '
                                       'simulator (a_bias updated − b, paired)'),
                        ('pair_drop', 'Difference of rewiring effects: (a_rewired − a_original) − (b_rewired − b_original)')):
        out += ['', f'## {title}', '', '| Set | a | b | Condition | Criterion | Δ [95% CI] | p (Holm) |',
                '|---|---|---|---|---|---|---|']
        for r in res['rows']:
            if r['kind'] == kind:
                out.append(f"| {SET_LABEL[r['set']]} | {MODELS[r['a']]} | {MODELS[r['b']]} | {COND_LABEL[r['condition']]} | "
                           f"{CRITERIA[r['criterion']][1]} | {_ci(r)} | {_p(r['p_holm'])} |")
    out += ['', '## By attacker (mean ± sd over seeds)', '',
            '| Set | Model | Condition | Attacker | ' + ' | '.join(lab for _, lab in CRITERIA.values()) + ' | root→clean |',
            '|---|---|---|---|' + '---|' * (len(CRITERIA) + 1)]
    for s in res['sets']:
        for name, lab in MODELS.items():
            for c, per in res['by_red'].get(name, {}).get(s, {}).items():
                for rn, m in per.items():
                    out.append(f"| {SET_LABEL[s]} | {lab} | {COND_LABEL[c]} | {RED_LABEL[rn]} | "
                               + ' | '.join(_ms(m[k]) for k in CRITERIA) + f" | {_ms(m['errors']['root_to_clean'])} |")
    out += ['', '## Error types (rate within the true class; mean ± sd over seeds)', '',
            '| Set | Model | Condition | ' + ' | '.join(ERROR_LABEL.values()) + ' |', '|---|---|---|' + '---|' * len(ERRORS)]
    for s in res['sets']:
        for name, lab in MODELS.items():
            for c, m in res['models'].get(name, {}).get(s, {}).items():
                out.append(f"| {SET_LABEL[s]} | {lab} | {COND_LABEL[c]} | "
                           + ' | '.join(_ms(m['errors'][e]) for e in ERRORS) + ' |')
    out += ['', '## Silent recall by host slot group (mean ± sd over seeds)', '',
            '| Set | Model | Condition | ' + ' | '.join(GROUPS) + ' |', '|---|---|---|' + '---|' * len(GROUPS)]
    for s, ds in res['sets'].items():
        for c, d in ds.items():
            if c == 'rewired_prior':
                continue
            out.append(f"| {SET_LABEL[s]} | silent host-steps | {COND_LABEL[c]} | "
                       + ' | '.join(f"{d['silent_by_group'][g]:,}" for g in GROUPS) + ' |')
        for name, lab in MODELS.items():
            for c, m in res['models'].get(name, {}).get(s, {}).items():
                out.append(f"| {SET_LABEL[s]} | {lab} | {COND_LABEL[c]} | " + ' | '.join(_ms(m['groups'][g]) for g in GROUPS) + ' |')
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
        collect(args.device, tuple(args.models), tuple(args.seeds))
    else:
        analyze(args.n_boot, tuple(args.models), tuple(args.seeds))


if __name__ == '__main__':
    main()

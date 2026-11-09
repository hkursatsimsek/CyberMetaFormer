"""Host permütasyonu OOD'si (revizyon madde 6 (a)): host sırası ve kimliği değişince modellerin tahmini.

Yeniden eğitim yok: v1'de eğitilmiş `model.pt` (asama3, s0–s4). Kümeler: v1 test (eğitim politikaları) ve ood_test
(ε = 0.3).

Permütasyonlar (π[i]: slot i'ye verisi gelen host; def ve user0 sabit):
    P1  topolojide eşdeğer hostlar içinde: user1↔user2, user3↔user4, ophost0←ophost1←ophost2←ophost0. Saldırı grafı
        kenarları ve alt ağlar korunur.
    P2  sınıflar arası: user1↔user3, user2↔user4. Alt ağlar korunur, kenar hizası bozulur: slot user1/2 artık ent1'e
        bağlı hostların verisini taşır (yeniden bağlanmış topolojinin veri içindeki karşılığı; gerçek simülasyonu
        ood_topoloji.py).
    P1s, P2s  aynı permütasyonlar, ama decoy sayısı sütunları slotta kalır (yalnız 4 bit, tarama geçmişi ve etiketler
        taşınır).
    P3  yalnız model tarafında: host gömmesi E_host'un satırları π2 ile karıştırılır, veri değişmez. Yalnız host-token
        modelleri.
P1 ve P2'de gözlem sütunları (4 bit, tarama geçmişi, decoy sayısı) ve etiketler birlikte permüte edilir. Decoy
kapasitesi hosta bağlıdır (başlangıçta user1/2: 4, user3: 2, user4: 1), yani verisiyle birlikte taşınır. Böylece
permüte slotta eğitimde görülmemiş bir değer oluşabilir (P1'de user3↔user4, P2'de bütün user'lar). P1s ve P2s bu
etkiyi ayırır: decoy kapasitesi slotun özelliği olarak kalır, gerçek yeniden bağlamada (ood_topoloji) olduğu gibi.
Decoy sayısının değiştiği politikalarda (restore_decoys, random, ε'lu politikaların rastgele aksiyonları) decoy geçmişi
slotta, etkinlik ise taşınan hostta kalır. Bu küçük tutarsızlık tanılama amacıyla kabul edilir.

Eşdeğerlikler (testte doğrulanır; P1 ve P2 için): host-token modellerinde veri permütasyonu, E_host'un ve ilişki matrisinin (topoloji
sapması) aynı permütasyonuna denktir. P1 ilişkileri korur, yani P1 ≡ E_host'un sınıf içinde karıştırılması. Sapmasız
(attn) modellerde P2 ≡ P3. Topoloji sapmalı modellerde P2 − P3 farkı, yanlış hizalanmış topoloji önselinin etkisidir.

Modeller: CMF (conv+SSD, topoloji sapması), CMF host_tam (tam dikkat), gru_host_topo, gru_host_attn ve GRU tam. GRU tam
düz 78 boyutlu girdiyle çalışır, permüte veriyi aynen görür; P3 onda tanımsızdır. Ölçütler argmax üzerinden
hesaplanır; sıcaklık argmax'ı değiştirmez. Ana ölçütler: host makro-F1, sessiz recall, N_geçmiş recall. Ek olarak hata
türleri (root→clean, root→user, user→clean, temiz yanlış alarm) ve host grubu başına sessiz recall (slot düzeyinde).
Dilimler her koşulun kendi gözleminden hesaplanır. Permütasyon yalnız slotları değiştirdiği için bunlar aynı
host-adımlarıdır (sayılar denetlenir).

Eşli bölüm bootstrap'ı: (i) her model için Pk − özdeş; (ii) her koşulda model çiftleri (PAIRS); (iii) çiftlerin
permütasyon düşüşü farkı, yani (a_Pk − a_id) − (b_Pk − b_id). Holm bütün satırlar üzerinden.

    python -m cybermetaformer.hidden_state.ood_permutasyon collect    # asama3/ood_permutasyon/preds/ (git dışı)
    python -m cybermetaformer.hidden_state.ood_permutasyon analyze    # makale/ood_permutasyon/{ozet.json, tablolar.md}
"""
import argparse
import contextlib
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
from cybermetaformer.hidden_state.duzeltilmis_gozlem import _stat as member_stat, _value as member_value
from cybermetaformer.hidden_state.models.cybermetaformer import HOSTS

RESULTS = os.path.join(D.ROOT, 'results', 'hidden_state')
PRED_DIR = os.path.join(RESULTS, 'asama3', 'ood_permutasyon', 'preds')
OUT_DIR = os.path.join(RESULTS, 'makale', 'ood_permutasyon')
MODELS = {'cybermetaformer': 'CyberMetaFormer',
          'cybermetaformer-host_tam': 'CMF, full attention (no topology bias)',
          'cybermetaformer-gru_host_topo': 'Host-token GRU, topology-biased attention',
          'cybermetaformer-gru_host_attn': 'Host-token GRU, attention without bias',
          'gru_full': 'GRU (full history)'}
HOST_TOKEN = tuple(m for m in MODELS if m != 'gru_full')
PAIRS = [('cybermetaformer', 'gru_full'), ('cybermetaformer', 'cybermetaformer-host_tam'),
         ('cybermetaformer-gru_host_topo', 'cybermetaformer-gru_host_attn')]
SEEDS = (0, 1, 2, 3, 4)
SETS = {'test': 'test', 'ood': 'ood_test'}
SET_LABEL = {'test': 'Test', 'ood': 'OOD (ε = 0.3)'}
CRITERIA = {'host_macro_f1': ('conf', 'Host macro-F1'), 'silent_recall': ('silent', 'Silent recall'),
            'never_recall': ('never', 'N_hist recall')}
# hata türü → (gerçek sınıf, tahmin edilen sınıflar); oran gerçek sınıfın host-adımları içinde
ERRORS = {'root_to_clean': (2, (0,)), 'root_to_user': (2, (1,)), 'user_to_clean': (1, (0,)),
          'clean_false_alarm': (0, (1, 2))}
ERROR_LABEL = {'root_to_clean': 'root→clean', 'root_to_user': 'root→user', 'user_to_clean': 'user→clean',
               'clean_false_alarm': 'clean false alarm'}
LABEL_NAMES = [HOSTS[h] for h in D.LABEL_HOSTS]
GROUPS = {'user1–2': ('user1', 'user2'), 'user3–4': ('user3', 'user4'), 'ent0': ('ent0',), 'ent1': ('ent1',),
          'ent2': ('ent2',), 'op': ('ophost0', 'ophost1', 'ophost2', 'opserv')}


def _perm(*cycles):
    """Döngülerden host permütasyonu (13 host, HOSTS sırası): döngüdeki her slot bir sonraki hostun verisini alır."""
    pi = list(range(len(HOSTS)))
    for cyc in cycles:
        idx = [HOSTS.index(h) for h in cyc]
        for a, b in zip(idx, idx[1:] + idx[:1]):
            pi[a] = b
    return np.array(pi)


PERMS = {'P1': _perm(('user1', 'user2'), ('user3', 'user4'), ('ophost0', 'ophost1', 'ophost2')),
         'P2': _perm(('user1', 'user3'), ('user2', 'user4'))}
P3_PERM = PERMS['P2']
# veri koşulu → (permütasyon, decoy sütunları da taşınır mı)
DATA_CONDITIONS = {'P1': (PERMS['P1'], True), 'P1s': (PERMS['P1'], False),
                   'P2': (PERMS['P2'], True), 'P2s': (PERMS['P2'], False)}
CONDITIONS = ('id', 'P1', 'P1s', 'P2', 'P2s', 'P3')
COND_LABEL = {'id': 'identity', 'P1': 'P1 (within class)', 'P1s': 'P1, decoy count stays in slot',
              'P2': 'P2 (across classes)', 'P2s': 'P2, decoy count stays in slot', 'P3': 'P3 (E_host only)'}


def obs_columns(pi, move_decoys=True):
    """Permüte gözlem = obs[..., obs_columns(π)]: slot i'nin 4 biti, tarama geçmişi ve (move_decoys ise) decoy sayısı
    host π(i)'den; move_decoys=False ise decoy sayısı slotta kalır."""
    pi, n = np.asarray(pi), len(pi)
    bits = (4 * pi[:, None] + np.arange(4)).reshape(-1)
    return np.concatenate([bits, 4 * n + pi, 5 * n + (pi if move_decoys else np.arange(n))])


def label_columns(pi):
    """Permüte etiket = labels[..., label_columns(π)] (12 etiket hostu; user0 sabit olmalı)."""
    lh = list(D.LABEL_HOSTS)
    assert all(int(pi[h]) in lh for h in lh), 'permütasyon user0 ile etiket hostlarını karıştırmamalı'
    return np.array([lh.index(int(pi[h])) for h in lh])


def permute(obs, labels, pi, move_decoys=True):
    return obs[..., obs_columns(pi, move_decoys)], labels[..., label_columns(pi)]


class ArrayStore:
    """WindowData'ya dizilerden store: obs (E, T+1, 78), labels (E, T, 12), prev_action (E, T+1)."""

    def __init__(self, obs, labels, prev_action):
        self.obs, self.labels, self.prev_action = obs, labels, prev_action


@contextlib.contextmanager
def shuffled_host_emb(model, pi):
    """P3: slot i, host π(i)'nin gömmesini alır; çıkışta ağırlık geri konur."""
    saved = model.host_emb.data.clone()
    model.host_emb.data = saved[torch.as_tensor(np.asarray(pi), device=saved.device)]
    try:
        yield model
    finally:
        model.host_emb.data = saved


def predict(model, cfg, obs, labels, prev_action, device):
    """Argmax tahmini (E, T, 12) int8."""
    wd = D.WindowData(ArrayStore(obs, labels, prev_action), np.arange(len(obs)), device=device)
    pred = E.predict_logits(model, wd, O.K, cfg['scale'], cfg['eval_batch_size']).argmax(-1).astype(np.int8)
    del wd
    return pred


def pred_path(name, seed, pred_dir=PRED_DIR):
    return os.path.join(pred_dir, f'{name}_k{O.K}_s{seed}.npz')


def check_against_record(name, seed, y_test, pred_test, y_ood=None, pred_ood=None):
    """Özdeş koşulun test (ve OOD) F1'i koşunun metrics.json'undakiyle aynı olmalı (aynı ağırlık, aynı tahmin)."""
    rep = json.load(open(os.path.join(O.run_dir(name, seed), 'metrics.json')))['reports']
    out = {}
    for key, y, p in (('test', y_test, pred_test), ('ood_test', y_ood, pred_ood)):
        if y is None or key not in rep:
            continue
        f1 = M.host_macro_f1(y.reshape(-1, 12), p.reshape(-1, 12))
        assert abs(f1 - rep[key]['host_macro_f1']) < 1e-4, f'{name} s{seed} {key}: F1 {f1} ≠ {rep[key]["host_macro_f1"]}'
        out[key] = f1
    return out


# ── toplama ──────────────────────────────────────────────────────────────

def open_sets():
    st = D.EpisodeStore(D.DEFAULT_DATA)
    with st.final_evaluation():
        idx = {s: st.split(sp) for s, sp in SETS.items()}
    return st, idx


def member_keys(name):
    return [f'{s}__{c}' for s in SETS for c in CONDITIONS if c != 'P3' or name in HOST_TOKEN]


def collect(device, models=tuple(MODELS), seeds=SEEDS):
    """Her üye ve kümede bütün koşulların tahminleri. Var olan dosyaya yalnız eksik koşullar eklenir."""
    st, idx = open_sets()
    data = {s: (st.obs[i], st.labels[i], st.prev_action[i]) for s, i in idx.items()}
    os.makedirs(PRED_DIR, exist_ok=True)
    for name in models:
        for seed in seeds:
            path = pred_path(name, seed)
            out = dict(np.load(path)) if os.path.exists(path) else {}
            todo = [k for k in member_keys(name) if k not in out]
            if not todo:
                continue
            model, cfg = O.load_model(name, seed, device)
            assert not getattr(model, 'action_input', False), 'aksiyon girdili modellerde aksiyonlar da permüte edilmeli'
            t0 = time.time()
            for key in todo:
                s, c = key.split('__')
                obs, y, pa = data[s]
                if c == 'id':
                    out[key] = predict(model, cfg, obs, y, pa, device)
                elif c == 'P3':
                    with shuffled_host_emb(model, P3_PERM):
                        out[key] = predict(model, cfg, obs, y, pa, device)
                else:
                    out[key] = predict(model, cfg, *permute(obs, y, *DATA_CONDITIONS[c]), pa, device)
            f1 = check_against_record(name, seed, data['test'][1], out['test__id'], data['ood'][1], out['ood__id'])
            np.savez_compressed(path, **out)
            print(f'{name} s{seed}: {time.time() - t0:.0f} s, {len(todo)} koşul; özdeş F1 = kayıt '
                  + ', '.join(f'{k} {v:.4f}' for k, v in f1.items()), flush=True)
            del model
            if device == 'mps':
                torch.mps.empty_cache()


# ── analiz ───────────────────────────────────────────────────────────────

def records(y, pred, alert, age):
    """Bir üyenin bölüm başına sayımları (eşli bootstrap için) ve host başına sessiz sayımları."""
    cnt, names = M.episode_slice_counts(y, pred, alert, age)
    silent, hit = (y > 0) & ~alert, pred > 0
    return {'conf': M.episode_confusion(y, pred).astype(np.int32),
            'silent': cnt[:, names.index('sessiz')], 'never': cnt[:, names.index('hiç')],
            'silent_host': np.stack([silent.sum((0, 1)), (silent & hit).sum((0, 1))], -1)}


def _msd(x):
    if any(v is None for v in x):
        return None
    return {'mean': float(np.mean(x)), 'sd': float(np.std(x, ddof=1)) if len(x) > 1 else 0.0, 'seeds': list(x)}


def group_recall(silent_host, hosts):
    n = silent_host[[LABEL_NAMES.index(h) for h in hosts]].sum(0)
    return float(n[1] / n[0]) if n[0] else None


def error_rates(C):
    """Havuzlanmış (3, 3) karışıklık → hata türü oranları."""
    return {e: float(C[t, list(ps)].sum() / max(C[t].sum(), 1)) for e, (t, ps) in ERRORS.items()}


def summarize(runs, ep_mask=None):
    """Seed listesi → ölçütler, hata türleri ve grup recall'ları (ortalama ± sd). ep_mask: bölüm alt kümesi."""
    if ep_mask is not None:
        runs = [{k: (v[ep_mask] if k != 'silent_host' else v) for k, v in r.items()} for r in runs]
    out = {c: _msd([member_value(r, key) for r in runs]) for c, (key, _) in CRITERIA.items()}
    confs = [r['conf'].sum((0, 1)) for r in runs]
    out['errors'] = {e: _msd([error_rates(C)[e] for C in confs]) for e in ERRORS}
    if ep_mask is None:
        out['groups'] = {g: _msd([group_recall(r['silent_host'], hs) for r in runs]) for g, hs in GROUPS.items()}
    return out


def set_stats(y, alert, age, red=None, red_names=None):
    silent = (y > 0) & ~alert
    out = {'n_episodes': int(len(y)), 'n_host_steps': int(y.size),
           'class_counts': {c: int((y == i).sum()) for i, c in enumerate(('clean', 'user', 'root'))},
           'n_silent': int(silent.sum()), 'n_never': int((silent & (age == F.NEVER)).sum()),
           'silent_by_group': {g: int(silent[..., [LABEL_NAMES.index(h) for h in hs]].sum()) for g, hs in GROUPS.items()}}
    if red is not None:
        out['episodes_by_red'] = {r: int((red == i).sum()) for i, r in enumerate(red_names)}
    return out


def bootstrap_rows(runs, sizes, n_boot, ka, kb, base=None, side=None, **tags):
    """ka − kb (base verilirse (ka − base[0]) − (kb − base[1])) her ölçüt için; tags satıra yazılır.

    side: koşu anahtarı → bootstrap ağırlık anahtarı (sizes'taki küme). Verilmezse tek küme (eşli). İki ayrı küme
    verilirse kümeler birbirinden bağımsız örneklenir (eşsiz iki örneklem)."""
    rows = []
    side = side or {k: next(iter(sizes)) for k in (ka, kb) + tuple(base or ())}
    for c, (key, _) in CRITERIA.items():
        f = {k: member_stat(runs[k], key, side[k]) for k in side}
        if base is None:
            fa, fb = f[ka], f[kb]
        else:
            fa = lambda W, f=f: f[ka](W) - f[base[0]](W)          # noqa: E731
            fb = lambda W, f=f: f[kb](W) - f[base[1]](W)          # noqa: E731
        rows.append({**tags, 'criterion': c, **M.paired_bootstrap(fa, fb, sizes, n_boot=n_boot, seed=0)})
    return rows


def analyze(n_boot=10_000, models=tuple(MODELS), seeds=SEEDS, log=print):
    st, idx = open_sets()
    preds = {(name, seed): dict(np.load(pred_path(name, seed))) for name in models for seed in seeds}
    res = {'seeds': list(seeds), 'n_boot': n_boot,
           'permutations': {c: {HOSTS[i]: HOSTS[j] for i, j in enumerate(pi) if i != j} for c, pi in PERMS.items()},
           'p3_host_emb_permutation': {HOSTS[i]: HOSTS[j] for i, j in enumerate(P3_PERM) if i != j},
           'sets': {}, 'models': {}, 'rows': []}
    runs = {}
    for s, i in idx.items():
        y0, obs0 = st.labels[i].astype(np.int64), st.obs[i]
        for c in CONDITIONS:
            obs, y = permute(obs0, y0, *DATA_CONDITIONS[c]) if c in DATA_CONDITIONS else (obs0, y0)
            alert = F.episode_alerts(obs)
            age = F.alert_age(alert)
            stats = set_stats(y, alert, age)
            if c == 'id':
                res['sets'][s] = stats
            else:                                   # permütasyon yalnız slotları değiştirir
                for k in ('n_host_steps', 'class_counts', 'n_silent', 'n_never'):
                    assert stats[k] == res['sets'][s][k], f'{s} {c}: {k} değişti'
                res['sets'][s].setdefault('silent_by_group_permuted', {})[c] = stats['silent_by_group']
            for name in models:
                if c == 'P3' and name not in HOST_TOKEN:
                    continue
                runs[s, c, name] = [records(y, preds[name, seed][f'{s}__{c}'], alert, age) for seed in seeds]
                res['models'].setdefault(name, {}).setdefault(s, {})[c] = summarize(runs[s, c, name])
        for name in models:
            m = res['models'][name][s]
            log(f'{name:30s} {s:5s} ' + '  '.join(
                f"{c}: F1 {m[c]['host_macro_f1']['mean']:.4f} sessiz {m[c]['silent_recall']['mean']:.4f} "
                f"N {m[c]['never_recall']['mean']:.4f}" for c in CONDITIONS if c in m))

        sizes = {s: len(i)}
        R = {(c, name): v for (ss, c, name), v in runs.items() if ss == s}
        for name in models:
            for c in CONDITIONS[1:]:
                if (c, name) in R:
                    res['rows'] += bootstrap_rows(R, sizes, n_boot, (c, name), ('id', name),
                                                  kind='perm_minus_id', set=s, model=name, condition=c)
        for a, b in PAIRS:
            if a not in models or b not in models:
                continue
            for c in CONDITIONS:
                if (c, a) in R and (c, b) in R:
                    res['rows'] += bootstrap_rows(R, sizes, n_boot, (c, a), (c, b),
                                                  kind='pair', set=s, a=a, b=b, condition=c)
                    if c != 'id':
                        res['rows'] += bootstrap_rows(R, sizes, n_boot, (c, a), (c, b), base=(('id', a), ('id', b)),
                                                      kind='pair_drop', set=s, a=a, b=b, condition=c)
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
    return f"{d['mean']:.3f} ± {d['sd']:.3f}" if d else '–'


def _ci(r):
    return f"{r['diff']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]"


def _p(p):
    return '< 0.001' if p < 0.001 else f'{p:.3f}'


def _perm_text(d):
    seen, parts = set(), []
    for a, b in d.items():
        if a in seen:
            continue
        if d.get(b) == a:
            parts.append(f'{a}↔{b}')
            seen |= {a, b}
        else:
            parts.append(f'{a}←{b}')
            seen.add(a)
    return ', '.join(parts)


def tables_md(res):
    n = len(res['seeds'])
    out = ['# Host permutation OOD', '',
           f"Models trained on the original host order; evaluated with permuted hosts. P1 = "
           f"{_perm_text(res['permutations']['P1'])} (attack-graph edges and subnets preserved); P2 = "
           f"{_perm_text(res['permutations']['P2'])} (subnets preserved, user→enterprise edges misaligned); "
           f"P3 = only the host-embedding rows permuted as in P2, data unchanged (host-token models). "
           f"Observation columns (4 bits, scan history, decoy count) and labels move together; slot a←b holds host b's "
           f"data. Decoy capacity is host-specific (user1/2: 4, user3: 2, user4: 1 at the start), so in P1 and P2 it "
           f"moves with the host; in P1s and P2s the decoy-count columns stay with the slot. Mean ± sd over {n} seeds (argmax; temperature does not change it). Differences: paired episode "
           f"bootstrap ({res['n_boot']:,} resamples), Holm over all {len(res['rows'])} rows.", '',
           '## Sets', '', '| Set | Episodes | Host-steps | Clean | User | Root | Silent | N_hist |',
           '|---|---|---|---|---|---|---|---|']
    for s, d in res['sets'].items():
        cc = d['class_counts']
        out.append(f"| {SET_LABEL[s]} | {d['n_episodes']:,} | {d['n_host_steps']:,} | {cc['clean']:,} | {cc['user']:,} | "
                   f"{cc['root']:,} | {d['n_silent']:,} | {d['n_never']:,} |")
    out += ['', '## Metrics', '', '| Set | Model | Condition | ' + ' | '.join(lab for _, lab in CRITERIA.values()) + ' |',
            '|---|---|---|' + '---|' * len(CRITERIA)]
    for s in res['sets']:
        for name, lab in MODELS.items():
            for c, m in res['models'].get(name, {}).get(s, {}).items():
                out.append(f"| {SET_LABEL[s]} | {lab} | {COND_LABEL[c]} | " + ' | '.join(_ms(m[k]) for k in CRITERIA) + ' |')
    out += ['', '## Permuted − identity (same model)', '', '| Set | Model | Condition | Criterion | Δ [95% CI] | p (Holm) |',
            '|---|---|---|---|---|---|']
    for r in res['rows']:
        if r['kind'] == 'perm_minus_id':
            out.append(f"| {SET_LABEL[r['set']]} | {MODELS[r['model']]} | {COND_LABEL[r['condition']]} | "
                       f"{CRITERIA[r['criterion']][1]} | {_ci(r)} | {_p(r['p_holm'])} |")
    for kind, title in (('pair', 'Model pairs per condition (a − b)'),
                        ('pair_drop', 'Difference of permutation effects: (a_P − a_id) − (b_P − b_id)')):
        out += ['', f'## {title}', '', '| Set | a | b | Condition | Criterion | Δ [95% CI] | p (Holm) |',
                '|---|---|---|---|---|---|---|']
        for r in res['rows']:
            if r['kind'] == kind:
                out.append(f"| {SET_LABEL[r['set']]} | {MODELS[r['a']]} | {MODELS[r['b']]} | {COND_LABEL[r['condition']]} | "
                           f"{CRITERIA[r['criterion']][1]} | {_ci(r)} | {_p(r['p_holm'])} |")
    out += ['', '## Error types (rate within the true class; mean ± sd over seeds)', '',
            '| Set | Model | Condition | ' + ' | '.join(ERROR_LABEL.values()) + ' |', '|---|---|---|' + '---|' * len(ERRORS)]
    for s in res['sets']:
        for name, lab in MODELS.items():
            for c, m in res['models'].get(name, {}).get(s, {}).items():
                out.append(f"| {SET_LABEL[s]} | {lab} | {COND_LABEL[c]} | "
                           + ' | '.join(_ms(m['errors'][e]) for e in ERRORS) + ' |')
    out += ['', '## Silent recall by host slot group (mean ± sd over seeds)', '',
            '| Set | Model | Condition | ' + ' | '.join(GROUPS) + ' |', '|---|---|---|' + '---|' * len(GROUPS)]
    for s, d in res['sets'].items():
        out.append(f"| {SET_LABEL[s]} | silent host-steps | identity | "
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

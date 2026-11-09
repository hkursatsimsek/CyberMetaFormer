"""Müdahale deneyleri (revizyon madde 7): hedef hostun kestirimi diğer hostların gözlemine ne kadar dayanıyor?

Yeniden eğitim yok: `model.pt` (asama3, s0–s4). Küme: v1 test (2000 bölüm, eğitim politikaları; dağılım içi).

Hedef host h'nin gözlem sütunları (4 bit, tarama geçmişi, decoy sayısı) olduğu gibi kalır. Bir S host kümesinin sütunları
bölümün bütün adımlarında değiştirilir; model nedensel olduğu için t'deki tahmin S'nin ≤ t geçmişini görür. Üç müdahale:
    empty    S'nin her adımı kendi t = 0 gözlemi (aktivite ve güvenlik bitleri 0, tarama yok, decoy başlangıçtaki
             gibi): "o hostlarda hiçbir şey olmadı"
    swap     S'nin sütunları, aynı Red ve Blue politikasından rastgele başka bir bölümün aynı hostlarından alınır.
             Her hostun marjinali korunur, hedefle ortak dağılım bozulur
    shuffle  S'nin sütunları bölüm içinde zamanda karıştırılır (t ≥ 1; S'nin hostları aynı permütasyonla, kendi
             aralarındaki eşzamanlılık korunur, hedefle zaman hizası ve sıralama bozulur)
Host kümeleri (eğitim topolojisinin ilişki matrisinden):
    related  h'nin saldırı grafı komşuları ve aynı alt ağdaki hostlar
    random   h ile ilişkisiz hostlardan (ilişki türü "diğer") eşit büyüklükte RANDOM_DRAWS rastgele küme
Hedefler: TARGETS (ent0–2, opserv, user1–4). def hiç ihlal edilmez; ophost'ların sessiz ihlali çok az olduğu için
hedef değildir (ama küme üyesi olabilir). Müdahale rastlantısı modelden bağımsızdır ve sabit seed'lidir; bütün modeller
ve seed'ler aynı müdahale edilmiş girdileri görür.

Ölçüler (hedef host-adımlarında; ilk satır birincil):
    Δ sessiz recall ve Δ N_geçmiş recall (müdahale − özgün; sessiz/N_geçmiş hedefin kendi gözleminden, değişmez)
    karar değişimi oranı (argmax), ΔP(ihlal) (üyenin val sıcaklığıyla, metrics.json)
    kayma kontrolü: (a) hedefin kendi uyarısı olan ihlal adımlarında Δ recall ve temiz adımlarda Δ yanlış alarm
    (hedefin kanıtı yerindeyken değişim, müdahalenin yarattığı yapay girdi kaymasına işaret eder); (b) müdahale edilmiş
    gözlemlerin eğitim kümesinde hiç görülmemiş tam gözlem (78 boyut) oranı ve host başına görülmemiş ardışık geçiş
    oranı; (c) S'de bölüm başına uyarı sayısı, özgün ve müdahale sonrası (girdinin ne kadar değiştiği)
Belirsizlik: seed sd ve eşli bölüm bootstrap'ı (seed ortalaması; random kümeler çekilişler üzerinden havuzlanır):
    (i) her model, müdahale ve küme türü için Δ (müdahale − özgün); (ii) related − random; (iii) çift farkı
    (CMF'nin Δ'sı − ötekinin Δ'sı). Holm bütün satırlar üzerinden. Sonuçlar karşı-olgusal kanıt gibi yorumlanmaz:
    müdahaleler gözlem uzayında yapılır, simülatörde değil.

    python -m cybermetaformer.hidden_state.mudahale collect    # asama3/mudahale/preds/ (git dışı), shift.json; MPS
    python -m cybermetaformer.hidden_state.mudahale analyze    # makale/mudahale/{ozet.json, tablolar.md, fig_mudahale.*}
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
from cybermetaformer.hidden_state.duzeltilmis_gozlem import _stat as member_stat
from cybermetaformer.hidden_state.models.cybermetaformer import HOSTS, REL_EDGE, REL_OTHER, REL_SUBNET, host_relations
from cybermetaformer.hidden_state.ood_permutasyon import ArrayStore, _ci, _ms, _msd, _p

RESULTS = os.path.join(D.ROOT, 'results', 'hidden_state')
STAGE = os.path.join(RESULTS, 'asama3', 'mudahale')
PRED_DIR = os.path.join(STAGE, 'preds')
OUT_DIR = os.path.join(RESULTS, 'makale', 'mudahale')
MODELS = {'cybermetaformer': 'CyberMetaFormer', 'cybermetaformer-host_tam': 'CMF, full attention (no topology bias)',
          'gru_full': 'GRU (full history)'}
PAIRS = [('cybermetaformer', 'gru_full'), ('cybermetaformer', 'cybermetaformer-host_tam')]
SEEDS = (0, 1, 2, 3, 4)
TARGETS = ('ent0', 'ent1', 'ent2', 'opserv', 'user1', 'user2', 'user3', 'user4')
INTERVENTIONS = ('empty', 'swap', 'shuffle')
IV_LABEL = {'empty': 'empty (t = 0 observation)', 'swap': 'swap (other episode, same policies)',
            'shuffle': 'shuffle (in time)'}
RANDOM_DRAWS = 3
SET_KINDS = ('related', 'random')
CRITERIA = {'silent': 'Silent recall', 'never': 'N_hist recall', 'alerted': 'Recall on alerted breaches',
            'clean_fa': 'False-alarm rate on clean steps'}
PRIMARY = ('silent', 'never')


def host_columns(h):
    n = D.NUM_HOSTS
    return np.array([4 * h, 4 * h + 1, 4 * h + 2, 4 * h + 3, 4 * n + h, 5 * n + h])


def host_sets(seed=0):
    """{hedef: {'related': [...], 'random': [[...] × RANDOM_DRAWS]}} host indeksleriyle (eğitim topolojisi)."""
    rel = host_relations().numpy()
    rng = np.random.default_rng(seed)
    out = {}
    for name in TARGETS:
        h = HOSTS.index(name)
        related = [j for j in range(D.NUM_HOSTS) if j != h and rel[h, j] in (REL_EDGE, REL_SUBNET)]
        pool = [j for j in range(D.NUM_HOSTS) if rel[h, j] == REL_OTHER]
        assert len(pool) >= len(related), name
        out[name] = {'related': related,
                     'random': [sorted(rng.choice(pool, len(related), replace=False).tolist())
                                for _ in range(RANDOM_DRAWS)]}
    return out


def set_keys(sets):
    """(hedef, küme anahtarı, host listesi): related ve random0..k."""
    for t, d in sets.items():
        yield t, 'related', d['related']
        for i, s in enumerate(d['random']):
            yield t, f'random{i}', s


def intervene(obs, hosts, kind, red, blue, rng):
    """obs (E, T+1, 78) → kopya; yalnız `hosts`'un sütunları değişir."""
    cols = np.concatenate([host_columns(h) for h in hosts])
    out = obs.copy()
    E, S = obs.shape[:2]
    if kind == 'empty':
        out[:, :, cols] = obs[:, :1, cols]
    elif kind == 'swap':
        partner = np.empty(E, np.int64)
        for key in np.unique(red * 100 + blue):                      # aynı Red ve Blue politikası içinde türetme
            idx = np.flatnonzero(red * 100 + blue == key)
            perm = rng.permutation(len(idx))
            while len(idx) > 1 and (perm == np.arange(len(idx))).any():
                perm = rng.permutation(len(idx))
            partner[idx] = idx[perm]
        out[:, :, cols] = obs[partner][:, :, cols]
    elif kind == 'shuffle':
        order = np.argsort(rng.random((E, S - 1)), axis=1) + 1          # t ≥ 1 karışır, t = 0 yerinde
        order = np.concatenate([np.zeros((E, 1), np.int64), order], 1)
        out[:, :, cols] = obs[np.arange(E)[:, None], order][:, :, cols]
    else:
        raise ValueError(kind)
    return out


def iv_rng(target, key, kind):
    """Müdahale rastlantısı: (hedef, küme, müdahale) başına sabit; modelden ve seed'den bağımsız."""
    set_id = 0 if key == 'related' else 1 + int(key[len('random'):])
    return np.random.default_rng([HOSTS.index(target), set_id, INTERVENTIONS.index(kind), 7])


def open_set():
    st = D.EpisodeStore(D.DEFAULT_DATA)
    return st, st.split('test')


def pred_path(name, seed):
    return os.path.join(PRED_DIR, f'{name}_k{O.K}_s{seed}.npz')


def temperature(name, seed):
    return float(json.load(open(os.path.join(O.run_dir(name, seed), 'metrics.json')))['reports']['temperature'])


@torch.no_grad()
def predict_target(model, cfg, obs, labels, prev_action, device, temp):
    """(E, T, 12) argmax ve P(ihlal) (float16)."""
    wd = D.WindowData(ArrayStore(obs, labels, prev_action), np.arange(len(obs)), device=device)
    lg = E.predict_logits(model, wd, O.K, cfg['scale'], cfg['eval_batch_size'])
    del wd
    p = torch.softmax(torch.as_tensor(lg) / temp, -1)
    return lg.argmax(-1).astype(np.int8), (1 - p[..., 0]).numpy().astype(np.float16)


# ── toplama ──────────────────────────────────────────────────────────────

def collect(device, models=tuple(MODELS), seeds=SEEDS):
    """Özgün ve her (hedef, küme, müdahale) için hedef hostun tahmini. Var olan dosya atlanır."""
    st, idx = open_set()
    obs, y, pa = st.obs[idx], st.labels[idx], st.prev_action[idx]
    red, blue = st.red_policy[idx], st.blue_policy[idx]
    sets = host_sets()
    lab = {t: D.LABEL_HOSTS.index(HOSTS.index(t)) for t in TARGETS}
    os.makedirs(PRED_DIR, exist_ok=True)
    for name in models:
        for seed in seeds:
            path = pred_path(name, seed)
            if os.path.exists(path):
                continue
            model, cfg = O.load_model(name, seed, device)
            temp, t0 = temperature(name, seed), time.time()
            pred, pb = predict_target(model, cfg, obs, y, pa, device, temp)
            out = {'base_pred': pred, 'base_p': pb}
            f1 = M.host_macro_f1(y.reshape(-1, 12), pred.reshape(-1, 12))
            rep = json.load(open(os.path.join(O.run_dir(name, seed), 'metrics.json')))['reports']['test']['host_macro_f1']
            assert abs(f1 - rep) < 1e-4, f'{name} s{seed}: test F1 {f1} ≠ {rep}'
            for t, key, hosts in set_keys(sets):
                for kind in INTERVENTIONS:
                    o = intervene(obs, hosts, kind, red, blue, iv_rng(t, key, kind))
                    p, q = predict_target(model, cfg, o, y, pa, device, temp)
                    out[f'{t}__{key}__{kind}__pred'] = p[..., lab[t]]
                    out[f'{t}__{key}__{kind}__p'] = q[..., lab[t]]
            np.savez_compressed(path, **out)
            print(f'{name} s{seed}: {time.time() - t0:.0f} s; test F1 {f1:.4f} = kayıt', flush=True)
            del model
            if device == 'mps':
                torch.mps.empty_cache()
    shift_path = os.path.join(STAGE, 'shift.json')
    if not os.path.exists(shift_path):
        with open(shift_path, 'w') as f:
            json.dump(shift_stats(st, idx, sets), f, indent=1)


def _row_hash(x, w, chunk=200_000):
    """(N, 78) int8 → (N,) uint64 özet (tam gözlem kimliği; çakışma olasılığı ihmal edilir). Parça parça."""
    return np.concatenate([(x[i:i + chunk].astype(np.uint64) + np.uint64(3)) @ w for i in range(0, len(x), chunk)])


def shift_stats(st, idx, sets):
    """Müdahale edilmiş girdilerin eğitim kümesine göre yeniliği ve S'deki uyarı sayısı (modelden bağımsız)."""
    rng_w = np.random.default_rng(12345)
    w = rng_w.integers(1, 2 ** 63, D.OBS_DIM, dtype=np.uint64) | np.uint64(1)
    train = st.obs[st.split('train')]
    seen_rows = np.unique(_row_hash(train.reshape(-1, D.OBS_DIM), w))
    tok = lambda o, h: o[..., host_columns(h)].astype(np.int64) @ (8 ** np.arange(6))     # noqa: E731
    seen_trans = {h: np.unique(tok(train[:, :-1], h) * 8 ** 6 + tok(train[:, 1:], h)) for h in range(D.NUM_HOSTS)}
    obs = st.obs[idx]
    red, blue = st.red_policy[idx], st.blue_policy[idx]
    alerts = F.alerts(obs[:, 1:], label_hosts=False)                                     # (E, T, 13)

    def novelty(o, hosts):
        rows = ~np.isin(_row_hash(o.reshape(-1, D.OBS_DIM), w), seen_rows)
        trans = np.mean([~np.isin(tok(o[:, :-1], h) * 8 ** 6 + tok(o[:, 1:], h), seen_trans[h]) for h in hosts])
        return float(rows.mean()), float(trans)

    out = {'original': dict(zip(('novel_rows', 'novel_transitions'), novelty(obs, range(D.NUM_HOSTS))))}
    for t, key, hosts in set_keys(sets):
        for kind in INTERVENTIONS:
            o = intervene(obs, hosts, kind, red, blue, iv_rng(t, key, kind))
            nr, nt = novelty(o, hosts)
            a = F.alerts(o[:, 1:], label_hosts=False)
            out[f'{t}__{key}__{kind}'] = {'novel_rows': nr, 'novel_transitions': nt,
                                         'alerts_per_episode_before': float(alerts[..., hosts].sum((1, 2)).mean()),
                                         'alerts_per_episode_after': float(a[..., hosts].sum((1, 2)).mean())}
    return out


# ── analiz ───────────────────────────────────────────────────────────────

def masks(y, alert, age):
    """Hedef host-adım maskeleri (E, T)."""
    return {'silent': (y > 0) & ~alert, 'never': (y > 0) & ~alert & (age == F.NEVER), 'alerted': (y > 0) & alert,
            'clean_fa': y == 0}


def counts(pred, mk):
    """{ölçüt: (E, 2)} [n, isabet]: recall'larda tahmin > 0, temiz adımda yanlış alarm (tahmin > 0)."""
    hit = pred > 0
    return {c: np.stack([m.sum(1), (m & hit).sum(1)], -1) for c, m in mk.items()}


def analyze(n_boot=100_000, models=tuple(MODELS), seeds=SEEDS, log=print):
    st, idx = open_set()
    y_all = st.labels[idx].astype(np.int64)
    alert_all = F.episode_alerts(st.obs[idx])
    age_all = F.alert_age(alert_all)
    sets = host_sets()
    preds = {(n, s): dict(np.load(pred_path(n, s))) for n in models for s in seeds}
    shift = json.load(open(os.path.join(STAGE, 'shift.json')))
    E_ = len(idx)
    res = {'seeds': list(seeds), 'n_boot': n_boot, 'n_episodes': E_, 'host_sets': {
        t: {'related': [HOSTS[j] for j in d['related']], 'random': [[HOSTS[j] for j in s] for s in d['random']]}
        for t, d in sets.items()}, 'targets': {}, 'models': {}, 'per_target': {}, 'shift': {}, 'rows': []}
    # (model, seed) → {(koşul): {ölçüt: (E, 2)}}; koşul = 'base' ya da (müdahale, küme türü); hedefler havuzlanır
    pooled = {}
    extra = {}                                       # karar değişimi ve ΔP, seed başına
    for t in TARGETS:
        j = D.LABEL_HOSTS.index(HOSTS.index(t))
        y, a, g = y_all[..., j], alert_all[..., j], age_all[..., j]
        mk = masks(y, a, g)
        res['targets'][t] = {c: int(m.sum()) for c, m in mk.items()}
        for n in models:
            for s in seeds:
                P = preds[n, s]
                base = P['base_pred'][..., j]
                pool = pooled.setdefault((n, s), {})
                for c, v in counts(base, mk).items():
                    pool.setdefault('base', {}).setdefault(c, np.zeros((E_, 2), np.int64))
                    pool['base'][c] += v
                for kind in INTERVENTIONS:
                    for sk in SET_KINDS:
                        keys = ['related'] if sk == 'related' else [f'random{i}' for i in range(RANDOM_DRAWS)]
                        for key in keys:
                            p = P[f'{t}__{key}__{kind}__pred']
                            for c, v in counts(p, mk).items():
                                pool.setdefault((kind, sk), {}).setdefault(c, np.zeros((E_, 2), np.int64))
                                pool[kind, sk][c] += v
                            ch = p != base
                            dp = P[f'{t}__{key}__{kind}__p'].astype(np.float32) - P['base_p'][..., j].astype(np.float32)
                            e = extra.setdefault((n, s, kind, sk), {'changed_silent': [0, 0], 'changed_all': [0, 0],
                                                                    'dp_silent': [0.0, 0]})
                            e['changed_silent'][0] += int(mk['silent'].sum())
                            e['changed_silent'][1] += int((ch & mk['silent']).sum())
                            e['changed_all'][0] += ch.size
                            e['changed_all'][1] += int(ch.sum())
                            e['dp_silent'][0] += float(dp[mk['silent']].sum())
                            e['dp_silent'][1] += int(mk['silent'].sum())
                            pt = res['per_target'].setdefault(n, {}).setdefault(t, {}).setdefault(f'{kind}|{sk}', [])
                            m = mk['silent']
                            pt.append(float((p[m] > 0).mean() - (base[m] > 0).mean()) if m.any() else None)
    # random kümelerde sayımlar RANDOM_DRAWS kez toplandı: oranlar havuzlanmış çekilişlerin ortalamasıdır (n de
    # aynı katla büyür), base ile karşılaştırma oran düzeyinde yapılır.

    def runs(n, cond, c):
        return [{'c': pooled[n, s][cond][c]} for s in seeds]

    def value(n, cond, c):
        v = [pooled[n, s][cond][c].sum(0) for s in seeds]
        return [float(x[1] / x[0]) if x[0] else None for x in v]

    for n in models:
        m = res['models'].setdefault(n, {})
        m['base'] = {c: _msd(value(n, 'base', c)) for c in CRITERIA}
        for kind in INTERVENTIONS:
            for sk in SET_KINDS:
                d = {c: _msd([a - b for a, b in zip(value(n, (kind, sk), c), value(n, 'base', c))]) for c in CRITERIA}
                ex = [extra[n, s, kind, sk] for s in seeds]
                d['decision_change_silent'] = _msd([e['changed_silent'][1] / e['changed_silent'][0] for e in ex])
                d['decision_change_all'] = _msd([e['changed_all'][1] / e['changed_all'][0] for e in ex])
                d['dp_silent'] = _msd([e['dp_silent'][0] / e['dp_silent'][1] for e in ex])
                m[f'{kind}|{sk}'] = d
        for t, per in res['per_target'][n].items():
            res['per_target'][n][t] = {k: float(np.mean([x for x in v if x is not None])) for k, v in per.items()}

    sizes = {'s': E_}
    for kind in INTERVENTIONS:
        for c in CRITERIA:
            for n in models:
                for sk in SET_KINDS:                                          # (i) müdahale − özgün
                    r = M.paired_bootstrap(member_stat(runs(n, (kind, sk), c), 'c', 's'),
                                           member_stat(runs(n, 'base', c), 'c', 's'), sizes, n_boot=n_boot, seed=0)
                    res['rows'].append({'kind': 'iv_minus_base', 'intervention': kind, 'set_kind': sk, 'model': n,
                                        'criterion': c, **r})
                if c in PRIMARY:                                              # (ii) related − random
                    r = M.paired_bootstrap(member_stat(runs(n, (kind, 'related'), c), 'c', 's'),
                                           member_stat(runs(n, (kind, 'random'), c), 'c', 's'), sizes, n_boot=n_boot,
                                           seed=0)
                    res['rows'].append({'kind': 'related_minus_random', 'intervention': kind, 'model': n,
                                        'criterion': c, **r})
            if c not in PRIMARY:
                continue
            for a, b in PAIRS:                                                # (iii) Δ_a − Δ_b
                if a not in models or b not in models:
                    continue
                for sk in SET_KINDS:
                    fa = member_stat(runs(a, (kind, sk), c), 'c', 's')
                    ga = member_stat(runs(a, 'base', c), 'c', 's')
                    fb = member_stat(runs(b, (kind, sk), c), 'c', 's')
                    gb = member_stat(runs(b, 'base', c), 'c', 's')
                    r = M.paired_bootstrap(lambda W: fa(W) - ga(W), lambda W: fb(W) - gb(W), sizes, n_boot=n_boot,
                                           seed=0)
                    res['rows'].append({'kind': 'pair_delta', 'intervention': kind, 'set_kind': sk, 'a': a, 'b': b,
                                        'criterion': c, **r})
    for r, p in zip(res['rows'], M.holm([r['p'] for r in res['rows']])):
        r['p_holm'] = p
    for kind in INTERVENTIONS:                                               # kayma kontrolü (b)–(c), hedef ortalaması
        for sk in SET_KINDS:
            keys = ['related'] if sk == 'related' else [f'random{i}' for i in range(RANDOM_DRAWS)]
            vals = [shift[f'{t}__{k}__{kind}'] for t in TARGETS for k in keys]
            res['shift'][f'{kind}|{sk}'] = {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}
    res['shift']['original'] = shift['original']
    for n in models:
        log(f'{n:26s} ' + '  '.join(f"{k}: Δsessiz {res['models'][n][k]['silent']['mean']:+.4f}"
                                    for k in res['models'][n] if k != 'base'))
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, 'ozet.json'), 'w') as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    with open(os.path.join(OUT_DIR, 'tablolar.md'), 'w') as f:
        f.write(tables_md(res))
    figure(res)
    log('→', OUT_DIR)
    return res


# ── çıktılar ─────────────────────────────────────────────────────────────

def _d(d, digits=3):
    return f"{d['mean']:+.{digits}f} ± {d['sd']:.{digits}f}" if d else '–'


def tables_md(res):
    n = len(res['seeds'])
    sets = res['host_sets']
    out = ['# Intervention experiments: does the target-host estimate use other hosts\' observations?', '',
           f"Test set ({res['n_episodes']:,} episodes, training policies). The target host's own observation columns are "
           f"kept; the columns of a host set S are replaced at every step: empty = S's own t = 0 observation; swap = "
           f"S's columns from a random other episode with the same Red and Blue policy; shuffle = S's columns permuted "
           f"in time within the episode (t ≥ 1). S = related hosts (attack-graph neighbours and same subnet in the "
           f"training topology) or {RANDOM_DRAWS} random equal-size sets of unrelated hosts (pooled). Targets: "
           f"{', '.join(TARGETS)}; host-steps pooled over targets. Silent and N_hist are defined by the target's own "
           f"observation and are unchanged by the intervention. Δ = intervened − original; mean ± sd over {n} seeds; "
           f"CIs from a paired episode bootstrap of the seed mean ({res['n_boot']:,} resamples), Holm over all "
           f"{len(res['rows'])} rows. Interventions act on observations, not on the simulator, so the results show how "
           f"much the estimates depend on other hosts' inputs; they are not counterfactual evidence about the "
           f"environment.", '',
           '## Host sets', '', '| Target | Related | Random draws |', '|---|---|---|']
    for t, d in sets.items():
        out.append(f"| {t} | {', '.join(d['related'])} | " + ' / '.join(', '.join(s) for s in d['random']) + ' |')
    out += ['', '## Target host-steps', '', '| Target | ' + ' | '.join(CRITERIA.values()) + ' |',
            '|---|' + '---|' * len(CRITERIA)]
    for t, d in res['targets'].items():
        out.append(f"| {t} | " + ' | '.join(f'{d[c]:,}' for c in CRITERIA) + ' |')
    out += ['', '## Original values', '', '| Model | ' + ' | '.join(CRITERIA.values()) + ' |', '|---|' + '---|' * len(CRITERIA)]
    for nm, lab in MODELS.items():
        if nm in res['models']:
            out.append(f"| {lab} | " + ' | '.join(_ms(res['models'][nm]['base'][c]) for c in CRITERIA) + ' |')
    out += ['', '## Effects (Δ = intervened − original; mean ± sd over seeds)', '',
            '| Intervention | Hosts | Model | Δ silent recall | Δ N_hist recall | Decision change (silent) | '
            'ΔP(breach) (silent) | Δ recall, alerted | Δ false alarm, clean |', '|---|---|---|---|---|---|---|---|---|']
    for kind in INTERVENTIONS:
        for sk in SET_KINDS:
            for nm, lab in MODELS.items():
                if nm not in res['models']:
                    continue
                d = res['models'][nm][f'{kind}|{sk}']
                out.append(f"| {IV_LABEL[kind]} | {sk} | {lab} | {_d(d['silent'])} | {_d(d['never'])} | "
                           f"{_ms(d['decision_change_silent'])} | {_d(d['dp_silent'])} | {_d(d['alerted'])} | "
                           f"{_d(d['clean_fa'])} |")
    title = {'iv_minus_base': 'Intervened − original (paired episode bootstrap)',
             'related_minus_random': 'Related − random hosts (same intervention)',
             'pair_delta': 'Difference of effects between models (Δ_a − Δ_b)'}
    for kind, t in title.items():
        rows = [r for r in res['rows'] if r['kind'] == kind]
        if kind == 'pair_delta':
            out += ['', f'## {t}', '', '| Intervention | Hosts | a | b | Criterion | Δ [95% CI] | p (Holm) |',
                    '|---|---|---|---|---|---|---|']
            out += [f"| {IV_LABEL[r['intervention']]} | {r['set_kind']} | {MODELS[r['a']]} | {MODELS[r['b']]} | "
                    f"{CRITERIA[r['criterion']]} | {_ci(r)} | {_p(r['p_holm'])} |" for r in rows]
        else:
            out += ['', f'## {t}', '', '| Intervention | Hosts | Model | Criterion | Δ [95% CI] | p (Holm) |',
                    '|---|---|---|---|---|---|']
            out += [f"| {IV_LABEL[r['intervention']]} | {r.get('set_kind', 'related − random')} | {MODELS[r['model']]} | "
                    f"{CRITERIA[r['criterion']]} | {_ci(r)} | {_p(r['p_holm'])} |" for r in rows]
    out += ['', '## Input shift control', '',
            'Novel rows = share of intervened full observations (78-dim) never seen in the training split; novel '
            'transitions = share of consecutive per-host token pairs of the intervened hosts never seen in training. '
            'Alerts = alerts on the intervened hosts per episode, before and after.', '',
            '| Intervention | Hosts | Novel rows | Novel transitions | Alerts before | Alerts after |',
            '|---|---|---|---|---|---|']
    o = res['shift']['original']
    out.append(f"| none (original test set) | all | {o['novel_rows']:.3f} | {o['novel_transitions']:.3f} | – | – |")
    for kind in INTERVENTIONS:
        for sk in SET_KINDS:
            d = res['shift'][f'{kind}|{sk}']
            out.append(f"| {IV_LABEL[kind]} | {sk} | {d['novel_rows']:.3f} | {d['novel_transitions']:.3f} | "
                       f"{d['alerts_per_episode_before']:.1f} | {d['alerts_per_episode_after']:.1f} |")
    out += ['', '## Δ silent recall per target (seed mean; random = mean over draws)', '',
            '| Model | Target | ' + ' | '.join(f'{k} {s}' for k in INTERVENTIONS for s in SET_KINDS) + ' |',
            '|---|---|' + '---|' * (len(INTERVENTIONS) * len(SET_KINDS))]
    for nm, lab in MODELS.items():
        for t, d in res['per_target'].get(nm, {}).items():
            out.append(f"| {lab} | {t} | " + ' | '.join(f"{d[f'{k}|{s}']:+.3f}" for k in INTERVENTIONS for s in SET_KINDS)
                       + ' |')
    return '\n'.join(out) + '\n'


def figure(res):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from cybermetaformer.hidden_state.rapor.makale_sekiller import CMF_C, GRU_C, OTHER_C, RC, WIDTH_IN, WHITE, _ygrid
    plt.rcParams.update(RC)
    colors = {'cybermetaformer': CMF_C, 'cybermetaformer-host_tam': OTHER_C, 'gru_full': GRU_C}
    short = {'cybermetaformer': 'CMF', 'cybermetaformer-host_tam': 'CMF, no topology bias', 'gru_full': 'GRU (full)'}
    boot = {(r['intervention'], r['set_kind'], r['model']): r for r in res['rows']
            if r['kind'] == 'iv_minus_base' and r['criterion'] == 'silent'}
    models = [m for m in MODELS if m in res['models']]
    fig, axes = plt.subplots(1, len(INTERVENTIONS), figsize=(WIDTH_IN, 2.4), sharey=True)
    w = 0.8 / len(models)
    for ax, kind in zip(axes, INTERVENTIONS):
        for i, nm in enumerate(models):
            xs = np.arange(len(SET_KINDS)) + (i - (len(models) - 1) / 2) * w
            r = [boot[kind, sk, nm] for sk in SET_KINDS]
            vals = [x['diff'] for x in r]
            err = np.array([[v - x['lo'] for v, x in zip(vals, r)], [x['hi'] - v for v, x in zip(vals, r)]])
            ax.bar(xs, vals, w * 0.9, color=colors[nm], label=short[nm], yerr=err, capsize=2,
                   error_kw={'elinewidth': 0.8, 'capthick': 0.8})
            seeds = [res['models'][nm][f'{kind}|{sk}']['silent']['seeds'] for sk in SET_KINDS]
            for x, sv in zip(xs, seeds):
                ax.scatter([x] * len(sv), sv, s=4, color='#222222', zorder=3, linewidths=0)
        ax.axhline(0, color='#888888', lw=0.8)
        ax.set_xticks(range(len(SET_KINDS)))
        ax.set_xticklabels(['related hosts', 'random hosts'])
        ax.set_title(kind)
        _ygrid(ax)
    axes[0].set_ylabel('Δ silent recall (target host)')
    axes[-1].legend(frameon=False, loc='lower right')
    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ('png', 'svg'):
        fig.savefig(os.path.join(OUT_DIR, f'fig_mudahale.{ext}'), dpi=300, facecolor=WHITE)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['collect', 'analyze'])
    p.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    p.add_argument('--n-boot', type=int, default=100_000)
    p.add_argument('--models', nargs='+', default=list(MODELS))
    p.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    args = p.parse_args()
    if args.cmd == 'collect':
        collect(args.device, tuple(args.models), tuple(args.seeds))
    else:
        analyze(args.n_boot, tuple(args.models), tuple(args.seeds))


if __name__ == '__main__':
    main()

"""Adım 0.6: sessiz ihlal kök neden analizi (yalnızca VM; CybORG gerekir).

"Sessiz ihlal": host ele geçirilmiş (etiket > 0) ama hedef adımın gözleminde uyarı yok (features.alerts, 'export').

Yöntem:
1. `TracedCAGE(FixedCAGE)`: `_process_state` içindeki rastgele çekilişler (Green taraması, %5 tespit kaçağı) ve
   `self.detection` dizisi her adımda kaydedilir. Depo dosyalarına dokunulmaz; çekilişler aynı sırayla yapıldığı
   için RNG akışı değişmez.
2. minicage_v1'in 24 shard'ı aynı seed'lerle yeniden üretilir; gözlem, gerçek durum ve aksiyonlar kayıtlı veriyle
   bayt bayt karşılaştırılır (izlemenin veriyi değiştirmediğinin ve etiketlerin veri setine ait olduğunun kanıtı).
3. Her (bölüm, host) için "güvenlik bayrağının neden görünmediği" adım adım izlenir ve her sessiz host-adıma bir neden
   atanır (CAUSES).
4. Kod okumasından çıkan iddialar kontrollü deneylerle (probe) ayrıca sınanır.

    venv_cyborg/bin/python -m cybermetaformer.hidden_state.silent_cause
Çıktılar: minicage/data/minicage_v1_aux/silent_cause.npz (bölüm × adım × 12 host neden kodu, veri setiyle hizalı),
cybermetaformer/results/hidden_state/asama0/silent_cause.json

Başka bir veri sürümü (M3, test-only politikalar; probe'lar koşmaz, minicage_v1_aux'a dokunulmaz):
    venv_ts/bin/python -m cybermetaformer.hidden_state.silent_cause --data minicage/data/minicage_v1_shift
Çıktılar: minicage/data/<sürüm>_aux/silent_cause.npz, cybermetaformer/results/hidden_state/asama3/silent_cause_<sürüm>.json
"""
import argparse
import json
import os
import time

import numpy as np

from CybORG_plus_plus.mini_CAGE.minimal import EXPLOITS
from minicage.dataset import loaders as L
from minicage.dataset.env import FixedCAGE
from minicage.dataset.policies import make_red, make_blue
from cybermetaformer.hidden_state import data as D
from cybermetaformer.hidden_state import features as F

NH = 13
CAUSES = ['yok', 'kacak_5', 'green_maske', 'remove_maske', 'restore', 'bilinmiyor']
NONE, MISS_MASK, MISS_GREEN, REMOVE_MASK, RESTORE, UNKNOWN = range(len(CAUSES))
CAUSE_TEXT = {
    'kacak_5': 'İstismar %5 olasılıkla kaydedilmedi (SSH dışı istismar; minimal.py:1251)',
    'green_maske': 'Aynı adımda Green aynı hostu taradı, istismar görünmedi (minimal.py:1252)',
    'remove_maske': "Blue 'remove' uyguladı; gözlem [1,0] 'kaldırıldı' gösterdi ama host hâlâ ele geçirilmiş (minimal.py:1308-1316)",
    'restore': 'Restore gözlemdeki bayrağı sildi ama host ele geçirilmiş kaldı',
    'bilinmiyor': 'Yukarıdakilerin hiçbiri',
}
# detection[h]'yi en son yazan Red aksiyonu (minimal.py:1272 her Red aksiyonunda (a-4) mod 13 hostuna yazar)
WRITERS = ['hic', 'exploit_tespit', 'exploit_kacak', 'network', 'escalate', 'impact', 'sleep_alias', 'remote_alias']
OUT_DIR = os.path.join(D.ROOT, 'results', 'hidden_state', 'asama0')
M3_DIR = os.path.join(D.ROOT, 'results', 'hidden_state', 'asama3')
AUX_DIR = os.path.join(D.DATA_ROOT, 'minicage_v1_aux')


class TracedCAGE(FixedCAGE):
    """_process_state'teki np.random.choice çekilişlerini ve detection dizisini kaydeder."""

    def _process_state(self, state, logged_decoys, red_action=None, blue_action=None):
        if red_action is None:
            return super()._process_state(state, logged_decoys, red_action, blue_action)
        calls, orig = [], np.random.choice

        def recorder(*args, **kwargs):
            out = orig(*args, **kwargs)
            calls.append((args, kwargs, out))
            return out

        det_before = self.detection.copy()
        np.random.choice = recorder
        try:
            res = super()._process_state(state, logged_decoys, red_action, blue_action)
        finally:
            np.random.choice = orig
        n = state.shape[0]
        green_host = -np.ones(n, np.int64)
        exploit_mask = None
        green_scan = None
        for args, kwargs, out in calls:
            p = kwargs.get('p')
            if p is not None and list(p) == [0.5, 0.5]:
                green_scan = np.asarray(out).astype(bool)
            elif p is None and list(args[0]) == [1, 2, 3, 8, 9, 10, 11, 12]:
                green_host[green_scan] = np.asarray(out).astype(np.int64)
            elif p is not None and list(p) == [0.05, 0.95]:
                exploit_mask = np.asarray(out).astype(bool)
        assert green_scan is not None and exploit_mask is not None, 'beklenen çekilişler bulunamadı'
        self.trace = {'green_host': green_host, 'exploit_mask': exploit_mask,
                      'det_before': det_before, 'det_after': self.detection.copy()}
        return res


def traced_shard(red_name, blue_name, n, steps, remove_bugs, seed, blue_fn=None):
    """generate.run_shard ile aynı döngü + iz. blue_fn(state, red_action, env) verilirse Blue onu kullanır (probe)."""
    np.random.seed(seed)
    env = TracedCAGE(num_envs=n, remove_bugs=remove_bugs)
    red, blue = make_red(red_name), make_blue(blue_name)
    state, _ = env.reset()
    rec = {k: [] for k in ['obs', 'ts', 'act_red', 'act_blue', 'red_success', 'blue_success', 'sel_exploit',
                           'green_host', 'exploit_mask', 'det_before', 'det_after']}
    rec['obs'].append(state['Blue'].astype(np.int8))
    rec['ts'].append(env.state.astype(np.int8))
    for _ in range(steps):
        red_action = red.get_action(state['Red'])
        blue_action = blue.get_action(state['Blue']) if blue_fn is None else blue_fn(state, red_action, env)
        state, _, _, _ = env.step(red_action=red_action, blue_action=blue_action, red_agent=red)
        rec['obs'].append(state['Blue'].astype(np.int8))
        rec['ts'].append(env.state.astype(np.int8))
        rec['act_red'].append(red_action[:, 0].astype(np.int64))
        rec['act_blue'].append(blue_action[:, 0].astype(np.int64))
        rec['red_success'].append(env.red_success[:, 0].copy())
        rec['blue_success'].append(env.blue_success[:, 0].copy())
        rec['sel_exploit'].append(np.asarray(env.selected_exploit).reshape(-1).copy())
        for key in ['green_host', 'exploit_mask', 'det_before', 'det_after']:
            rec[key].append(env.trace[key])
    return {k: np.stack(v, axis=1) for k, v in rec.items()}       # (n, T(+1), …)


def _onehot_host(host, valid):
    m = np.zeros((len(host), NH), bool)
    m[np.arange(len(host))[valid], host[valid]] = True
    return m


def attribute(tr):
    """İzden neden kodları: (n, T, 13) cause, writer; ayrıca yeniden-bayraklama engeli tanıları."""
    obs, ts = tr['obs'].astype(np.int64), tr['ts'].astype(np.int64)
    n, T = tr['act_red'].shape
    comp = L.host_labels(ts) > 0                                     # (n, T+1, 13)
    bits = obs[..., :4 * NH].reshape(n, T + 1, NH, 4)
    alert = (bits[..., 1] == 1) | (bits[..., 3] == 1)
    flag = bits[..., 3] == 1
    fcause = np.zeros((n, NH), np.int64)
    writer = np.zeros((n, NH), np.int64)
    cause = np.zeros((n, T, NH), np.int8)
    writer_at = np.zeros((n, T, NH), np.int8)
    reassert_blocked = np.zeros((n, T, NH), np.int8)    # 0 yok, 1 detection False, 2 aynı hosta Red aksiyonu, 3 açıklanamayan
    ssh = len(EXPLOITS) - 1
    for t in range(T):
        a_r, a_b = tr['act_red'][:, t], tr['act_blue'][:, t]
        rh = (a_r - 4) % NH                                          # minimal.py:1197 (her Red aksiyonu için)
        is_exploit = (a_r > 3 + NH) & (a_r < 2 * NH + 4)
        green = tr['green_host'][:, t] == rh
        detected = (tr['exploit_mask'][:, t] | (tr['sel_exploit'][:, t] == ssh)) & ~green
        ok = tr['red_success'][:, t] == 1
        new_comp = _onehot_host(rh, is_exploit & ok) & ~comp[:, t] & comp[:, t + 1]
        miss_green = new_comp & _onehot_host(rh, green)
        miss_mask = new_comp & _onehot_host(rh, ~green & ~detected)
        seen = new_comp & _onehot_host(rh, detected)

        bh = (a_b - 1) % NH
        remove = _onehot_host(bh, (a_b > 2 * NH) & (a_b < 3 * NH + 1))
        restore = _onehot_host(bh, a_b > 3 * NH)

        fcause[seen] = NONE
        fcause[miss_mask] = MISS_MASK
        fcause[miss_green] = MISS_GREEN
        fcause[remove & comp[:, t + 1]] = REMOVE_MASK
        fcause[restore & comp[:, t + 1]] = RESTORE
        fcause[flag[:, t + 1]] = NONE

        # detection[rh] yazarı (minimal.py:1272)
        w = np.select(
            [is_exploit & ok & detected, is_exploit, a_r == 0, (a_r >= 1) & (a_r <= 3),
             (a_r >= 4) & (a_r < 4 + NH), (a_r >= 4 + 2 * NH) & (a_r < 4 + 3 * NH), a_r >= 4 + 3 * NH],
            [1, 2, 6, 7, 3, 4, 5], 0)
        writer[np.arange(n), rh] = w

        silent = comp[:, t + 1] & ~alert[:, t + 1]
        c = np.where(silent, np.where(fcause == NONE, UNKNOWN, fcause), NONE)
        cause[:, t] = c
        writer_at[:, t] = writer
        # yeniden bayraklama (minimal.py:1200-1227) neden olmadı? Bu adımın başındaki detection'a bakılır.
        same = _onehot_host(rh, ~is_exploit)
        blocked = np.where(~tr['det_before'][:, t], 1, np.where(same, 2, 3))
        reassert_blocked[:, t] = np.where(silent & np.isin(c, [REMOVE_MASK, RESTORE]), blocked, 0)
    lh = L.LABEL_HOSTS
    return cause[..., lh], writer_at[..., lh], reassert_blocked[..., lh]


# ── kontrollü deneyler (probe) ──────────────────────────────────────────

def probe_restore_vs_exploit(n=2000, steps=40, seed=123):
    """Blue, Red'in aynı adımda istismar ettiği hostu restore ederse: kim kazanır?"""
    def blue_fn(state, red_action, env):
        a = red_action[:, 0]
        is_exp = (a > 3 + NH) & (a < 2 * NH + 4)
        out = np.zeros((len(a), 1), int)
        h = (a - 4) % NH
        out[is_exp & (h != 8), 0] = 40 + h[is_exp & (h != 8)]
        return out
    tr = traced_shard('bline', 'sleep', n, steps, True, seed, blue_fn)
    a_r, a_b = tr['act_red'], tr['act_blue']
    is_exp = (a_r > 3 + NH) & (a_r < 2 * NH + 4) & (a_b > 3 * NH)
    h = (a_r - 4) % NH
    lab = L.host_labels(tr['ts'].astype(np.int64))
    before = np.take_along_axis(lab[:, :-1], h[..., None], 2)[..., 0]
    after = np.take_along_axis(lab[:, 1:], h[..., None], 2)[..., 0]
    m = is_exp & (before == 0) & (tr['red_success'] == 1)
    return {'deneme': int(m.sum()), 'restore_sonrasi_ele_gecirilmis': int((after[m] > 0).sum()),
            'yorum': 'restore, Red güncellemesinden SONRA uygulanır (update_blue updated_state üzerinde çalışır)'}


def probe_remove_on_root(n=2000, steps=60, seed=321):
    """Root'lu hosta remove: gerçek durum değişmez, gözlem yine de [1,0] 'kaldırıldı' gösterir mi?"""
    def blue_fn(state, red_action, env):
        lab = L.host_labels(env.state)                    # kâhin Blue: gerçek durumu görür
        root = lab == 2
        root[:, 8] = False
        out = np.zeros((len(lab), 1), int)
        has = root.any(1)
        out[has, 0] = 27 + root[has].argmax(1)
        return out
    tr = traced_shard('bline', 'sleep', n, steps, True, seed, blue_fn)
    a_b = tr['act_blue']
    is_rm = (a_b > 2 * NH) & (a_b < 3 * NH + 1)
    h = (a_b - 1) % NH
    lab = L.host_labels(tr['ts'].astype(np.int64))
    root_before = np.take_along_axis(lab[:, :-1], h[..., None], 2)[..., 0] == 2
    root_after = np.take_along_axis(lab[:, 1:], h[..., None], 2)[..., 0] == 2
    bits = tr['obs'][:, 1:, :4 * NH].reshape(n, steps, NH, 4).astype(np.int64)
    safety = np.take_along_axis(bits[..., 2:], h[..., None, None].repeat(2, -1), 2)[:, :, 0]
    m = is_rm & root_before
    shown_removed = (safety[m] == [1, 0]).all(-1)
    return {'deneme': int(m.sum()), 'root_kaldi': int(root_after[m].sum()),
            'gozlem_[1,0]_gosterdi': int(shown_removed.sum()),
            'blue_success_-1_iken': int((tr['blue_success'][m] == -1).sum()),
            'yorum': "remove root'a etkisiz (update_blue:741) ama _process_state remove'u "
                     "logical_and(blue_success, remove) ile süzer; -1 doğru (truthy) sayıldığı için bayrak her durumda [1,0]"}


def probe_detection_overwrite(n=2000, steps=100, seed=77):
    """Tespit edilen istismardan sonra aynı hosta (veya sleep/remote takma adına) gelen Red aksiyonu detection'ı siler mi?"""
    out = {}
    for red_name in ['bline', 'meander']:
        tr = traced_shard(red_name, 'sleep', n, steps, True, seed)
        a_r = tr['act_red']
        rh = (a_r - 4) % NH
        db = np.take_along_axis(tr['det_before'], rh[..., None], 2)[..., 0]
        da = np.take_along_axis(tr['det_after'], rh[..., None], 2)[..., 0]
        is_exploit = (a_r > 3 + NH) & (a_r < 2 * NH + 4)
        esc = (a_r >= 4 + 2 * NH) & (a_r < 4 + 3 * NH)
        for name, m in [('escalate', esc), ('sleep (→ user1)', a_r == 0),
                        ('remote_sub* (→ user2-4)', (a_r >= 1) & (a_r <= 3)),
                        ('network/impact', ~is_exploit & ~esc & (a_r >= 4))]:
            mm = m & db
            d = out.setdefault(name, {'detection_True_iken': 0, 'False_yapildi': 0})
            d['detection_True_iken'] += int(mm.sum())
            d['False_yapildi'] += int((mm & ~da).sum())
    return out


# ── ana akış ────────────────────────────────────────────────────────────

def _share(cause, mask=None):
    c = cause if mask is None else cause[mask]
    silent = c[c != NONE]
    tot = max(len(silent), 1)
    return {name: float((silent == i).sum() / tot) for i, name in enumerate(CAUSES) if i != NONE} | {'n_sessiz': int(len(silent))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=D.DEFAULT_DATA, help='veri sürümü klasörü (varsayılan minicage_v1)')
    args = p.parse_args()
    data_dir = os.path.abspath(args.data)
    default = data_dir == os.path.abspath(D.DEFAULT_DATA)
    name = os.path.basename(data_dir)
    aux_dir = AUX_DIR if default else data_dir + '_aux'
    out_dir, out_name = (OUT_DIR, 'silent_cause.json') if default else (M3_DIR, f'silent_cause_{name}.json')
    meta = L.load_meta(data_dir)
    raw, _ = L.load_raw(data_dir, fields=['obs_blue', 'true_state', 'act_red', 'act_blue'])
    causes, writers, blocked, mismatch = [], [], [], []
    t0, start = time.time(), 0
    for sh in meta['shards']:
        tr = traced_shard(sh['red'], sh['blue'], sh['episodes'], meta['steps_per_episode'], meta['remove_bugs'], sh['seed'])
        sl = slice(start, start + sh['episodes'])
        same = (np.array_equal(tr['obs'], raw['obs_blue'][sl]) and np.array_equal(tr['ts'], raw['true_state'][sl])
                and np.array_equal(tr['act_red'], raw['act_red'][sl]) and np.array_equal(tr['act_blue'], raw['act_blue'][sl]))
        if not same:
            mismatch.append(sh['file'])
        c, w, b = attribute(tr)
        causes.append(c), writers.append(w), blocked.append(b)
        print(f"{sh['file']} {sh['red']:8s} x {sh['blue']:22s} veriyle aynı: {same}  ({time.time() - t0:.0f} s)", flush=True)
        start += sh['episodes']
    cause, writer, blocked = np.concatenate(causes), np.concatenate(writers), np.concatenate(blocked)
    assert not mismatch, f'izlenen üretim kayıtlı veriden farklı: {mismatch}'

    # sessiz tanımı features.py ile aynı mı? (bağımsız kontrol)
    labels = L.host_labels(raw['true_state'][:, 1:].astype(np.int64))[..., L.LABEL_HOSTS]
    silent_ref = (labels > 0) & ~F.episode_alerts(raw['obs_blue'])
    assert np.array_equal(silent_ref, cause != NONE), 'neden etiketleri sessiz tanımıyla örtüşmüyor'

    os.makedirs(aux_dir, exist_ok=True)
    np.savez_compressed(os.path.join(aux_dir, 'silent_cause.npz'), cause=cause, writer=writer,
                        reassert_blocked=blocked, causes=np.array(CAUSES), writers=np.array(WRITERS))

    bp = np.concatenate([np.full(s['episodes'], meta['blue_policies'].index(s['blue'])) for s in meta['shards']])
    rp = np.concatenate([np.full(s['episodes'], meta['red_policies'].index(s['red'])) for s in meta['shards']])
    blues = [b for b in meta['blue_policies'] if b in {s['blue'] for s in meta['shards']}]   # veride olanlar
    res = {'veri': name, 'veriyle_ayni': True, 'causes': CAUSES, 'cause_text': CAUSE_TEXT, 'genel': _share(cause)}
    if default:                                                     # diğer sürümlerin bölmeleri anlamsız (yalnızca test)
        sp = L.load_splits(data_dir)
        res['split'] = {s: _share(cause[sp[s]]) for s in ['train', 'val', 'test', 'ood_test']}
    # sessiz oranı: ihlalli host-adımlarının sessiz payı (neden paylarının paydası n_sessiz)
    comp_n = (labels > 0)
    res['sessiz_orani'] = {b: float((cause[bp == meta['blue_policies'].index(b)] != NONE).sum()
                                    / max(comp_n[bp == meta['blue_policies'].index(b)].sum(), 1)) for b in blues}
    res['blue'] = {b: _share(cause[bp == meta['blue_policies'].index(b)]) for b in blues}
    res['red'] = {r: _share(cause[rp == i]) for i, r in enumerate(meta['red_policies'])}
    res['red_blue'] = {f'{r}×{b}': _share(cause[(rp == i) & (bp == meta['blue_policies'].index(b))])
                       for i, r in enumerate(meta['red_policies']) for b in blues}
    hosts = [meta['hosts'][h] for h in L.LABEL_HOSTS]
    res['host'] = {h: _share(cause[..., i]) for i, h in enumerate(hosts)}
    rm = cause == REMOVE_MASK
    res['remove_maske_yeniden_bayraklama'] = {
        'detection_false': float((blocked[rm] == 1).mean()) if rm.any() else None,
        'ayni_hosta_red_aksiyonu': float((blocked[rm] == 2).mean()) if rm.any() else None,
        'aciklanamayan': float((blocked[rm] == 3).mean()) if rm.any() else None,
        'detection_yazari': {WRITERS[i]: float((writer[rm] == i).mean()) for i in range(len(WRITERS))} if rm.any() else None}
    # sessiz koşuların süresi ve bölüm sonuna kadar sürme oranı (neden bazında)
    runs = {}
    for ci, cname in enumerate(CAUSES):
        if ci == NONE:
            continue
        m = cause == ci
        starts = m & ~np.concatenate([np.zeros_like(m[:, :1]), m[:, :-1]], axis=1)
        ends_at_T = m[:, -1]
        n_runs = int(starts.sum())
        runs[cname] = {'kosu': n_runs, 'ort_sure': float(m.sum() / max(n_runs, 1)),
                       'bolum_sonuna_kadar': float(ends_at_T.sum() / max(n_runs, 1))}
    res['kosular'] = runs
    if default:
        print('probe: restore × istismar ...', flush=True)
        res['probe'] = {'restore_vs_exploit': probe_restore_vs_exploit(),
                        'remove_on_root': probe_remove_on_root(),
                        'detection_overwrite': probe_detection_overwrite()}
    res['sure_s'] = round(time.time() - t0, 1)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, out_name), 'w') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: res[k] for k in ['genel', 'split', 'sessiz_orani', 'blue', 'probe'] if k in res},
                     indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()

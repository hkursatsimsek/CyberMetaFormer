"""Gözlemden türetilen özellikler: uyarılar ve uyarı yaşı (numpy, CybORG'a bağımlı değil).

Uyarı tanımları (host başına 4 bit: aktivite [tarama, istismar] + güvenlik [g0, g1]):
- 'export': istismar aktivitesi (bit 1) VEYA güvenlik g1 (bit 3). `loaders.export_tabular` ve makaledeki
  "sessiz ihlal" tanımı budur. Ana tanım.
- 'flag': yalnızca bit 3 (güvenlik [0,1] istismar edildi ya da [1,1] root). `baselines.py` uyarı kuralı budur.
"""
import numpy as np

from minicage.dataset import loaders as L

NEVER = -1          # alert_age: bölüm başından beri bu hostta hiç uyarı yok


def host_bits(obs):
    """(..., 78) → (..., 13, 4)."""
    return np.asarray(obs)[..., :4 * L.NUM_HOSTS].reshape(*np.shape(obs)[:-1], L.NUM_HOSTS, 4)


def alerts(obs, kind='export', label_hosts=True):
    """(..., 78) → (..., 12) bool (label_hosts=False ise 13)."""
    b = host_bits(obs)
    if kind == 'export':
        a = (b[..., 1] == 1) | (b[..., 3] == 1)
    elif kind == 'flag':
        a = b[..., 3] == 1
    else:
        raise ValueError(f'Bilinmeyen uyarı tanımı: {kind}')
    return a[..., L.LABEL_HOSTS] if label_hosts else a


def alert_age(alert_seq):
    """(E, T, H) bool → (E, T, H) int32: t'de uyarı varsa 0, yoksa son uyarıdan bu yana geçen adım,
    bölüm başından beri hiç uyarı yoksa NEVER (-1)."""
    E, T, H = alert_seq.shape
    age = np.empty((E, T, H), np.int32)
    last = np.full((E, H), -1, np.int64)
    for t in range(T):
        last = np.where(alert_seq[:, t], t, last)
        age[:, t] = np.where(last < 0, NEVER, t - last)
    return age


def episode_alerts(obs_blue, kind='export'):
    """Bölüm dizisi (E, T+1, 78) → hedef adımlarla hizalı uyarılar (E, T, 12) (obs[1..T])."""
    return alerts(obs_blue[:, 1:], kind=kind)


# ── XGB-TF: pencere içi zamansal özet özellikler ───────────────

TF_EVENTS = ['export', 'flag', 'root', 'removed']      # yaş özelliği çıkarılan olaylar
TF_HOST_FEATS = ['age_export', 'cnt_export', 'age_flag', 'age_root', 'age_removed', 'cnt_scan', 'last_sec']
SEC_CODE = {(0, 0): 0, (0, 1): 1, (1, 1): 2, (1, 0): 3}  # last_sec: 0 yok, 1 istismar, 2 root, 3 remove


def tf_feature_names(hosts):
    """hosts: meta['hosts'] (13 ad). Sıra: önce son gözlemin 78 ham özelliği, sonra host × TF_HOST_FEATS."""
    raw = [f'obs_{i}' for i in range(4 * L.NUM_HOSTS)] + [f'scan_{h}' for h in hosts] + [f'decoy_{h}' for h in hosts]
    return raw + [f'{f}_{h}' for h in hosts for f in TF_HOST_FEATS]


def _last_index(ev):
    """(E, S, H) bool → (E, S, H) int64: s'ye kadar (dahil) olayın son görüldüğü adım; hiç yoksa -10**6."""
    E, S, H = ev.shape
    idx = np.where(ev, np.arange(S)[None, :, None], -10 ** 6)
    return np.maximum.accumulate(idx, axis=1)


def temporal_features(obs, k):
    """Bölüm dizisi (E, T+1, 78) → (E, T, 78 + 13·7) int8; t = 1..T hedef adımları, pencere obs[t-k+1 .. t].

    Host başına (13 host, user0 dahil; gözlemde bilgi taşır):
    - age_<olay>: penceredeki son olaydan bu yana geçen adım (0 = şimdi); pencerede yoksa k
    - cnt_export, cnt_scan: penceredeki uyarı / tarama aktivitesi sayısı
    - last_sec: penceredeki en son boş olmayan güvenlik durumu (SEC_CODE), yoksa 0
    Pencere bölüm başında kısalır (dolgu = olay yok), WindowData ile aynı hizalama.
    """
    obs = np.asarray(obs)
    E, S, _ = obs.shape
    b = host_bits(obs).astype(bool)                                           # (E, S, 13, 4)
    ev = {'export': b[..., 1] | b[..., 3], 'flag': b[..., 3], 'root': b[..., 2] & b[..., 3],
          'removed': b[..., 2] & ~b[..., 3]}
    t = np.arange(S)[None, :, None]
    feats = {}
    for name in TF_EVENTS:
        age = t - _last_index(ev[name])
        feats[f'age_{name}'] = np.where(age < k, age, k)
    for name, e in [('export', ev['export']), ('scan', b[..., 0])]:
        cs = np.concatenate([np.zeros((E, 1, L.NUM_HOSTS), np.int64), np.cumsum(e, axis=1)], axis=1)   # cs[s+1] = Σ_{≤s}
        lo = np.maximum(np.arange(S) - k + 1, 0)
        feats[f'cnt_{name}'] = cs[:, 1:] - cs[:, lo]
    sec = b[..., 2].astype(np.int64) * 2 + b[..., 3]                          # 0 [0,0], 1 [0,1], 2 [1,0], 3 [1,1]
    code = np.array([0, SEC_CODE[(0, 1)], SEC_CODE[(1, 0)], SEC_CODE[(1, 1)]])[sec]
    last = _last_index(sec > 0)
    in_win = (t - last) < k
    got = np.take_along_axis(code, np.clip(last, 0, None), axis=1)
    feats['last_sec'] = np.where(in_win, got, 0)
    tf = np.stack([feats[f] for f in TF_HOST_FEATS], axis=-1)                # (E, S, 13, 7)
    out = np.concatenate([obs, tf.reshape(E, S, -1)], axis=-1)[:, 1:]
    assert out.max() <= 127
    return out.astype(np.int8)

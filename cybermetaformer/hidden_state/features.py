"""Features derived from the observation: alerts and alert age (numpy, independent of CybORG).

Alert definitions (4 bits per host: activity [scan, exploit] + safety [g0, g1]):
- 'export': exploit activity (bit 1) OR safety g1 (bit 3). This is the definition in `loaders.export_tabular` and
  of "silent compromise" in the paper. Main definition.
- 'flag': bit 3 only (safety [0,1] exploited or [1,1] root). This is the alert rule of `baselines.py`.
"""
import numpy as np

from minicage.dataset import loaders as L

NEVER = -1          # alert_age: no alert on this host since the start of the episode


def host_bits(obs):
    """(..., 78) → (..., 13, 4)."""
    return np.asarray(obs)[..., :4 * L.NUM_HOSTS].reshape(*np.shape(obs)[:-1], L.NUM_HOSTS, 4)


def alerts(obs, kind='export', label_hosts=True):
    """(..., 78) → (..., 12) bool (13 if label_hosts=False)."""
    b = host_bits(obs)
    if kind == 'export':
        a = (b[..., 1] == 1) | (b[..., 3] == 1)
    elif kind == 'flag':
        a = b[..., 3] == 1
    else:
        raise ValueError(f'Unknown alert definition: {kind}')
    return a[..., L.LABEL_HOSTS] if label_hosts else a


def alert_age(alert_seq):
    """(E, T, H) bool → (E, T, H) int32: 0 if there is an alert at t, otherwise steps since the last alert,
    NEVER (-1) if there has been no alert since the start of the episode."""
    E, T, H = alert_seq.shape
    age = np.empty((E, T, H), np.int32)
    last = np.full((E, H), -1, np.int64)
    for t in range(T):
        last = np.where(alert_seq[:, t], t, last)
        age[:, t] = np.where(last < 0, NEVER, t - last)
    return age


def episode_alerts(obs_blue, kind='export'):
    """Episode array (E, T+1, 78) → alerts aligned with the target steps (E, T, 12) (obs[1..T])."""
    return alerts(obs_blue[:, 1:], kind=kind)


# ── XGB-TF: temporal summary features within the window ──────

TF_EVENTS = ['export', 'flag', 'root', 'removed']      # events whose age feature is extracted
TF_HOST_FEATS = ['age_export', 'cnt_export', 'age_flag', 'age_root', 'age_removed', 'cnt_scan', 'last_sec']
SEC_CODE = {(0, 0): 0, (0, 1): 1, (1, 1): 2, (1, 0): 3}  # last_sec: 0 none, 1 exploit, 2 root, 3 remove


def tf_feature_names(hosts):
    """hosts: meta['hosts'] (13 names). Order: first the 78 raw features of the last observation, then host × TF_HOST_FEATS."""
    raw = [f'obs_{i}' for i in range(4 * L.NUM_HOSTS)] + [f'scan_{h}' for h in hosts] + [f'decoy_{h}' for h in hosts]
    return raw + [f'{f}_{h}' for h in hosts for f in TF_HOST_FEATS]


def _last_index(ev):
    """(E, S, H) bool → (E, S, H) int64: step at which the event was last seen up to s (inclusive); -10**6 if never."""
    E, S, H = ev.shape
    idx = np.where(ev, np.arange(S)[None, :, None], -10 ** 6)
    return np.maximum.accumulate(idx, axis=1)


def temporal_features(obs, k):
    """Episode array (E, T+1, 78) → (E, T, 78 + 13·7) int8; target steps t = 1..T, window obs[t-k+1 .. t].

    Per host (13 hosts, including user0; it carries information in the observation):
    - age_<event>: steps since the last event in the window (0 = now); k if absent from the window
    - cnt_export, cnt_scan: number of alerts / scan activities in the window
    - last_sec: the most recent non-empty safety state in the window (SEC_CODE), otherwise 0
    The window shortens at the episode start (padding = no event), same alignment as WindowData.
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

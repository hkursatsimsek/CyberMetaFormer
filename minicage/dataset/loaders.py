"""Ham MiniCAGE veri setini yükler ve görev görünümlerine dönüştürür.

    from minicage.dataset import loaders as L
    raw, meta = L.load_raw('minicage/data/minicage_v1')
    splits = L.load_splits('minicage/data/minicage_v1')
    X, y = L.detection_view(raw, splits['train'], k=4)
"""
import csv
import gzip
import json
import os

import numpy as np

NUM_HOSTS = 13
USER0 = 8                 # Red'in sabit dayanağı; Blue müdahale edemez → etiketlerden çıkarılır
LABEL_HOSTS = [h for h in range(NUM_HOSTS) if h != USER0]
RED_ACTION_TYPES = ['sleep', 'remote', 'network', 'exploit', 'escalate', 'impact']
OOD_BLUE_POLICY = 'react_restore_eps0.3'


def load_meta(data_dir):
    with open(os.path.join(data_dir, 'meta.json')) as f:
        return json.load(f)


def load_raw(data_dir, fields=None):
    """Tüm shard'ları bölüm ekseninde birleştirir. fields=None → tüm alanlar."""
    meta = load_meta(data_dir)
    parts = []
    for shard in meta['shards']:
        with np.load(os.path.join(data_dir, shard['file'])) as z:
            keys = fields or z.files
            parts.append({k: z[k] for k in keys})
    raw = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    return raw, meta


# ── etiketler ───────────────────────────────────────────────────────────

def host_labels(true_state):
    """(..., 39) gerçek durum → (..., 13) sınıf: 0 temiz, 1 user erişimi, 2 root erişimi."""
    s = true_state.reshape(*true_state.shape[:-1], NUM_HOSTS, 3)
    labels = np.zeros(s.shape[:-1], np.int8)
    labels[s[..., 1] == 1] = 1
    labels[s[..., 2] == 1] = 2
    return labels


def red_action_type(act_red):
    """56 Red aksiyonu → 6 tip (sleep, remote, network, exploit, escalate, impact)."""
    a = np.asarray(act_red).astype(np.int64)
    out = np.zeros_like(a)
    out[(a >= 1) & (a <= 3)] = 1
    host_actions = a >= 4
    out[host_actions] = 2 + (a[host_actions] - 4) // NUM_HOSTS
    return out


# ── bölümler ────────────────────────────────────────────────────────────

def make_splits(data_dir, seed=0, frac=(0.8, 0.1, 0.1)):
    """Bölüm düzeyinde, (red, blue) kombinasyonuna göre tabakalı train/val/test + OOD test."""
    raw, meta = load_raw(data_dir, fields=['red_policy', 'blue_policy'])
    rng = np.random.default_rng(seed)
    ood_id = meta['blue_policies'].index(OOD_BLUE_POLICY)
    splits = {'train': [], 'val': [], 'test': [], 'ood_test': []}
    combos = raw['red_policy'].astype(int) * 100 + raw['blue_policy'].astype(int)
    for combo in np.unique(combos):
        idx = rng.permutation(np.where(combos == combo)[0])
        if combo % 100 == ood_id:
            splits['ood_test'].extend(idx)
            continue
        n_train, n_val = int(len(idx) * frac[0]), int(len(idx) * frac[1])
        splits['train'].extend(idx[:n_train])
        splits['val'].extend(idx[n_train:n_train + n_val])
        splits['test'].extend(idx[n_train + n_val:])
    splits = {k: np.sort(np.array(v, np.int64)) for k, v in splits.items()}
    np.savez(os.path.join(data_dir, 'splits.npz'), **splits)
    return splits


def load_splits(data_dir):
    with np.load(os.path.join(data_dir, 'splits.npz')) as z:
        return {k: z[k] for k in z.files}


# ── görev görünümleri ───────────────────────────────────────────────────

def _windows(obs, k):
    """(E, T+1, D) → (E, T, k*D): her t ≥ 1 için obs[t-k+1 .. t], başta sıfır dolgu."""
    E, T1, D = obs.shape
    padded = np.concatenate([np.zeros((E, k - 1, D), obs.dtype), obs], axis=1)
    win = np.stack([padded[:, j:j + T1] for j in range(k)], axis=2)  # (E, T+1, k, D)
    return win[:, 1:].reshape(E, T1 - 1, k * D)


def detection_view(raw, idx, k=1):
    """Saldırı tespiti. X: son k Blue gözlemi; y: (N, 12) host sınıfı {0,1,2} (user0 hariç).

    Dönen y_any: ağda herhangi bir ihlal (user0 dışında user/root) var mı (ikili).
    """
    X = _windows(raw['obs_blue'][idx], k)
    y = host_labels(raw['true_state'][idx][:, 1:])[..., LABEL_HOSTS]
    X, y = X.reshape(-1, X.shape[-1]), y.reshape(-1, len(LABEL_HOSTS))
    return X, y, (y > 0).any(axis=1).astype(np.int8)


def attacker_id_view(raw, idx, n_steps=10):
    """Saldırgan türü tanıma. X: (N, n_steps, 78) ilk n_steps Blue gözlemi; y: red_policy."""
    return raw['obs_blue'][idx][:, 1:n_steps + 1], raw['red_policy'][idx].astype(np.int64)


def next_red_action_view(raw, idx, k=4, coarse=True):
    """Red'in sıradaki aksiyonu. X: t'ye kadarki son k Blue gözlemi; y: act_red[t]."""
    obs = raw['obs_blue'][idx]
    E, T1, D = obs.shape
    padded = np.concatenate([np.zeros((E, k - 1, D), obs.dtype), obs], axis=1)
    X = np.stack([padded[:, j:j + T1 - 1] for j in range(k)], axis=2).reshape(E * (T1 - 1), k * D)
    y = raw['act_red'][idx].reshape(-1)
    return X, (red_action_type(y) if coarse else y.astype(np.int64))


def d4rl_view(raw, idx, blue_policies=None, meta=None):
    """Offline RL / BC için D4RL biçiminde düz geçiş sözlüğü (Blue perspektifi)."""
    idx = np.asarray(idx)
    if blue_policies is not None:
        ids = [meta['blue_policies'].index(p) for p in blue_policies]
        idx = idx[np.isin(raw['blue_policy'][idx], ids)]
    obs = raw['obs_blue'][idx].astype(np.float32)
    E, T1, D = obs.shape
    T = T1 - 1
    timeouts = np.zeros((E, T), bool)
    timeouts[:, -1] = True
    return {
        'observations': obs[:, :-1].reshape(-1, D),
        'actions': raw['act_blue'][idx].reshape(-1).astype(np.int64),
        'rewards': raw['rew_blue'][idx].reshape(-1),
        'next_observations': obs[:, 1:].reshape(-1, D),
        'terminals': np.zeros(E * T, bool),
        'timeouts': timeouts.reshape(-1),
        'episode': np.repeat(idx, T),
    }


def risk_view(raw, idx, horizon=10, k=4, host=7):
    """Erken uyarı. X: son k Blue gözlemi; y: host (varsayılan OpServer) önümüzdeki `horizon`
    adımda impact edilecek mi. Zaten impact altındaki ve ufku bölüm sonunu aşan adımlar çıkarılır."""
    obs = raw['obs_blue'][idx]
    E, T1, _ = obs.shape
    T = T1 - 1
    imp = np.concatenate([np.zeros((E, 1)), raw['impacted'][idx][:, :, host]], axis=1)   # obs ile hizalı
    c = np.concatenate([np.zeros((E, 1)), np.cumsum(imp, axis=1)], axis=1)                # c[b]-c[a] = sum imp[a:b]
    s = np.arange(1, T + 1)
    valid = s <= T - horizon
    s = s[valid]
    future = (c[:, s + horizon + 1] - c[:, s + 1]) > 0
    keep = imp[:, s] == 0
    X = _windows(obs, k)[:, valid]
    return X[keep], future[keep].astype(np.int8)


def dynamics_view(raw, idx):
    """Dünya modeli / dinamik öğrenme: (gözlem, gerçek durum, Blue aksiyonu) → sonraki gözlem ve gerçek durum."""
    obs, ts = raw['obs_blue'][idx], raw['true_state'][idx]
    D, S = obs.shape[-1], ts.shape[-1]
    return {
        'obs': obs[:, :-1].reshape(-1, D), 'true_state': ts[:, :-1].reshape(-1, S),
        'blue_action': raw['act_blue'][idx].reshape(-1).astype(np.int64),
        'red_action': raw['act_red'][idx].reshape(-1).astype(np.int64),
        'next_obs': obs[:, 1:].reshape(-1, D), 'next_true_state': ts[:, 1:].reshape(-1, S),
        'reward': raw['rew_blue'][idx].reshape(-1),
    }


def export_tabular(data_dir, out_path=None, sample_episodes=None, seed=0):
    """Adım başına bir satır, okunabilir sütunlarla csv.gz yazar."""
    raw, meta = load_raw(data_dir)
    hosts, n = meta['hosts'], NUM_HOSTS
    E, T = raw['act_blue'].shape
    episodes = np.arange(E)
    if sample_episodes:
        episodes = np.sort(np.random.default_rng(seed).choice(E, sample_episodes, replace=False))
    out_path = out_path or os.path.join(data_dir, 'tabular.csv.gz')

    red_names = np.array(meta['action_names']['Red'])
    blue_names = np.array(meta['action_names']['Blue'])
    host_cols = [h for i, h in enumerate(hosts) if i != USER0]
    header = (['episode', 'step', 'red_policy', 'blue_policy', 'red_action', 'red_action_type',
               'blue_action', 'red_success', 'blue_success', 'reward', 'cum_reward',
               'n_user_hosts', 'n_root_hosts', 'opserv_impacted', 'n_alerts']
              + [f'alert_{h}' for h in host_cols] + [f'label_{h}' for h in host_cols])

    with gzip.open(out_path, 'wt', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for start in range(0, len(episodes), 500):
            ep = episodes[start:start + 500]
            labels = host_labels(raw['true_state'][ep][:, 1:])                    # (e, T, 13)
            safety = raw['obs_blue'][ep][:, 1:, :4 * n].reshape(len(ep), T, n, 4)
            alerts = (safety[..., 3] == 1) | (safety[..., 1] == 1)               # istismar uyarısı
            alerts, labels = alerts[..., LABEL_HOSTS], labels[..., LABEL_HOSTS]
            cum = np.cumsum(raw['rew_blue'][ep], axis=1)
            for i, e in enumerate(ep):
                rp = meta['red_policies'][raw['red_policy'][e]]
                bp = meta['blue_policies'][raw['blue_policy'][e]]
                for t in range(T):
                    ar = raw['act_red'][e, t]
                    w.writerow([
                        e, t, rp, bp, red_names[ar], RED_ACTION_TYPES[red_action_type(ar)],
                        blue_names[raw['act_blue'][e, t]], raw['red_success'][e, t],
                        raw['blue_success'][e, t], round(float(raw['rew_blue'][e, t]), 2),
                        round(float(cum[i, t]), 2), int((labels[i, t] == 1).sum()),
                        int((labels[i, t] == 2).sum()), raw['impacted'][e, t, 7],
                        int(alerts[i, t].sum())]
                        + alerts[i, t].astype(int).tolist() + labels[i, t].tolist())
    return out_path

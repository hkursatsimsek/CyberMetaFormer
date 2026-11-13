"""Rule-based references. No training needed; there is no seed.

- majority: the most frequent class per host in train ("clean" for every host; the majority reference in baselines.json).
- persist (k): "root if an alert was seen in the last k observations, otherwise clean". k=1 is the classic alert rule. Alert
  definition 'export' (features.py). Root choice: in train most compromised steps on every host are root (user 3–15× rarer).
  Shows the "free" value of memory: as k grows, how many silent compromises are caught by the alert history alone.
"""
import numpy as np

from cybermetaformer.hidden_state import data as D
from cybermetaformer.hidden_state import features as F

RULES = ['majority', 'persist']


def majority_classes(store, idx):
    """Most frequent class per host in train, (12,)."""
    y = store.labels[idx].reshape(-1, len(D.LABEL_HOSTS))
    return np.array([np.bincount(y[:, h], minlength=D.N_CLASSES).argmax() for h in range(y.shape[1])])


def persist_pred(obs, k, kind='export', cls=2):
    """Episode array (E, T+1, 78) → (E, T, 12): cls if there is an alert within window obs[t-k+1 .. t], otherwise 0."""
    a = F.alerts(obs, kind=kind)                                   # (E, T+1, 12), obs[0] dahil
    t = np.arange(a.shape[1])[None, :, None]
    last = np.maximum.accumulate(np.where(a, t, -10 ** 6), axis=1)
    return np.where(t - last < k, cls, 0)[:, 1:].astype(np.int64)


def rule_predictor(name, k, store, train_idx):
    """predict(idx) → (None, pred) for final_reports. Rules have no logits (calibration is not reported)."""
    if name == 'majority':
        cls = majority_classes(store, train_idx)
        return lambda idx: (None, np.broadcast_to(cls, (len(idx), store.T, len(cls))).copy())
    if name == 'persist':
        return lambda idx: (None, persist_pred(store.obs[idx], k))
    raise ValueError(f'Bilinmeyen kural: {name}')

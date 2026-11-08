"""Kural tabanlı referanslar. Eğitim gerektirmez; seed yoktur.

- majority: host başına train'deki en sık sınıf (her host için "temiz"; baselines.json'daki çoğunluk referansı).
- persist (k): "son k gözlemde uyarı görüldüyse root, yoksa temiz". k=1 klasik uyarı kuralıdır. Uyarı tanımı 'export'
  (features.py). Root seçimi: train'de her hostta ele geçirilmiş adımların çoğunluğu root (user 3–15 kat daha az).
  Hafızanın "bedava" değerini gösterir: k büyüdükçe sessiz ihlallerin ne kadarı yalnızca uyarı geçmişiyle yakalanıyor.
"""
import numpy as np

from cybermetaformer.hidden_state import data as D
from cybermetaformer.hidden_state import features as F

RULES = ['majority', 'persist']


def majority_classes(store, idx):
    """Host başına train'deki en sık sınıf, (12,)."""
    y = store.labels[idx].reshape(-1, len(D.LABEL_HOSTS))
    return np.array([np.bincount(y[:, h], minlength=D.N_CLASSES).argmax() for h in range(y.shape[1])])


def persist_pred(obs, k, kind='export', cls=2):
    """Bölüm dizisi (E, T+1, 78) → (E, T, 12): pencere obs[t-k+1 .. t] içinde uyarı varsa cls, yoksa 0."""
    a = F.alerts(obs, kind=kind)                                   # (E, T+1, 12), obs[0] dahil
    t = np.arange(a.shape[1])[None, :, None]
    last = np.maximum.accumulate(np.where(a, t, -10 ** 6), axis=1)
    return np.where(t - last < k, cls, 0)[:, 1:].astype(np.int64)


def rule_predictor(name, k, store, train_idx):
    """final_reports için predict(idx) → (None, pred). Kuralların logit'i yok (kalibrasyon raporlanmaz)."""
    if name == 'majority':
        cls = majority_classes(store, train_idx)
        return lambda idx: (None, np.broadcast_to(cls, (len(idx), store.T, len(cls))).copy())
    if name == 'persist':
        return lambda idx: (None, persist_pred(store.obs[idx], k))
    raise ValueError(f'Bilinmeyen kural: {name}')

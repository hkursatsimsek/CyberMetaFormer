"""XGBoost baseline'ları.

- `xgb`: düzleştirilmiş son k gözlem (k×78), WindowData ile aynı hizalama (eskiden yeniye, baştaki dolgu sıfır).
- `xgb_tf`: son gözlem (78) + host başına pencere içi zamansal özetler (13×7; features.temporal_features).

Host başına çok sınıflı model (hist, multi:softprob), val mlogloss ile erken durdurma. Özellik matrisi parça parça
nicemlenir (QuantileDMatrix + DataIter; k=32'de bile tam float32 kopya oluşmaz) ve 12 host için yalnızca etiket
değiştirilerek yeniden kullanılır. Train'de hiç ele geçirilmemiş host (def) için sabit "temiz" tahmini kullanılır.

    model, info = train_xgb({'model': 'xgb_tf', 'k': 8, 'seed': 0}, store)
    logits = model.margins(store, idx)            # (E, T, 12, 3)
"""
import time

import numpy as np
import xgboost as xgb

from cybermetaformer.hidden_state import data as D
from cybermetaformer.hidden_state import features as F
from cybermetaformer.hidden_state import metrics as M

KINDS = {'xgb': 'raw', 'xgb_tf': 'tf'}
DEFAULTS = {
    'model': 'xgb_tf', 'k': 8, 'seed': 0, 'train_episodes': None, 'train_subsample_seed': 0, 'val_episodes': None,
    'nthread': 0, 'num_boost_round': 2000, 'early_stopping_rounds': 50, 'chunk_episodes': 1000,
    'params': {'max_depth': 6, 'eta': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 1.0,
               'reg_lambda': 1.0, 'max_bin': 256},
}
CONST_CLEAN = np.array([0.0, -20.0, -20.0], np.float32)       # hiç ele geçirilmemiş host: sabit temiz


def feature_matrix(store, idx, k, kind):
    """(len(idx)·T, d) float32, bölüm-ana sıra (WindowData / predict_logits ile aynı)."""
    obs = store.obs[np.asarray(idx)]
    if kind == 'tf':
        X = F.temporal_features(obs, k)
    elif kind == 'raw':
        n, S, d = obs.shape
        padded = np.concatenate([np.zeros((n, k - 1, d), obs.dtype), obs], axis=1)
        win = np.lib.stride_tricks.sliding_window_view(padded, k, axis=1)          # (n, S, d, k)
        X = win.transpose(0, 1, 3, 2)[:, 1:]                                        # (n, T, k, d), eskiden yeniye
    else:
        raise ValueError(kind)
    return X.reshape(-1, int(np.prod(X.shape[2:]))).astype(np.float32)


class _Chunks(xgb.DataIter):
    def __init__(self, store, idx, k, kind, chunk):
        self.store, self.k, self.kind = store, k, kind
        self.parts = [idx[i:i + chunk] for i in range(0, len(idx), chunk)]
        self.i = 0
        super().__init__(cache_prefix=None)

    def next(self, input_data):
        if self.i == len(self.parts):
            return False
        input_data(data=feature_matrix(self.store, self.parts[self.i], self.k, self.kind))
        self.i += 1
        return True

    def reset(self):
        self.i = 0


def quantile_matrix(store, idx, k, kind, chunk, max_bin, nthread, ref=None):
    return xgb.QuantileDMatrix(_Chunks(store, np.asarray(idx), k, kind, chunk), max_bin=max_bin, nthread=nthread,
                               ref=ref)


class XGBHosts:
    """12 hostun booster'ları (ya da None = sabit temiz)."""

    def __init__(self, boosters, kind, k, chunk):
        self.boosters, self.kind, self.k, self.chunk = boosters, kind, k, chunk

    def _margins_X(self, X):
        out = np.empty((len(X), len(self.boosters), D.N_CLASSES), np.float32)
        for h, bst in enumerate(self.boosters):
            if bst is None:
                out[:, h] = CONST_CLEAN
            else:
                out[:, h] = bst.inplace_predict(X, iteration_range=(0, bst.best_iteration + 1),
                                                predict_type='margin')
        return out

    def margins(self, store, idx):
        """(E, T, 12, 3) ham skorlar (softmax öncesi; kalibrasyon ve sıcaklık için logit yerine geçer)."""
        idx = np.asarray(idx)
        parts = [self._margins_X(feature_matrix(store, idx[i:i + self.chunk], self.k, self.kind))
                 for i in range(0, len(idx), self.chunk)]
        return np.concatenate(parts).reshape(len(idx), store.T, len(self.boosters), D.N_CLASSES)

    def shap_summary(self, store, idx, hosts, rows=20000, seed=0):
        """TreeSHAP (pred_contribs) özetleri: hangi hostun ve hangi özellik türünün katkısı (satır normalize).

        by_host[h][j]: label hostu h'nin tahmininde gözlem hostu j'nin özelliklerinin |SHAP| payı (13 sütun).
        by_type[tür][h]: özellik türünün payı. Ham XGB'de ayrıca by_lag[h][gecikme] (0 = son gözlem)."""
        X = feature_matrix(store, idx, self.k, self.kind)
        X = X[np.random.default_rng(seed).choice(len(X), min(rows, len(X)), replace=False)]
        d = X.shape[1]
        if self.kind == 'tf':
            n_tf = len(F.TF_HOST_FEATS)
            src = np.concatenate([np.arange(52) // 4, np.arange(13), np.arange(13), np.repeat(np.arange(13), n_tf)])
            typ = np.array(['bit'] * 52 + ['scan'] * 13 + ['decoy'] * 13 + F.TF_HOST_FEATS * 13)
            lag = None
        else:
            j = np.arange(d) % D.OBS_DIM
            src = np.where(j < 52, j // 4, (j - 52) % 13)
            typ = np.where(j < 52, 'bit', np.where(j < 65, 'scan', 'decoy'))
            lag = self.k - 1 - np.arange(d) // D.OBS_DIM
        types = list(dict.fromkeys(typ.tolist()))
        by_host, by_type, by_lag = [], {t: [] for t in types}, []
        dm = xgb.DMatrix(X)
        for bst in self.boosters:
            if bst is None:
                by_host.append(None)
                for t in types:
                    by_type[t].append(None)
                by_lag.append(None)
                continue
            c = bst.predict(dm, pred_contribs=True, iteration_range=(0, bst.best_iteration + 1))   # (n, 3, d+1)
            imp = np.abs(c[..., :d]).mean(0).sum(0)
            imp = imp / max(imp.sum(), 1e-12)
            by_host.append(np.bincount(src, imp, minlength=13).round(5).tolist())
            for t in types:
                by_type[t].append(round(float(imp[typ == t].sum()), 5))
            if lag is not None:
                by_lag.append(np.bincount(lag, imp, minlength=self.k).round(5).tolist())
        label_hosts = [hosts[h] for h in D.LABEL_HOSTS]
        out = {'label_hosts': label_hosts, 'obs_hosts': list(hosts), 'by_host': by_host, 'by_type': by_type,
               'rows': int(len(X))}
        if lag is not None:
            out['by_lag'] = by_lag
        return out


def train_xgb(cfg, store, log=print):
    """Host başına booster eğitir. Dönüş: (XGBHosts, info)."""
    cfg = {**DEFAULTS, **cfg, 'params': {**DEFAULTS['params'], **cfg.get('params', {})}}
    kind, k, p = KINDS[cfg['model']], cfg['k'], dict(cfg['params'])
    max_bin = p.pop('max_bin')
    tr_idx = D.subsample_episodes(store.split('train'), cfg['train_episodes'], cfg['train_subsample_seed'])
    va_idx = D.subsample_episodes(store.split('val'), cfg['val_episodes'], 1)
    store.assert_no_ood(tr_idx)
    store.assert_no_ood(va_idx)
    t0 = time.time()
    dtr = quantile_matrix(store, tr_idx, k, kind, cfg['chunk_episodes'], max_bin, cfg['nthread'])
    dva = quantile_matrix(store, va_idx, k, kind, cfg['chunk_episodes'], max_bin, cfg['nthread'], ref=dtr)
    t_data = time.time() - t0
    ytr = store.labels[tr_idx].reshape(-1, len(D.LABEL_HOSTS))
    yva = store.labels[va_idx].reshape(-1, len(D.LABEL_HOSTS))
    params = {'objective': 'multi:softprob', 'num_class': D.N_CLASSES, 'tree_method': 'hist',
              'eval_metric': 'mlogloss', 'seed': cfg['seed'], 'nthread': cfg['nthread'], 'max_bin': max_bin, **p}
    boosters, rounds = [], []
    for h in range(len(D.LABEL_HOSTS)):
        if not (ytr[:, h] > 0).any():
            boosters.append(None)
            rounds.append(0)
            continue
        dtr.set_label(ytr[:, h])
        dva.set_label(yva[:, h])
        bst = xgb.train(params, dtr, cfg['num_boost_round'], evals=[(dva, 'val')],
                        early_stopping_rounds=cfg['early_stopping_rounds'], verbose_eval=False)
        boosters.append(bst)
        rounds.append(bst.best_iteration + 1)
        log(f"  host {h:2d}: {rounds[-1]} tur, val mlogloss {bst.best_score:.4f}")
    model = XGBHosts(boosters, kind, k, cfg['chunk_episodes'])
    val_pred = model.margins(store, va_idx).argmax(-1).reshape(-1, len(D.LABEL_HOSTS))
    info = {'device': 'cpu', 'train_episodes_hash': D.episode_hash(tr_idx), 'n_train_episodes': len(tr_idx),
            'n_features': int(dtr.num_col()), 'rounds_per_host': rounds, 'data_seconds': t_data,
            'train_seconds': time.time() - t0, 'best_val_host_macro_f1': M.host_macro_f1(yva, val_pred),
            'n_trees': int(sum(len(b.get_dump()) for b in boosters if b is not None))}
    return model, info


def xgb_predictor(model, store):
    def predict(idx):
        logits = model.margins(store, idx)
        return logits, logits.argmax(-1)
    return predict

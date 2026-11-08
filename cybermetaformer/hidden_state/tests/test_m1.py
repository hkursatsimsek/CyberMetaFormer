"""Aşama 1 bileşenleri: XGB-TF özellikleri, kurallar, tam geçmiş RNN, XGBoost, HPO yaması, nihai değerlendirme."""
import numpy as np
import optuna
import pytest
import torch

from cybermetaformer.hidden_state import data as D
from cybermetaformer.hidden_state import features as F
from cybermetaformer.hidden_state import metrics as M
from cybermetaformer.hidden_state import rules as R
from cybermetaformer.hidden_state.evaluate import final_reports
from cybermetaformer.hidden_state.hpo import SPACES, overrides_from_params
from cybermetaformer.hidden_state.models import build_model
from cybermetaformer.hidden_state.run_grid import planned_runs


def _brute_tf(obs, k):
    """temporal_features'ın döngüyle yazılmış karşılığı (tek bölüm)."""
    S = obs.shape[0]
    b = F.host_bits(obs).astype(bool)
    rows = []
    for t in range(1, S):
        lo = max(t - k + 1, 0)
        feats = []
        for h in range(13):
            w = b[lo:t + 1, h]                                       # eskiden yeniye
            ev = {'export': w[:, 1] | w[:, 3], 'flag': w[:, 3], 'root': w[:, 2] & w[:, 3], 'removed': w[:, 2] & ~w[:, 3]}

            def age(e):
                hits = np.where(e)[0]
                return (len(e) - 1 - hits[-1]) if len(hits) else k
            sec = [(int(r[2]), int(r[3])) for r in w]
            nonzero = [s for s in sec if s != (0, 0)]
            feats += [age(ev['export']), int(ev['export'].sum()), age(ev['flag']), age(ev['root']),
                      age(ev['removed']), int(w[:, 0].sum()), F.SEC_CODE[nonzero[-1]] if nonzero else 0]
        rows.append(np.concatenate([obs[t], feats]))
    return np.array(rows)


@pytest.mark.parametrize('k', [1, 3, 8, 32, 101])
def test_temporal_features_match_brute_force(store, small_idx, k):
    obs = store.obs[small_idx[:3]]
    got = F.temporal_features(obs, k)
    assert got.shape == (3, store.T, 78 + 13 * len(F.TF_HOST_FEATS))
    assert len(F.tf_feature_names(store.meta['hosts'])) == got.shape[-1]
    for e in range(3):
        np.testing.assert_array_equal(got[e], _brute_tf(obs[e], k))


@pytest.mark.parametrize('k', [1, 4, 8, 32])
def test_persist_rule_brute_force(store, small_idx, k):
    obs = store.obs[small_idx[:5]]
    a = F.alerts(obs)                                                  # (5, T+1, 12)
    want = np.zeros((5, store.T, 12), np.int64)
    for t in range(1, store.T + 1):
        want[:, t - 1] = np.where(a[:, max(t - k + 1, 0):t + 1].any(1), 2, 0)
    np.testing.assert_array_equal(R.persist_pred(obs, k), want)


def test_persist_k1_is_alert_rule(store, small_idx):
    obs = store.obs[small_idx]
    np.testing.assert_array_equal(R.persist_pred(obs, 1) > 0, F.episode_alerts(obs))


@pytest.mark.parametrize('cell', ['lstm_full', 'gru_full'])
def test_episode_mode_equals_unpadded_window(store, small_idx, cell):
    """Tam geçmiş RNN'in t. çıktısı = k = t+1 dolgusuz pencereyle RNN'in son çıktısı."""
    torch.manual_seed(0)
    model = build_model(cell, 101).eval()
    wd = D.WindowData(store, small_idx[:2])
    obs, _ = wd.episodes(torch.tensor([0, 1]))
    x = D.scale_obs(obs)
    with torch.no_grad():
        seq = model.forward_episode(x)                                  # (2, T+1, 12, 3)
        for t in [1, 7, 50, 100]:
            win, mask = wd.windows(torch.tensor([0, 1]), torch.tensor([t, t]), t + 1)
            assert mask.all()
            last = model(D.scale_obs(win), mask)
            torch.testing.assert_close(seq[:, t], last, rtol=1e-5, atol=1e-5)


def test_episode_mode_training_runs(store):
    from cybermetaformer.hidden_state.train import train
    cfg = {'model': 'gru_full', 'mode': 'episode', 'k': 101, 'train_episodes': 64, 'val_episodes': 32,
           'epochs': 2, 'episodes_per_batch': 8, 'device': 'cpu', 'model_args': {'hidden': 32}}
    model, history, info = train(cfg, store, log=lambda s: None)
    assert len(history) == 2 and info['epochs_run'] == 2
    with pytest.raises(ValueError):
        train({**cfg, 'mode': 'window'}, store, log=lambda s: None)


def test_xgb_raw_features_match_windows(store, small_idx):
    from cybermetaformer.hidden_state.xgb import feature_matrix
    idx = small_idx[:6]
    wd = D.WindowData(store, idx)
    for k in [1, 8, 33]:
        ref = []
        for ep, t in D.eval_indices(wd.n, wd.T, 997):
            x, _ = wd.windows(ep, t, k)
            ref.append(x.reshape(len(ep), -1).numpy())
        np.testing.assert_array_equal(feature_matrix(store, idx, k, 'raw'), np.concatenate(ref).astype(np.float32))


def test_xgb_tiny_deterministic(store):
    from cybermetaformer.hidden_state.xgb import train_xgb
    cfg = {'model': 'xgb_tf', 'k': 4, 'train_episodes': 150, 'val_episodes': 50, 'num_boost_round': 10,
           'chunk_episodes': 64, 'nthread': 2}
    m1, info = train_xgb(cfg, store, log=lambda s: None)
    m2, _ = train_xgb(cfg, store, log=lambda s: None)
    idx = store.split('val')[:20]
    lg = m1.margins(store, idx)
    assert lg.shape == (20, store.T, 12, 3)
    np.testing.assert_array_equal(lg, m2.margins(store, idx))
    assert info['rounds_per_host'][0] == 0                               # def hiç ele geçirilmiyor → sabit temiz
    assert (lg[..., 0, :].argmax(-1) == 0).all()


@pytest.mark.parametrize('model', sorted(SPACES))
def test_hpo_overrides_roundtrip(model):
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=1))
    trial = study.ask()
    over = SPACES[model](trial)
    assert overrides_from_params(model, trial.params) == over


def test_final_reports_majority_matches_baselines(store):
    """Nihai değerlendirme hattı (val → sıcaklık yok, test/OOD + bootstrap): çoğunluk = 0.285 / 0.311."""
    predict = R.rule_predictor('majority', 0, store, store.split('train'))
    reports, outs = final_reports(store, predict, splits=('test', 'ood_test'), log=lambda s: None)
    assert f"{reports['test']['host_macro_f1']:.3f}" == '0.285'
    assert f"{reports['ood_test']['host_macro_f1']:.3f}" == '0.311'
    b = reports['test']['bootstrap']
    assert b['lo'] <= b['point'] <= b['hi']
    assert outs['test']['ep_conf'].sum() == reports['test']['n_episodes'] * store.T * 12
    counts, names = outs['test']['ep_slices']
    assert names[0] == 'sessiz' and counts[:, 0, 0].sum() == reports['test']['n_silent']
    assert counts[..., 1].sum() == 0                              # çoğunluk hiç pozitif tahmin etmez
    assert reports['test']['silent_recall'] == 0.0
    assert set(reports['test']['cause_recall']) <= set(store.cause_names[1:])


def test_cause_recall_counts(store):
    with store.final_evaluation():
        idx = store.split('test')
    c = store.cause[idx]
    y = store.labels[idx]
    rep = M.cause_recall(y, c, store.cause_names)                        # tahmin = gerçek → recall 1
    assert sum(v['n'] for v in rep.values()) == int(((y > 0) & ~F.episode_alerts(store.obs[idx])).sum())
    assert all(v['recall'] == 1.0 for v in rep.values())


def test_planned_runs_names():
    runs = planned_runs(['majority', 'persist', 'gru', 'lstm_full'], [1, 8], [0, 1])
    names = [r[3] for r in runs]
    assert names == ['majority', 'persist_k1', 'persist_k8', 'gru_k1_s0', 'gru_k1_s1', 'gru_k8_s0', 'gru_k8_s1',
                     'lstm_full_k101_s0', 'lstm_full_k101_s1']

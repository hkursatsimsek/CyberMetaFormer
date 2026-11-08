"""metrics.py ↔ baselines.py denkliği, baselines.json kural değerleri ve sessiz ihlal ön ölçümü (adım 0.5)."""
import json
import os

import numpy as np
import pytest

from minicage.dataset import loaders as L
from cybermetaformer.hidden_state import data as D
from cybermetaformer.hidden_state import features as F
from cybermetaformer.hidden_state import metrics as M

baselines = pytest.importorskip('minicage.dataset.baselines')   # CybORG ister → yalnız VM'de koşar


def _ref_host_f1(y, p):
    H = y.shape[1]
    return np.mean([baselines.macro_f1(y[:, h], p[:, h], [0, 1, 2]) for h in range(H) if y[:, h].any()])


def _random_case(seed, N=5000, H=12):
    rng = np.random.default_rng(seed)
    y = rng.choice(3, size=(N, H), p=[0.7, 0.05, 0.25])
    y[:, 0] = 0                                       # def gibi hiç ele geçirilmeyen host
    y[:, 5] = rng.choice([0, 2], size=N)              # user sınıfı hiç yok
    p = np.where(rng.random((N, H)) < 0.8, y, rng.choice(3, size=(N, H)))
    p[:, 7] = 0                                       # hiç pozitif tahmin yok
    return y, p


@pytest.mark.parametrize('seed', range(5))
def test_host_macro_f1_matches_baselines(seed):
    y, p = _random_case(seed)
    assert M.host_macro_f1(y, p) == _ref_host_f1(y, p)          # birebir (aynı işlem sırası)


@pytest.mark.parametrize('seed', range(3))
def test_breach_f1_matches_baselines(seed):
    y, p = _random_case(seed)
    ya, pa = (y > 0).any(1).astype(np.int8), (p > 0).any(1).astype(np.int8)
    assert M.breach_f1(y, p) == baselines.macro_f1(ya, pa, [1])


def test_episode_confusion_and_bootstrap():
    y, p = _random_case(0, N=40 * 100)
    ye, pe = y.reshape(40, 100, 12), p.reshape(40, 100, 12)
    C = M.episode_confusion(ye, pe)
    np.testing.assert_array_equal(C.sum(0), M.confusion(y, p))
    ci = M.bootstrap_ci(C, n_boot=200)
    assert ci['point'] == pytest.approx(M.host_macro_f1(y, p), abs=1e-12)
    assert ci['lo'] <= ci['point'] <= ci['hi']


def _baseline_value(model, split, task_prefix):
    path = os.path.join(D.DEFAULT_DATA, 'baselines.json')
    for r in json.load(open(path)):
        if r['model'] == model and r['split'] == split and r['task'].startswith(task_prefix):
            return r
    raise KeyError((model, split, task_prefix))


@pytest.mark.parametrize('split', ['test', 'ood_test'])
def test_rule_values_match_baselines_json(store, split):
    """Çoğunluk (0.285 / 0.311) ve uyarı kuralı (0.833 / 0.802), baselines.json'daki referans metinleriyle."""
    with store.final_evaluation():
        idx = store.split(split)
    y = store.labels[idx].reshape(-1, 12)
    maj = M.host_macro_f1(y, np.zeros_like(y))
    rule = F.alerts(store.obs[idx][:, 1:], kind='flag').reshape(-1, 12)
    r1 = _baseline_value('MLP, k=1', split, 'tespit (host')
    r2 = _baseline_value('MLP, k=1', split, 'tespit (ağda')
    assert f'{maj:.3f}' == r1['reference'].split(': ')[1]
    assert f'{M.breach_f1(y, rule.astype(np.int64)):.3f}' == r2['reference'].split(': ')[1]


def test_silent_share_matches_plan(store):
    """Sessiz ihlal ön ölçümü: sessiz / ele geçirilmiş = 0.408 (train), 0.404 (test), 0.593 (OOD)."""
    expected = {'train': 0.408, 'test': 0.404, 'ood_test': 0.593}
    with store.final_evaluation():
        for split, val in expected.items():
            idx = store.split(split)
            y = store.labels[idx]
            a = F.episode_alerts(store.obs[idx])
            s = M.silent_metrics(y, y, a)
            assert round(s['silent_share'], 3) == val


def test_alert_age_matches_appendix_a(store, small_idx):
    """Ek A'daki gap hesabıyla aynı (hiç uyarı yok → NEVER)."""
    a = F.episode_alerts(store.obs[small_idx])
    E, T, H = a.shape
    last = np.full((E, H), -10 ** 6)
    gap = np.zeros((E, T, H), np.int64)
    for t in range(T):
        last = np.where(a[:, t], t, last)
        gap[:, t] = t - last
    age = F.alert_age(a)
    np.testing.assert_array_equal(age == F.NEVER, gap > 10 ** 5)
    np.testing.assert_array_equal(age[age != F.NEVER], gap[age != F.NEVER])


def test_calibration_and_temperature():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 20000)
    perfect = np.eye(3)[y]
    c = M.calibration(perfect, y)
    assert c['ece'] == pytest.approx(0) and c['brier'] == pytest.approx(0)
    true_logits = rng.normal(size=(20000, 3)) * 3
    y = np.array([rng.choice(3, p=pr) for pr in M.softmax(true_logits)])
    t = M.fit_temperature(true_logits * 2.5, y)          # 2.5 kat aşırı güvenli → T ≈ 2.5
    assert t == pytest.approx(2.5, rel=0.1)


def test_latency_and_persistence_toy():
    y = np.zeros((1, 10, 1), int)
    y[0, 2:8, 0] = 2                          # dönem: t=2..7
    p = np.zeros_like(y)
    p[0, 4:7, 0] = 2                          # ilk doğru tahmin t=4 → gecikme 2
    a = np.zeros_like(y, bool)
    a[0, 2:4, 0] = True                       # uyarı t=2,3; sessiz koşu t=4..7 (4 adım)
    lat = M.detection_latency(y, p)
    assert lat['mean_latency'] == 2 and lat['n_spells'] == 1 and lat['detected_spells'] == 1
    per = M.silent_persistence(y, p, a)
    assert per['n_runs'] == 1 and per['mean_run'] == 4 and per['mean_held'] == 3

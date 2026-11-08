"""Belirsizlik modülü: bootstrap bileşenleri raporlarla aynı, seçici tahmin tutarlı, MC dropout kipi ve tahmini."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cybermetaformer.hidden_state import data as D
from cybermetaformer.hidden_state import evaluate as E
from cybermetaformer.hidden_state import metrics as M
from cybermetaformer.hidden_state import uncertainty as U


def _probs_and_labels(seed=0, Ne=30, T=20, H=12, informative=True):
    rng = np.random.default_rng(seed)
    y = rng.choice(3, size=(Ne, T, H), p=[0.7, 0.1, 0.2])
    logits = rng.normal(size=(Ne, T, H, 3))
    if informative:                                   # doğru sınıf öne çıkar; güven hataları ayırt eder
        logits += 2.5 * np.eye(3)[y] * rng.random((Ne, T, H, 1))
    y[..., 0] = 0
    alert = rng.random((Ne, T, H)) < 0.4
    es = SimpleNamespace(y=y, alert=alert, E=Ne, n=y.size)
    return M.softmax(logits).astype(np.float32), es


def test_bootstrap_stats_match_reports():
    P, es = _probs_and_labels()
    fns = U.stat_fns(P, es)
    W = np.ones((1, es.E))
    cal = M.calibration(P, es.y)
    assert fns['ece'](W)[0] == pytest.approx(cal['ece'], abs=1e-6)
    assert fns['nll'](W)[0] == pytest.approx(cal['nll'], abs=1e-5)
    pred = P.argmax(-1)
    assert fns['host_macro_f1'](W)[0] == pytest.approx(M.host_macro_f1(es.y.reshape(-1, 12), pred.reshape(-1, 12)), abs=1e-12)


def test_bootstrap_stats_resample_episodes():
    P, es = _probs_and_labels()
    fns = U.stat_fns(P, es)
    W = np.random.default_rng(0).integers(0, 3, (4, es.E)).astype(np.float64)
    # ağırlıklı istatistik = bölümleri ağırlık kadar tekrarlayarak hesaplanan istatistik
    rep = np.repeat(np.arange(es.E), W[0].astype(int))
    cal = M.calibration(P[rep], es.y[rep])
    assert fns['ece'](W)[0] == pytest.approx(cal['ece'], abs=1e-6)
    assert fns['nll'](W)[0] == pytest.approx(cal['nll'], abs=1e-5)


def test_selective_prediction_is_consistent():
    P, es = _probs_and_labels()
    rep, ep_conf = U.evaluate_probs(P, es, curve=True)
    err = float((P.argmax(-1) != es.y).mean())
    assert rep['err@0.8'] < rep['err@0.9'] < rep['err@0.95'] < err               # güven bilgilendirici: dışlamak hatayı azaltır
    assert rep['curve']['risk'][-1] == pytest.approx(err, abs=1e-4)             # kapsama 1'de risk = genel hata
    assert 0 < rep['aurc'] < err
    assert ep_conf.shape == (es.E, 12, 3, 3)
    assert sum(rep['reliability']['count']) == es.n


def test_confidence_is_uninformative_for_flat_probabilities():
    P, es = _probs_and_labels(informative=False)
    rep, _ = U.evaluate_probs(P, es)
    err = float((P.argmax(-1) != es.y).mean())
    assert abs(rep['err@0.8'] - err) < 0.03                                      # güvenden hata öngörülemez


def test_rescale_is_temperature_scaling():
    logits = np.random.default_rng(0).normal(size=(5, 4, 12, 3)).astype(np.float32)
    P = U.probs(logits)
    np.testing.assert_allclose(U.rescale(P, 1.0), P, atol=1e-5)
    np.testing.assert_allclose(U.rescale(P, 2.0), U.probs(logits, 2.0), atol=1e-5)


class _Tiny(torch.nn.Module):
    def __init__(self, k, dropout):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(k * D.OBS_DIM, 64), torch.nn.BatchNorm1d(64),
                                       torch.nn.ReLU(), torch.nn.Dropout(dropout), torch.nn.Linear(64, 36))

    def forward(self, x, mask):
        return self.net(x).view(-1, 12, 3)


def test_mc_mode_keeps_batchnorm_eval():
    model = _Tiny(2, 0.3)
    U.mc_mode(model)
    mods = {type(m).__name__: m for m in model.modules()}
    assert mods['Dropout'].training and not mods['BatchNorm1d'].training


def test_mc_mode_keeps_attention_dropout_off():
    layer = torch.nn.TransformerEncoderLayer(16, 2, 32, 0.3, batch_first=True, norm_first=True)
    model = torch.nn.Sequential(torch.nn.TransformerEncoder(layer, 2, enable_nested_tensor=False))
    U.mc_mode(model)
    attn = [m for m in model.modules() if isinstance(m, torch.nn.MultiheadAttention)]
    drops = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
    assert attn and not any(m.training for m in attn)       # MPS: SDPA dropout desteklenmiyor
    assert drops and all(m.training for m in drops)
    out = model(torch.randn(4, 5, 16))                       # eğitim kipinde dikkat dropout'suz ileri geçiş çalışır
    assert out.shape == (4, 5, 16)


@pytest.mark.parametrize('dropout', [0.0, 0.5])
def test_predict_mc(store, small_idx, dropout):
    k = 3
    torch.manual_seed(0)
    model = _Tiny(k, dropout).eval()
    wd = D.WindowData(store, small_idx[:6])
    det = M.softmax(E.predict_logits(model, wd, k))
    mc = np.exp(U.predict_mc(model, wd, k, passes=8))
    np.testing.assert_allclose(mc.sum(-1), 1.0, atol=1e-5)
    if dropout == 0.0:
        np.testing.assert_allclose(mc, det, atol=1e-5)
    else:
        assert np.abs(mc - det).max() > 1e-3

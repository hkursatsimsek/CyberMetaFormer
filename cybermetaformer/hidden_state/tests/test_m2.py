"""Stage 2 components: vendored TSLib, adapter layer, validity matrix, compile path, weight saving."""
import hashlib
import os

import numpy as np
import pytest
import torch

from cybermetaformer.hidden_state import data as D
from cybermetaformer.hidden_state.models import NOT_APPLICABLE, build_model
from cybermetaformer.hidden_state.run_grid import planned_runs

TSLIB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'third_party', 'tslib')


def test_vendored_tslib_unchanged():
    """third_party/tslib/ files match SHA256SUMS (including the import changes listed in the README)."""
    with open(os.path.join(TSLIB, 'SHA256SUMS')) as f:
        want = dict(reversed(line.split()) for line in f if line.strip())
    for rel, digest in want.items():
        with open(os.path.join(TSLIB, rel), 'rb') as f:
            assert hashlib.sha256(f.read()).hexdigest() == digest, rel
    on_disk = {os.path.relpath(os.path.join(d, n), TSLIB) for d, _, ns in os.walk(TSLIB) for n in ns
               if n.endswith('.py') or n == 'LICENSE'}
    assert on_disk == {r.removeprefix('./') for r in want}


def _x(k, B=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 3, (B, k, D.OBS_DIM), generator=g).float()
    mask = torch.ones(B, k, dtype=torch.bool)
    mask[0, :k - 1] = False
    return torch.where(mask[..., None], x, torch.zeros(())), mask


@pytest.mark.parametrize('k', [4, 8, 32])
def test_patchtst_is_tslib_classification_without_norm(k):
    """Forward pass of the adapter = TSLib's classification with instance normalisation removed (T1)."""
    torch.manual_seed(0)
    model = build_model('patchtst', k).eval()
    x, mask = _x(k)
    with torch.no_grad():
        mean = x.mean(1, keepdim=True)
        normed = (x - mean) / torch.sqrt(torch.var(x - mean, dim=1, keepdim=True, unbiased=False) + 1e-5)
        torch.testing.assert_close(model(normed, mask).reshape(len(x), -1), model.net.classification(x, mask.float()))


def test_k_dependent_settings():
    """Validity matrix: PatchTST patch/stride, TimesNet top_k, TimeMixer number of scales."""
    assert [(build_model('patchtst', k).patch_len, build_model('patchtst', k).stride) for k in [4, 8, 16, 32]] == \
        [(2, 1), (4, 2), (8, 4), (16, 8)]
    assert (build_model('patchtst', 32, patch_ratio=0.25).patch_len, build_model('patchtst', 8, patch_ratio=0.25).patch_len) == (8, 2)
    assert [build_model('timesnet', k, top_k=5).top_k for k in [4, 8, 32]] == [1, 4, 5]
    assert [build_model('timemixer', k).down_sampling_layers for k in [2, 4, 8, 16, 32, 64]] == [0, 1, 2, 3, 3, 3]


def test_not_applicable_cells():
    assert set(NOT_APPLICABLE) == {('patchtst', 1), ('timesnet', 1), ('timemixer', 1)}
    names = [r[3] for r in planned_runs(['patchtst', 'timemixer', 'tsmixer'], [1, 8], [0])]
    assert names == ['patchtst_k8_s0', 'timemixer_k8_s0', 'tsmixer_k1_s0', 'tsmixer_k8_s0']


@pytest.mark.parametrize('readout', ['last', 'cls'])
@pytest.mark.parametrize('causal', [False, True])
def test_transformer_ignores_padding(readout, causal):
    """The content of padding steps does not change the output (key_padding_mask; CLS included)."""
    torch.manual_seed(0)
    model = build_model('transformer', 8, readout=readout, causal=causal).eval()
    x, mask = _x(8)
    noisy = torch.where(mask[..., None], x, torch.randn_like(x))
    with torch.no_grad():
        torch.testing.assert_close(model(x, mask), model(noisy, mask))


def test_compile_path(store, monkeypatch):
    """compile: true → the training forward pass uses the module returned by torch.compile; dynamo is reset per run.
    An identity wrapper stands in for real compilation: the result must equal the one with compile off."""
    from cybermetaformer.hidden_state.train import train
    calls = {'compile': [], 'reset': 0}

    def fake_compile(model, **kw):
        calls['compile'].append(kw)
        return model

    def fake_reset():
        calls['reset'] += 1
    monkeypatch.setattr(torch, 'compile', fake_compile)
    monkeypatch.setattr(torch._dynamo, 'reset', fake_reset)
    cfg = {'model': 'tsmixer', 'k': 4, 'train_episodes': 32, 'val_episodes': 16, 'epochs': 1, 'device': 'cpu',
           'model_args': {'d_model': 16, 'layers': 1}}
    _, h_eager, _ = train(cfg, store, log=lambda s: None)
    _, h_comp, _ = train({**cfg, 'compile': True}, store, log=lambda s: None)
    assert calls == {'compile': [{'dynamic': False}], 'reset': 1}
    assert h_eager[0]['loss'] == h_comp[0]['loss']
    assert h_eager[0]['val_host_macro_f1'] == h_comp[0]['val_host_macro_f1']


def test_save_run_writes_loadable_weights(tmp_path):
    from cybermetaformer.hidden_state.train import save_run
    torch.manual_seed(0)
    model = build_model('itransformer', 4, d_model=16, heads=2, layers=1, ff=32)
    save_run(str(tmp_path), {'model': 'itransformer', 'k': 4}, None, {}, {}, model=model)
    again = build_model('itransformer', 4, d_model=16, heads=2, layers=1, ff=32)
    again.load_state_dict(torch.load(tmp_path / 'model.pt'))
    x, mask = _x(4)
    with torch.no_grad():
        np.testing.assert_array_equal(model.eval()(x, mask).numpy(), again.eval()(x, mask).numpy())


def test_m2_budget_protocol():
    """M2 budget: 15 trials in stage 2; PatchTST d 32, TimesNet d 16, ≤ 2 layers, default patch ratio."""
    import optuna
    from cybermetaformer.hidden_state.hpo import SPACES, protocol
    assert [protocol(s, False)['n_trials'] for s in ['asama1', 'asama2', 'asama2_smoke']] == [25, 15, 15]
    assert protocol('asama2_smoke', True)['n_trials'] == 2
    sp, st = (optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0)) for _ in range(2))
    for _ in range(30):
        pt, tn = SPACES['patchtst'](sp.ask())['model_args'], SPACES['timesnet'](st.ask())['model_args']
        assert pt['d_model'] == 32 and pt['layers'] <= 2 and 'patch_ratio' not in pt
        assert tn['d_model'] == 16 and tn['layers'] <= 2


def test_m1_refs_select_by_val():
    """Stage 2's M1 references are chosen on val; test/OOD do not enter the choice; full history is not a window."""
    from cybermetaformer.hidden_state.aggregate import m1_refs

    def g(model, k, val, test):
        return {'model': model, 'k': k, 'seeds': [0, 1], 'val_host_macro_f1': {'mean': val, 'std': 0.0, 'n': 2},
                'test_host_macro_f1': {'mean': test, 'std': 0.0, 'n': 2}}
    ref = {f'{m}_k{k}': g(m, k, v, t) for m, k, v, t in [
        ('mlp', 8, 0.93, 0.99), ('gru', 8, 0.94, 0.90), ('xgb_tf', 64, 0.95, 0.95), ('gru', 32, 0.96, 0.96),
        ('gru_full', 101, 0.98, 0.98), ('persist', 8, 0.99, 0.99)]}
    R = m1_refs(ref)
    assert R['best'][8]['model'] == 'gru'                         # best on val, not the MLP best on test
    assert R['best'][64]['model'] == 'xgb_tf' and 'persist' not in {b['model'] for b in R['best'].values()}
    assert R['best_window']['k'] == 32 and R['gru_full']['k'] == 101 and set(R['mlp']) == {8}


def test_cd_summary_best_first():
    """CD summary: rank 1 = best first (autorank's `rankdf` is reversed); the episode-F1 mean is the column mean."""
    import pandas as pd
    from autorank import autorank
    from cybermetaformer.hidden_state.cd import summary_rows
    rng = np.random.default_rng(0)
    base = rng.random(200)
    df = pd.DataFrame({'bad': base, 'good': base + 0.2, 'mid': base + 0.1})
    rows = summary_rows(df, autorank(df, verbose=False, order='descending', force_mode='nonparametric'))
    assert [r['model'] for r in rows] == ['good', 'mid', 'bad']
    assert rows[0]['meanrank'] == pytest.approx(1.0) and rows[0]['mean'] == pytest.approx(df['good'].mean())


def test_pareto_front():
    """Low cost and high accuracy: dominated points are not on the frontier; at equal cost only the better one."""
    from cybermetaformer.hidden_state.efficiency import pareto_front
    pts = [(1.0, 0.80), (2.0, 0.85), (2.0, 0.84), (3.0, 0.83), (0.5, 0.70), (4.0, 0.90)]
    assert pareto_front(pts) == [4, 0, 1, 5]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS not available')
def test_patchtst_mps_large_batch_matches_cpu():
    """MPS Conv1d is wrong for batch > 2^16 (torch 2.14.1); PatchTST's patched convolution must match the CPU."""
    from cybermetaformer.hidden_state.models.tslib_adapter import _ChunkedConv1d
    torch.manual_seed(0)
    cpu = build_model('patchtst', 8, d_model=16, heads=4, layers=1, ff=32).eval()
    assert all(isinstance(m, _ChunkedConv1d) for l in cpu.net.encoder.attn_layers for m in (l.conv1, l.conv2))
    assert any(k.endswith('conv1.weight') for k in cpu.state_dict())
    mps = build_model('patchtst', 8, d_model=16, heads=4, layers=1, ff=32)
    mps.load_state_dict(cpu.state_dict())
    mps = mps.to('mps').eval()
    x, mask = torch.rand(900, 8, D.OBS_DIM), torch.ones(900, 8, dtype=torch.bool)     # 900·78 = 70 200 > 2^16
    xm = x.to('mps').requires_grad_(True)
    xc = x.clone().requires_grad_(True)
    out_c, out_m = cpu(xc, mask), mps(xm, mask.to('mps'))
    torch.testing.assert_close(out_m.cpu(), out_c, atol=1e-4, rtol=1e-4)
    out_c.pow(2).sum().backward()
    out_m.pow(2).sum().backward()
    torch.testing.assert_close(mps.net.encoder.attn_layers[0].conv1.weight.grad.cpu(),
                               cpu.net.encoder.attn_layers[0].conv1.weight.grad, atol=1e-3, rtol=1e-3)

"""Episode-level paired comparison: vectorised host macro-F1, slice counts, paired bootstrap, Holm, compare.py."""
import json

import numpy as np
import pytest

from cybermetaformer.hidden_state import compare as Cmp
from cybermetaformer.hidden_state import metrics as M


def _episodes(seed, E=60, T=100, H=12, noise=0.2):
    rng = np.random.default_rng(seed)
    y = rng.choice(3, size=(E, T, H), p=[0.7, 0.05, 0.25])
    y[..., 0] = 0                                                 # a host never compromised, like def
    p = np.where(rng.random((E, T, H)) < 1 - noise, y, rng.choice(3, size=(E, T, H)))
    alert = rng.random((E, T, H)) < 0.4
    age = np.where(rng.random((E, T, H)) < 0.2, -1, rng.integers(1, 90, (E, T, H)))
    cause = np.where((y > 0) & ~alert, rng.integers(1, 4, (E, T, H)), 0)
    return y, p, alert, age, cause


@pytest.mark.parametrize('seed', range(3))
def test_host_macro_f1_batch_matches_scalar(seed):
    y, p, *_ = _episodes(seed)
    C = M.episode_confusion(y, p)                                 # (E, H, 3, 3): each episode is one "sample"
    C[0, 5, 1] = 0                                                # no user class on one host
    batch = M.host_macro_f1_batch(C)
    ref = [M.host_macro_f1_from_confusion(c) for c in C]
    np.testing.assert_allclose(batch, ref, rtol=0, atol=1e-12)


def test_episode_slice_counts_match_reports():
    y, p, alert, age, cause = _episodes(0)
    names3 = ['yok', 'kacak_5', 'green_maske', 'remove_maske']
    counts, names = M.episode_slice_counts(y, p, alert, age, cause, names3)
    assert counts.shape == (len(y), len(names), 2)
    tot = counts.sum(0)
    rec = dict(zip(names, tot[:, 1] / tot[:, 0]))
    sm = M.silent_metrics(y, p, alert)
    assert tot[0, 0] == sm['n_silent']
    assert rec['sessiz'] == pytest.approx(sm['silent_recall'], abs=1e-12)
    for b, v in M.alert_age_recall(y, p, alert, age).items():
        assert rec[b] == pytest.approx(v['recall'], abs=1e-12)
    for c, v in M.cause_recall(p, cause, names3).items():
        assert rec[c] == pytest.approx(v['recall'], abs=1e-12)


def _f1_stat(C):
    return lambda W: M.host_macro_f1_batch((W['x'] @ C.reshape(len(C), -1)).reshape(-1, *C.shape[1:]))


def test_paired_bootstrap_identical_models():
    y, p, *_ = _episodes(1)
    C = M.episode_confusion(y, p)
    r = M.paired_bootstrap(_f1_stat(C), _f1_stat(C), {'x': len(C)}, n_boot=300)
    assert r['diff'] == 0 and r['lo'] == 0 and r['hi'] == 0 and r['p'] == 1.0


def test_paired_bootstrap_detects_difference():
    y, p_good, *_ = _episodes(2, noise=0.1)
    _, p_bad, *_ = _episodes(2, noise=0.3)                        # same y (same seed), noisier prediction
    Ca, Cb = M.episode_confusion(y, p_good), M.episode_confusion(y, p_bad)
    r = M.paired_bootstrap(_f1_stat(Ca), _f1_stat(Cb), {'x': len(Ca)}, n_boot=500)
    assert r['diff'] == pytest.approx(M.host_macro_f1_from_confusion(Ca.sum(0))
                                      - M.host_macro_f1_from_confusion(Cb.sum(0)), abs=1e-12)
    assert 0 < r['lo'] <= r['diff'] <= r['hi']
    assert r['p'] == pytest.approx(2 / 501)                       # a > b in every sample


def test_holm():
    assert M.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert M.holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])


def _run(seed, noise, h='abc'):
    out = {'seed': seed, 'conf': {}, 'silent': {}, 'hash': {'test': h, 'ood_test': h}}
    for s, off in (('test', 0), ('ood_test', 100)):
        y, p, alert, age, _ = _episodes(off + seed, noise=noise)
        out['conf'][s] = M.episode_confusion(y, p).astype(np.uint8)
        out['silent'][s] = M.episode_slice_counts(y, p, alert, age)[0][:, 0].astype(np.uint16)
    return out


def test_compare_rq3_criteria():
    a = [_run(s, 0.1) for s in range(2)]
    b = [_run(s, 0.3) for s in range(2)]
    res, has_silent = Cmp.compare(a, b, n_boot=200)
    assert has_silent and list(res) == ['ood_host_macro_f1', 'ood_rel_drop', 'ood_silent_recall']
    assert res['ood_host_macro_f1']['a_better'] and not res['ood_host_macro_f1']['b_better']
    assert res['ood_host_macro_f1']['a'] == pytest.approx(np.mean(
        [M.host_macro_f1_from_confusion(r['conf']['ood_test'].sum(0).astype(np.int64)) for r in a]), abs=1e-12)
    assert all(v['p_holm'] >= v['p'] for v in res.values())
    b[0]['silent'] = None                                         # old run: no slice file → silent recall is skipped
    res, has_silent = Cmp.compare(a, b, n_boot=50)
    assert not has_silent and 'ood_silent_recall' not in res
    b[1]['hash'] = {'test': 'abc', 'ood_test': 'baska'}
    with pytest.raises(AssertionError):
        Cmp.compare(a, b, n_boot=50)


def test_compare_selected_criteria():
    """Stage 2: test criteria are selectable and returned in order; the silent one is skipped without a slice file."""
    a = [_run(s, 0.1) for s in range(2)]
    b = [_run(s, 0.3) for s in range(2)]
    res, _ = Cmp.compare(a, b, n_boot=100, criteria=['test_host_macro_f1', 'test_silent_recall'])
    assert list(res) == ['test_host_macro_f1', 'test_silent_recall'] and res['test_host_macro_f1']['a_better']
    b[0]['silent'] = None
    res, _ = Cmp.compare(a, b, n_boot=50, criteria=['test_host_macro_f1', 'test_silent_recall'])
    assert list(res) == ['test_host_macro_f1']
    assert Cmp._resolve('asama2', 'asama1/gru_k32') == ('asama1', 'gru_k32', None)
    assert Cmp._resolve('asama2', 'tsmixer_k8') == ('asama2', 'tsmixer_k8', None)
    assert Cmp._resolve('asama2', 'asama3/cybermetaformer_k101@0,1,2') == ('asama3', 'cybermetaformer_k101', (0, 1, 2))


def test_load_runs_seed_pin(tmp_path):
    """M4: `@0,1,2` reads only those seeds; a missing seed raises an error."""
    for s in range(5):
        d = tmp_path / f'm_k101_s{s}'
        d.mkdir()
        (d / 'metrics.json').write_text(json.dumps({'reports': {'test': {'episodes_hash': 'h'}}}))
        np.savez(d / 'episode_confusion.npz', test=np.zeros((2, 12, 3, 3)))
    assert [r['seed'] for r in Cmp.load_runs(str(tmp_path), 'm_k101')] == [0, 1, 2, 3, 4]
    assert [r['seed'] for r in Cmp.load_runs(str(tmp_path), 'm_k101', (0, 1, 2))] == [0, 1, 2]
    with pytest.raises(SystemExit):
        Cmp.load_runs(str(tmp_path), 'm_k101', (0, 7))


def _run_m3(seed, noise, h='abc'):
    """Stage 3 run: all slices and the shift episodes of (ii)."""
    out = {'seed': seed, 'conf': {}, 'silent': {}, 'slices': {'sessiz': {}, 'never': {}}, 'hash': {}}
    for s, off in (('test', 0), ('ood_test', 100), ('test_shift', 200), ('ood_test_shift', 300)):
        y, p, alert, age, _ = _episodes(off + seed, noise=noise)
        counts, names = M.episode_slice_counts(y, p, alert, age)
        out['conf'][s] = M.episode_confusion(y, p).astype(np.uint8)
        for n in ('sessiz', 'never'):
            out['slices'][n][s] = counts[:, names.index(n)].astype(np.uint16)
        out['silent'][s] = out['slices']['sessiz'][s]
        out['hash'][s] = h
    return out


def test_compare_m3_criteria():
    """Stage 3: the never-alerted slice and the shift episodes; the test/OOD bootstrap is the same as in M2."""
    a = [_run_m3(s, 0.1) for s in range(2)]
    b = [_run_m3(s, 0.3) for s in range(2)]
    crit = ['ood_never_recall', 'test_shift_host_macro_f1', 'ood_test_shift_silent_recall']
    res, _ = Cmp.compare(a, b, n_boot=200, criteria=crit)            # Holm × 3: the smallest p is 3 · 2 / 201
    assert list(res) == crit and res['test_shift_host_macro_f1']['a_better']
    rec = lambda c: c[:, 1].sum() / c[:, 0].sum()                  # noqa: E731
    assert res['ood_never_recall']['a'] == pytest.approx(np.mean([rec(r['slices']['never']['ood_test']) for r in a]))
    assert res['ood_test_shift_silent_recall']['b'] == pytest.approx(
        np.mean([rec(r['slices']['sessiz']['ood_test_shift']) for r in b]))
    m3, _ = Cmp.compare(a, b, n_boot=100, criteria=['ood_host_macro_f1'])
    m2, _ = Cmp.compare([_run(s, 0.1) for s in range(2)], [_run(s, 0.3) for s in range(2)], n_boot=100,
                        criteria=['ood_host_macro_f1'])
    assert m3['ood_host_macro_f1'] == m2['ood_host_macro_f1']       # extra episodes leave test/OOD as is
    b[1]['hash']['test_shift'] = 'baska'
    with pytest.raises(AssertionError):
        Cmp.compare(a, b, n_boot=50, criteria=['test_shift_host_macro_f1'])
    Cmp.compare(a, b, n_boot=50, criteria=['ood_host_macro_f1'])     # no shift criterion → no shift-summary check
    for r in b:
        r['slices'] = None                                          # old run: silent slice only
    res, _ = Cmp.compare(a, b, n_boot=50, criteria=['ood_silent_recall', 'ood_never_recall'])
    assert list(res) == ['ood_silent_recall']

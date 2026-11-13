"""data.py ↔ loaders.detection_view equivalence and split protection."""
import numpy as np
import pytest
import torch

from minicage.dataset import loaders as L
from cybermetaformer.hidden_state import data as D


def _all_windows(wd, k, bs=997):
    xs, ys = [], []
    for ep, t in D.eval_indices(wd.n, wd.T, bs):
        x, mask = wd.windows(ep, t, k)
        xs.append(x.reshape(len(ep), -1).numpy())
        ys.append(wd.targets(ep, t).numpy())
    return np.concatenate(xs), np.concatenate(ys)


@pytest.mark.parametrize('k', [1, 4, 8, 16, 32, 64, 100])
def test_windows_match_detection_view(store, raw, small_idx, k):
    X, y, _ = L.detection_view(raw, small_idx, k)
    Xd, yd = _all_windows(D.WindowData(store, small_idx), k)
    assert Xd.dtype == X.dtype
    np.testing.assert_array_equal(Xd, X)
    np.testing.assert_array_equal(yd, y)


def test_padding_mask(store, small_idx):
    wd = D.WindowData(store, small_idx)
    x, mask = wd.windows(torch.tensor([0, 1]), torch.tensor([1, 100]), 8)
    assert mask[0].tolist() == [False] * 6 + [True] * 2      # t=1: obs0, obs1 real, 6 padding
    assert mask[1].all()
    assert (x[0, :6] == 0).all()


def test_padding_mask_partial(store, small_idx):
    wd = D.WindowData(store, small_idx)
    x, mask = wd.windows(torch.tensor([0]), torch.tensor([5]), 8)
    assert mask[0].tolist() == [False, False] + [True] * 6   # obs[-2], obs[-1] padding; obs0..obs5 real
    np.testing.assert_array_equal(x[0, 2:].numpy(), store.obs[small_idx[0], 0:6])


def test_episode_mode(store, small_idx):
    wd = D.WindowData(store, small_idx)
    obs, lab = wd.episodes(torch.tensor([3, 7]))
    np.testing.assert_array_equal(obs.numpy(), store.obs[small_idx[[3, 7]]])
    np.testing.assert_array_equal(lab.numpy(), store.labels[small_idx[[3, 7]]])
    # the last position of the full history = the window k=T+1
    x, mask = wd.windows(torch.tensor([3]), torch.tensor([wd.T]), wd.T + 1)
    assert mask.all()
    np.testing.assert_array_equal(x[0].numpy(), obs[0].numpy())


def test_host_tokens(store, small_idx):
    x = torch.as_tensor(store.obs[small_idx[:5]]).float()          # (5, 101, 78)
    tok = D.to_host_tokens(x)
    assert tok.shape == (5, 101, 13, 6)
    for h in range(13):
        np.testing.assert_array_equal(tok[..., h, :4].numpy(), x[..., 4 * h:4 * h + 4].numpy())
        np.testing.assert_array_equal(tok[..., h, 4].numpy(), x[..., 52 + h].numpy())
        np.testing.assert_array_equal(tok[..., h, 5].numpy(), x[..., 65 + h].numpy())


def test_scale(store, small_idx):
    x = torch.as_tensor(store.obs[small_idx[:3]])
    s = D.scale_obs(x)
    assert torch.equal(s[..., :52], x[..., :52].float())
    assert torch.allclose(s[..., 52:65], x[..., 52:65].float() / 2)
    assert torch.allclose(s[..., 65:], x[..., 65:].float() / 4)
    assert torch.equal(D.scale_obs(x, 'none'), x.float())


def test_split_guard(store):
    with pytest.raises(D.OODAccessError):
        store.split('ood_test')
    with store.final_evaluation():
        ood = store.split('ood_test')
    assert len(ood) == 4000
    with pytest.raises(D.OODAccessError):
        store.split('ood_test')
    for name in ['train', 'val', 'test']:
        store.assert_no_ood(store.split(name))
    with pytest.raises(D.OODAccessError):
        store.assert_no_ood(np.concatenate([store.split('train')[:5], ood[:1]]))


def test_batch_indices_cover_all():
    seen = torch.cat([ep * 100 + (t - 1) for ep, t in D.batch_indices(7, 100, 64)])
    assert sorted(seen.tolist()) == list(range(700))


def test_batch_indices_subsample():
    g = torch.Generator().manual_seed(0)
    pairs = [(e, t) for ep, tt in D.batch_indices(9, 100, 50, steps_per_episode=10, generator=g)
             for e, t in zip(ep.tolist(), tt.tolist())]
    assert len(pairs) == 90 and len(set(pairs)) == 90
    assert all(sum(e == i for e, _ in pairs) == 10 for i in range(9))


def test_permutation_matches_baselines_fit():
    """baselines.fit: torch.randperm(len(X)); we make the same call → same batch order with the same seed."""
    torch.manual_seed(0)
    ref = torch.randperm(4000 * 100)
    torch.manual_seed(0)
    ours = torch.cat([ep * 100 + (t - 1) for ep, t in D.batch_indices(4000, 100, 1024)])
    assert torch.equal(ref, ours)


def test_subsample_matches_baselines(splits):
    baselines = pytest.importorskip('minicage.dataset.baselines')   # needs CybORG → runs only on the VM
    tr = splits['train']
    np.testing.assert_array_equal(D.subsample_episodes(tr, 4000), tr[baselines.subsample(len(tr), 4000)])

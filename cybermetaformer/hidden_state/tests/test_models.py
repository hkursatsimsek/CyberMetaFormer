"""Model shape tests and the single-batch memorisation (overfit) test."""
import pytest
import torch

from cybermetaformer.hidden_state import data as D
from cybermetaformer.hidden_state.models import NOT_APPLICABLE, REGISTRY, build_model


@pytest.mark.parametrize('name', sorted(REGISTRY))
@pytest.mark.parametrize('k', [1, 4, 8, 32])
def test_output_shape(name, k):
    if (name, k) in NOT_APPLICABLE:
        with pytest.raises(ValueError):
            build_model(name, k)
        return
    torch.manual_seed(0)
    model = build_model(name, k)
    x = torch.randn(5, k, D.OBS_DIM)
    mask = torch.ones(5, k, dtype=torch.bool)
    mask[0, :k - 1] = False                      # padding at the episode start
    out = model(x, mask)
    assert out.shape == (5, len(D.LABEL_HOSTS), D.N_CLASSES)
    assert torch.isfinite(out).all()


# Heavy M2 models are built small in this test (default sizes take minutes on the CPU); the aim is that the
# gradient flows and the model can memorise one batch.
SMALL = {'patchtst': {'d_model': 32, 'heads': 4, 'layers': 1, 'ff': 64},
         'itransformer': {'d_model': 32, 'heads': 4, 'layers': 1, 'ff': 64},
         'timesnet': {'d_model': 16, 'ff': 16, 'layers': 1, 'kernels': 3}}
# PatchTST splits each sample into 78 series; the backward pass of the 1×1 Conv1d in the TSLib encoder loops over
# samples on the CPU (not on MPS), so the batch is small.
BATCH = {'patchtst': 16}


@pytest.mark.parametrize('name', sorted(REGISTRY))
def test_overfit_one_batch(store, small_idx, name):
    torch.manual_seed(0)
    k = 8
    wd = D.WindowData(store, small_idx)
    ep, t = next(D.batch_indices(wd.n, wd.T, BATCH.get(name, 64)))
    x, mask = wd.windows(ep, t, k)
    x, y = D.scale_obs(x), wd.targets(ep, t)
    model = build_model(name, k, **SMALL.get(name, {}))
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(300):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x, mask).reshape(-1, 3), y.reshape(-1))
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        acc = (model(x, mask).argmax(-1) == y).float().mean().item()
    assert acc > 0.97, f'{name}: could not memorise one batch (accuracy {acc:.3f})'

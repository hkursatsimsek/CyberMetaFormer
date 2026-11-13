"""MPS validation: are the forward passes and parameter gradients of the models on MPS the same as on CPU? (2026-10-21)

That all M2 runs of PatchTST were trained with a broken gradient was noticed through the completeness check of IG.
This script extends that check to all models and records two MPS bugs separately:

1. **Models.** For every learning window model, the seed 0 config (M1: MLP, 1D-CNN, LSTM, GRU; M2: all six), k ∈ {8, 32,
   64}. A training batch (1024 windows from the train episodes), the same initial weights, eval mode (dropout off).
   CPU eager is the reference; MPS eager and `torch.compile` as in training (M2, except TimesNet). Measures: the largest
   absolute difference of the output, the relative difference and cosine of all parameter gradients.
2. **Conv1d, batch 2^16 limit.** Batches of 65 536 and 65 537 rows; forward, input and weight gradients.
3. **AvgPool1d, permuted input.** The layout in TimeMixer's downsampling; forward and input gradient.

    python -m cybermetaformer.hidden_state.mps_check     # asama2/mps_dogrulama.json

4. **Episode mode (M3; `--m3`).** The full-history GRU and the main CyberMetaFormer config together with variants that
   cover all temporal/host mixer types (`M3_VARIANTS`); 16 train episodes, `forward_episode`, the same measures. Run before
   an M3 model is trained on MPS for the first time.

    python -m cybermetaformer.hidden_state.mps_check --m3   # asama3/mps_dogrulama.json
"""
import argparse
import glob
import json
import os

import torch
import yaml

from cybermetaformer.hidden_state import data as D
from cybermetaformer.hidden_state.models import build_model

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), 'results', 'hidden_state')
WINDOW_M1 = ('mlp', 'cnn1d', 'lstm', 'gru')
NO_COMPILE = ('timesnet',)                 # train.py: TimesNet without compile (FFT)


def _diff(a, b):
    return float((a - b).abs().max())


def compare_model(cfg, x, m, y, compile_mps):
    """→ {device mode: {out, grad_rel, grad_cos}}; reference CPU eager. If m is None, episode mode (forward_episode)."""
    name, k = cfg['model'], int(cfg['k'])
    torch.manual_seed(0)
    init = build_model(name, k, **(cfg.get('model_args') or {})).state_dict()
    res, ref = {}, None
    for dev, comp in [('cpu', False), ('mps', False)] + ([('mps', True)] if compile_mps else []):
        model = build_model(name, k, **(cfg.get('model_args') or {}))
        model.load_state_dict(init)
        model = model.to(dev).eval()
        fwd = torch.compile(model, dynamic=False) if comp else model
        out = model.forward_episode(x.to(dev)) if m is None else fwd(x.to(dev), m.to(dev))
        torch.nn.functional.cross_entropy(out.reshape(-1, D.N_CLASSES), y.to(dev).reshape(-1)).backward()
        g = torch.cat([p.grad.flatten().cpu() for p in model.parameters() if p.grad is not None])
        if ref is None:
            ref = (out.detach().cpu(), g)
            continue
        res['mps_compile' if comp else 'mps_eager'] = {
            'out_max_abs': _diff(out.detach().cpu(), ref[0]), 'grad_rel': float((g - ref[1]).norm() / ref[1].norm()),
            'grad_cos': float(torch.nn.functional.cosine_similarity(g, ref[1], dim=0))}
    return res


def check_models():
    store = D.EpisodeStore()
    wd = D.WindowData(store, D.subsample_episodes(store.split('train'), 50, 0), device='cpu')
    ep, t = next(D.batch_indices(wd.n, wd.T, 1024, generator=torch.Generator().manual_seed(0)))
    y = wd.targets(ep, t)
    rows = []
    for stage in ('asama1', 'asama2'):
        for path in sorted(glob.glob(os.path.join(RESULTS, stage, '*_s0', 'config.yaml'))):
            cfg = yaml.safe_load(open(path))
            k = int(cfg['k'])
            if k not in (8, 32, 64) or (stage == 'asama1' and cfg['model'] not in WINDOW_M1):
                continue
            x, m = wd.windows(ep, t, k)
            x = D.scale_obs(x, cfg.get('scale', 'global'))
            compile_mps = bool(cfg.get('compile')) and cfg['model'] not in NO_COMPILE
            r = {'stage': stage, 'model': cfg['model'], 'k': k, **compare_model(cfg, x, m, y, compile_mps)}
            rows.append(r)
            print(r, flush=True)
    return rows


def _op(make, x, gout):
    res = {}
    for dev in ('cpu', 'mps'):
        torch.manual_seed(1)
        mod = make().to(dev)
        xx = x.clone().to(dev).requires_grad_(True)
        out = mod(xx)
        grads = torch.autograd.grad(out, [xx] + list(mod.parameters()), gout.to(dev))
        res[dev] = [out.detach().cpu()] + [g.cpu() for g in grads]
    names = ['out', 'grad_input'] + [f'grad_{n}' for n, _ in make().named_parameters()]
    return {n: float((a - b).abs().max() / (a.abs().max() + 1e-12)) for n, a, b in zip(names, res['cpu'], res['mps'])}


def check_ops():
    torch.manual_seed(0)
    out = {}
    for n in (2 ** 16, 2 ** 16 + 1):
        out[f'conv1d_k1_batch{n}'] = _op(lambda: torch.nn.Conv1d(32, 64, 1), torch.randn(n, 32, 4), torch.randn(n, 64, 4))
    x, g = torch.randn(64, 32, 78), torch.randn(64, 78, 16)

    class Pool(torch.nn.Module):
        def __init__(self, contiguous):
            super().__init__()
            self.contiguous = contiguous

        def forward(self, x):
            z = x.permute(0, 2, 1)
            return torch.nn.functional.avg_pool1d(z.contiguous() if self.contiguous else z, 2)

    out['avgpool1d_permuted'] = _op(lambda: Pool(False), x, g)
    out['avgpool1d_contiguous'] = _op(lambda: Pool(True), x, g)
    for key, v in out.items():
        print(key, v, flush=True)
    return out


M3_VARIANTS = [                            # patch to the model_args of configs/cybermetaformer.yaml
    {},
    {'temporal': ['conv', 'attn']},
    {'temporal': ['conv', 'attn'], 'window': 32},
    {'temporal': ['conv', 'gru']},
    {'temporal': ['pool', 'mlp'], 'host_mixer': 'gcn', 'age': False, 'pos': 'none'},
    {'temporal': ['conv', 'ssd'], 'host_mixer': 'attn'},
    {'temporal': ['conv', 'ssd'], 'host_mixer': 'topo', 'topo_mask': True},
    {'temporal': ['conv', 'ssd'], 'host_mixer': 'none', 'tokens': 'flat'},
]


def check_episode_models():
    store = D.EpisodeStore()
    wd = D.WindowData(store, D.subsample_episodes(store.split('train'), 16, 0), device='cpu')
    obs, y = wd.episodes(torch.arange(wd.n))
    x = D.scale_obs(obs)
    base = yaml.safe_load(open(os.path.join(HERE, 'configs', 'cybermetaformer.yaml')))
    cfgs = [yaml.safe_load(open(os.path.join(HERE, 'configs', 'gru_full.yaml')))]
    cfgs += [{**base, 'model_args': {**base['model_args'], **v}} for v in M3_VARIANTS]
    rows = []
    for cfg in cfgs:
        r = {'model': cfg['model'], 'model_args': cfg.get('model_args'),
             **compare_model(cfg, x, None, y_episode(y), compile_mps=False)}
        rows.append(r)
        print(r, flush=True)
    return rows


def y_episode(y):
    """WindowData.episodes targets (B, T, 12) → a leading ignored row to align with forward_episode(...)[:, 1:]."""
    return torch.cat([torch.full_like(y[:, :1], -100), y], 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--m3', action='store_true', help='episode-mode models only (asama3/mps_dogrulama.json)')
    args = p.parse_args()
    if args.m3:
        res = {'torch': torch.__version__, 'episode_models': check_episode_models()}
        path = os.path.join(RESULTS, 'asama3', 'mps_dogrulama.json')
        with open(path, 'w') as f:
            json.dump(res, f, indent=1)
        print('→', path)
        return
    res = {'torch': torch.__version__, 'ops': check_ops(), 'models': check_models()}
    path = os.path.join(RESULTS, 'asama2', 'mps_dogrulama.json')
    with open(path, 'w') as f:
        json.dump(res, f, indent=1)
    print('→', path)


if __name__ == '__main__':
    main()

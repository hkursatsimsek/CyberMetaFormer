"""Compute benchmark and total budget estimate; repeated at the start of M2 with the TSLib models.

    venv_cyborg/bin/python -m cybermetaformer.hidden_state.bench --device cpu      # VM
    python -m cybermetaformer.hidden_state.bench --device mps                      # Mac (venv_ts)
    python -m cybermetaformer.hidden_state.bench --device cpu                      # Mac CPU (for comparison)
    python -m cybermetaformer.hidden_state.bench --device mps --m2                 # M2: 6 model, eager + compile

With real data batches (including the lazy window gather), the training step time, inference throughput and
single-sample latency are measured for model family × k ∈ {8, 32}; the total time is computed from the planned grid (GRID).
Output: cybermetaformer/results/hidden_state/asama0/bench_<device>_<machine>.json

`--m2`: the 6 M2 models with their base configurations (`configs/<model>.yaml` = constructor defaults), k ∈ {8, 32, 64}
(for the k=64 extension decision), training step both eager and with `torch.compile`; inference eager (as in train.py).
The budget uses the validity matrix (no N/A cells) and the faster mode per model.
Output: cybermetaformer/results/hidden_state/asama2/bench_m2_<device>_<machine>.json

`--m2-sizes`: training epoch time of the heavy pair (PatchTST, TimesNet) at the ends of the HPO space (`hpo.space_*`),
training only, in the mode of configs. Sensitivity of the budget to the size HPO will choose. Output:
cybermetaformer/results/hidden_state/asama2/bench_m2_boyut_<device>_<machine>.json

`--m3`: episode-mode (full-history) models, 16-episode batches as in `train.py`: full GRU (reference to compare with
the real run time), the main CyberMetaFormer config, the ends of the HPO space (fitted to B1 with `fit_width`) and
the ablation variants (`M3_BENCH`). Training step, val evaluation (162-episode chunks) and single-episode step
latency. Output: cybermetaformer/results/hidden_state/asama3/bench_m3_<device>_<machine>.json

`--compile-check`: compile equivalence trial. Vanilla Transformer k=8, base
configuration, full train, val only: eager s0, eager s1 (scale of the seed noise) and compile s0. Output:
cybermetaformer/results/hidden_state/asama2/compile_esdeger_<device>.json
"""
import argparse
import json
import os
import platform
import socket
import statistics
import time

import torch

from cybermetaformer.hidden_state import data as D
from cybermetaformer.hidden_state.models import NOT_APPLICABLE, build_model
from cybermetaformer.hidden_state.train import release_device_memory, resolve_device

OUT_DIR = os.path.join(D.ROOT, 'results', 'hidden_state', 'asama0')
FAMILIES = ['mlp', 'cnn1d', 'lstm', 'gru', 'transformer']
KS = [8, 32]
N_TRAIN, N_VAL = 16000 * 100, 2000 * 100          # sample counts (windows, not host-steps)
GRID = {                                           # planned experiment grid
    'k_values': [1, 4, 8, 16, 32], 'seeds': 5, 'max_epochs': 30, 'expected_epochs': 15,
    'hpo_trials': 25, 'hpo_k': [8, 32], 'hpo_train_frac': 0.25, 'hpo_max_epochs': 10, 'hpo_expected_epochs': 6,
    'm1_models': ['mlp', 'cnn1d', 'lstm', 'gru'],
    # the 6 M2 models are not vendored yet (TSLib); assumption: vanilla Transformer time × factor
    'm2_models': {'transformer': 1.0, 'patchtst': 1.0, 'itransformer': 1.0, 'tsmixer': 0.5, 'timemixer': 1.5, 'timesnet': 4.0},
}
M2_OUT_DIR = os.path.join(D.ROOT, 'results', 'hidden_state', 'asama2')
M2_FAMILIES = ['transformer', 'patchtst', 'itransformer', 'tsmixer', 'timesnet', 'timemixer']
M2_KS = [8, 32, 64]
# (model, k, model_args patch): base configuration and the ends of the HPO space (ff = d_model × ff_mult); PatchTST's
# time does not depend on k (~4 patches per channel), TimesNet's grows with k
M2_SIZES = [
    ('patchtst', 8, {}),
    ('patchtst', 8, {'d_model': 32, 'ff': 64, 'layers': 1}),
    ('patchtst', 8, {'d_model': 64, 'ff': 128}),
    ('patchtst', 8, {'patch_ratio': 0.25}),
    ('patchtst', 8, {'ff': 512, 'layers': 3, 'patch_ratio': 0.25}),
    ('timesnet', 32, {}),
    ('timesnet', 32, {'d_model': 16, 'ff': 16, 'layers': 1, 'top_k': 2}),
    ('timesnet', 32, {'d_model': 16, 'ff': 32}),
    ('timesnet', 32, {'d_model': 64, 'ff': 64}),
    ('timesnet', 32, {'d_model': 64, 'ff': 128}),
    ('timesnet', 32, {'d_model': 64, 'ff': 128, 'layers': 3}),
]


def _sync(device):
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()


def bench_one(name, k, wd, device, bs=1024, warmup=5, steps=30, compile=False, infer=True, model_args=None):
    """compile: training forward pass with torch.compile (compiled during the warm-up steps). infer=False: training only.
    model_args: instead of the constructor defaults (`--m2-sizes`)."""
    model = build_model(name, k, **(model_args or {})).to(device)
    fwd = model
    if compile:
        torch._dynamo.reset()
        fwd = torch.compile(model, dynamic=False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    it = D.batch_indices(wd.n, wd.T, bs)
    batches = [next(it) for _ in range(warmup + steps)]
    model.train()
    times = []
    for i, (ep, t) in enumerate(batches):
        _sync(device)
        t0 = time.perf_counter()
        x, mask = wd.windows(ep, t, k)
        y = wd.targets(ep, t)
        loss = torch.nn.functional.cross_entropy(fwd(D.scale_obs(x), mask).reshape(-1, 3), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        _sync(device)
        if i >= warmup:
            times.append(time.perf_counter() - t0)
    train_per_sample = statistics.median(times) / bs
    if not infer:
        return {'params': sum(p.numel() for p in model.parameters() if p.requires_grad),
                'train_s_per_sample': train_per_sample, 'epoch_minutes': train_per_sample * N_TRAIN / 60}

    model.eval()
    with torch.no_grad():
        ebs = 8192
        ep, t = next(D.eval_indices(wd.n, wd.T, ebs))
        for _ in range(2):
            x, mask = wd.windows(ep, t, k)
            model(D.scale_obs(x), mask)
        _sync(device)
        t0 = time.perf_counter()
        for _ in range(3):
            x, mask = wd.windows(ep, t, k)
            model(D.scale_obs(x), mask)
        _sync(device)
        infer_per_sample = (time.perf_counter() - t0) / (3 * ebs)
        lat = []
        x1, m1 = wd.windows(ep[:1], t[:1], k)
        x1 = D.scale_obs(x1)
        for i in range(60):
            _sync(device)
            t0 = time.perf_counter()
            model(x1, m1)
            _sync(device)
            if i >= 10:
                lat.append(time.perf_counter() - t0)
    return {'params': sum(p.numel() for p in model.parameters()),
            'train_s_per_sample': train_per_sample, 'infer_s_per_sample': infer_per_sample,
            'latency_ms_bs1': 1000 * statistics.median(lat),
            'epoch_minutes': train_per_sample * N_TRAIN / 60, 'val_eval_minutes': infer_per_sample * N_VAL / 60}


def run_hours(r, epochs, train_frac=1.0):
    return epochs * (r['epoch_minutes'] * train_frac + r['val_eval_minutes']) / 60


def budget(res):
    g = res['grid']

    def t(name, k):          # k=1,4 → k=8 measurement; k=16 → k=32 measurement (conservative)
        return res['models'][name][str(8 if k <= 8 else 32)]

    out = {}
    for label, epochs_key, hpo_key in [('ust_sinir', 'max_epochs', 'hpo_max_epochs'), ('beklenen', 'expected_epochs', 'hpo_expected_epochs')]:
        m1 = sum(run_hours(t(m, k), g[epochs_key]) * g['seeds'] for m in g['m1_models'] for k in g['k_values'])
        m2 = sum(run_hours(t('transformer', k), g[epochs_key]) * f * g['seeds']
                 for f in g['m2_models'].values() for k in g['k_values'])
        models = g['m1_models'] + list(g['m2_models'])
        hpo = sum(run_hours(t(m if m in res['models'] else 'transformer', k), g[hpo_key], g['hpo_train_frac'])
                  * g['hpo_trials'] * (g['m2_models'].get(m, 1.0)) for m in models for k in g['hpo_k'])
        out[label] = {'m1_final_saat': m1, 'm2_final_saat': m2, 'hpo_saat': hpo, 'toplam_saat': m1 + m2 + hpo}
    return out


def _m2_k(k):
    return str(8 if k <= 8 else 32 if k <= 32 else 64)          # k=1,4 → k=8 measurement; k=16 → k=32 (conservative)


def budget_m2(res):
    """M2 budget (hours): per model final (5 k × 5 seeds, excluding N/A cells), HPO (k ∈ {8, 32} × 25 trials) and the k=64
    extension (5 seeds; HPO from k=32), eager and compile separately. `secilen` = per model the final + the mode with the shorter HPO;
    eager if compile failed."""
    g, out = res['grid'], {}
    for label, ek, hk in [('beklenen', 'expected_epochs', 'hpo_expected_epochs'),
                          ('ust_sinir', 'max_epochs', 'hpo_max_epochs')]:
        rows = {}
        for m, r in res['models'].items():
            row = {}
            for mode in ('eager', 'compile'):
                if any('error' in r[_m2_k(k)][mode] for k in M2_KS):
                    continue

                def t(k, mode=mode):
                    x = r[_m2_k(k)]
                    return {'epoch_minutes': x[mode]['epoch_minutes'],
                            'val_eval_minutes': x['eager']['val_eval_minutes']}
                ks = [k for k in g['k_values'] if (m, k) not in NOT_APPLICABLE]
                row[mode] = {
                    'final': sum(run_hours(t(k), g[ek]) * g['seeds'] for k in ks),
                    'hpo': sum(run_hours(t(k), g[hk], g['hpo_train_frac']) * g['hpo_trials'] for k in g['hpo_k']),
                    'k64': run_hours(t(64), g[ek]) * g['seeds']}
            row['secilen'] = min((md for md in ('eager', 'compile') if md in row),
                                 key=lambda md: row[md]['final'] + row[md]['hpo'])
            rows[m] = row
        sel = [rw[rw['secilen']] for rw in rows.values()]
        out[label] = {'modeller': rows, 'toplam_saat': {
            'eager': sum(rw['eager']['final'] + rw['eager']['hpo'] for rw in rows.values()),
            'secilen': sum(x['final'] + x['hpo'] for x in sel), 'k64_secilen': sum(x['k64'] for x in sel)}}
    return out


def main_m2(device, wd, res):
    res['grid'] = {**GRID, 'm2_ks': M2_KS}
    for name in M2_FAMILIES:
        res['models'][name] = {}
        for k in M2_KS:
            r = {'eager': bench_one(name, k, wd, device)}
            release_device_memory()
            try:
                r['compile'] = bench_one(name, k, wd, device, compile=True, infer=False)
            except Exception as e:                                    # a model that fails to compile stays eager
                r['compile'] = {'error': repr(e)[:500]}
            release_device_memory()
            res['models'][name][str(k)] = r
            e, c = r['eager'], r['compile']
            ctxt = ('HATA ' + c['error'][:60] if 'error' in c
                    else f"{c['epoch_minutes']:6.2f} dk (×{e['epoch_minutes'] / c['epoch_minutes']:.2f})")
            print(f"{name:12s} k={k:<3d} param {e['params']:>8d}  epoch eager {e['epoch_minutes']:6.2f} dk  "
                  f"compile {ctxt}  val {e['val_eval_minutes']:5.2f} dk  step_latency {e['latency_ms_bs1']:.2f} ms",
                  flush=True)
    res['budget'] = budget_m2(res)
    for label, b in res['budget'].items():
        print(label, '  '.join(f'{k}={v:.1f}' for k, v in b['toplam_saat'].items()),
              '| kip:', ', '.join(f"{m}={rw['secilen']}" for m, rw in b['modeller'].items()))
    os.makedirs(M2_OUT_DIR, exist_ok=True)
    path = os.path.join(M2_OUT_DIR, f"bench_m2_{device}_{res['machine']}.json")
    with open(path, 'w') as f:
        json.dump(res, f, indent=2)
    print('Kaydedildi:', path)


def sizes_m2(device, wd, res):
    from cybermetaformer.hidden_state.hpo import load_base
    res['boyut'] = []
    for name, k, patch in M2_SIZES:
        base = load_base(name)
        args = {**base['model_args'], **patch}
        r = bench_one(name, k, wd, device, compile=base['compile'], infer=False, model_args=args)
        release_device_memory()
        res['boyut'].append({'model': name, 'k': k, 'compile': base['compile'], 'model_args': args, **r})
        print(f"{name:9s} k={k:<3d} {patch or 'temel'}  param {r['params']:>8d}  epoch {r['epoch_minutes']:6.2f} dk",
              flush=True)
    os.makedirs(M2_OUT_DIR, exist_ok=True)
    path = os.path.join(M2_OUT_DIR, f"bench_m2_boyut_{device}_{res['machine']}.json")
    with open(path, 'w') as f:
        json.dump(res, f, indent=2)
    print('Kaydedildi:', path)


M3_OUT_DIR = os.path.join(D.ROOT, 'results', 'hidden_state', 'asama3')
N_TRAIN_EP, N_VAL_EP, EP_BATCH = 16000, 2000, 16
# (label, model_args patch; 'fit' = width fitted to B1)
M3_BENCH = [
    ('ana', {}),
    ('hpo_min', {'layers': 2, 'mlp_ratio': 2, 'state': 8, 'heads': 2, 'kernel': 3, 'fit': True}),
    ('hpo_max', {'layers': 4, 'mlp_ratio': 4, 'state': 32, 'heads': 4, 'kernel': 6, 'fit': True}),
    ('conv_attn', {'temporal': ['conv', 'attn']}),
    ('conv_gru', {'temporal': ['conv', 'gru']}),
    ('pencere32', {'temporal': ['conv', 'attn'], 'window': 32}),
    ('pool_mlp', {'temporal': ['pool', 'mlp']}),
    ('host_attn', {'host_mixer': 'attn'}),
    ('host_gcn', {'host_mixer': 'gcn'}),
    ('host_yok', {'host_mixer': 'none'}),
    ('flat', {'host_mixer': 'none', 'tokens': 'flat'}),
]


def bench_episode(name, model_args, wd, device, warmup=5, steps=30):
    model = build_model(name, 101, **model_args).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(0)
    batches = [torch.randperm(wd.n, generator=gen)[:EP_BATCH] for _ in range(warmup + steps)]
    model.train()
    times = []
    for i, ep in enumerate(batches):
        _sync(device)
        t0 = time.perf_counter()
        obs, y = wd.episodes(ep)
        out = model.forward_episode(D.scale_obs(obs))[:, 1:]
        loss = torch.nn.functional.cross_entropy(out.reshape(-1, 3), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        _sync(device)
        if i >= warmup:
            times.append(time.perf_counter() - t0)
    step = statistics.median(times)
    model.eval()
    with torch.no_grad():
        n = min(162, wd.n)
        obs, _ = wd.episodes(torch.arange(n))
        x = D.scale_obs(obs)
        model.forward_episode(x)
        _sync(device)
        t0 = time.perf_counter()
        for _ in range(3):
            model.forward_episode(x)
        _sync(device)
        infer_ep = (time.perf_counter() - t0) / (3 * n)
    return {'params': sum(p.numel() for p in model.parameters()), 'step_s': step,
            'epoch_minutes': step * N_TRAIN_EP / EP_BATCH / 60, 'val_eval_minutes': infer_ep * N_VAL_EP / 60}


def main_m3(device, store, res):
    from cybermetaformer.hidden_state.hpo import load_base
    from cybermetaformer.hidden_state.models.cybermetaformer import fit_width
    wd = D.WindowData(store, store.split('train')[:2000], device=device)
    rows = [('gru_full', 'gru_full', load_base('gru_full')['model_args'])]
    base = load_base('cybermetaformer')['model_args']
    for label, patch in M3_BENCH:
        args = {**base, **{k: v for k, v in patch.items() if k != 'fit'}}
        if patch.get('fit'):
            args['d'] = fit_width('b1', **{k: v for k, v in args.items() if k != 'd'})
        rows.append((label, 'cybermetaformer', args))
    res['m3'] = []
    for label, name, args in rows:
        r = bench_episode(name, args, wd, device)
        release_device_memory()
        res['m3'].append({'label': label, 'model': name, 'model_args': args, **r})
        print(f"{label:10s} param {r['params']:>8d}  step {1000 * r['step_s']:6.1f} ms  epoch {r['epoch_minutes']:6.2f} min  "
              f"val {r['val_eval_minutes']:5.2f} min", flush=True)
    os.makedirs(M3_OUT_DIR, exist_ok=True)
    path = os.path.join(M3_OUT_DIR, f"bench_m3_{device}_{res['machine']}.json")
    with open(path, 'w') as f:
        json.dump(res, f, indent=2)
    print('Kaydedildi:', path)


def compile_check(device, store):
    from cybermetaformer.hidden_state.hpo import load_base
    from cybermetaformer.hidden_state.train import train
    out = {'model': 'transformer', 'k': 8, 'device': device, 'torch': torch.__version__, 'kosular': {}}
    for name, seed, comp in [('eager_s0', 0, False), ('eager_s1', 1, False), ('compile_s0', 0, True)]:
        cfg = {**load_base('transformer'), 'k': 8, 'seed': seed, 'device': device, 'compile': comp}
        _, history, info = train(cfg, store, log=lambda msg, name=name: print(f'[{name}] {msg}', flush=True))
        out['kosular'][name] = {'val_host_macro_f1': info['best_val_host_macro_f1'], 'epochs': info['epochs_run'],
                                'train_seconds': info['train_seconds'], 'history': history}
        release_device_memory()
    r = {n: v['val_host_macro_f1'] for n, v in out['kosular'].items()}
    out['fark_compile_eager'] = r['compile_s0'] - r['eager_s0']
    out['fark_seed'] = r['eager_s1'] - r['eager_s0']
    per_epoch = {n: v['train_seconds'] / v['epochs'] for n, v in out['kosular'].items()}
    out['epoch_hizi'] = per_epoch['eager_s0'] / per_epoch['compile_s0']         # the total time depends on early stopping
    print(f"val host macro-F1: {r}  compile − eager {out['fark_compile_eager']:+.4f}  "
          f"seed difference {out['fark_seed']:+.4f}  speed per epoch ×{out['epoch_hizi']:.2f}")
    os.makedirs(M2_OUT_DIR, exist_ok=True)
    path = os.path.join(M2_OUT_DIR, f'compile_esdeger_{device}.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print('Kaydedildi:', path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='auto')
    p.add_argument('--threads', type=int, default=None)
    p.add_argument('--m2', action='store_true', help="M2'nin 6 modeli, k ∈ {8, 32, 64}, eager + compile")
    p.add_argument('--m2-sizes', action='store_true', help='PatchTST and TimesNet, epoch time at the ends of the HPO space')
    p.add_argument('--m3', action='store_true', help='M3: episode-mode models (full GRU, CyberMetaFormer variants)')
    p.add_argument('--compile-check', action='store_true', help='compile equivalence trial (Transformer k=8)')
    args = p.parse_args()
    device = resolve_device(args.device)
    if args.threads:
        torch.set_num_threads(args.threads)
    store = D.EpisodeStore()
    if args.compile_check:
        return compile_check(device, store)
    if args.m3:
        res = {'device': device, 'machine': socket.gethostname(), 'platform': platform.platform(),
               'torch': torch.__version__}
        return main_m3(device, store, res)
    wd = D.WindowData(store, store.split('train')[:2000], device=device)
    res = {'device': device, 'machine': socket.gethostname(), 'platform': platform.platform(),
           'processor': platform.processor(), 'torch': torch.__version__, 'threads': torch.get_num_threads(),
           'grid': GRID, 'models': {}}
    if args.m2:
        return main_m2(device, wd, res)
    if args.m2_sizes:
        return sizes_m2(device, wd, res)
    for name in FAMILIES:
        res['models'][name] = {}
        for k in KS:
            r = bench_one(name, k, wd, device)
            res['models'][name][str(k)] = r
            print(f"{name:12s} k={k:<3d} param {r['params']:>8d}  epoch {r['epoch_minutes']:6.2f} dk  "
                  f"val {r['val_eval_minutes']:5.2f} dk  step_latency {r['latency_ms_bs1']:.2f} ms", flush=True)
    res['budget'] = budget(res)
    for label, b in res['budget'].items():
        print(label, '  '.join(f'{k}={v:.1f}' for k, v in b.items()))
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"bench_{device}_{res['machine']}.json")
    with open(path, 'w') as f:
        json.dump(res, f, indent=2)
    print('Kaydedildi:', path)


if __name__ == '__main__':
    main()

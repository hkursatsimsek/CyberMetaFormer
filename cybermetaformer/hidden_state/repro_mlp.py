"""Adım 0.7: orijinal MLP baseline'ının yeni altyapıyla yeniden üretimi.

Orijinal protokol (minicage/dataset/baselines.py): train'den 4000 bölüm (seed 0), ham girdi (ölçekleme yok),
MLP 256-256, Adam 1e-3, 5 epoch, batch 1024, tek torch seed'i (0) ve k=1 → 4 → 8 aynı RNG akışında.

İki kip:
- zincir: orijinal RNG akışı birebir izlenir → baselines.json değerleriyle aynı sonuç beklenir
- bağımsız: her (k, seed) taze tohumla → seed oynaklığı ve ±0.005 toleransının anlamı

    venv_cyborg/bin/python -m cybermetaformer.hidden_state.repro_mlp [--seeds 0 1 2]
"""
import argparse
import json
import os
import time

import torch

from cybermetaformer.hidden_state import data as D
from cybermetaformer.hidden_state.evaluate import evaluate_split
from cybermetaformer.hidden_state.train import RESULTS, train

ORIGINAL = {
    'model': 'mlp', 'model_args': {}, 'scale': 'none', 'optimizer': 'adam', 'lr': 1e-3,
    'epochs': 5, 'batch_size': 1024, 'schedule': 'none', 'train_episodes': 4000, 'train_subsample_seed': 0,
    'early_stopping': None, 'loss': 'ce', 'device': 'cpu', 'eval_batch_size': 65536,
}
KS = [1, 4, 8]
TOL = 0.005


def reference():
    rows = json.load(open(os.path.join(D.DEFAULT_DATA, 'baselines.json')))
    ref = {}
    for r in rows:
        if r['model'].startswith('MLP, k=') and r['task'].startswith('tespit'):
            k = int(r['model'].split('=')[1])
            key = 'host_macro_f1' if r['task'].startswith('tespit (host') else 'breach_f1'
            ref[(k, r['split'], key)] = r['value']
    return ref


def run_one(store, k, seed):
    cfg = {**ORIGINAL, 'k': k, 'seed': seed}
    t0 = time.time()
    model, _, info = train(cfg, store, log=lambda s: None)
    out = {'train_seconds': time.time() - t0, 'n_params': info['n_params']}
    with store.final_evaluation():
        for split in ['test', 'ood_test']:
            rep, _ = evaluate_split(model, store, split, k, 'none', 'cpu', 65536, full=False)
            out[split] = {m: rep[m] for m in ['host_macro_f1', 'breach_f1']}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    args = p.parse_args()
    torch.set_num_threads(4)                       # baselines.py ile aynı
    store, ref = D.EpisodeStore(), reference()
    res = {'protocol': ORIGINAL, 'tolerance': TOL, 'chain': {}, 'independent': {}}

    torch.manual_seed(0)                           # baselines.py modül düzeyinde bir kez
    for k in KS:
        res['chain'][k] = run_one(store, k, seed=None)
        for split in ['test', 'ood_test']:
            for m in ['host_macro_f1', 'breach_f1']:
                got, want = res['chain'][k][split][m], ref[(k, split, m)]
                print(f'zincir k={k} {split:8s} {m:13s} yeni {got:.6f}  orijinal {want:.6f}  fark {got - want:+.2e}')

    for k in KS:
        for s in args.seeds:
            res['independent'][f'{k}_{s}'] = r = run_one(store, k, seed=s)
            print(f"bağımsız k={k} seed={s}: test {r['test']['host_macro_f1']:.4f}  "
                  f"ood {r['ood_test']['host_macro_f1']:.4f}  ({r['train_seconds']:.0f} s)")

    res['reference'] = {f'{k}_{s}_{m}': v for (k, s, m), v in ref.items()}
    out_dir = os.path.join(RESULTS, 'asama0')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'mlp_repro.json'), 'w') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print('Kaydedildi:', os.path.join(out_dir, 'mlp_repro.json'))


if __name__ == '__main__':
    main()

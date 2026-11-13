"""Paper (M4, item 2): equivalence of the step mode (Algorithm 1, `CyberMetaFormer.step`) and the episode mode
(`forward_episode`), with trained weights, on all episodes of v1 test and ood_test.

For each (model, seed, setting) the two modes are compared on the same device and at the same precision. Evaluated
positions are t = 1..T × 12 label hosts (the t = 0 output is used neither in training nor in evaluation). Measured:
    max |Δz|         largest logit difference
    max |Δp|         largest difference of the softmax probabilities at the run's val temperature
    argmax diff.     number of host-steps whose decision changes / total
    floor            largest difference of the episode-mode logits of the setting from the CPU float64 episode mode
                     (only if cpu64 was also run). To compare the mode difference with the numerical precision floor.
Settings: cpu64 (arithmetic equivalence), cpu32 (the CPU latency setting of efficiency.py), mps32 (training device).
The alert-age channel is computed in float32 in both modes, then cast to the model precision; it is the same for both.

Results are cached per run (runs/<model>_s<seed>_<setting>.json). Adding a seed or setting does not rerun the old
ones; `--table` only rewrites the summary table.
Output: results/hidden_state/makale/step_equivalence/{runs/, summary.json, table.md}

    python -m cybermetaformer.hidden_state.step_equivalence --settings cpu64 cpu32 --seeds 0 1 2
    python -m cybermetaformer.hidden_state.step_equivalence --settings cpu64 cpu32 mps32 --seeds 0 1 2 3 4   # paper table
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from cybermetaformer.hidden_state import data as D
from cybermetaformer.hidden_state.ood_suite import load_model, run_dir

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'results', 'hidden_state', 'makale', 'step_equivalence')
SETTINGS = {'cpu64': ('cpu', torch.float64), 'cpu32': ('cpu', torch.float32), 'mps32': ('mps', torch.float32)}
SPLITS = {'test': 'test', 'ood': 'ood_test'}
LABEL = {'cybermetaformer': 'CyberMetaFormer (B1)', 'cybermetaformer-gru_host_topo': 'Host-token GRU + topo attention',
         'cybermetaformer-gru_host_attn': 'Host-token GRU + attention', 'cybermetaformer-b2': 'CyberMetaFormer (B2)'}
SETTING_LABEL = {'cpu64': 'CPU, FP64', 'cpu32': 'CPU, FP32', 'mps32': 'MPS, FP32'}
EP_BATCH, STEP_BATCH = 64, 512                     # intermediate (S×S) SSD tensors in the episode mode; none in the step mode


@torch.no_grad()
def logits_both(model, obs, device, dtype):
    """obs (E, T+1, 78) int8 numpy → (episode-mode, step-mode) logits, (E, T, 12, 3) float64 numpy."""
    ep, st = [], []
    for i in range(0, len(obs), EP_BATCH):
        x = D.scale_obs(torch.as_tensor(obs[i:i + EP_BATCH], device=device)).to(dtype)
        ep.append(model.forward_episode(x)[:, 1:].cpu().double())       # MPS float64 desteklemez
    for i in range(0, len(obs), STEP_BATCH):
        x = D.scale_obs(torch.as_tensor(obs[i:i + STEP_BATCH], device=device)).to(dtype)
        state = model.init_state(x.shape[0], device)
        out = [model.step(x[:, t], state) for t in range(x.shape[1])]
        st.append(torch.stack(out[1:], 1).cpu().double())
    return torch.cat(ep).numpy(), torch.cat(st).numpy()


def softmax(z, T):
    z = z / T
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def compare(ep, st, T, ref=None):
    dz = np.abs(ep - st)
    dp = np.abs(softmax(ep, T) - softmax(st, T))
    flips = int((ep.argmax(-1) != st.argmax(-1)).sum())
    res = {'max_abs_dlogit': float(dz.max()), 'mean_abs_dlogit': float(dz.mean()), 'max_abs_dp': float(dp.max()),
           'argmax_flips': flips, 'n_host_steps': int(ep.shape[0] * ep.shape[1] * ep.shape[2]),
           'n_episodes': int(ep.shape[0])}
    if ref is not None:
        res['floor_max_abs_dlogit_vs_cpu64'] = float(np.abs(ep - ref).max())
        res['argmax_flips_vs_cpu64'] = int((ep.argmax(-1) != ref.argmax(-1)).sum())
    return res


def run(models, seeds, settings, threads, max_episodes=None):
    torch.set_num_threads(threads)
    suffix = '' if max_episodes is None else f'_smoke{max_episodes}'
    store = D.EpisodeStore()
    with store.final_evaluation():
        obs = {s: store.obs[store.split(split)[:max_episodes]] for s, split in SPLITS.items()}
    os.makedirs(os.path.join(OUT, 'runs'), exist_ok=True)
    for name in models:
        for seed in seeds:
            if not os.path.exists(os.path.join(run_dir(name, seed), 'model.pt')):
                print(f'skipped (no model.pt): {name} s{seed}')
                continue
            T = json.load(open(os.path.join(run_dir(name, seed), 'metrics.json')))['reports']['temperature']
            ref = {}
            paths = {st: os.path.join(OUT, 'runs', f'{name}_s{seed}_{st}{suffix}.json') for st in settings}
            todo = [st for st in settings if not os.path.exists(paths[st])]
            if 'cpu64' in settings and 'cpu64' not in todo and todo:
                todo.append('cpu64')                                     # the reference is recomputed for the floor
            for setting in sorted(todo, key=lambda st: st != 'cpu64'):  # cpu64 first
                path = paths[setting]
                device, dtype = SETTINGS[setting]
                model, _ = load_model(name, seed, 'cpu')
                model = model.to(dtype).to(device).eval()
                t0, res = time.time(), {}
                for s in SPLITS:
                    ep, st = logits_both(model, obs[s], device, dtype)
                    if setting == 'cpu64':
                        ref[s] = ep
                    res[s] = compare(ep, st, T, ref.get(s) if setting != 'cpu64' else None)
                res.update(model=name, seed=seed, setting=setting, temperature=T, torch=torch.__version__,
                           threads=torch.get_num_threads(), seconds=round(time.time() - t0, 1))
                if os.path.exists(path):                                 # recomputed only for the floor
                    continue
                with open(path, 'w') as f:
                    json.dump(res, f, indent=2)
                print(f"{name} s{seed} {setting}: test max|Δz|={res['test']['max_abs_dlogit']:.2e} "
                      f"ood max|Δz|={res['ood']['max_abs_dlogit']:.2e} flips={res['test']['argmax_flips']}/"
                      f"{res['ood']['argmax_flips']} ({res['seconds']} s)", flush=True)


def table():
    rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(OUT, 'runs', '*.json')))
            if '_smoke' not in os.path.basename(p)]
    groups = {}
    for r in rows:
        groups.setdefault((r['model'], r['setting']), []).append(r)
    order = {s: i for i, s in enumerate(SETTINGS)}
    lines = ['| Model | Setting | Seeds | Split | Episodes | max \\|Δz\\| | max \\|Δp\\| | Argmax flips | '
             'Precision floor max \\|Δz\\| |', '|---|---|---|---|---:|---:|---:|---:|---:|']
    summary = {}
    for (name, setting), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], order[kv[0][1]])):
        seeds = sorted(r['seed'] for r in rs)
        for s in SPLITS:
            agg = {'max_abs_dlogit': max(r[s]['max_abs_dlogit'] for r in rs),
                   'max_abs_dp': max(r[s]['max_abs_dp'] for r in rs),
                   'argmax_flips': sum(r[s]['argmax_flips'] for r in rs),
                   'n_host_steps': sum(r[s]['n_host_steps'] for r in rs), 'n_episodes': rs[0][s]['n_episodes']}
            floors = [r[s]['floor_max_abs_dlogit_vs_cpu64'] for r in rs if 'floor_max_abs_dlogit_vs_cpu64' in r[s]]
            agg['floor_max_abs_dlogit_vs_cpu64'] = max(floors) if len(floors) == len(rs) else None
            summary.setdefault(name, {}).setdefault(setting, {'seeds': seeds})[s] = agg
            floor = '—' if agg['floor_max_abs_dlogit_vs_cpu64'] is None else \
                f"{agg['floor_max_abs_dlogit_vs_cpu64']:.1e}"
            lines.append(f"| {LABEL.get(name, name)} | {SETTING_LABEL[setting]} | {', '.join(map(str, seeds))} | "
                         f"{'Test' if s == 'test' else 'OOD'} | {agg['n_episodes']:,} | {agg['max_abs_dlogit']:.1e} | "
                         f"{agg['max_abs_dp']:.1e} | {agg['argmax_flips']:,} / {agg['n_host_steps']:,} | {floor} |")
    note = ('Note. Δ compares the step mode (Algorithm 1, recurrent SSD and convolution buffer) with the episode mode '
            '(parallel form, Eq. 7) on the same device and precision, over all episodes, steps t = 1, …, T and the 12 '
            'label hosts; maxima and flip counts are taken over seeds. Probabilities use each run\'s validation '
            'temperature. The precision floor is the largest difference between the episode-mode logits in that '
            'setting and the CPU FP64 episode-mode logits.')
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'table.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n\n' + note + '\n')
    with open(os.path.join(OUT, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print('\n'.join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models', nargs='+', default=['cybermetaformer'])
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    p.add_argument('--settings', nargs='+', choices=list(SETTINGS), default=['cpu64', 'cpu32'])
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--table', action='store_true', help='only rewrite the summary table')
    p.add_argument('--max-episodes', type=int, help='smoke test (runs/*_smoke*.json, not in the table)')
    a = p.parse_args()
    if not a.table:
        run(a.models, a.seeds, a.settings, a.threads, a.max_episodes)
    table()


if __name__ == '__main__':
    main()

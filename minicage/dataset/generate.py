"""MiniCAGE'den ham bölüm-ana veri seti üretir.

Kullanım (<repo> içinden; Mac'te venv_ts, VM'de venv_cyborg):
    venv_ts/bin/python -m minicage.dataset.generate --smoke --out <boş_klasör>
    venv_ts/bin/python -m minicage.dataset.generate --episodes-per-config 2000 --out minicage/data/<yeni_surum>

Var olan veri sürümlerinin üzerine yazılmaz: hedef klasör (--out ya da varsayılanı) boş değilse üretim başlamaz.
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np

from CybORG_plus_plus.mini_CAGE.minimal import HOSTS
from minicage.dataset.env import FixedCAGE
from minicage.dataset.policies import (
    RED_POLICIES, BLUE_POLICIES, V1_BLUE_POLICIES, make_red, make_blue, blue_epsilon)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_shard(red_name, blue_name, num_episodes, steps, remove_bugs, seed):
    """Tek bir (red, blue) kombinasyonunu num_episodes paralel ortamda koşturur."""
    np.random.seed(seed)
    env = FixedCAGE(num_envs=num_episodes, remove_bugs=remove_bugs)
    red, blue = make_red(red_name), make_blue(blue_name)
    state, _ = env.reset()

    E, T = num_episodes, steps
    out = {
        'obs_blue': np.zeros((E, T + 1, state['Blue'].shape[1]), np.int8),
        'obs_red': np.zeros((E, T + 1, state['Red'].shape[1]), np.int8),
        'true_state': np.zeros((E, T + 1, env.state.shape[1]), np.int8),
        'act_blue': np.zeros((E, T), np.int16),
        'act_red': np.zeros((E, T), np.int16),
        'rew_blue': np.zeros((E, T), np.float32),
        'blue_success': np.zeros((E, T), np.int8),
        'red_success': np.zeros((E, T), np.int8),
        'impacted': np.zeros((E, T, env.num_nodes), np.int8),
    }
    out['obs_blue'][:, 0] = state['Blue']
    out['obs_red'][:, 0] = state['Red']
    out['true_state'][:, 0] = env.state

    for t in range(T):
        red_action = red.get_action(state['Red'])
        blue_action = blue.get_action(state['Blue'])
        state, reward, _, info = env.step(
            red_action=red_action, blue_action=blue_action, red_agent=red)
        out['act_red'][:, t] = red_action[:, 0]
        out['act_blue'][:, t] = blue_action[:, 0]
        out['rew_blue'][:, t] = reward['Blue'][:, 0]
        out['blue_success'][:, t] = env.blue_success[:, 0]
        out['red_success'][:, t] = env.red_success[:, 0]
        out['impacted'][:, t] = info['impacted']
        out['obs_blue'][:, t + 1] = state['Blue']
        out['obs_red'][:, t + 1] = state['Red']
        out['true_state'][:, t + 1] = env.state

    n = env.num_nodes
    out['decoys_left'] = out['obs_blue'][:, :, 5 * n:6 * n].copy()
    out['red_policy'] = np.full(E, RED_POLICIES.index(red_name), np.int8)
    out['blue_policy'] = np.full(E, BLUE_POLICIES.index(blue_name), np.int8)
    out['epsilon'] = np.full(E, blue_epsilon(blue_name), np.float32)
    out['remove_bugs'] = np.full(E, remove_bugs, bool)
    out['seed'] = np.full(E, seed, np.int64)
    return out, env.action_mapping


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--episodes-per-config', type=int, default=2000)
    p.add_argument('--shard-size', type=int, default=1000)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--remove-bugs', default='true', choices=['true', 'false'])
    p.add_argument('--red', nargs='+', default=RED_POLICIES, choices=RED_POLICIES)
    p.add_argument('--blue', nargs='+', default=V1_BLUE_POLICIES, choices=BLUE_POLICIES)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default=None)
    p.add_argument('--smoke', action='store_true', help='2 kombinasyon x 20 bölüm')
    args = p.parse_args()

    remove_bugs = args.remove_bugs == 'true'
    if args.smoke:
        configs = [('bline', 'react_restore'), ('meander', 'random')]
        args.episodes_per_config, args.shard_size = 20, 20
        out_dir = args.out or os.path.join(ROOT, 'data', 'smoke')
    else:
        configs = [(r, b) for r in args.red for b in args.blue]
        out_dir = args.out or os.path.join(
            ROOT, 'data', 'minicage_v1' if remove_bugs else 'minicage_v1_bugged')
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        sys.exit(f'{out_dir} boş değil; var olan veri sürümlerinin üzerine yazılmaz. '
                 '--out ile yeni (ya da boş) bir klasör verin.')
    os.makedirs(out_dir, exist_ok=True)

    shards, shard_id, t0 = [], 0, time.time()
    for red_name, blue_name in configs:
        remaining = args.episodes_per_config
        while remaining > 0:
            n = min(args.shard_size, remaining)
            seed = args.seed * 100_000 + shard_id
            data, action_mapping = run_shard(
                red_name, blue_name, n, args.steps, remove_bugs, seed)
            fname = f'shard_{shard_id:03d}.npz'
            np.savez_compressed(os.path.join(out_dir, fname), **data)
            shards.append({'file': fname, 'red': red_name, 'blue': blue_name,
                           'episodes': n, 'seed': seed})
            mean = data['rew_blue'].sum(1).mean()
            print(f'{fname}: {red_name:8s} x {blue_name:22s} {n} bölüm, '
                  f'ort. ödül {mean:8.1f}  ({time.time() - t0:.0f} s)', flush=True)
            remaining -= n
            shard_id += 1

    n = len(HOSTS)
    meta = {
        'name': os.path.basename(out_dir),
        'source': 'CybORG++ MiniCAGE (FixedCAGE: SimplifiedCAGE + ödül çift-sayım düzeltmesi)',
        'red_agents': {'bline': 'mini_CAGE/red_bline_agent.py::B_line_minimal (ortam başına)', 'meander': 'mini_CAGE/test_agent.py::Meander_minimal'},
        'steps_per_episode': args.steps,
        'remove_bugs': remove_bugs,
        'base_seed': args.seed,
        'hosts': HOSTS,
        'red_policies': RED_POLICIES,
        'blue_policies': BLUE_POLICIES,
        'action_names': action_mapping,
        'obs_blue_layout': {
            'host_activity_safety': [0, 4 * n, 'host başına 4: aktivite [scan, exploit] + güvenlik ([0,0] temiz/bilinmiyor, [0,1] istismar edildi, [1,1] root (analyse ile), [1,0] remove)'],
            'scan_history': [4 * n, 5 * n, '0 yok, 1 önceden tarandı, 2 bu adımda tarandı'],
            'decoys_left': [5 * n, 6 * n, 'host başına kalan decoy sayısı'],
        },
        'obs_red_layout': '1 success bayrağı + host başına 3 (scanned, user, priv)',
        'true_state_layout': 'host başına 3: [scanned, user, priv]; -1 = Red bilmiyor',
        'shards': shards,
        'versions': {'python': platform.python_version(), 'numpy': np.__version__},
        'generation_seconds': round(time.time() - t0, 1),
        'command': ' '.join(sys.argv),
    }
    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    from minicage.dataset.loaders import make_splits
    make_splits(out_dir)
    print(f'Bitti: {out_dir} ({meta["generation_seconds"]} s)')


if __name__ == '__main__':
    main()

"""Re-simulates a recorded data version with the same seeds; the Blue actions are forced from the record.

Used to measure, on the same episodes (paired), the effect of a fix that changes only the Blue observation and not
the environment dynamics. The default environment is FixedObsCAGE: failed removes/restores are not reflected (env.py).

The policies are still called, but their outputs are discarded. The RNG consumption of the Blue policies does not depend
on the observation: React_restore, Restore_decoys, Sleep, ReactRestoreAnalyse and ReactRemove are deterministic; RandomBlue
and EpsilonBlue draw a fixed-size sample at every step. The Red observation is independent of the Blue observation. So the
random-number stream and the true state must stay identical to the record; the gate checks this per shard.

Gate (exit code 1 if any condition fails):
  1. All fields other than obs_blue are byte-identical to the record (including dtype and shape).
  2. Every difference chain, i.e. the step at which a difference starts on a host, is tied to a cause: a failed
     (blue_success = -1) remove or restore on that host at the previous step, or a scan-bit difference on any host in
     the same episode at the previous step (cross-host coupling: batch activity reset in _process_state, minimal.py:1186-1194).
  3. Difference types match the cause: the safety bits (last two of the 4 bits per host) are changed only by remove and
     restore; the activity bits and scan history only by restore and cross-host coupling; the decoy columns never
     change.
With --env FixedCAGE there must be no difference at all (reproduction test with forced actions).

Usage (from the repository root):
    venv_ts/bin/python -m minicage.dataset.resimulate --src minicage/data/minicage_v1 --out minicage/data/minicage_v1_fixedobs
    venv_ts/bin/python -m minicage.dataset.resimulate --src minicage/data/minicage_v1 --out <empty_folder> --env FixedCAGE

Existing data versions are never overwritten: does not start if --out is not empty. splits.npz is copied from the source
(the episode ids are the same).
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time

import numpy as np

from minicage.dataset.env import FixedCAGE, FixedObsCAGE
from minicage.dataset.generate import run_shard
from minicage.dataset.policies import NUM_HOSTS, REMOVE, RESTORE, USER0

ENVS = {'FixedObsCAGE': FixedObsCAGE, 'FixedCAGE': FixedCAGE}
FIX_NOTE = ('_process_state sees only successful Blue actions (blue_success > 0): a failed remove does not write [1,0] '
            '"removed"; an invalid user0 restore does not clear the safety flag and activity bits of user0 (nor, therefore, '
            'its scan history). True state, Red observation, reward and the recorded blue_success are unchanged.')
NONE, BY_REMOVE, BY_RESTORE, BY_COUPLING = 0, 1, 2, 4     # cause of a difference chain (bit mask)


def gate(src, new):
    """Checks the three gate conditions. src, new: field name → array. Returns: (problems, counts)."""
    problems = []
    if set(src) != set(new):
        problems.append(f'field sets differ: {sorted(set(src) ^ set(new))}')
    for k in sorted((set(src) & set(new)) - {'obs_blue'}):
        a, b = src[k], new[k]
        if a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a, b):
            problems.append(f'{k} differs from the record')
    ob0, ob1 = src['obs_blue'], new['obs_blue']
    if ob0.shape != ob1.shape or ob0.dtype != ob1.dtype:
        return problems + ['obs_blue shape differs'], {}

    N, (E, T1) = NUM_HOSTS, ob0.shape[:2]
    diff = ob0 != ob1
    per_host = diff[:, :, :4 * N].reshape(E, T1, N, 4)
    d_act, d_safe = per_host[..., :2].any(-1), per_host[..., 2:].any(-1)        # (episode, observation step, host)
    d_scan, d_decoy = diff[:, :, 4 * N:5 * N], diff[:, :, 5 * N:6 * N]
    if d_decoy.any():
        problems.append(f'obs_blue: {int(d_decoy.sum())} differences in the decoy columns')
    host = d_act | d_safe | d_scan

    # The cause of a failed action is written to the observation step it produces (t + 1) and to its target host.
    a, s = src['act_blue'].astype(int), src['blue_success']
    cause = np.zeros(host.shape, np.int8)
    for lo, hi, bit in ((REMOVE, RESTORE, BY_REMOVE), (RESTORE, RESTORE + N, BY_RESTORE)):
        e, t = np.nonzero((a >= lo) & (a < hi) & (s == -1))
        cause[e, t + 1, a[e, t] - lo] |= bit
    # Cross-host coupling: if any host had a scan bit (activity bit 0) at the previous step, _process_state resets the
    # activity of all hosts in that episode, including this step's green scan (minimal.py:1186-1194). A scan-bit difference
    # on one host can thus pass to the activity and scan columns of other hosts at the next step.
    coupled = per_host[:, :-1, :, 0].any(-1)                       # (episode, observation step - 1)
    cause[:, 1:] |= np.where(coupled, BY_COUPLING, NONE)[:, :, None].astype(np.int8)

    # Chain: consecutive differing steps on one host. Its label is the union of the causes seen along the chain.
    start = host.copy()
    start[:, 1:] &= ~host[:, :-1]
    if start[:, 0].any():
        problems.append('difference in the first observation')
    orphan = start & (cause == NONE)
    if orphan.any():
        problems.append(f'{int(orphan.sum())} difference chains do not start with a failed remove/restore')
    origin = np.zeros(host.shape, np.int8)
    label = np.zeros((E, N), np.int8)
    for i in range(T1):
        label = np.where(host[:, i], label | cause[:, i], NONE)
        origin[:, i] = label
    # remove writes only the safety flag; restore also resets the activity bits (the scan history is fed from them);
    # cross-host coupling carries only activity.
    outside = (d_act | d_scan) & ((origin & (BY_RESTORE | BY_COUPLING)) == 0)
    if outside.any():
        problems.append(f'obs_blue: {int(outside.sum())} activity/scan differences not caused by restore or '
                        'cross-host coupling')
    unsafe = d_safe & ((origin & (BY_REMOVE | BY_RESTORE)) == 0)
    if unsafe.any():
        problems.append(f'obs_blue: {int(unsafe.sum())} safety-bit differences not caused by remove/restore')

    own = (cause & (BY_REMOVE | BY_RESTORE)) > 0
    label_hosts = np.arange(N) != USER0
    counts = {
        'failed_remove_actions': int(((a >= REMOVE) & (a < RESTORE) & (s == -1)).sum()),
        'failed_restore_actions': int(((a >= RESTORE) & (s == -1)).sum()),
        'chains_remove': int((start & ((cause & BY_REMOVE) > 0)).sum()),
        'chains_restore': int((start & ((cause & BY_RESTORE) > 0)).sum()),
        'chains_coupling_only': int((start & ~own & (cause > 0)).sum()),
        'host_steps_changed_remove': int((origin == BY_REMOVE).sum()),
        'host_steps_changed_restore': int((origin == BY_RESTORE).sum()),
        'host_steps_changed_other': int(((origin != NONE) & (origin != BY_REMOVE) & (origin != BY_RESTORE)).sum()),
        'label_host_steps_changed': int(host[:, :, label_hosts].sum()),
        'label_host_safety_steps_changed': int(d_safe[:, :, label_hosts].sum()),
        'label_host_activity_scan_steps_changed': int((d_act | d_scan)[:, :, label_hosts].sum()),
        'user0_steps_changed': int(host[:, :, USER0].sum()),
        'episodes_changed': int(host.any(axis=(1, 2)).sum()),
    }
    return problems, counts


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description='Re-simulates a recorded data version with forced Blue actions.')
    p.add_argument('--src', required=True, help='source data version (meta.json, shard_*.npz, splits.npz)')
    p.add_argument('--out', required=True, help='empty or non-existent target folder')
    p.add_argument('--env', default='FixedObsCAGE', choices=sorted(ENVS))
    args = p.parse_args()

    if os.path.isdir(args.out) and os.listdir(args.out):
        sys.exit(f'{args.out} is not empty; existing data versions are never overwritten.')
    meta_path = os.path.join(args.src, 'meta.json')
    with open(meta_path) as f:
        meta = json.load(f)
    os.makedirs(args.out, exist_ok=True)

    totals, per_shard, all_problems, t0 = {}, [], [], time.time()
    for s in meta['shards']:
        with np.load(os.path.join(args.src, s['file'])) as z:
            src = {k: z[k] for k in z.files}
        data, _ = run_shard(s['red'], s['blue'], s['episodes'], meta['steps_per_episode'], meta['remove_bugs'],
                            s['seed'], env_cls=ENVS[args.env], blue_actions=src['act_blue'])
        problems, counts = gate(src, data)
        np.savez_compressed(os.path.join(args.out, s['file']), **data)
        per_shard.append({'file': s['file'], 'red': s['red'], 'blue': s['blue'], 'problems': problems, **counts})
        for k, v in counts.items():
            totals[k] = totals.get(k, 0) + v
        all_problems += [f"{s['file']}: {q}" for q in problems]
        status = 'gate passed' if not problems else 'GATE: ' + '; '.join(problems)
        print(f"{s['file']}: {s['red']:8s} x {s['blue']:22s} changed episodes {counts.get('episodes_changed', 0):4d}, "
              f"chains remove/restore {counts.get('chains_remove', 0):5d}/{counts.get('chains_restore', 0):5d}  "
              f"{status}  ({time.time() - t0:.0f} s)", flush=True)

    shutil.copyfile(os.path.join(args.src, 'splits.npz'), os.path.join(args.out, 'splits.npz'))
    src_name = meta.get('name', os.path.basename(os.path.normpath(args.src)))
    out = dict(meta)
    out.update({
        'name': os.path.basename(os.path.normpath(args.out)),
        'source': (f'CybORG++ MiniCAGE ({args.env}); {src_name} record re-simulated, '
                   'Blue actions forced from the record'),
        'observation_fix': FIX_NOTE if args.env == 'FixedObsCAGE' else None,
        'resimulated_from': {'name': src_name, 'meta_sha256': _sha256(meta_path), 'command': meta.get('command')},
        'gate': {'passed': not all_problems, 'problems': all_problems, 'totals': totals, 'shards': per_shard},
        'versions': {'python': platform.python_version(), 'numpy': np.__version__},
        'generation_seconds': round(time.time() - t0, 1),
        'command': ' '.join(sys.argv),
    })
    with open(os.path.join(args.out, 'meta.json'), 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print('toplam:', json.dumps(totals, ensure_ascii=False))
    if all_problems:
        print(f'GATE FAILED ({len(all_problems)} problems); data left under {args.out} for inspection.')
        sys.exit(1)
    print(f'gate passed: {args.out} ({out["generation_seconds"]} s)')


if __name__ == '__main__':
    main()

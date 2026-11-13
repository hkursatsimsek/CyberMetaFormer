"""Host-permutation OOD (revision item 6 (a)): model predictions when host order and identity change.

No retraining: the `model.pt` trained on v1 (asama3, s0–s4). Sets: v1 test (training policies) and ood_test
(ε = 0.3).

Permutations (π[i]: the host whose data goes to slot i; def and user0 fixed):
    P1  within topologically equivalent hosts: user1↔user2, user3↔user4, ophost0←ophost1←ophost2←ophost0. The
        attack-graph edges and the subnets are kept.
    P2  across classes: user1↔user3, user2↔user4. Subnets are kept, the edge alignment is broken: slot user1/2 now
        carries the data of hosts linked to ent1 (the in-data counterpart of a rewired topology; its real simulation is
        ood_topology.py).
    P1s, P2s  the same permutations, but the decoy-count columns stay in the slot (only the 4 bits, scan history and
        labels move).
    P3  model side only: the rows of the host embedding E_host are shuffled with π2, the data do not change. Host-token
        models only.
In P1 and P2 the observation columns (4 bits, scan history, decoy count) and the labels are permuted together. Decoy
capacity belongs to the host (at the start user1/2: 4, user3: 2, user4: 1), so it moves with its data. A permuted
slot can therefore hold a value never seen in training (user3↔user4 in P1, all users in P2). P1s and P2s separate this
effect: decoy capacity stays a property of the slot, as in a real rewiring (ood_topology).
In policies where the decoy count changes (restore_decoys, random, the random actions of the ε policies) the decoy
history stays in the slot while the activity stays with the moved host. This small inconsistency is accepted for the
diagnostic purpose.

Equivalences (verified in the tests; for P1 and P2): in host-token models a data permutation equals the same
permutation of E_host and of the relation matrix (topology bias). P1 keeps the relations, so P1 ≡ shuffling E_host
within classes. In models without bias (attn) P2 ≡ P3. In topology-biased models the P2 − P3 difference is the effect
of the misaligned topology prior.

Models: CMF (conv+SSD, topology bias), CMF host_tam (full attention), gru_host_topo, gru_host_attn and the full GRU. The
full GRU works on the flat 78-dim input and sees the permuted data as they are; P3 is undefined for it. Criteria are
computed on the argmax; temperature does not change the argmax. Main criteria: host macro-F1, silent recall, N_hist
recall. Also the error types (root→clean, root→user, user→clean, clean false alarm) and silent recall per host group
(at slot level). The slices are computed from each condition's own observation. The permutation only changes the slots,
so these are the same host-steps (the counts are checked).

Paired episode bootstrap: (i) Pk − identity for each model; (ii) model pairs in each condition (PAIRS); (iii) the
difference of the pairs' permutation drops, i.e. (a_Pk − a_id) − (b_Pk − b_id). Holm over all rows.

    python -m cybermetaformer.hidden_state.ood_permutation collect    # asama3/ood_permutation/preds/ (outside git)
    python -m cybermetaformer.hidden_state.ood_permutation analyze    # makale/ood_permutation/{ozet.json, tablolar.md}
"""
import argparse
import contextlib
import json
import os
import time

import numpy as np
import torch

from cybermetaformer.hidden_state import data as D
from cybermetaformer.hidden_state import evaluate as E
from cybermetaformer.hidden_state import features as F
from cybermetaformer.hidden_state import metrics as M
from cybermetaformer.hidden_state import ood_suite as O
from cybermetaformer.hidden_state.corrected_obs import _stat as member_stat, _value as member_value
from cybermetaformer.hidden_state.models.cybermetaformer import HOSTS

RESULTS = os.path.join(D.ROOT, 'results', 'hidden_state')
PRED_DIR = os.path.join(RESULTS, 'asama3', 'ood_permutation', 'preds')
OUT_DIR = os.path.join(RESULTS, 'makale', 'ood_permutation')
MODELS = {'cybermetaformer': 'CyberMetaFormer',
          'cybermetaformer-host_tam': 'CMF, full attention (no topology bias)',
          'cybermetaformer-gru_host_topo': 'Host-token GRU, topology-biased attention',
          'cybermetaformer-gru_host_attn': 'Host-token GRU, attention without bias',
          'gru_full': 'GRU (full history)'}
HOST_TOKEN = tuple(m for m in MODELS if m != 'gru_full')
PAIRS = [('cybermetaformer', 'gru_full'), ('cybermetaformer', 'cybermetaformer-host_tam'),
         ('cybermetaformer-gru_host_topo', 'cybermetaformer-gru_host_attn')]
SEEDS = (0, 1, 2, 3, 4)
SETS = {'test': 'test', 'ood': 'ood_test'}
SET_LABEL = {'test': 'Test', 'ood': 'OOD (ε = 0.3)'}
CRITERIA = {'host_macro_f1': ('conf', 'Host macro-F1'), 'silent_recall': ('silent', 'Silent recall'),
            'never_recall': ('never', 'N_hist recall')}
# error type → (true class, predicted classes); rate within the host-steps of the true class
ERRORS = {'root_to_clean': (2, (0,)), 'root_to_user': (2, (1,)), 'user_to_clean': (1, (0,)),
          'clean_false_alarm': (0, (1, 2))}
ERROR_LABEL = {'root_to_clean': 'root→clean', 'root_to_user': 'root→user', 'user_to_clean': 'user→clean',
               'clean_false_alarm': 'clean false alarm'}
LABEL_NAMES = [HOSTS[h] for h in D.LABEL_HOSTS]
GROUPS = {'user1–2': ('user1', 'user2'), 'user3–4': ('user3', 'user4'), 'ent0': ('ent0',), 'ent1': ('ent1',),
          'ent2': ('ent2',), 'op': ('ophost0', 'ophost1', 'ophost2', 'opserv')}


def _perm(*cycles):
    """Host permutation from cycles (13 hosts, HOSTS order): each slot in a cycle takes the data of the next host."""
    pi = list(range(len(HOSTS)))
    for cyc in cycles:
        idx = [HOSTS.index(h) for h in cyc]
        for a, b in zip(idx, idx[1:] + idx[:1]):
            pi[a] = b
    return np.array(pi)


PERMS = {'P1': _perm(('user1', 'user2'), ('user3', 'user4'), ('ophost0', 'ophost1', 'ophost2')),
         'P2': _perm(('user1', 'user3'), ('user2', 'user4'))}
P3_PERM = PERMS['P2']
# data condition → (permutation, whether the decoy columns move too)
DATA_CONDITIONS = {'P1': (PERMS['P1'], True), 'P1s': (PERMS['P1'], False),
                   'P2': (PERMS['P2'], True), 'P2s': (PERMS['P2'], False)}
CONDITIONS = ('id', 'P1', 'P1s', 'P2', 'P2s', 'P3')
COND_LABEL = {'id': 'identity', 'P1': 'P1 (within class)', 'P1s': 'P1, decoy count stays in slot',
              'P2': 'P2 (across classes)', 'P2s': 'P2, decoy count stays in slot', 'P3': 'P3 (E_host only)'}


def obs_columns(pi, move_decoys=True):
    """Permuted observation = obs[..., obs_columns(π)]: the 4 bits, scan history and (if move_decoys) decoy count of
    slot i come from host π(i); with move_decoys=False the decoy count stays in the slot."""
    pi, n = np.asarray(pi), len(pi)
    bits = (4 * pi[:, None] + np.arange(4)).reshape(-1)
    return np.concatenate([bits, 4 * n + pi, 5 * n + (pi if move_decoys else np.arange(n))])


def label_columns(pi):
    """Permuted labels = labels[..., label_columns(π)] (12 label hosts; user0 must stay fixed)."""
    lh = list(D.LABEL_HOSTS)
    assert all(int(pi[h]) in lh for h in lh), 'the permutation must not mix user0 with the label hosts'
    return np.array([lh.index(int(pi[h])) for h in lh])


def permute(obs, labels, pi, move_decoys=True):
    return obs[..., obs_columns(pi, move_decoys)], labels[..., label_columns(pi)]


class ArrayStore:
    """Store from arrays for WindowData: obs (E, T+1, 78), labels (E, T, 12), prev_action (E, T+1)."""

    def __init__(self, obs, labels, prev_action):
        self.obs, self.labels, self.prev_action = obs, labels, prev_action


@contextlib.contextmanager
def shuffled_host_emb(model, pi):
    """P3: slot i gets the embedding of host π(i); the weight is restored on exit."""
    saved = model.host_emb.data.clone()
    model.host_emb.data = saved[torch.as_tensor(np.asarray(pi), device=saved.device)]
    try:
        yield model
    finally:
        model.host_emb.data = saved


def predict(model, cfg, obs, labels, prev_action, device):
    """Argmax tahmini (E, T, 12) int8."""
    wd = D.WindowData(ArrayStore(obs, labels, prev_action), np.arange(len(obs)), device=device)
    pred = E.predict_logits(model, wd, O.K, cfg['scale'], cfg['eval_batch_size']).argmax(-1).astype(np.int8)
    del wd
    return pred


def pred_path(name, seed, pred_dir=PRED_DIR):
    return os.path.join(pred_dir, f'{name}_k{O.K}_s{seed}.npz')


def check_against_record(name, seed, y_test, pred_test, y_ood=None, pred_ood=None):
    """The test (and OOD) F1 of the identity condition must match the run's metrics.json (same weights, same predictions)."""
    rep = json.load(open(os.path.join(O.run_dir(name, seed), 'metrics.json')))['reports']
    out = {}
    for key, y, p in (('test', y_test, pred_test), ('ood_test', y_ood, pred_ood)):
        if y is None or key not in rep:
            continue
        f1 = M.host_macro_f1(y.reshape(-1, 12), p.reshape(-1, 12))
        assert abs(f1 - rep[key]['host_macro_f1']) < 1e-4, f'{name} s{seed} {key}: F1 {f1} ≠ {rep[key]["host_macro_f1"]}'
        out[key] = f1
    return out


# ── toplama ──────────────────────────────────────────────────────────────

def open_sets():
    st = D.EpisodeStore(D.DEFAULT_DATA)
    with st.final_evaluation():
        idx = {s: st.split(sp) for s, sp in SETS.items()}
    return st, idx


def member_keys(name):
    return [f'{s}__{c}' for s in SETS for c in CONDITIONS if c != 'P3' or name in HOST_TOKEN]


def collect(device, models=tuple(MODELS), seeds=SEEDS):
    """Predictions of all conditions for each member and set. Only missing conditions are added to an existing file."""
    st, idx = open_sets()
    data = {s: (st.obs[i], st.labels[i], st.prev_action[i]) for s, i in idx.items()}
    os.makedirs(PRED_DIR, exist_ok=True)
    for name in models:
        for seed in seeds:
            path = pred_path(name, seed)
            out = dict(np.load(path)) if os.path.exists(path) else {}
            todo = [k for k in member_keys(name) if k not in out]
            if not todo:
                continue
            model, cfg = O.load_model(name, seed, device)
            assert not getattr(model, 'action_input', False), 'models with action input would need permuted actions too'
            t0 = time.time()
            for key in todo:
                s, c = key.split('__')
                obs, y, pa = data[s]
                if c == 'id':
                    out[key] = predict(model, cfg, obs, y, pa, device)
                elif c == 'P3':
                    with shuffled_host_emb(model, P3_PERM):
                        out[key] = predict(model, cfg, obs, y, pa, device)
                else:
                    out[key] = predict(model, cfg, *permute(obs, y, *DATA_CONDITIONS[c]), pa, device)
            f1 = check_against_record(name, seed, data['test'][1], out['test__id'], data['ood'][1], out['ood__id'])
            np.savez_compressed(path, **out)
            print(f'{name} s{seed}: {time.time() - t0:.0f} s, {len(todo)} conditions; identity F1 = record '
                  + ', '.join(f'{k} {v:.4f}' for k, v in f1.items()), flush=True)
            del model
            if device == 'mps':
                torch.mps.empty_cache()


# ── analysis ─────────────────────────────────────────────────────────────

def records(y, pred, alert, age):
    """Per-episode counts of a member (for the paired bootstrap) and per-host silent counts."""
    cnt, names = M.episode_slice_counts(y, pred, alert, age)
    silent, hit = (y > 0) & ~alert, pred > 0
    return {'conf': M.episode_confusion(y, pred).astype(np.int32),
            'silent': cnt[:, names.index('sessiz')], 'never': cnt[:, names.index('never')],
            'silent_host': np.stack([silent.sum((0, 1)), (silent & hit).sum((0, 1))], -1)}


def _msd(x):
    if any(v is None for v in x):
        return None
    return {'mean': float(np.mean(x)), 'sd': float(np.std(x, ddof=1)) if len(x) > 1 else 0.0, 'seeds': list(x)}


def group_recall(silent_host, hosts):
    n = silent_host[[LABEL_NAMES.index(h) for h in hosts]].sum(0)
    return float(n[1] / n[0]) if n[0] else None


def error_rates(C):
    """Pooled (3, 3) confusion → error-type rates."""
    return {e: float(C[t, list(ps)].sum() / max(C[t].sum(), 1)) for e, (t, ps) in ERRORS.items()}


def summarize(runs, ep_mask=None):
    """Seed list → criteria, error types and group recalls (mean ± sd). ep_mask: episode subset."""
    if ep_mask is not None:
        runs = [{k: (v[ep_mask] if k != 'silent_host' else v) for k, v in r.items()} for r in runs]
    out = {c: _msd([member_value(r, key) for r in runs]) for c, (key, _) in CRITERIA.items()}
    confs = [r['conf'].sum((0, 1)) for r in runs]
    out['errors'] = {e: _msd([error_rates(C)[e] for C in confs]) for e in ERRORS}
    if ep_mask is None:
        out['groups'] = {g: _msd([group_recall(r['silent_host'], hs) for r in runs]) for g, hs in GROUPS.items()}
    return out


def set_stats(y, alert, age, red=None, red_names=None):
    silent = (y > 0) & ~alert
    out = {'n_episodes': int(len(y)), 'n_host_steps': int(y.size),
           'class_counts': {c: int((y == i).sum()) for i, c in enumerate(('clean', 'user', 'root'))},
           'n_silent': int(silent.sum()), 'n_never': int((silent & (age == F.NEVER)).sum()),
           'silent_by_group': {g: int(silent[..., [LABEL_NAMES.index(h) for h in hs]].sum()) for g, hs in GROUPS.items()}}
    if red is not None:
        out['episodes_by_red'] = {r: int((red == i).sum()) for i, r in enumerate(red_names)}
    return out


def bootstrap_rows(runs, sizes, n_boot, ka, kb, base=None, side=None, **tags):
    """ka − kb ((ka − base[0]) − (kb − base[1]) if base is given) for each criterion; tags are written to the row.

    side: run key → bootstrap weight key (a set in sizes). Without it, a single set (paired). With two different
    sets, the sets are resampled independently of each other (unpaired two-sample)."""
    rows = []
    side = side or {k: next(iter(sizes)) for k in (ka, kb) + tuple(base or ())}
    for c, (key, _) in CRITERIA.items():
        f = {k: member_stat(runs[k], key, side[k]) for k in side}
        if base is None:
            fa, fb = f[ka], f[kb]
        else:
            fa = lambda W, f=f: f[ka](W) - f[base[0]](W)          # noqa: E731
            fb = lambda W, f=f: f[kb](W) - f[base[1]](W)          # noqa: E731
        rows.append({**tags, 'criterion': c, **M.paired_bootstrap(fa, fb, sizes, n_boot=n_boot, seed=0)})
    return rows


def analyze(n_boot=10_000, models=tuple(MODELS), seeds=SEEDS, log=print):
    st, idx = open_sets()
    preds = {(name, seed): dict(np.load(pred_path(name, seed))) for name in models for seed in seeds}
    res = {'seeds': list(seeds), 'n_boot': n_boot,
           'permutations': {c: {HOSTS[i]: HOSTS[j] for i, j in enumerate(pi) if i != j} for c, pi in PERMS.items()},
           'p3_host_emb_permutation': {HOSTS[i]: HOSTS[j] for i, j in enumerate(P3_PERM) if i != j},
           'sets': {}, 'models': {}, 'rows': []}
    runs = {}
    for s, i in idx.items():
        y0, obs0 = st.labels[i].astype(np.int64), st.obs[i]
        for c in CONDITIONS:
            obs, y = permute(obs0, y0, *DATA_CONDITIONS[c]) if c in DATA_CONDITIONS else (obs0, y0)
            alert = F.episode_alerts(obs)
            age = F.alert_age(alert)
            stats = set_stats(y, alert, age)
            if c == 'id':
                res['sets'][s] = stats
            else:                                   # the permutation only changes the slots
                for k in ('n_host_steps', 'class_counts', 'n_silent', 'n_never'):
                    assert stats[k] == res['sets'][s][k], f'{s} {c}: {k} changed'
                res['sets'][s].setdefault('silent_by_group_permuted', {})[c] = stats['silent_by_group']
            for name in models:
                if c == 'P3' and name not in HOST_TOKEN:
                    continue
                runs[s, c, name] = [records(y, preds[name, seed][f'{s}__{c}'], alert, age) for seed in seeds]
                res['models'].setdefault(name, {}).setdefault(s, {})[c] = summarize(runs[s, c, name])
        for name in models:
            m = res['models'][name][s]
            log(f'{name:30s} {s:5s} ' + '  '.join(
                f"{c}: F1 {m[c]['host_macro_f1']['mean']:.4f} silent {m[c]['silent_recall']['mean']:.4f} "
                f"N {m[c]['never_recall']['mean']:.4f}" for c in CONDITIONS if c in m))

        sizes = {s: len(i)}
        R = {(c, name): v for (ss, c, name), v in runs.items() if ss == s}
        for name in models:
            for c in CONDITIONS[1:]:
                if (c, name) in R:
                    res['rows'] += bootstrap_rows(R, sizes, n_boot, (c, name), ('id', name),
                                                  kind='perm_minus_id', set=s, model=name, condition=c)
        for a, b in PAIRS:
            if a not in models or b not in models:
                continue
            for c in CONDITIONS:
                if (c, a) in R and (c, b) in R:
                    res['rows'] += bootstrap_rows(R, sizes, n_boot, (c, a), (c, b),
                                                  kind='pair', set=s, a=a, b=b, condition=c)
                    if c != 'id':
                        res['rows'] += bootstrap_rows(R, sizes, n_boot, (c, a), (c, b), base=(('id', a), ('id', b)),
                                                      kind='pair_drop', set=s, a=a, b=b, condition=c)
    for r, p in zip(res['rows'], M.holm([r['p'] for r in res['rows']])):
        r['p_holm'] = p
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, 'ozet.json'), 'w') as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    with open(os.path.join(OUT_DIR, 'tablolar.md'), 'w') as f:
        f.write(tables_md(res))
    log('→', OUT_DIR)
    return res


# ── outputs ──────────────────────────────────────────────────────────────

def _ms(d):
    return f"{d['mean']:.3f} ± {d['sd']:.3f}" if d else '–'


def _ci(r):
    return f"{r['diff']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]"


def _p(p):
    return '< 0.001' if p < 0.001 else f'{p:.3f}'


def _perm_text(d):
    seen, parts = set(), []
    for a, b in d.items():
        if a in seen:
            continue
        if d.get(b) == a:
            parts.append(f'{a}↔{b}')
            seen |= {a, b}
        else:
            parts.append(f'{a}←{b}')
            seen.add(a)
    return ', '.join(parts)


def tables_md(res):
    n = len(res['seeds'])
    out = ['# Host permutation OOD', '',
           f"Models trained on the original host order; evaluated with permuted hosts. P1 = "
           f"{_perm_text(res['permutations']['P1'])} (attack-graph edges and subnets preserved); P2 = "
           f"{_perm_text(res['permutations']['P2'])} (subnets preserved, user→enterprise edges misaligned); "
           f"P3 = only the host-embedding rows permuted as in P2, data unchanged (host-token models). "
           f"Observation columns (4 bits, scan history, decoy count) and labels move together; slot a←b holds host b's "
           f"data. Decoy capacity is host-specific (user1/2: 4, user3: 2, user4: 1 at the start), so in P1 and P2 it "
           f"moves with the host; in P1s and P2s the decoy-count columns stay with the slot. Mean ± sd over {n} seeds (argmax; temperature does not change it). Differences: paired episode "
           f"bootstrap ({res['n_boot']:,} resamples), Holm over all {len(res['rows'])} rows.", '',
           '## Sets', '', '| Set | Episodes | Host-steps | Clean | User | Root | Silent | N_hist |',
           '|---|---|---|---|---|---|---|---|']
    for s, d in res['sets'].items():
        cc = d['class_counts']
        out.append(f"| {SET_LABEL[s]} | {d['n_episodes']:,} | {d['n_host_steps']:,} | {cc['clean']:,} | {cc['user']:,} | "
                   f"{cc['root']:,} | {d['n_silent']:,} | {d['n_never']:,} |")
    out += ['', '## Metrics', '', '| Set | Model | Condition | ' + ' | '.join(lab for _, lab in CRITERIA.values()) + ' |',
            '|---|---|---|' + '---|' * len(CRITERIA)]
    for s in res['sets']:
        for name, lab in MODELS.items():
            for c, m in res['models'].get(name, {}).get(s, {}).items():
                out.append(f"| {SET_LABEL[s]} | {lab} | {COND_LABEL[c]} | " + ' | '.join(_ms(m[k]) for k in CRITERIA) + ' |')
    out += ['', '## Permuted − identity (same model)', '', '| Set | Model | Condition | Criterion | Δ [95% CI] | p (Holm) |',
            '|---|---|---|---|---|---|']
    for r in res['rows']:
        if r['kind'] == 'perm_minus_id':
            out.append(f"| {SET_LABEL[r['set']]} | {MODELS[r['model']]} | {COND_LABEL[r['condition']]} | "
                       f"{CRITERIA[r['criterion']][1]} | {_ci(r)} | {_p(r['p_holm'])} |")
    for kind, title in (('pair', 'Model pairs per condition (a − b)'),
                        ('pair_drop', 'Difference of permutation effects: (a_P − a_id) − (b_P − b_id)')):
        out += ['', f'## {title}', '', '| Set | a | b | Condition | Criterion | Δ [95% CI] | p (Holm) |',
                '|---|---|---|---|---|---|---|']
        for r in res['rows']:
            if r['kind'] == kind:
                out.append(f"| {SET_LABEL[r['set']]} | {MODELS[r['a']]} | {MODELS[r['b']]} | {COND_LABEL[r['condition']]} | "
                           f"{CRITERIA[r['criterion']][1]} | {_ci(r)} | {_p(r['p_holm'])} |")
    out += ['', '## Error types (rate within the true class; mean ± sd over seeds)', '',
            '| Set | Model | Condition | ' + ' | '.join(ERROR_LABEL.values()) + ' |', '|---|---|---|' + '---|' * len(ERRORS)]
    for s in res['sets']:
        for name, lab in MODELS.items():
            for c, m in res['models'].get(name, {}).get(s, {}).items():
                out.append(f"| {SET_LABEL[s]} | {lab} | {COND_LABEL[c]} | "
                           + ' | '.join(_ms(m['errors'][e]) for e in ERRORS) + ' |')
    out += ['', '## Silent recall by host slot group (mean ± sd over seeds)', '',
            '| Set | Model | Condition | ' + ' | '.join(GROUPS) + ' |', '|---|---|---|' + '---|' * len(GROUPS)]
    for s, d in res['sets'].items():
        out.append(f"| {SET_LABEL[s]} | silent host-steps | identity | "
                   + ' | '.join(f"{d['silent_by_group'][g]:,}" for g in GROUPS) + ' |')
        for name, lab in MODELS.items():
            for c, m in res['models'].get(name, {}).get(s, {}).items():
                out.append(f"| {SET_LABEL[s]} | {lab} | {COND_LABEL[c]} | " + ' | '.join(_ms(m['groups'][g]) for g in GROUPS) + ' |')
    return '\n'.join(out) + '\n'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['collect', 'analyze'])
    p.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    p.add_argument('--n-boot', type=int, default=10_000)
    p.add_argument('--models', nargs='+', default=list(MODELS))
    p.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    args = p.parse_args()
    if args.cmd == 'collect':
        collect(args.device, tuple(args.models), tuple(args.seeds))
    else:
        analyze(args.n_boot, tuple(args.models), tuple(args.seeds))


if __name__ == '__main__':
    main()

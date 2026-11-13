"""OOD failure analysis (revision item 6, first step): breakdown of the errors by type on mechanisms unseen in
training.

No retraining. The saved ood_suite logits (asama3, s0–s4) are used: CMF, full GRU, gru_host_topo and
gru_host_attn. The sets are the same as in ood_suite:
    eps0.1   the react_restore_eps0.1 episodes of v1 test (training policy, reference)
    ood      v1 ood_test (ε = 0.3)
    analyse  react_restore_analyse (periyodik analyse; minicage_v1_shift)
    remove   react_remove (remove instead of restore; minicage_v1_shift)

In each set:
1. Class counts (clean / user / root host-steps), silent host-steps per class, N_hist.
2. Error types per model (seed mean ± sd). On all host-steps: root→clean, root→user, user→clean,
   user→root, clean false alarm. On silent host-steps: root→clean, root→user, user→clean.
3. Silent host-step slices: attacker × host group (subnet), alert age, type of the last alert × whether the host
   was restored after that alert (explain_m3.alert_context), silent-compromise cause (silent_cause). Per slice n,
   share, root share; per model recall, root→clean and user→clean.
4. Silent-recall differences CMF − full GRU and CMF − gru_host_topo. A paired episode bootstrap per slice, on the
   slices with n ≥ MIN_BOOT host-steps; Holm over all rows.
Silent = label > 0 and no 'export' alert at that step; caught = prediction > 0. Predictions are argmax (temperature
does not change them).

    python -m cybermetaformer.hidden_state.failure_analysis     # makale/failure_analysis/{ozet.json, tablolar.md}
"""
import argparse
import json
import os

import numpy as np

from cybermetaformer.hidden_state import data as D
from cybermetaformer.hidden_state import metrics as M
from cybermetaformer.hidden_state import ood_suite as O
from cybermetaformer.hidden_state.corrected_obs import _stat as member_stat
from cybermetaformer.hidden_state.explain_m3 import LAST_KIND, alert_context
from cybermetaformer.hidden_state.models.cybermetaformer import SUBNET
from cybermetaformer.hidden_state.ood_permutation import ERROR_LABEL, ERRORS, _ci, _ms, _msd, _p, error_rates

OUT_DIR = os.path.join(D.ROOT, 'results', 'hidden_state', 'makale', 'failure_analysis')
MODELS = {'cybermetaformer': 'CyberMetaFormer', 'gru_full': 'GRU (full history)',
          'cybermetaformer-gru_host_topo': 'Host-token GRU, topology-biased attention',
          'cybermetaformer-gru_host_attn': 'Host-token GRU, attention without bias'}
PAIRS = [('cybermetaformer', 'gru_full'), ('cybermetaformer', 'cybermetaformer-gru_host_topo')]
SEEDS = (0, 1, 2, 3, 4)
SETS = ('eps0.1', 'ood', 'analyse', 'remove')
SET_LABEL = {'eps0.1': 'ε = 0.1 (training policy)', 'ood': 'ε = 0.3 (OOD)', 'analyse': 'react_restore_analyse',
             'remove': 'react_remove'}
ALL_ERRORS = {**ERRORS, 'user_to_root': (1, (2,))}
ALL_ERROR_LABEL = {**ERROR_LABEL, 'user_to_root': 'user→root'}
SILENT_ERRORS = ('root_to_clean', 'root_to_user', 'user_to_clean')
GROUP_LABEL = {'ent': 'enterprise', 'op': 'operational', 'user': 'user'}
RED_LABEL = {'bline': 'B-line', 'meander': 'Meander'}
KIND_LABEL = dict(zip(LAST_KIND, ['no alert so far', 'exploit', 'root flag (no activity)', 'other']))
CAUSE_LABEL = {'kacak_5': 'cause: unlogged exploit (5%)', 'green_maske': 'cause: masked by Green scan',
               'remove_maske': 'cause: masked by failed remove', 'restore': 'cause: after restore',
               'bilinmiyor': 'cause: unknown'}
MIN_SLICE, MIN_BOOT = 200, 1000


def slices(st, ep, es):
    """{(kind, name): mask (E, T, 12)} within the silent host-steps; kind ∈ attacker_group, age, last_alert, cause."""
    sil = (es.y > 0) & ~es.alert
    group = np.array(['ent', 'op', 'user'])[np.array(SUBNET)[D.LABEL_HOSTS]]       # def is always clean
    out = {}
    for r, red in enumerate(st.meta['red_policies']):
        for g in ('ent', 'op', 'user'):
            out['attacker_group', f'{RED_LABEL[red]} × {GROUP_LABEL[g]}'] = \
                sil & (st.red_policy[ep] == r)[:, None, None] & (group == g)
    for name, lo, hi in M.ALERT_AGE_BINS:
        lab = 'never alerted (N_hist)' if lo < 0 else f'{name} steps since last alert'
        out['age', lab] = sil & ((es.age < 0) if lo < 0 else (es.age >= lo) & (es.age <= hi))
    kind, rest = alert_context(st, ep, es)
    for k, kn in enumerate(LAST_KIND):
        for r in (False, True):
            out['last_alert', f"{KIND_LABEL[kn]}{', restored since' if r else ''}"] = sil & (kind == k) & (rest == r)
    if es.cause is not None:
        for code, cn in enumerate(es.cause_names):
            if cn in CAUSE_LABEL:
                out['cause', CAUSE_LABEL[cn]] = sil & (es.cause == code)
    return sil, out


def slice_rates(y, pred, m):
    """Recall (prediction > 0) and silent error types (rate within the true class) in a slice."""
    out = {'recall': float((pred[m] > 0).mean()) if m.any() else None}
    for e in SILENT_ERRORS:
        t, ps = ALL_ERRORS[e]
        mt = m & (y == t)
        out[e] = float(np.isin(pred[mt], ps).mean()) if mt.any() else None
    return out


def episode_counts(pred, m):
    """(E, 2): number of host-steps in the slice and caught ones (for the episode bootstrap)."""
    return np.stack([m.sum((1, 2)), (m & (pred > 0)).sum((1, 2))], -1)


def analyze(n_boot=10_000, seeds=SEEDS, log=print):
    stores = O.open_stores()
    idx = O.base_indices(stores)
    sets = O.eval_sets(stores, idx)
    members = {n: [O.load_member(n, s) for s in seeds] for n in MODELS}
    res = {'seeds': list(seeds), 'n_boot': n_boot, 'min_slice': MIN_SLICE, 'min_boot': MIN_BOOT, 'sets': {},
           'models': {}, 'slices': {}, 'rows': []}
    for s in SETS:
        es = sets[s]
        b = O.SETS[s][0]
        st = stores[O.BASE[b][0]]
        ep = idx[b][es.pos]
        y = es.y
        preds = {n: [m[b][es.pos].astype(np.float32).argmax(-1) for m in ms] for n, ms in members.items()}
        sil, sl = slices(st, ep, es)
        red = st.red_policy[ep]
        res['sets'][s] = {
            'n_episodes': int(es.E), 'episodes_by_red': {RED_LABEL[r]: int((red == i).sum())
                                                        for i, r in enumerate(st.meta['red_policies'])},
            'n_host_steps': int(y.size),
            'class_counts': {c: int((y == i).sum()) for i, c in enumerate(('clean', 'user', 'root'))},
            'silent_by_class': {'user': int((sil & (y == 1)).sum()), 'root': int((sil & (y == 2)).sum())},
            'n_never': int((sil & (es.age < 0)).sum())}
        for n, ps in preds.items():
            confs = [M.episode_confusion(y, p).sum((0, 1)) for p in ps]
            sil_rates = [slice_rates(y, p, sil) for p in ps]
            res['models'].setdefault(n, {})[s] = {
                'host_macro_f1': _msd([float(M.host_macro_f1_batch(M.episode_confusion(y, p).sum(0)[None])[0])
                                       for p in ps]),
                'all_errors': {e: _msd([float(C[t, list(q)].sum() / max(C[t].sum(), 1)) for C in confs])
                               for e, (t, q) in ALL_ERRORS.items()},
                'silent': {k: _msd([r[k] for r in sil_rates]) for k in ('recall',) + SILENT_ERRORS}}
        rows = {}
        for (kind, name), m in sl.items():
            if m.sum() < MIN_SLICE:
                continue
            rows[f'{kind}|{name}'] = {
                'kind': kind, 'name': name, 'n': int(m.sum()), 'share': float(m.sum() / sil.sum()),
                'root_share': float((m & (y == 2)).sum() / m.sum()),
                'models': {n: {k: _msd([slice_rates(y, p, m)[k] for p in ps]) for k in ('recall',) + SILENT_ERRORS}
                           for n, ps in preds.items()}}
            if m.sum() >= MIN_BOOT:
                runs = {n: [{'c': episode_counts(p, m)} for p in ps] for n, ps in preds.items()}
                for a, bb in PAIRS:
                    r = M.paired_bootstrap(member_stat(runs[a], 'c', s), member_stat(runs[bb], 'c', s), {s: es.E},
                                           n_boot=n_boot, seed=0)
                    res['rows'].append({'set': s, 'slice': f'{kind}|{name}', 'a': a, 'b': bb, **r})
        res['slices'][s] = rows
        log(f"{s:8s} " + '  '.join(f"{n} silent {res['models'][n][s]['silent']['recall']['mean']:.3f} "
                                   f"r→c {res['models'][n][s]['silent']['root_to_clean']['mean']:.3f}" for n in MODELS))
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

def _m(d):
    return f"{d['mean']:.3f}" if d else '–'


SLICE_KIND_LABEL = {'attacker_group': 'Attacker × host group (subnet)', 'age': 'Alert age (steps since the last alert)',
                    'last_alert': 'Last alert type × restore since then', 'cause': 'Root cause of the silent breach'}


def tables_md(res):
    n = len(res['seeds'])
    names = list(MODELS)
    out = ['# OOD failure analysis by error type', '',
           f"Saved logits of the M3/M4 runs (no retraining); argmax predictions; mean ± sd over {n} seeds. Silent = "
           f"label > 0 and no 'export' alert at that step; detected = prediction > 0. Error rates are within the true "
           f"class (e.g. root→clean = share of true-root host-steps predicted clean). Slices with fewer than "
           f"{res['min_slice']} silent host-steps are omitted. CMF − baseline differences of silent recall: paired "
           f"episode bootstrap ({res['n_boot']:,} resamples) on slices with ≥ {res['min_boot']:,} host-steps, Holm over "
           f"all {len(res['rows'])} rows.", '',
           '## Sets', '',
           '| Set | Episodes (B-line / Meander) | Host-steps | Clean | User | Root | Silent user | Silent root | N_hist |',
           '|---|---|---|---|---|---|---|---|---|']
    for s, d in res['sets'].items():
        cc, sc, er = d['class_counts'], d['silent_by_class'], d['episodes_by_red']
        out.append(f"| {SET_LABEL[s]} | {d['n_episodes']:,} ({er['B-line']:,} / {er['Meander']:,}) | {d['n_host_steps']:,} | "
                   f"{cc['clean']:,} | {cc['user']:,} | {cc['root']:,} | {sc['user']:,} | {sc['root']:,} | {d['n_never']:,} |")
    out += ['', '## Error types, all host-steps', '',
            '| Set | Model | Host macro-F1 | ' + ' | '.join(ALL_ERROR_LABEL[e] for e in ALL_ERRORS) + ' |',
            '|---|---|---|' + '---|' * len(ALL_ERRORS)]
    for s in res['sets']:
        for nm in names:
            m = res['models'][nm][s]
            out.append(f"| {SET_LABEL[s]} | {MODELS[nm]} | {_ms(m['host_macro_f1'])} | "
                       + ' | '.join(_ms(m['all_errors'][e]) for e in ALL_ERRORS) + ' |')
    out += ['', '## Error types, silent host-steps', '',
            '| Set | Model | Silent recall | ' + ' | '.join(ALL_ERROR_LABEL[e] for e in SILENT_ERRORS) + ' |',
            '|---|---|---|' + '---|' * len(SILENT_ERRORS)]
    for s in res['sets']:
        for nm in names:
            m = res['models'][nm][s]['silent']
            out.append(f"| {SET_LABEL[s]} | {MODELS[nm]} | {_ms(m['recall'])} | "
                       + ' | '.join(_ms(m[e]) for e in SILENT_ERRORS) + ' |')
    short = {'cybermetaformer': 'CMF', 'gru_full': 'GRU', 'cybermetaformer-gru_host_topo': 'HT-GRU topo',
             'cybermetaformer-gru_host_attn': 'HT-GRU attn'}
    boot = {(r['set'], r['slice'], r['a'], r['b']): r for r in res['rows']}
    for s, rows in res['slices'].items():
        out += ['', f'## Silent slices: {SET_LABEL[s]}', '',
                'Recall and root→clean / user→clean per model (seed means). Share = share of the set\'s silent '
                'host-steps; root = share of true root in the slice.', '',
                '| Slice | n | Share | Root | ' + ' | '.join(f'{short[nm]} recall' for nm in names) + ' | '
                + ' | '.join(f'{short[nm]} r→c / u→c' for nm in names) + ' | CMF − GRU [95% CI] | p (Holm) |',
                '|---|---|---|---|' + '---|' * (2 * len(names)) + '---|---|']
        for kind, klab in SLICE_KIND_LABEL.items():
            sel = [(k, r) for k, r in rows.items() if r['kind'] == kind]
            if not sel:
                continue
            out.append(f"| *{klab}* |  |  |  |" + ' |' * (2 * len(names)) + '  |  |')
            for k, r in sel:
                b = boot.get((s, k, 'cybermetaformer', 'gru_full'))
                mm = r['models']
                out.append(f"| {r['name']} | {r['n']:,} | {r['share']:.3f} | {r['root_share']:.2f} | "
                           + ' | '.join(_m(mm[nm]['recall']) for nm in names) + ' | '
                           + ' | '.join(f"{_m(mm[nm]['root_to_clean'])} / {_m(mm[nm]['user_to_clean'])}" for nm in names)
                           + f" | {_ci(b) if b else '–'} | {_p(b['p_holm']) if b else '–'} |")
    rows = [r for r in res['rows'] if r['b'] != 'gru_full']
    if rows:
        out += ['', '## CMF − host-token GRU (topology bias), silent recall per slice', '',
                '| Set | Slice | Δ [95% CI] | p (Holm) |', '|---|---|---|---|']
        for r in rows:
            out.append(f"| {SET_LABEL[r['set']]} | {r['slice'].split('|', 1)[1]} | {_ci(r)} | {_p(r['p_holm'])} |")
    return '\n'.join(out) + '\n'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n-boot', type=int, default=10_000)
    p.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    args = p.parse_args()
    analyze(args.n_boot, tuple(args.seeds))


if __name__ == '__main__':
    main()

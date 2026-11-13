"""Paired episode-level comparison of two models.

The test unit is the episode: both models are evaluated on the same test/OOD episodes, so each bootstrap sample
gives them the same episode weights (paired). Each model enters with the mean over its seeds. With 5 paired seeds
Wilcoxon cannot go below 0.05 (smallest two-sided p 0.0625), so this method was chosen (decided 2026-10-18).

Criteria (RQ3): OOD host macro-F1, relative OOD drop (test and OOD episodes resampled independently), OOD silent recall.
p is two-sided, Holm-corrected over the three criteria. With 4,000 OOD episodes even small differences come out
significant; the decision rests on the size of the difference and its CI.

Stage 2 adds the criteria test host macro-F1 and test silent recall (`--criteria`) and cross-stage
comparisons: a name given as `<stage>/<run prefix>` is read from that stage. `--plan m2` runs all pairwise tests of
stage 2 (`m2_plan`) and writes `asama2/karsilastirma/ozet.md`.

Stage 3 adds the RQ4 slice "never alerted" (`*_never_recall`) and the episodes of the (ii)
attacker shift (`test_shift_*`, `ood_test_shift_*`; only in runs with `train_red`) as criteria. `--plan m3`
runs `m3_plan` and writes `asama3/karsilastirma/ozet.md`. A criterion resamples only its own episodes; the bootstrap
stream of the test/OOD criteria is the same as in M2.

M4 revision: an `@0,1,2` suffix on the name fixes the seed set (`asama3/cybermetaformer_k101@0,1,2`).
CMF, the full GRU (asama3) and ana_m4 now have 5 seeds, so `--plan m3` fixes the 3 seeds of M3 and reproduces the M3
record exactly. `--plan m3_5seed` runs the same plan with the 5-seed rule: main pairs with all seeds; ablations and
3-seed variants against s0–s2 of CMF. `--plan m4` holds the host-token GRU pairs of item 4 (a separate Holm family).
Both write to `asama3/karsilastirma_m4/<plan>/`; the M3 record does not change.

Input: `episode_confusion.npz` and `episode_slices.npz` in the run folders (outside git, only on the Mac). Silent
recall needs `episode_slices.npz` in every seed of both models (runs after 2026-10-18).

    python -m cybermetaformer.hidden_state.compare --a gru_k32 --b gru_full_k101      # b = reference
    python -m cybermetaformer.hidden_state.compare --stage asama2 --a transformer_k64 --b asama1/gru_k32 \
        --criteria test_host_macro_f1 ood_host_macro_f1
    python -m cybermetaformer.hidden_state.compare --plan m2
    python -m cybermetaformer.hidden_state.compare --plan m3
    python -m cybermetaformer.hidden_state.compare --plan m3_5seed --n-boot 100000
    python -m cybermetaformer.hidden_state.compare --plan m4 --n-boot 100000
"""
import argparse
import functools
import json
import os
import re

import numpy as np

from cybermetaformer.hidden_state import metrics as M

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), 'results', 'hidden_state')
SPLITS = ('test', 'ood_test')
SLICES = ('sessiz', 'never')                 # silent compromise; never alerted (RQ4, H4)


def load_runs(stage_dir, name, seeds=None):
    """<name>_s<seed> (or seedless <name>) folders → per-seed confusion, slice counts and episode summary.
    All splits in the confusion file are read (test, ood_test; in (ii) runs also test_in/_shift, ood_test_in/_shift).
    If seeds is given, only those seeds are read and all of them must exist."""
    pat = re.compile(rf'^{re.escape(name)}(_s(\d+))?$')
    runs = []
    for d in sorted(os.listdir(stage_dir)):
        m, path = pat.match(d), os.path.join(stage_dir, d)
        if not m or not os.path.exists(os.path.join(path, 'metrics.json')):
            continue
        if seeds is not None and (m.group(2) is None or int(m.group(2)) not in seeds):
            continue
        reports = json.load(open(os.path.join(path, 'metrics.json')))['reports']
        conf = np.load(os.path.join(path, 'episode_confusion.npz'))
        splits = [s for s in conf.files if s != 'val']
        sl_path = os.path.join(path, 'episode_slices.npz')
        sl = np.load(sl_path) if os.path.exists(sl_path) else None
        slices = None if sl is None else {n: {s: sl[s][:, list(sl['names']).index(n)] for s in splits if s in sl.files}
                                          for n in SLICES}
        runs.append({'seed': int(m.group(2)) if m.group(2) else None,
                     'conf': {s: conf[s] for s in splits},
                     'silent': None if slices is None else slices['sessiz'],
                     'slices': slices,
                     'hash': {s: reports[s]['episodes_hash'] for s in splits}})
    if not runs:
        raise SystemExit(f'Run not found: {stage_dir}/{name}[_s<seed>]')
    if seeds is not None and sorted(r['seed'] for r in runs) != sorted(seeds):
        raise SystemExit(f'{stage_dir}/{name}: requested seeds {sorted(seeds)}, found {[r["seed"] for r in runs]}')
    return runs


def _f1(conf, w):
    return M.host_macro_f1_batch((w @ conf.reshape(len(conf), -1)).reshape(-1, *conf.shape[1:]))


def _recall(counts, w):
    n_hit = w @ counts.astype(np.float64)
    return n_hit[:, 1] / np.maximum(n_hit[:, 0], 1)


def _slice(r, name, split):
    return r['silent'][split] if name == 'sessiz' else r['slices'][name][split]


# criterion → (good direction: +1 larger is better, -1 smaller is better; per-seed statistic)
CRITERIA = {
    'test_host_macro_f1': (+1, lambda r, W: _f1(r['conf']['test'], W['test'])),
    'ood_host_macro_f1': (+1, lambda r, W: _f1(r['conf']['ood_test'], W['ood_test'])),
    'ood_rel_drop': (-1, lambda r, W: 1 - _f1(r['conf']['ood_test'], W['ood_test']) / _f1(r['conf']['test'], W['test'])),
    'ood_silent_recall': (+1, lambda r, W: _recall(r['silent']['ood_test'], W['ood_test'])),
    'test_silent_recall': (+1, lambda r, W: _recall(r['silent']['test'], W['test'])),
}
for _c, _s in (('test', 'test'), ('ood', 'ood_test')):
    CRITERIA[f'{_c}_never_recall'] = (+1, lambda r, W, s=_s: _recall(_slice(r, 'never', s), W[s]))
SHIFT_SPLITS = {}                          # criterion → the splits it resamples (default SPLITS)
for _s in ('test_shift', 'ood_test_shift'):
    CRITERIA[f'{_s}_host_macro_f1'] = (+1, lambda r, W, s=_s: _f1(r['conf'][s], W[s]))
    CRITERIA[f'{_s}_silent_recall'] = (+1, lambda r, W, s=_s: _recall(_slice(r, 'sessiz', s), W[s]))
    SHIFT_SPLITS.update({f'{_s}_host_macro_f1': (_s,), f'{_s}_silent_recall': (_s,)})
RQ3 = ['ood_host_macro_f1', 'ood_rel_drop', 'ood_silent_recall']
SILENT = {'ood_silent_recall', 'test_silent_recall'}
# slice of the criterion (needs the slice file)
SLICE_OF = {**{c: 'sessiz' for c in CRITERIA if c.endswith('silent_recall')},
            **{c: 'never' for c in CRITERIA if c.endswith('never_recall')}}


def _has_slice(r, name):
    return r['silent'] is not None if name == 'sessiz' else bool(r.get('slices'))


def compare(runs_a, runs_b, n_boot=10_000, seed=0, criteria=RQ3):
    for s in dict.fromkeys([*SPLITS, *(s for c in criteria for s in SHIFT_SPLITS.get(c, ()))]):
        hashes = {r['hash'][s] for r in runs_a + runs_b}
        assert len(hashes) == 1, f'{s}: the two models were not evaluated on the same episodes'
    has_silent = all(r['silent'] is not None for r in runs_a + runs_b)
    out = {}
    for name in criteria:
        better, stat = CRITERIA[name]
        if name in SLICE_OF and not all(_has_slice(r, SLICE_OF[name]) for r in runs_a + runs_b):
            continue
        sizes = {s: len(runs_a[0]['conf'][s]) for s in SHIFT_SPLITS.get(name, SPLITS)}
        ones = {s: np.ones((1, E)) for s, E in sizes.items()}
        sa = lambda W, stat=stat: np.mean([stat(r, W) for r in runs_a], 0)     # noqa: E731
        sb = lambda W, stat=stat: np.mean([stat(r, W) for r in runs_b], 0)     # noqa: E731
        res = M.paired_bootstrap(sa, sb, sizes, n_boot=n_boot, seed=seed)
        out[name] = {'a': float(sa(ones)[0]), 'b': float(sb(ones)[0]), **res, 'better': better}
    for name, p in zip(out, M.holm([v['p'] for v in out.values()])):
        v = out[name]
        v['p_holm'] = p
        v['a_better'] = bool(p < 0.05 and np.sign(v['diff']) == v['better'])
        v['b_better'] = bool(p < 0.05 and np.sign(v['diff']) == -v['better'])
    return out, has_silent


def _resolve(stage, name):
    """`<stage>/<prefix>[@s,s,…]` or just a prefix (default stage) → (stage, prefix, seeds or None)."""
    name, _, pin = name.partition('@')
    seeds = tuple(int(x) for x in pin.split(',')) if pin else None
    st, n = tuple(name.split('/', 1)) if '/' in name else (stage, name)
    return st, n, seeds


def run_pair(stage, a, b, criteria=RQ3, n_boot=10_000, seed=0, cache=None, verbose=True, out_dir=None):
    """a − b test; the result is written to `<out_dir>/<a>__<b>.json` (default `<stage>/karsilastirma`; a name from
    another stage becomes `<stage>-<prefix>`). Fixed seeds (`@…`) do not enter the file name or the a/b fields;
    they are recorded in seeds_a/seeds_b."""
    cache = {} if cache is None else cache
    (sa, na, pa), (sb, nb, pb) = _resolve(stage, a), _resolve(stage, b)
    for key in [(sa, na, pa), (sb, nb, pb)]:
        if key not in cache:
            cache[key] = load_runs(os.path.join(RESULTS, key[0]), key[1], key[2])
    runs_a, runs_b = cache[sa, na, pa], cache[sb, nb, pb]
    res, has_silent = compare(runs_a, runs_b, n_boot, seed, criteria)
    if verbose:
        print(f"{a} ({len(runs_a)} seeds) − {b} ({len(runs_b)} seeds), paired episode bootstrap, {n_boot} samples, Holm")
        print(f"{'criterion':20s} {'a':>7s} {'b':>7s} {'a − b':>8s}  {'95% CI':19s} {'p':>7s} {'p Holm':>7s}")
        for name, v in res.items():
            print(f"{name:20s} {v['a']:7.4f} {v['b']:7.4f} {v['diff']:+8.4f}  [{v['lo']:+.4f}, {v['hi']:+.4f}] "
                  f"{v['p']:7.4f} {v['p_holm']:7.4f}")
        if not has_silent and SILENT & set(criteria):
            print('Silent recall skipped: episode_slices.npz missing in at least one run.')
    out_dir = out_dir or os.path.join(RESULTS, stage, 'karsilastirma')
    os.makedirs(out_dir, exist_ok=True)
    tag = lambda st, n: n if st == stage else f'{st}-{n}'      # noqa: E731
    path = os.path.join(out_dir, f'{tag(sa, na)}__{tag(sb, nb)}.json')
    rec = {'a': a.partition('@')[0], 'b': b.partition('@')[0], 'seeds_a': [r['seed'] for r in runs_a], 'seeds_b': [r['seed'] for r in runs_b],
           'n_boot': n_boot, 'bootstrap_seed': seed, 'results': res}
    with open(path, 'w') as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)
    if verbose:
        print('→', path)
    return rec


F1 = ['test_host_macro_f1', 'ood_host_macro_f1']


def m2_plan():
    """Pairwise tests of stage 2: [(family, a, b, criteria)], names `<stage>/<prefix>`. References are chosen by val
    as decided on 2026-10-18 (`aggregate.m1_refs`; `asama1/ozet/ozet.json` and `asama2/ozet/ozet.json`).
    - k=8 and k=32 (the k values where all six models exist): each M2 model − the best M1 model at the same k;
    - k=64: the light four − the best windowed M1 model (in M1 only XGB-TF exists at k=64);
    - RQ3: each M2 model at its best k by val − the full-history GRU (OOD F1, relative drop, OOD silent recall)."""
    from cybermetaformer.hidden_state.aggregate import M2_MODELS, _best, m1_refs
    s1, s2 = (json.load(open(os.path.join(RESULTS, st, 'ozet', 'ozet.json'))) for st in ['asama1', 'asama2'])
    R = m1_refs(s1)
    plan = []
    for k in (8, 32):
        b = R['best'][k]
        plan += [(f'k={k}: best M1 at the same k', f'asama2/{m}_k{k}', f"asama1/{b['model']}_k{k}", F1)
                 for m in M2_MODELS]
    bw = R['best_window']
    plan += [('k=64: best windowed M1', f'asama2/{m}_k64', f"asama1/{bw['model']}_k{bw['k']}", F1)
             for m in M2_MODELS if f'{m}_k64' in s2]
    plan += [('RQ3: full-history GRU', f"asama2/{m}_k{_best(s2, m)['k']}", 'asama1/gru_full_k101', RQ3)
             for m in M2_MODELS]
    return plan


CMF = 'cybermetaformer'
M3_CORE = ['duz78', 'host_yok', 'host_tam', 'host_gcn', 'konv_dikkat', 'hafiza_yok', 'yalniz_yas', 'yalniz_ssd',
           'konum_yok']
M3_EXTRA = ['kol_konv', 'kol_ssd', 'konv_gru', 'havuz_mlp']
M3_DG = ['dg_groupdro', 'dg_vrex', 'dg_coral', 'dg_dann', 'art_uyari', 'art_zaman']
ABL = ['test_host_macro_f1', 'ood_host_macro_f1', 'ood_silent_recall']
NEVER = ['test_never_recall', 'ood_never_recall']
SHIFT = ['test_shift_host_macro_f1', 'ood_test_shift_host_macro_f1', 'ood_test_shift_silent_recall']


def m3_plan(five=False):
    """Pairwise tests of stage 3: [(family, a, b, criteria)]. Family labels are data (ozet.json `family`).
    five=False: the M3 record; CMF, ana_m4 and the asama3 full GRU, later completed to 5 seeds, are fixed to s0–s2 of M3.
    five=True (M4 revision, §A5): RQ3 and the cross-machine repeat with all seeds; the reference facing the 3-seed
    variants is fixed to s0–s2 (same seed set). Comparisons on the same machine: ablations against
    the main model on the M3 Pro, domain generalisation / augmentation / action / B2 against `ana_m4` on the M4 Pro; the
    gru_full action against the 3-seed full GRU of asama3 (M3 Pro; cross-machine).
    - RQ3: main model − baselines (M2 models at their best k by val);
    - RQ4 (H4): models with the host mixer − without the mixer / flat model, recall on the "never alerted" slice;
    - (ii): models trained with the same Red, on the test and OOD episodes of the unseen Red."""
    from cybermetaformer.hidden_state.aggregate import _best, m1_refs
    s1, s2 = (json.load(open(os.path.join(RESULTS, st, 'ozet', 'ozet.json'))) for st in ['asama1', 'asama2'])
    R = m1_refs(s1)
    S3 = '@0,1,2'
    ana3, m43, gru3 = f'asama3/{CMF}_k101{S3}', f'asama3/{CMF}-ana_m4_k101{S3}', f'asama3/gru_full_k101{S3}'
    ana, m4 = (f'asama3/{CMF}_k101', f'asama3/{CMF}-ana_m4_k101') if five else (ana3, m43)
    refs = ['asama1/gru_full_k101', f"asama1/{R['best'][32]['model']}_k32",
            *(f"asama2/{m}_k{_best(s2, m)['k']}" for m in ('timemixer', 'transformer', 'patchtst'))]
    plan = [('RQ3: main model − baselines', ana, b, ['test_host_macro_f1', *RQ3]) for b in refs]
    ana, m4 = ana3, m43                                        # from here on against the 3-seed variants
    plan += [('Ablation core (M3 Pro, against the main model)', f'asama3/{CMF}-{v}_k101', ana, ABL) for v in M3_CORE]
    plan += [('Ablation extension (M3 Pro, against the main model)', f'asama3/{CMF}-{v}_k101', ana, ABL) for v in M3_EXTRA]
    plan += [('RQ4 (H4): never-alerted slice', a, f'asama3/{CMF}-{b}_k101', NEVER)
             for a, b in [(ana, 'host_yok'), (ana, 'duz78'), (f'asama3/{CMF}-host_tam_k101', 'host_yok'),
                          (f'asama3/{CMF}-host_gcn_k101', 'host_yok')]]
    plan += [('Domain generalisation and augmentation (M4 Pro, against ana_m4)', f'asama3/{CMF}-{v}_k101', m4,
              ['test_host_macro_f1', *RQ3]) for v in M3_DG]
    plan += [('Action input', f'asama3/{CMF}-aksiyon_k101', m4, [*ABL, 'test_silent_recall']),
             ('Action input', 'asama3/gru_full-aksiyon_k101', gru3, [*ABL, 'test_silent_recall'])]
    plan += [('Scale and cross-machine replication', f'asama3/{CMF}-b2_k101', m4, ABL),
             ('Scale and cross-machine replication', *((f'asama3/{CMF}-ana_m4_k101', f'asama3/{CMF}_k101') if five
                                                   else (m4, ana)), ABL)]
    for red in ('bline', 'meander'):
        a = f'asama3/{CMF}-saldirgan_{red}_k101'
        plan += [(f'(ii) attacker shift: trained only on {red}', a, b, SHIFT)
                 for b in (f'asama3/gru_full-saldirgan_{red}_k101', f'asama3/timemixer-saldirgan_{red}_k64')]
    return plan


M4_ALL = ['test_host_macro_f1', 'ood_host_macro_f1', 'ood_rel_drop', 'test_silent_recall', 'ood_silent_recall',
          'test_never_recall', 'ood_never_recall']


def m4_plan():
    """M4 revision, item 4: host-token GRU + cross-host attention references, 5 seeds. By the same-machine
    rule gru_host_* (M4 Pro) against ana_m4 on the M4 Pro; the comparison with the full GRU (M3 Pro) is
    cross-machine (the ana_m4 − main repeat is in m3_5seed)."""
    m4, gru = f'asama3/{CMF}-ana_m4_k101', 'asama1/gru_full_k101'
    topo, attn = f'asama3/{CMF}-gru_host_topo_k101', f'asama3/{CMF}-gru_host_attn_k101'
    return [('A4: host-token GRU − CMF (M4 Pro, against ana_m4)', topo, m4, M4_ALL),
            ('A4: host-token GRU − CMF (M4 Pro, against ana_m4)', attn, m4, M4_ALL),
            ('A4: topology bias within the host-token GRU (M4 Pro)', topo, attn, M4_ALL),
            ('A4: host-token GRU − full-history GRU (across machines)', topo, gru, M4_ALL),
            ('A4: host-token GRU − full-history GRU (across machines)', attn, gru, M4_ALL)]


# plan → (stage, plan builder, title, output folder (relative to the stage), seed column in the table)
PLANS = {'m2': ('asama2', m2_plan, 'M2', 'karsilastirma', False),
         'm3': ('asama3', m3_plan, 'M3', 'karsilastirma', False),
         'm3_5seed': ('asama3', functools.partial(m3_plan, five=True), 'M3, 5 seeds (M4 revision, item 5)',
                      os.path.join('karsilastirma_m4', 'm3_5seed'), True),
         'm4': ('asama3', m4_plan, 'M4 revision, item 4', os.path.join('karsilastirma_m4', 'm4'), True)}


def run_plan(name, n_boot=10_000, seed=0, out_dir=None):
    """Runs the plan; one JSON per pair, plus `<output folder>/ozet.md` (within-pair and plan-wide Holm)."""
    stage, make, title, sub, show_seeds = PLANS[name]
    out_dir = out_dir or os.path.join(RESULTS, stage, sub)
    cache, rows = {}, []
    plan = make()
    for i, (fam, a, b, crit) in enumerate(plan, 1):
        print(f'[{i}/{len(plan)}] {a} − {b}', flush=True)
        rec = run_pair(stage, a, b, crit, n_boot, seed, cache, verbose=False, out_dir=out_dir)
        rows += [(fam, a, b, name, v, rec) for name, v in rec['results'].items()]
    for row, p in zip(rows, M.holm([r[4]['p'] for r in rows])):
        row[4]['p_holm_plan'] = p
    lab = lambda n: n.partition('@')[0].split('/', 1)[1]        # noqa: E731
    seed_note = ("Each model enters with the mean over the given seeds (column: number of seeds). "
                 if show_seeds else "Each model enters with the mean over its seeds. ")
    lines = [f'# {title} pairwise tests (paired episode bootstrap)', '',
             f"{n_boot} samples, bootstrap seed {seed}. {seed_note}a − b; 95% CI. "
             "p Holm: pair = over the criteria of the pair; plan = over all rows. Smallest p 2 / (n + 1). "
             f"For the relative drop smaller is better. `python -m cybermetaformer.hidden_state.compare --plan {name}`.", '']
    if show_seeds:
        lines[-1:-1] = [f'Plan-wide Holm family: {len(rows)} rows; smallest raw p {2 / (n_boot + 1):.2g}, smallest '
                        f'plan-wide corrected p {min(1.0, len(rows) * 2 / (n_boot + 1)):.2g}.']
    for fam in dict.fromkeys(r[0] for r in rows):
        if show_seeds:
            lines += [f'## {fam}', '', '| a | b | seeds a / b | criterion | a | b | a − b | 95% CI | p Holm (pair) | '
                      'p Holm (plan) |', '|---|---|---|---|---:|---:|---:|---|---:|---:|']
        else:
            lines += [f'## {fam}', '', '| a | b | criterion | a | b | a − b | 95% CI | p Holm (pair) | p Holm (plan) |',
                      '|---|---|---|---:|---:|---:|---|---:|---:|']
        for _, a, b, name, v, rec in [r for r in rows if r[0] == fam]:
            sd = (f" {len(rec['seeds_a'])} / {len(rec['seeds_b'])} |" if show_seeds else '')
            lines.append(f"| {lab(a)} | {lab(b)} |{sd} {name} | {v['a']:.4f} | {v['b']:.4f} | {v['diff']:+.4f} | "
                         f"[{v['lo']:+.4f}, {v['hi']:+.4f}] | {v['p_holm']:.4f} | {v['p_holm_plan']:.4f} |")
        lines.append('')
    path = os.path.join(out_dir, 'ozet.md')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    # rows with plan-wide Holm in one file (the paper tables also read the M3 record from here)
    with open(os.path.join(out_dir, 'ozet.json'), 'w') as f:
        json.dump({'plan': name, 'n_boot': n_boot, 'bootstrap_seed': seed, 'n_rows': len(rows),
                   'rows': [{'family': fam, 'model_a': rec['a'], 'model_b': rec['b'], 'seeds_a': rec['seeds_a'],
                             'seeds_b': rec['seeds_b'], 'criterion': c, **v}
                            for fam, _, _, c, v, rec in rows]}, f, indent=1, ensure_ascii=False)
    print('→', path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--a', help='compared model, run name prefix (e.g. gru_k32; another stage: asama1/gru_k32)')
    p.add_argument('--b', help='reference model (e.g. gru_full_k101)')
    p.add_argument('--stage', default='asama1')
    p.add_argument('--criteria', nargs='+', default=RQ3, choices=list(CRITERIA))
    p.add_argument('--plan', choices=list(PLANS), help='run all tests of stage 2 / stage 3 / the M4 revision')
    p.add_argument('--n-boot', type=int, default=10_000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out-dir', help='only with --plan: output folder (default: from the plan definition)')
    args = p.parse_args()
    if args.plan:
        run_plan(args.plan, args.n_boot, args.seed, args.out_dir)
    elif args.a and args.b:
        run_pair(args.stage, args.a, args.b, args.criteria, args.n_boot, args.seed)
    else:
        p.error('--a and --b, or --plan, are required')


if __name__ == '__main__':
    main()

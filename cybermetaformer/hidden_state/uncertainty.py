"""Uncertainty and calibration (M3 preparation): deep ensemble, MC dropout, selective prediction.

With the saved M2 weights (`model.pt`), no retraining. The light four (Transformer, iTransformer, TSMixer,
TimeMixer; best k by val = 64), 3 seeds. TimesNet is left out: its output depends on the evaluation batch,
so the member probabilities are not defined per window. PatchTST is added once its rerun finishes.

Five evaluation sets (all with models trained on `minicage_v1`; temperature only on the v1 val):
    val, test, ood (v1 `ood_test`), bug_test, bug_ood (`minicage_v1_bugged`; simulator version shift)

Configurations (each with host macro-F1, breach F1, silent recall, ECE/Brier/NLL, selective prediction;
the names are keys in ozet.json):
    uye            per member, with its own val temperature (as reported in M2); seed mean
    topluluk_ham   mean of the member softmaxes (T=1)
    topluluk       mean of the member softmaxes, each scaled by its own temperature
    topluluk_T     topluluk_ham + the ensemble's own val temperature (on the log-probabilities)
    mc             single seed (0), N-pass MC dropout (dropout on, BatchNorm eval) + its own val temperature
    capraz         ensemble of 4 architectures × 3 seeds = 12 members (like topluluk_T)
Paired episode bootstrap (`metrics.paired_bootstrap`): topluluk_T − uye for ECE, NLL, host macro-F1.

Selective prediction: confidence = probability of the most likely class (per host-step); the least confident host-steps
are dropped. Report: area under the risk–coverage curve (AURC), risk at coverage 0.95/0.9/0.8 and host macro-F1 on the rest.

    python -m cybermetaformer.hidden_state.uncertainty collect            # asama2/belirsizlik/logits/ (outside git)
    python -m cybermetaformer.hidden_state.uncertainty mc --passes 16     # MC dropout (seed 0)
    python -m cybermetaformer.hidden_state.uncertainty analyze            # ozet.json, tablolar.md and figures
    python -m cybermetaformer.hidden_state.uncertainty figures            # figures only (from ozet.json)
"""
import argparse
import itertools
import json
import os
import time

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt                     # noqa: E402
import numpy as np                                  # noqa: E402
import torch                                        # noqa: E402
import yaml                                         # noqa: E402
from matplotlib.lines import Line2D                 # noqa: E402

from cybermetaformer.hidden_state import aggregate as A     # noqa: E402
from cybermetaformer.hidden_state import data as D          # noqa: E402
from cybermetaformer.hidden_state import evaluate as E      # noqa: E402
from cybermetaformer.hidden_state import features as F      # noqa: E402
from cybermetaformer.hidden_state import metrics as M       # noqa: E402
from cybermetaformer.hidden_state import train as T         # noqa: E402
from cybermetaformer.hidden_state.explain import STAGE_DIR, load_model   # noqa: E402

OUT_DIR = os.path.join(STAGE_DIR, 'belirsizlik')
LOGIT_DIR = os.path.join(OUT_DIR, 'logits')
BUGGED = os.path.join(D.DATA_ROOT, 'minicage_v1_bugged')
MODELS = ['transformer', 'itransformer', 'tsmixer', 'timemixer']
K = 64
SEEDS = (0, 1, 2)
SETS = {'val': ('v1', 'val'), 'test': ('v1', 'test'), 'ood': ('v1', 'ood_test'),
        'bug_test': ('bug', 'test'), 'bug_ood': ('bug', 'ood_test')}
EVAL_SETS = ['test', 'ood', 'bug_test', 'bug_ood']
COVERAGES = (0.95, 0.9, 0.8)
N_BINS = 15
LOGIT_FLOOR = -40.0           # against zero probability in the MC mean (float16 log-probability)


def run_name(name, k, seed, mc=0):
    return f'{name}_k{k}_s{seed}' + (f'_mc{mc}' if mc else '')


def open_stores():
    return {'v1': D.EpisodeStore(), 'bug': D.EpisodeStore(BUGGED)}


def set_indices(stores):
    """{set: episode indices}; the final-evaluation lock is opened for access to ood_test."""
    with stores['v1'].final_evaluation(), stores['bug'].final_evaluation():
        return {s: stores[src].split(split) for s, (src, split) in SETS.items()}


# ── prediction ───────────────────────────────────────────────────────────

def mc_mode(model):
    """Dropout on; BatchNorm eval (running statistics, not batch statistics). `nn.MultiheadAttention` is eval too:
    `scaled_dot_product_attention` on MPS does not support dropout (NotImplementedError), so in the vanilla Transformer
    the dropout of the attention weights does not enter MC; residual and FFN dropouts (nn.Dropout) are on."""
    model.train()
    for m in model.modules():
        if isinstance(m, (torch.nn.modules.batchnorm._BatchNorm, torch.nn.MultiheadAttention)):
            m.eval()


@torch.no_grad()
def predict_mc(model, wd, k, passes, scale='global', batch_size=65536):
    """MC dropout: log of the softmax mean over `passes` passes per window (n, T, 12, 3)."""
    mc_mode(model)
    batch_size = min(batch_size, max(2048, batch_size * 8 // max(k, 8)))
    out = []
    for ep, t in D.eval_indices(wd.n, wd.T, batch_size):
        x, mask = wd.windows(ep, t, k)
        x = D.scale_obs(x, scale)
        p = sum(torch.softmax(model(x, mask).float(), -1) for _ in range(passes)) / passes
        out.append(p.clamp_min(float(np.exp(LOGIT_FLOOR))).log().cpu())
    return torch.cat(out).view(wd.n, wd.T, len(D.LABEL_HOSTS), D.N_CLASSES).numpy()


def collect_run(name, k, seed, stores, idx, device, mc=0):
    """Logits of the five sets (float16) → belirsizlik/logits/<run>.npz. mc > 0: MC dropout log-probabilities."""
    path = os.path.join(LOGIT_DIR, run_name(name, k, seed, mc) + '.npz')
    if os.path.exists(path):
        return path
    cfg = {**T.DEFAULTS, **yaml.safe_load(open(os.path.join(STAGE_DIR, f'{name}_k{k}_s{seed}', 'config.yaml')))}
    model = load_model(name, k, seed, device)
    t0, out = time.time(), {}
    if mc:
        torch.manual_seed(1000 + seed)
    for s, (src, _) in SETS.items():
        wd = D.WindowData(stores[src], idx[s], device=device)
        if mc:
            lg = predict_mc(model, wd, k, mc, cfg['scale'], cfg['eval_batch_size'])
        else:
            lg = E.predict_logits(model, wd, k, cfg['scale'], cfg['eval_batch_size'])
        out[s] = lg.astype(np.float16)
    os.makedirs(LOGIT_DIR, exist_ok=True)
    np.savez_compressed(path, **out)
    print(f'{run_name(name, k, seed, mc)}: {time.time() - t0:.0f} s', flush=True)
    return path


def load_logits(name, k, seed, mc=0):
    with np.load(os.path.join(LOGIT_DIR, run_name(name, k, seed, mc) + '.npz')) as z:
        return {s: z[s] for s in z.files}


# ── evaluation set and metrics ───────────────────────────────────────────

class EvalSet:
    """Labels and alerts of a set (all (E, T, 12)); number of episodes and of host-steps."""

    def __init__(self, store, idx):
        self.y = store.labels[idx].astype(np.int64)
        self.alert = F.episode_alerts(store.obs[idx])
        self.E = len(idx)
        self.n = self.y.size


def selective(P, pred, es):
    """Risk–coverage curve over host-steps sorted by confidence: AURC, risk and host macro-F1 per coverage."""
    conf = P.max(-1).reshape(-1)
    err = (pred != es.y).reshape(-1)
    order = np.argsort(-conf, kind='stable')
    risk = np.cumsum(err[order]) / np.arange(1, len(err) + 1)
    out = {'aurc': float(risk.mean())}
    y2, p2 = es.y.reshape(-1, es.y.shape[-1]), pred.reshape(-1, es.y.shape[-1])
    for c in COVERAGES:
        m = int(round(c * len(err)))
        keep = np.zeros(len(err), bool)
        keep[order[:m]] = True
        out[f'err@{c}'] = float(risk[m - 1])
        out[f'f1@{c}'] = M.host_macro_f1(y2, p2, mask=keep.reshape(es.y.shape).reshape(-1, es.y.shape[-1]))
    grid = (np.linspace(0.5, 1.0, 101) * len(err)).astype(np.int64) - 1
    out['curve'] = {'coverage': np.linspace(0.5, 1.0, 101).tolist(), 'risk': risk[grid].tolist()}
    return out


def reliability(P, y, n_bins=N_BINS):
    conf, correct = P.max(-1).reshape(-1), (P.argmax(-1) == y).reshape(-1)
    bins = np.minimum((conf * n_bins).astype(np.int64), n_bins - 1)
    cnt = np.bincount(bins, minlength=n_bins)
    return {'count': cnt.tolist(), 'accuracy': (np.bincount(bins, correct, minlength=n_bins) / np.maximum(cnt, 1)).tolist(),
            'confidence': (np.bincount(bins, conf, minlength=n_bins) / np.maximum(cnt, 1)).tolist()}


def evaluate_probs(P, es, curve=False):
    """P (E, T, 12, 3) probabilities → (report, per-episode confusion (E, 12, 3, 3))."""
    pred = P.argmax(-1)
    H = es.y.shape[-1]
    rep = {'host_macro_f1': M.host_macro_f1(es.y.reshape(-1, H), pred.reshape(-1, H)), 'breach_f1': M.breach_f1(es.y, pred),
           'silent_recall': M.silent_metrics(es.y, pred, es.alert)['silent_recall']}
    rep.update(M.calibration(P, es.y))
    sel = selective(P, pred, es)
    if not curve:
        sel.pop('curve')
    rep.update(sel)
    if curve:
        rep['reliability'] = reliability(P, es.y)
    return rep, M.episode_confusion(es.y, pred)


def episode_calibration_stats(P, y, n_bins=N_BINS):
    """Per-episode ECE/NLL components (for the bootstrap): per-bin accuracy and confidence sums, NLL sum."""
    Ne = P.shape[0]
    conf, correct = P.max(-1).reshape(Ne, -1), (P.argmax(-1) == y).reshape(Ne, -1)
    bins = np.minimum((conf * n_bins).astype(np.int64), n_bins - 1)
    flat = (np.arange(Ne)[:, None] * n_bins + bins).reshape(-1)
    acc = np.bincount(flat, correct.reshape(-1), minlength=Ne * n_bins).reshape(Ne, n_bins)
    cs = np.bincount(flat, conf.reshape(-1), minlength=Ne * n_bins).reshape(Ne, n_bins)
    py = np.take_along_axis(P, y[..., None], -1)[..., 0].reshape(Ne, -1)
    return {'acc': acc, 'conf': cs, 'nll': -np.log(np.clip(py, 1e-12, None)).sum(1), 'n': conf.shape[1]}


def stat_fns(P, es):
    """Functions computing ECE, NLL and host macro-F1 of a probability array from episode weights W (n, E)."""
    cal = episode_calibration_stats(P, es.y)
    ep_conf = M.episode_confusion(es.y, P.argmax(-1))
    return {'ece': lambda W: np.abs(W @ cal['acc'] - W @ cal['conf']).sum(1) / (W.sum(1) * cal['n']),
            'nll': lambda W: (W @ cal['nll']) / (W.sum(1) * cal['n']),
            'host_macro_f1': lambda W: M.host_macro_f1_batch(np.tensordot(W, ep_conf, axes=1))}


# ── probabilities ────────────────────────────────────────────────────────

def probs(logits, temperature=1.0):
    return M.softmax(logits.astype(np.float32), temperature).astype(np.float32)


def val_temperature(logits_val, store_v1, idx_val):
    return E.fit_temperature(logits_val, store_v1.labels[idx_val])


def rescale(P, temperature):
    """The log of the probabilities is scaled like a logit (ensemble temperature)."""
    return probs(np.log(np.clip(P, 1e-12, None)), temperature)


def mc_passes(name):
    """Number of passes of the MC dropout run on disk (0 if none)."""
    prefix = f'{name}_k{K}_s0_mc'
    found = [f for f in (os.listdir(LOGIT_DIR) if os.path.isdir(LOGIT_DIR) else []) if f.startswith(prefix)]
    return int(found[0][len(prefix):-len('.npz')]) if found else 0


def analyze_group(name, members, mc, stores, idx, sets, n_boot, log=print):
    """members: logit dicts of the members; mc: MC dropout log-probability dict of seed 0, or None.
    Returns: {'temperatures', set: {configuration: report}}."""
    v1, idx_val = stores['v1'], idx['val']
    temps = [val_temperature(lg['val'], v1, idx_val) for lg in members]
    ham_val = np.mean([probs(lg['val']) for lg in members], 0)
    t_ens = E.fit_temperature(np.log(np.clip(ham_val, 1e-12, None)), v1.labels[idx_val])
    res = {'temperatures': {'uye': temps, 'topluluk_T': t_ens}}
    if mc is not None:
        res['temperatures']['mc'] = val_temperature(mc['val'], v1, idx_val)
    for s, es in sets.items():
        per = [probs(lg[s], t) for lg, t in zip(members, temps)]
        uye = [evaluate_probs(P, es)[0] for P in per]
        P_ham = np.mean([probs(lg[s]) for lg in members], 0)
        P_ens = rescale(P_ham, t_ens)
        out = {'uye': {key: [float(r[key]) for r in uye] for key in uye[0]},
               'topluluk_ham': evaluate_probs(P_ham, es)[0],
               'topluluk': evaluate_probs(np.mean(per, 0), es)[0],
               'topluluk_T': evaluate_probs(P_ens, es, curve=True)[0],
               'uye_seed0': evaluate_probs(per[0], es, curve=True)[0]}
        if mc is not None:
            out['mc_ham'] = evaluate_probs(probs(mc[s]), es)[0]
            out['mc'] = evaluate_probs(probs(mc[s], res['temperatures']['mc']), es, curve=True)[0]
        # topluluk_T − uye (mean of the members): paired episode bootstrap
        fns_ens, fns_members = stat_fns(P_ens, es), [stat_fns(P, es) for P in per]
        out['fark'] = {stat: M.paired_bootstrap(
            lambda W, f=fns_ens[stat]: f(W['s']),
            lambda W, st=stat: np.mean([f[st](W['s']) for f in fns_members], 0),
            {'s': es.E}, n_boot=n_boot) for stat in ('ece', 'nll', 'host_macro_f1')}
        res[s] = out
        log(f'  {name:12s} {s:9s} member F1 {np.mean(out["uye"]["host_macro_f1"]):.4f} ECE {np.mean(out["uye"]["ece"]):.4f}  '
            f'topluluk_T F1 {out["topluluk_T"]["host_macro_f1"]:.4f} ECE {out["topluluk_T"]["ece"]:.4f}')
    return res


def diversity_control(stores, idx, sets, n_draw=30, seed=0, log=print):
    """Does the ensemble gain come from the number of members or from architectural diversity? Size 3: the 3 seeds of
    one architecture (4 ensembles) and one member from each of three architectures (`n_draw` random draws out of 108
    possible triples; triples share members, so they are not independent: mean and range only). Each ensemble is scaled
    by its own val temperature. Sizes 1 and 12 for comparison."""
    names = [(n, s) for n in MODELS for s in SEEDS]
    cache = {}
    for n, s in names:
        lg = load_logits(n, K, s)
        cache[(n, s)] = {k: probs(lg[k]) for k in ('val', 'ood', 'bug_ood')}
    v1, idx_val = stores['v1'], idx['val']

    def evaluate(keys):
        t = E.fit_temperature(np.log(np.clip(np.mean([cache[m]['val'] for m in keys], 0), 1e-12, None)), v1.labels[idx_val])
        out = {}
        for s in ('ood', 'bug_ood'):
            P = rescale(np.mean([cache[m][s] for m in keys], 0), t)
            cal = M.calibration(P, sets[s].y)
            H = sets[s].y.shape[-1]
            out[s] = {'host_macro_f1': M.host_macro_f1(sets[s].y.reshape(-1, H), P.argmax(-1).reshape(-1, H)),
                      'ece': cal['ece'], 'nll': cal['nll']}
        return out

    rng = np.random.default_rng(seed)
    combos = list(itertools.combinations(MODELS, 3))
    cross = [[(n, int(rng.choice(SEEDS))) for n in combos[rng.integers(len(combos))]] for _ in range(n_draw)]
    groups = {'tek': [[m] for m in names], 'ayni_mimari_3': [[(n, s) for s in SEEDS] for n in MODELS],
              'capraz_mimari_3': cross, 'capraz_12': [names]}
    out = {}
    for g, ensembles in groups.items():
        rows = [evaluate(keys) for keys in ensembles]
        out[g] = {s: {stat: {'mean': float(np.mean([r[s][stat] for r in rows])),
                             'min': float(np.min([r[s][stat] for r in rows])), 'max': float(np.max([r[s][stat] for r in rows]))}
                      for stat in ('host_macro_f1', 'ece', 'nll')} for s in ('ood', 'bug_ood')}
        out[g]['n_topluluk'] = len(ensembles)
        log(f'  diversity {g:16s} ({len(ensembles):3d} ensembles) OOD F1 {out[g]["ood"]["host_macro_f1"]["mean"]:.4f} '
            f'ECE {out[g]["ood"]["ece"]["mean"]:.4f}')
    return out


def analyze(n_boot=4000, log=print):
    stores = open_stores()
    idx = set_indices(stores)
    with stores['v1'].final_evaluation(), stores['bug'].final_evaluation():
        sets = {s: EvalSet(stores[src], idx[s]) for s, (src, _) in SETS.items() if s != 'val'}
    out = {'k': K, 'seeds': list(SEEDS), 'episodes': {s: int(len(i)) for s, i in idx.items()}}
    for name in MODELS:
        passes = mc_passes(name)
        out[name] = analyze_group(name, [load_logits(name, K, s) for s in SEEDS],
                                  load_logits(name, K, 0, passes) if passes else None, stores, idx, sets, n_boot, log)
        out[name]['mc_passes'] = passes
    out['capraz'] = analyze_group('capraz', [load_logits(n, K, s) for n in MODELS for s in SEEDS], None, stores, idx,
                                  sets, n_boot, log)
    out['cesitlilik'] = diversity_control(stores, idx, sets, log=log)
    return out


# ── tables ───────────────────────────────────────────────────────────────

SET_LABEL = {'test': 'test (v1, seen defenders)', 'ood': 'OOD (v1 `ood_test`, ε=0.3)',
             'bug_test': 'bugged simulator, seen defenders', 'bug_ood': 'bugged simulator, OOD (ε=0.3)'}
GROUP_LABEL = {**{m: A.LABEL[m] for m in MODELS}, 'capraz': '4 architectures × 3 seeds (12 members)'}


def _delta(d, digits=4):
    star = '*' if d['lo'] > 0 or d['hi'] < 0 else ''
    return f"{d['diff']:+.{digits}f} [{d['lo']:+.{digits}f}, {d['hi']:+.{digits}f}]{star}"


def tables_md(res):
    groups = [g for g in (*MODELS, 'capraz') if g in res]
    mean = np.mean
    out = ['# Uncertainty and calibration: deep ensemble, MC dropout, selective prediction (generated)', '',
           f"`uncertainty.py analyze`. The light four, k={res['k']}, seeds {list(res['seeds'])}; episode counts "
           f"{res['episodes']}. *member*: seed mean of the metrics of the members scaled by their own val temperature; "
           f"*ens.*: `topluluk_T` (mean of the member T=1 softmaxes + the ensemble's val temperature). Δ = ens. − member, "
           f"paired episode bootstrap 95% CI; `*`: the CI excludes 0.", '']
    for s in EVAL_SETS:
        out += [f'## {SET_LABEL[s]}', '',
                '| Model | F1 member | F1 ens. | ΔF1 | ECE member | ECE ens. | ΔECE | NLL member | NLL ens. | ΔNLL |',
                '|---|---|---|---|---|---|---|---|---|---|']
        for g in groups:
            r = res[g][s]
            out.append(f"| {GROUP_LABEL[g]} | {mean(r['uye']['host_macro_f1']):.4f} | {r['topluluk_T']['host_macro_f1']:.4f} | "
                       f"{_delta(r['fark']['host_macro_f1'])} | {mean(r['uye']['ece']):.4f} | {r['topluluk_T']['ece']:.4f} | "
                       f"{_delta(r['fark']['ece'])} | {mean(r['uye']['nll']):.4f} | {r['topluluk_T']['nll']:.4f} | "
                       f"{_delta(r['fark']['nll'])} |")
        out.append('')
    out += ['## Silent recall and breach F1 (member / ens.)', '',
            '| Model | ' + ' | '.join(f'{s} silent' for s in EVAL_SETS) + ' | ' + ' | '.join(f'{s} breach F1' for s in EVAL_SETS) + ' |',
            '|---|' + '---|' * (2 * len(EVAL_SETS))]
    for g in groups:
        cells = [f"{mean(res[g][s]['uye'][key]):.4f} / {res[g][s]['topluluk_T'][key]:.4f}"
                 for key in ('silent_recall', 'breach_f1') for s in EVAL_SETS]
        out.append(f'| {GROUP_LABEL[g]} | ' + ' | '.join(cells) + ' |')
    out += ['', '## Calibration configurations: ECE', '',
            'Ensemble without temperature (`ham`), ensemble scaled by the member temperatures (`T_member`), ensemble '
            'temperature (`T_ens.`); MC dropout single seed (0), `member s0` the same seed.', '',
            '| Model | Set | member (T) | member s0 | ens. ham | ens. T_member | ens. T_ens. | MC ham | MC T |',
            '|---|---|---|---|---|---|---|---|---|']
    for g in groups:
        for s in ('ood', 'bug_ood'):
            r = res[g][s]
            mc = ('—', '—') if 'mc' not in r else (f"{r['mc_ham']['ece']:.4f}", f"{r['mc']['ece']:.4f}")
            out.append(f"| {GROUP_LABEL[g]} | {s} | {mean(r['uye']['ece']):.4f} | {r['uye_seed0']['ece']:.4f} | "
                       f"{r['topluluk_ham']['ece']:.4f} | {r['topluluk']['ece']:.4f} | {r['topluluk_T']['ece']:.4f} | "
                       f"{mc[0]} | {mc[1]} |")
    out += ['', '## Selective prediction (OOD and bugged OOD)', '',
            'Confidence is the probability of the most likely class; the least confident host-steps are dropped. AURC: area '
            'under the risk–coverage curve (smaller is better). F1@c: host macro-F1 on the remaining host-steps at coverage c. '
            'Three configurations: `member s0` / `ens.` / `MC` (— if no MC).', '',
            '| Model | Set | AURC | risk@0.9 | F1@0.95 | F1@0.9 | F1@0.8 |', '|---|---|---|---|---|---|---|']
    for g in groups:
        for s in ('ood', 'bug_ood'):
            r = res[g][s]
            cfgs = [r['uye_seed0'], r['topluluk_T'], r.get('mc')]

            def triple(key, d=4):
                return ' / '.join('—' if c is None else f'{c[key]:.{d}f}' for c in cfgs)
            out.append(f"| {GROUP_LABEL[g]} | {s} | {triple('aurc', 5)} | {triple('err@0.9', 5)} | {triple('f1@0.95')} | "
                       f"{triple('f1@0.9')} | {triple('f1@0.8')} |")
    if 'cesitlilik' in res:
        c = res['cesitlilik']
        out += ['', '## Does the gain come from the number of members or from diversity?', '',
                'Each ensemble is scaled by its own val temperature. Cell: mean [min, max] (per ensemble). '
                "Same architecture: the 3 seeds of one architecture (4 ensembles). Different architectures: one member "
                f"from each of three architectures ({c['capraz_mimari_3']['n_topluluk']} random draws out of 108 possible triples; "
                "triples share members, "
                'so the range is not an independent confidence interval).', '',
                '| Ensemble | Size | OOD F1 | OOD ECE | bugged OOD F1 | bugged OOD ECE |', '|---|---|---|---|---|---|']
        for g, label, size in (('tek', 'single member (mean over 12)', 1), ('ayni_mimari_3', 'same architecture', 3),
                               ('capraz_mimari_3', 'different architectures', 3), ('capraz_12', 'four architectures × 3 seeds', 12)):
            def cell(s, stat):
                d = c[g][s][stat]
                return f"{d['mean']:.4f}" + ('' if g == 'capraz_12' else f" [{d['min']:.4f}, {d['max']:.4f}]")
            out.append(f"| {label} | {size} | {cell('ood', 'host_macro_f1')} | {cell('ood', 'ece')} | "
                       f"{cell('bug_ood', 'host_macro_f1')} | {cell('bug_ood', 'ece')} |")
    out += ['', '## Temperatures (val)', '', '| Model | member (seed 0, 1, 2 ...) | ensemble | MC |', '|---|---|---|---|']
    for g in groups:
        t = res[g]['temperatures']
        mc = f"{t['mc']:.3f}" if 'mc' in t else '—'
        out.append(f"| {GROUP_LABEL[g]} | {', '.join(f'{x:.3f}' for x in t['uye'])} | {t['topluluk_T']:.3f} | {mc} |")
    return '\n'.join(out) + '\n'


# ── figures ──────────────────────────────────────────────────────────────

FIG_DIR = os.path.join(STAGE_DIR, 'figures')
MIN_BIN = 200                         # minimum host-steps for a bin to be shown in the reliability plot


def _color(g):
    return A.SLOT.get(g, A.INK)


def _panels(res, title, subtitle, fig_name, draw, legend, ylabel=None, sharey=True):
    groups = [g for g in (*MODELS, 'capraz') if g in res]
    fig, axes = plt.subplots(1, len(groups), figsize=(3.0 * len(groups) + 0.5, 4.2), sharey=sharey)
    for ax, g in zip(axes, groups):
        draw(ax, g, res[g])
        ax.set_title(GROUP_LABEL[g] if g != 'capraz' else '4 architectures × 3 seeds', loc='left', fontsize=10, color=A.INK)
        A._grid(ax)
    if ylabel:
        axes[0].set_ylabel(ylabel)
    fig.legend(handles=legend, loc='lower center', ncol=len(legend), fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, x=0.01, ha='left', fontsize=11, fontweight='bold')
    fig.text(0.01, 0.905, subtitle, ha='left', fontsize=9, color=A.INK2)
    fig.tight_layout(rect=(0, 0.08, 1, 0.9))
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, fig_name), dpi=180)
    plt.close(fig)


def _legend_pair(second, dashed=None):
    h = [Line2D([], [], color=A.INK2, lw=1.2, marker='o', ms=6, mfc=A.SURFACE, mec=A.INK2, mew=1.5, label='member (seed 0)'),
         Line2D([], [], color=A.INK, lw=2, marker='o', ms=6, mec=A.SURFACE, label=second)]
    if dashed:
        h.append(Line2D([], [], color=A.INK, lw=1.5, ls=':', label=dashed))
    return h


def fig_reliability(res, s='ood'):
    """Accuracy − confidence (per bin): negative = overconfident. Member (seed 0) and ensemble."""
    def draw(ax, g, r):
        ax.axhline(0, color=A.MUTED, lw=1, ls='--')
        for key, kw in (('uye_seed0', dict(color=A.INK2, mfc=A.SURFACE, mec=A.INK2, mew=1.5, lw=1.2)),
                        ('topluluk_T', dict(color=_color(g), mec=A.SURFACE, mew=1, lw=2))):
            rel = r[s][key]['reliability']
            cnt, acc, conf = (np.array(rel[x]) for x in ('count', 'accuracy', 'confidence'))
            show = cnt >= MIN_BIN
            ax.plot(conf[show], (acc - conf)[show], marker='o', ms=6, zorder=3, **kw)
        ax.set_xlabel('confidence (probability of the most likely class)')
    _panels(res, f'Calibration gap, {SET_LABEL[s]}'.replace('`', ''),
            f'Accuracy − confidence per bin (15 equal bins; bins with ≥ {MIN_BIN} host-steps). 0 = calibrated, negative = '
            'overconfident. Both scaled by the val temperature.',
            f'fig_belirsizlik_guvenilirlik_{s}.png', draw, _legend_pair('ensemble (model colour)'),
            ylabel='accuracy − confidence')


def fig_risk_coverage(res, s='ood'):
    def draw(ax, g, r):
        for key, kw in (('uye_seed0', dict(color=A.INK2, lw=1.2)), ('topluluk_T', dict(color=_color(g), lw=2)),
                        ('mc', dict(color=_color(g), lw=1.5, ls=':'))):
            if key in r[s]:
                c = r[s][key]['curve']
                ax.plot(c['coverage'], 100 * np.array(c['risk']), **kw)
        ax.set_xlabel('coverage (share of host-steps kept)')
    _panels(res, f'Selective prediction, {SET_LABEL[s]}'.replace('`', ''),
            'The least confident host-steps are dropped; risk on the remaining steps. Lower curve is better.',
            f'fig_belirsizlik_secici_{s}.png', draw, _legend_pair('ensemble (model colour)', 'MC dropout (seed 0)'),
            ylabel='risk (%)')


def fig_shift_ladder(res):
    """Change of ECE across the four sets: member (seed range) and ensemble."""
    order = ['test', 'bug_test', 'ood', 'bug_ood']
    names = ['test', 'bugged\ntest', 'OOD', 'bugged\nOOD']

    def draw(ax, g, r):
        x = np.arange(len(order))
        lo = np.array([min(r[s]['uye']['ece']) for s in order])
        hi = np.array([max(r[s]['uye']['ece']) for s in order])
        mid = np.array([np.mean(r[s]['uye']['ece']) for s in order])
        ens = np.array([r[s]['topluluk_T']['ece'] for s in order])
        ax.vlines(x, lo, hi, color=A.INK2, lw=1)
        ax.plot(x, mid, ls='none', marker='o', ms=7, mfc=A.SURFACE, mec=A.INK2, mew=1.5, zorder=3)
        ax.plot(x, ens, color=_color(g), lw=2, marker='o', ms=7, mec=A.SURFACE, zorder=4)
        ax.set_xticks(x, names, fontsize=8.5)
    _panels(res, 'Calibration under shift: ECE',
            'Member: mean and range over 3 seeds; ensemble: the model colour. Test → bugged test: simulator shift; '
            'test → OOD: defender shift; bugged OOD: both.',
            'fig_belirsizlik_kayma_ece.png', draw,
            [Line2D([], [], ls='none', marker='o', ms=7, mfc=A.SURFACE, mec=A.INK2, mew=1.5,
                    label='member (mean over seeds, bar = range)'),
             Line2D([], [], color=A.INK, lw=2, marker='o', ms=7, mec=A.SURFACE, label='ensemble (model colour)')],
            ylabel='ECE')


def figures(res):
    for s in ('ood', 'bug_ood'):
        fig_reliability(res, s)
        fig_risk_coverage(res, s)
    fig_shift_ladder(res)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['collect', 'mc', 'analyze', 'figures'])
    p.add_argument('--device', default='mps')
    p.add_argument('--models', nargs='+', default=MODELS)
    p.add_argument('--passes', type=int, default=16)
    p.add_argument('--boot', type=int, default=4000)
    args = p.parse_args()
    if args.cmd in ('collect', 'mc'):
        stores = open_stores()
        idx = set_indices(stores)
        for name in args.models:
            for seed in (SEEDS if args.cmd == 'collect' else (0,)):
                collect_run(name, K, seed, stores, idx, args.device, mc=args.passes if args.cmd == 'mc' else 0)
    elif args.cmd == 'analyze':
        res = analyze(args.boot)
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(os.path.join(OUT_DIR, 'ozet.json'), 'w') as f:
            json.dump(res, f, indent=1)
        with open(os.path.join(OUT_DIR, 'tablolar.md'), 'w') as f:
            f.write(tables_md(res))
        figures(res)
    else:
        with open(os.path.join(OUT_DIR, 'ozet.json')) as f:
            figures(json.load(f))


if __name__ == '__main__':
    main()

"""M3 explainability: attention of the host mixer and the silent-recall loss in `react_restore_analyse`.

With the saved weights (`model.pt`), no retraining; the sets and the silent-compromise definition are the same as in
`ood_suite` (silent = label > 0 and no 'export' alert at that step; "caught" = prediction > 0). Three parts:

1. **Ablations on analyse.** `ood_suite` covered only the main CMF and the full GRU. The logits of the host
   tokenisation, mixer type and memory ablations (3 seeds) on the v1 test, `ood_test` and analyse episodes are collected
   (the test F1 is checked against the run's metrics.json). Silent recall on eps0.1 (the training
   policy of v1 test), OOD and analyse.
2. **Slices of the analyse loss** (CMF, full GRU, flat-78 CMF, CMF without mixer): attacker × host group (ent / op /
   user), alert age and type of the last alert × whether that host was restored since
   (from the observation bits and `prev_action`).
   Share and seed-mean recall in each slice; the same slices on eps0.1 and OOD.
3. **Attention of the host mixer** (CMF topo and `host_tam`, 3 seeds; 400 random episodes per set, seed 0).
   The HostMixer input is taken with a forward_pre_hook and the attention probabilities (incl. the topology bias) are
   recomputed; head mean. (a) Share falling on each relation type (self / attack-graph edge / same subnet / other) and
   its share under uniform attention; (b) share that silent-compromise queries give to other hosts alerting in the last
   5 steps (uniform: number of those hosts / 13), separately for caught and missed compromises; (c) the relation biases
   learned by CMF. The output at position t estimates y[t−1] (obs[0] is the initial observation).

    python -m cybermetaformer.hidden_state.explain_m3 collect   # asama3/aciklama/logits/ (outside git), dikkat.json; MPS
    python -m cybermetaformer.hidden_state.explain_m3 analyze   # ozet.json, tablolar.md, figures
"""
import argparse
import json
import os
import time

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt                     # noqa: E402
import numpy as np                                  # noqa: E402
import torch                                        # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from cybermetaformer.hidden_state import data as D          # noqa: E402
from cybermetaformer.hidden_state import evaluate as E      # noqa: E402
from cybermetaformer.hidden_state import features as F      # noqa: E402
from cybermetaformer.hidden_state import metrics as M       # noqa: E402
from cybermetaformer.hidden_state import ood_suite as O    # noqa: E402
from cybermetaformer.hidden_state.aggregate import BLUES, INK, INK2, MUTED, SURFACE, _SLOTS, _grid  # noqa: E402
from cybermetaformer.hidden_state.models import cybermetaformer as C  # noqa: E402

OUT_DIR = os.path.join(O.STAGE_DIR, 'aciklama')
LOGIT_DIR = os.path.join(OUT_DIR, 'logits')
VARIANTS = ['host_yok', 'duz78', 'host_tam', 'host_gcn', 'hafiza_yok', 'yalniz_yas', 'kol_konv', 'kol_ssd', 'konum_yok']
MODELS = ['cybermetaformer', 'gru_full'] + [f'cybermetaformer-{v}' for v in VARIANTS]
LABEL = {'cybermetaformer': 'CMF (main)', 'gru_full': 'full GRU', **{f'cybermetaformer-{v}': v for v in VARIANTS}}
SLICE_MODELS = ['cybermetaformer', 'gru_full', 'cybermetaformer-duz78', 'cybermetaformer-host_yok']
ATTN_MODELS = ['cybermetaformer', 'cybermetaformer-host_tam']
SETS = ['eps0.1', 'ood', 'analyse']
SET_LABEL = {'eps0.1': 'ε=0.1 (training policy)', 'ood': 'ε=0.3 (OOD)', 'analyse': 'react_restore_analyse'}
REL_NAMES = ['self', 'edge', 'subnet', 'other']
N_ATTN_EP, RECENT = 400, 5
RESTORE, ANALYSE = 40, 1                          # Blue action codes: 1 + host analyse, 40 + host restore
LAST_KIND = ['no alert', 'exploit', 'root flag (no activity)', 'other']   # slice-name parts (data, ozet.json keys)


def logit_path(name, seed):
    return os.path.join(LOGIT_DIR, f'{name}_k{O.K}_s{seed}.npz')


# ── collection ───────────────────────────────────────────────────────────

def collect_logits(stores, idx, sets, device, log=print):
    os.makedirs(LOGIT_DIR, exist_ok=True)
    parts = {'test': ('v1', idx['test']), 'ood': ('v1', idx['ood']), 'analyse': ('shift', idx['shift'][sets['analyse'].pos])}
    for name in MODELS[2:]:
        for seed in O.SEEDS:
            if os.path.exists(logit_path(name, seed)):
                continue
            t0 = time.time()
            model, cfg = O.load_model(name, seed, device)
            out = {}
            for b, (src, ix) in parts.items():
                wd = D.WindowData(stores[src], ix, device=device)
                out[b] = E.predict_logits(model, wd, O.K, cfg['scale'], cfg['eval_batch_size'])
            f1 = M.host_macro_f1(stores['v1'].labels[idx['test']].reshape(-1, 12), out['test'].argmax(-1).reshape(-1, 12))
            ref = json.load(open(os.path.join(O.run_dir(name, seed), 'metrics.json')))['reports']['test']['host_macro_f1']
            assert abs(f1 - ref) < 1e-4, f'{name} s{seed}: test F1 {f1} ≠ {ref}'
            np.savez_compressed(logit_path(name, seed), **{b: v.astype(np.float16) for b, v in out.items()})
            log(f'{name} s{seed}: {time.time() - t0:.0f} s; test F1 {f1:.4f} (record {ref:.4f})')
            del model
            if device == 'mps':
                torch.mps.empty_cache()


def attention_probs(hm, x):
    """HostMixer input (B, S, 13, d) → attention probabilities (B, S, head, 13, 13); the same scores as HostMixer.forward
    (in topo, relation bias + mask)."""
    B, S, H, d = x.shape
    q, k, _ = hm.qkv(x.reshape(B * S, H, d)).view(B * S, H, 3, hm.heads, d // hm.heads).permute(2, 0, 3, 1, 4)
    s = q @ k.transpose(-1, -2) / (d // hm.heads) ** 0.5
    if hm.kind == 'topo':
        s = s + (hm.rel_bias[:, hm.rel] + hm.mask)[None]
    return s.softmax(-1).view(B, S, hm.heads, H, H)


def attention_stats(model, x, es_sel):
    """Per layer: (a) relation shares, (b) share of silent-compromise queries on other alerting hosts, mean matrix."""
    y, alert = es_sel
    caps = []
    hooks = [blk.host.register_forward_pre_hook(lambda m, a: caps.append(a[0])) for blk in model.blocks]
    with torch.no_grad():
        pred = model.forward_episode(x)[:, 1:].argmax(-1).cpu().numpy()
        A = [attention_probs(blk.host, c).mean(2)[:, 1:].cpu().numpy() for blk, c in zip(model.blocks, caps)]
    for h in hooks:
        h.remove()
    rel = C.host_relations().numpy()
    lh = np.array(D.LABEL_HOSTS)
    al13 = np.zeros(alert.shape[:2] + (D.NUM_HOSTS,), bool)
    al13[..., lh] = alert
    recent = np.stack([al13[:, max(0, t - RECENT + 1):t + 1].any(1) for t in range(al13.shape[1])], 1)
    e, t, hq = np.nonzero((y > 0) & ~alert)
    qi = lh[hq]
    other = recent[e, t]
    other[np.arange(len(qi)), qi] = False
    caught = pred[e, t, hq] > 0
    layers = []
    for a in A:
        row = a[e, t, qi]
        share = (row * other).sum(-1)
        layers.append({'rel_share': [float((a * (rel == r)).sum(-1).mean()) for r in range(4)],
                       'rel_uniform': [float((rel == r).sum(1).mean() / D.NUM_HOSTS) for r in range(4)],
                       'silent_self': float(row[np.arange(len(qi)), qi].mean()),
                       'silent_alerting': float(share.mean()),
                       'silent_alerting_uniform': float(other.sum(-1).mean() / D.NUM_HOSTS),
                       'silent_alerting_caught': float(share[caught].mean()),
                       'silent_alerting_missed': float(share[~caught].mean()),
                       'matrix': a.mean((0, 1)).round(4).tolist()})
    return {'n_silent': int(len(qi)), 'recall': float(caught.mean()), 'layers': layers}


def collect_attention(stores, idx, sets, device, log=print):
    out = {'n_episodes': N_ATTN_EP, 'recent': RECENT, 'models': {}}
    sel = {s: np.sort(np.random.default_rng(0).choice(sets[s].E, N_ATTN_EP, replace=False)) for s in SETS}
    for name in ATTN_MODELS:
        out['models'][name] = {}
        for seed in O.SEEDS:
            model, _ = O.load_model(name, seed, device)
            r = {}
            if model.blocks[0].host.kind == 'topo':
                r['rel_bias'] = [blk.host.rel_bias.detach().cpu().numpy().round(3).tolist() for blk in model.blocks]
            for s in SETS:
                es, b = sets[s], O.SETS[s][0]
                ep = idx[b][es.pos][sel[s]]
                x = D.scale_obs(torch.as_tensor(stores[O.BASE[b][0]].obs[ep], device=device))
                r[s] = attention_stats(model, x, (es.y[sel[s]], es.alert[sel[s]]))
                log(f"{name} s{seed} {s}: silent {r[s]['n_silent']}, recall {r[s]['recall']:.3f}, share on other alerting hosts "
                    + ' / '.join(f"{L['silent_alerting']:.3f}" for L in r[s]['layers']))
            out['models'][name][str(seed)] = r
            del model
            if device == 'mps':
                torch.mps.empty_cache()
    with open(os.path.join(OUT_DIR, 'dikkat.json'), 'w') as f:
        json.dump(out, f, ensure_ascii=False)


def collect(device):
    stores = O.open_stores()
    idx = O.base_indices(stores)
    sets = O.eval_sets(stores, idx)
    collect_logits(stores, idx, sets, device)
    collect_attention(stores, idx, sets, device)


# ── analysis ─────────────────────────────────────────────────────────────

def member_preds(name, seed, sets):
    """{set: prediction (E, T, 12)}; main CMF and full GRU from the ood_suite logits."""
    if name in O.MODELS:
        m = O.load_member(name, seed)
        lg = {'eps0.1': m['test'][sets['eps0.1'].pos], 'ood': m['ood'], 'analyse': m['shift'][sets['analyse'].pos]}
    else:
        with np.load(logit_path(name, seed)) as z:
            lg = {'eps0.1': z['test'][sets['eps0.1'].pos], 'ood': z['ood'], 'analyse': z['analyse']}
    return {s: v.astype(np.float32).argmax(-1) for s, v in lg.items()}


def alert_context(store, ep, es):
    """For each host-step, the type of the last alert (LAST_KIND index) and whether the host was restored since that alert."""
    lh = np.array(D.LABEL_HOSTS)
    bits = F.host_bits(store.obs[ep][:, 1:])[:, :, lh]
    act = store.prev_action[ep][:, 1:]
    restore = act[..., None] == RESTORE + lh
    Ep, T, H = es.y.shape
    kind_now = np.where(bits[..., 1] == 1, 1, np.where((bits[..., 2] == 1) & (bits[..., 3] == 1), 2, 3))
    last_kind, since = np.zeros((Ep, H), int), np.zeros((Ep, H), bool)
    kind, rest = np.zeros((Ep, T, H), int), np.zeros((Ep, T, H), bool)
    for t in range(T):
        a = es.alert[:, t]
        since = np.where(a, False, since | restore[:, t])
        last_kind = np.where(a, kind_now[:, t], last_kind)
        kind[:, t], rest[:, t] = last_kind, since
    return kind, rest


def slices(stores, idx, sets, s):
    """{slice name: mask (E, T, 12)} within the silent host-steps. Slice names are data (ozet.json keys)."""
    es, b = sets[s], O.SETS[s][0]
    st = stores[O.BASE[b][0]]
    ep = idx[b][es.pos]
    sil = (es.y > 0) & ~es.alert
    out = {}
    group = np.array(['ent', 'op', 'user'])[np.array(C.SUBNET)[D.LABEL_HOSTS]]     # subnet (def is always clean)
    for r, red in enumerate(st.meta['red_policies']):
        for g in ('ent', 'op', 'user'):
            out[f'{red} × {g}'] = sil & (st.red_policy[ep] == r)[:, None, None] & (group == g)
    for name, lo, hi in M.ALERT_AGE_BINS:
        out[f'age {name}'] = sil & ((es.age < 0) if lo < 0 else (es.age >= lo) & (es.age <= hi))
    kind, rest = alert_context(st, ep, es)
    for k, kn in enumerate(LAST_KIND):
        for r in (False, True):
            out[f"last alert: {kn}{', then restore' if r else ''}"] = sil & (kind == k) & (rest == r)
    return sil, out


def analyze(log=print):
    stores = O.open_stores()
    idx = O.base_indices(stores)
    sets = O.eval_sets(stores, idx)
    preds = {n: [member_preds(n, seed, sets) for seed in O.SEEDS] for n in MODELS}
    res = {'ablasyon': {}, 'dilimler': {}}
    for n in MODELS:
        r = {}
        for s in SETS:
            es = sets[s]
            sil = (es.y > 0) & ~es.alert
            rec = [float((p[s][sil] > 0).mean()) for p in preds[n]]
            f1 = [M.host_macro_f1(es.y.reshape(-1, 12), p[s].reshape(-1, 12)) for p in preds[n]]
            r[s] = {'silent_recall': rec, 'host_macro_f1': f1}
        res['ablasyon'][n] = r
        log(f"{LABEL[n]:12s} silent recall " + '  '.join(f"{s} {np.mean(r[s]['silent_recall']):.3f}" for s in SETS))
    for s in SETS:
        sil, sl = slices(stores, idx, sets, s)
        rows = {}
        for name, m in sl.items():
            if m.sum() < 200:
                continue
            rows[name] = {'n': int(m.sum()), 'pay': float(m.sum() / sil.sum()),
                          **{n: float(np.mean([(p[s][m] > 0).mean() for p in preds[n]])) for n in SLICE_MODELS}}
        res['dilimler'][s] = rows
    attn_path = os.path.join(OUT_DIR, 'dikkat.json')
    res['dikkat'] = json.load(open(attn_path)) if os.path.exists(attn_path) else None
    with open(os.path.join(OUT_DIR, 'ozet.json'), 'w') as f:
        json.dump({k: v for k, v in res.items() if k != 'dikkat'}, f, indent=1, ensure_ascii=False)
    with open(os.path.join(OUT_DIR, 'tablolar.md'), 'w') as f:
        f.write(tables_md(res))
    fig_slices(res, os.path.join(OUT_DIR, 'fig_analyse_dilimler.png'))
    if res['dikkat']:
        fig_attention(res['dikkat'], stores['v1'].meta['hosts'], os.path.join(OUT_DIR, 'fig_host_dikkat.png'))
    log('→', OUT_DIR)
    return res


# ── outputs ──────────────────────────────────────────────────────────────

def _ms(v):
    return f'{np.mean(v):.3f} ± {np.std(v, ddof=1):.3f}'


def tables_md(res):
    L = ['# M3 explainability: host mixer and the analyse loss', '',
         '`explain_m3.py`. Silent recall and host macro-F1 per member (argmax; temperature does not change predictions), mean '
         'over 3 seeds ± std (ddof=1). eps0.1 = the `react_restore_eps0.1` episodes of v1 test; OOD = `ood_test` (ε=0.3); analyse = '
         'the `react_restore_analyse` episodes of `minicage_v1_shift` (4,000).', '',
         '## 1. Ablations on analyse', '',
         '| Model | Silent recall eps0.1 | OOD | analyse | analyse seeds | analyse / eps0.1 | F1 analyse |',
         '|---|---:|---:|---:|---|---:|---:|']
    for n, r in res['ablasyon'].items():
        a = r['analyse']['silent_recall']
        L.append(f"| {LABEL[n]} | {_ms(r['eps0.1']['silent_recall'])} | {_ms(r['ood']['silent_recall'])} | {_ms(a)} | "
                 f"{' / '.join(f'{v:.3f}' for v in a)} | {np.mean(a) / np.mean(r['eps0.1']['silent_recall']):.2f} | "
                 f"{np.mean(r['analyse']['host_macro_f1']):.4f} |")
    L += ['', '## 2. Silent recall slices', '',
          'Share: share of the set\'s silent host-steps. Recall is the seed mean. Host groups are subnets: ent = '
          'def and ent0–2 (def is always clean), op = ophost0–2 and opserv, user = user1–4 (user0 is not labelled). '
          'Last-alert type from the observation bits of that step: '
          'exploit activity, root flag without activity `[1,1]` (the form analyse rewrites), other; "then '
          'restore" = the host was restored since the last alert (slice-name suffix). A slice has at least 200 host-steps.', '']
    for s, rows in res['dilimler'].items():
        L += [f'### {SET_LABEL[s]}', '', '| Slice | Share | ' + ' | '.join(LABEL[n] for n in SLICE_MODELS) + ' |',
              '|---|---:|' + '---:|' * len(SLICE_MODELS)]
        for name, r in rows.items():
            L.append(f"| {name} | {r['pay']:.3f} | " + ' | '.join(f'{r[n]:.3f}' for n in SLICE_MODELS) + ' |')
        L.append('')
    d = res['dikkat']
    if d:
        L += ['## 3. Attention of the host mixer', '',
              f"{d['n_episodes']} random episodes per set; head mean; mean over 3 seeds (± std). Relation share: "
              'all queries; under uniform attention self 0.077, edge 0.059, subnet 0.260, other 0.604. Alerting: total share '
              f"that silent-compromise queries give to other hosts alerting in the last {d['recent']} steps.", '',
              '| Model | Set | Layer | Self | Edge | Subnet | Other | Alerting (uniform) | Caught / missed |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for n, by_seed in d['models'].items():
            for s in SETS:
                for li in range(len(by_seed['0'][s]['layers'])):
                    ls = [by_seed[k][s]['layers'][li] for k in by_seed]
                    sh = np.array([x['rel_share'] for x in ls])
                    g = lambda key: np.mean([x[key] for x in ls])                                   # noqa: E731
                    L.append(f"| {LABEL[n]} | {s} | {li} | " + ' | '.join(f'{v:.3f}' for v in sh.mean(0)) +
                             f" | {g('silent_alerting'):.3f} ({g('silent_alerting_uniform'):.3f}) | "
                             f"{g('silent_alerting_caught'):.3f} / {g('silent_alerting_missed'):.3f} |")
        src = [i for i, t in enumerate(C.CONNECTED_HOSTS) if t is not None]
        dst = [C.HOSTS.index(C.CONNECTED_HOSTS[i]) for i in src]
        L += ['', '**Attack-graph edge** (ε=0.1): attention that query hosts with an edge (' +
              ', '.join(f'{C.HOSTS[i]}→{C.HOSTS[j]}' for i, j in zip(src, dst)) + ') give to their edge partner; '
              'uniform 1/13 = 0.077. Seed mean ± std.', '',
              '| Model | ' + ' | '.join(f'Layer {li}' for li in range(3)) + ' |', '|---|---:|---:|---:|']
        for n, by_seed in d['models'].items():
            v = np.array([[np.array(by_seed[k]['eps0.1']['layers'][li]['matrix'])[src, dst].mean() for li in range(3)]
                          for k in by_seed])
            L.append(f'| {LABEL[n]} | ' + ' | '.join(f'{a:.3f} ± {b:.3f}' for a, b in zip(v.mean(0), v.std(0, ddof=1))) + ' |')
        rb = np.array([d['models']['cybermetaformer'][k]['rel_bias'] for k in d['models']['cybermetaformer']])
        L += ['', "**Relation biases of CMF** (head mean; row = layer; mean over 3 seeds ± std; before softmax):", '',
              '| Layer | ' + ' | '.join(REL_NAMES) + ' |', '|---:|' + '---:|' * 4]
        m, sd = rb.mean(2).mean(0), rb.mean(2).std(0, ddof=1)
        for li in range(m.shape[0]):
            L.append(f'| {li} | ' + ' | '.join(f'{a:+.2f} ± {b:.2f}' for a, b in zip(m[li], sd[li])) + ' |')
    return '\n'.join(L) + '\n'


def fig_slices(res, path):
    """Silent recall in the attacker × host-group slices: three sets, three models (categorical slots 1–3)."""
    models = SLICE_MODELS[:3]
    cats = ['bline × ent', 'bline × user', 'meander × ent', 'meander × user']
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    w = 0.26
    for ax, s in zip(axes, SETS):
        rows = res['dilimler'][s]
        for i, n in enumerate(models):
            xs = [j + (i - 1) * w for j, c in enumerate(cats) if c in rows]
            ys = [rows[c][n] for c in cats if c in rows]
            ax.bar(xs, ys, w - 0.03, color=_SLOTS[i], label=LABEL[n], zorder=3)
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels([f"{c.replace(' × ', chr(10))}\n(share {rows[c]['pay']:.2f})" if c in rows else c for c in cats],
                           fontsize=8, color=INK2)
        ax.set_title(SET_LABEL[s], fontsize=10, color=INK, loc='left')
        ax.set_ylim(0, 1.02)
        _grid(ax)
    axes[0].set_ylabel('silent recall')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8.5, frameon=False, loc='upper right', ncol=3, bbox_to_anchor=(0.99, 0.99))
    fig.suptitle('Silent recall by attacker × host group (seed mean)', x=0.01, ha='left', fontsize=11,
                 fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)


def fig_attention(d, hosts, path):
    """Mean host attention (seed 0, eps0.1): row = query, column = key; subnet borders as lines."""
    cmap = LinearSegmentedColormap.from_list('blues', BLUES)
    fig, axes = plt.subplots(len(ATTN_MODELS), 3, figsize=(12, 8.2))
    vmax = max(np.max(L['matrix']) for n in ATTN_MODELS for L in d['models'][n]['0']['eps0.1']['layers'])
    bounds = [i - 0.5 for i in (4, 8)]
    for r, n in enumerate(ATTN_MODELS):
        for li, ax in enumerate(axes[r]):
            m = np.array(d['models'][n]['0']['eps0.1']['layers'][li]['matrix'])
            im = ax.imshow(m, cmap=cmap, vmin=0, vmax=vmax)
            for b in bounds:
                ax.axhline(b, color=MUTED, lw=0.8)
                ax.axvline(b, color=MUTED, lw=0.8)
            ax.set_xticks(range(len(hosts)))
            ax.set_yticks(range(len(hosts)))
            ax.set_xticklabels(hosts, rotation=90, fontsize=6.5, color=INK2)
            ax.set_yticklabels(hosts, fontsize=6.5, color=INK2)
            ax.set_title(f'{LABEL[n]}, layer {li}', fontsize=9, color=INK, loc='left')
    fig.colorbar(im, ax=axes, shrink=0.6, label='attention (query row sums to 1)')
    fig.suptitle('Host mixer: mean attention (ε=0.1 test, seed 0; lines = subnet borders)', x=0.01, ha='left',
                 fontsize=11, fontweight='bold')
    fig.savefig(path, dpi=180, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['collect', 'analyze'])
    p.add_argument('--device', default='mps')
    args = p.parse_args()
    if args.cmd == 'collect':
        collect(args.device)
    else:
        analyze()


if __name__ == '__main__':
    main()

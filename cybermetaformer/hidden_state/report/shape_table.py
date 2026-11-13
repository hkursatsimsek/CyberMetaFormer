"""Paper (M4, item 2): stage-by-stage tensor shapes of CyberMetaFormer and parameter counts per component.

The model is built with the best HPO setting (asama3/hpo/cybermetaformer_k101/best.yaml). The symbolic shapes in the
table are compared with the real shapes measured by forward hooks on a dummy episode. Parameters are grouped by their
`named_parameters` names. The script stops if the groups do not cover all model parameters or the total does not match.
The online state size is counted from `init_state`.

Output (English): results/hidden_state/makale/boyut/{table.md, summary.json}

    venv_ts/bin/python -m cybermetaformer.hidden_state.report.shape_table
"""
import json
import os

import torch
import yaml

from cybermetaformer.hidden_state.data import LABEL_HOSTS, NUM_HOSTS, OBS_DIM
from cybermetaformer.hidden_state.models.cybermetaformer import MAX_STEPS, CyberMetaFormer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BEST = os.path.join(ROOT, 'results', 'hidden_state', 'asama3', 'hpo', 'cybermetaformer_k101', 'best.yaml')
OUT = os.path.join(ROOT, 'results', 'hidden_state', 'makale', 'boyut')


def build():
    with open(BEST) as f:
        args = yaml.safe_load(f)['overrides']['model_args']
    return CyberMetaFormer(**args).eval(), args


def param_groups(model):
    """Parameter name → table row. Block parameters are summed over the block index (not multiplied by L)."""
    rules = [('inp.', 'proj'), ('host_emb', 'emb_host'), ('subnet_emb', 'emb_sub'), ('pos', 'emb_pos'),
             ('temporal.branches.0.', 'conv'), ('temporal.branches.1.', 'ssd'), ('temporal.gate.', 'gate'),
             ('n1.', 'ln1'), ('n2.', 'ln2'), ('host.', 'host'), ('n3.', 'ln3'), ('mlp.', 'mlp'),
             ('norm.', 'ln_f'), ('head.', 'head')]
    per_block, other = {}, {}
    for name, p in model.named_parameters():
        block = name.startswith('blocks.')
        key = name.split('.', 2)[2] if block else name
        group = next((g for prefix, g in rules if key.startswith(prefix)), None)
        if group is None:
            raise KeyError(f'ungrouped parameter: {name}')
        tgt = per_block if block else other
        if block:
            idx = int(name.split('.')[1])
            tgt.setdefault(group, {}).setdefault(idx, 0)
            tgt[group][idx] += p.numel()
        else:
            tgt[group] = tgt.get(group, 0) + p.numel()
    counts = {g: set(v.values()) for g, v in per_block.items()}
    if any(len(c) != 1 for c in counts.values()):
        raise ValueError(f'parameter counts differ between blocks: {counts}')
    return {g: c.pop() for g, c in counts.items()}, other


def measured_shapes(model, B=2):
    """Input/output shapes of the submodules of the first block on a dummy episode (B, S, 78)."""
    shapes = {}

    def hook(key):
        def f(_, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            shapes[key] = (tuple(inp[0].shape), tuple(o.shape))
        return f

    blk = model.blocks[0]
    mods = {'conv': blk.temporal.branches[0], 'ssd': blk.temporal.branches[1], 'gate': blk.temporal.gate,
            'temporal': blk.temporal, 'host': blk.host, 'mlp': blk.mlp, 'proj': model.inp, 'head': model.head}
    hs = [m.register_forward_hook(hook(k)) for k, m in mods.items()]
    x = torch.rand(B, MAX_STEPS, OBS_DIM)
    with torch.no_grad():
        out = model.forward_episode(x)
    for h in hs:
        h.remove()
    shapes['output'] = (tuple(x.shape), tuple(out.shape))
    return shapes


def state_size(model):
    st = model.init_state(batch=1)
    blocks = [[tuple(t.shape) for t in blk] for blk in st['blocks']]
    n_float = sum(t.numel() for blk in st['blocks'] for t in blk)
    return {'last_event': tuple(st['last'].shape), 'per_block': blocks[0], 'n_blocks': len(blocks),
            'floats': n_float, 'bytes_fp32': 4 * n_float}


def main():
    model, args = build()
    total = sum(p.numel() for p in model.parameters())
    blk, other = param_groups(model)
    L = len(model.blocks)
    d, H, heads, N, K = args['d'], NUM_HOSTS, args['heads'], args['state'], args['kernel']
    P, S, B = d // heads, MAX_STEPS, 2
    n_in = model.inp.in_features
    shp = measured_shapes(model, B)

    # symbolic shapes; compared with the measured shapes (B=2)
    sym = {'proj': (f'(B, S, H, {n_in})', '(B, S, H, d)', (B, S, H, n_in), (B, S, H, d)),
           'conv': ('(B·H, S, d)', '(B·H, S, d)', (B * H, S, d), (B * H, S, d)),
           'ssd': ('(B·H, S, d)', '(B·H, S, d)', (B * H, S, d), (B * H, S, d)),
           'gate': ('(B·H, S, d)', '(B·H, S, 2)', (B * H, S, d), (B * H, S, 2)),
           'host': ('(B, S, H, d)', '(B, S, H, d)', (B, S, H, d), (B, S, H, d)),
           'mlp': ('(B, S, H, d)', '(B, S, H, d)', (B, S, H, d), (B, S, H, d)),
           'head': ('(B, S, 12, d)', '(B, S, 12, 3)', (B, S, len(LABEL_HOSTS), d), (B, S, len(LABEL_HOSTS), 3)),
           'output': ('(B, S, 78)', '(B, S, 12, 3)', (B, S, OBS_DIM), (B, S, len(LABEL_HOSTS), 3))}
    for k, (_, _, i, o) in sym.items():
        if shp[k] != (i, o):
            raise AssertionError(f'{k}: measured {shp[k]} ≠ expected {(i, o)}')

    emb = other['proj'] + other['emb_host'] + other['emb_sub'] + other['emb_pos']
    block_total = sum(blk.values())
    out_total = other['ln_f'] + other['head']
    if emb + L * block_total + out_total != total:
        raise AssertionError(f'sum of groups {emb + L * block_total + out_total} ≠ {total}')
    st = state_size(model)

    def n(v):
        return f'{v:,}'

    rows = [
        ('Input', 'Scaled Blue observation $\\hat{o}_t$', '(B, S, 78)', '—', '—'),
        ('Tokenization', 'Host tokens $x_{t,h}$ (Eq. 1)', '(B, S, 78)', '(B, S, H, 6)', '0'),
        ('', 'Alert-age features $[\\phi_{t,h};\\nu_{t,h}]$ (Eq. 2)', '(B, S, 78)', '(B, S, H, 8)', '0'),
        ('Embedding', f'Input projection $W_{{\\mathrm{{in}}}}$ ({n_in}→d)', sym['proj'][0], sym['proj'][1],
         n(other['proj'])),
        ('', 'Host, subnet and position embeddings (Eq. 3)', '—', '(B, S, H, d)',
         f"{n(other['emb_host'])} + {n(other['emb_sub'])} + {n(other['emb_pos'])}"),
        (f'Block ℓ (×{L})', 'LayerNorm $\\mathrm{LN}_1$', '(B, S, H, d)', '(B, S, H, d)', n(blk['ln1'])),
        ('', f'Causal depthwise convolution branch, K = {K} (Eq. 4)', sym['conv'][0], sym['conv'][1],
         n(blk['conv'])),
        ('', f'SSD branch, {heads} heads, N = {N}, P = {P} (Eqs. 5–7)', sym['ssd'][0], sym['ssd'][1],
         n(blk['ssd'])),
        ('', 'Softmax branch gate (Eq. 8)', sym['gate'][0], sym['gate'][1], n(blk['gate'])),
        ('', 'LayerNorm $\\mathrm{LN}_2$ + topology-biased host attention (Eq. 9)', sym['host'][0], sym['host'][1],
         f"{n(blk['ln2'])} + {n(blk['host'])}"),
        ('', f"LayerNorm $\\mathrm{{LN}}_3$ + channel MLP, d→{args['mlp_ratio']}d→d (Eq. 10)", sym['mlp'][0],
         sym['mlp'][1], f"{n(blk['ln3'])} + {n(blk['mlp'])}"),
        ('', 'Block total', '', '', n(block_total)),
        ('Output', 'Final LayerNorm + shared head on 12 label hosts (Eq. 11)', '(B, S, H, d)', sym['head'][1],
         f"{n(other['ln_f'])} + {n(other['head'])}"),
        ('**Total**', '', sym['output'][0], sym['output'][1], f'**{n(total)}**'),
    ]
    lines = ['| Stage | Operation | Input shape | Output shape | Parameters |', '|---|---|---|---|---:|']
    lines += [f'| {" | ".join(r)} |' for r in rows]
    per_block_state = ' and '.join(f'({", ".join(map(str, s))})' for s in st['per_block'])
    note = (f'Note. B: episodes per batch; S = T + 1 = {S} steps; H = {H} hosts; d = {d}. Inside the temporal mixer '
            f'every host is an independent sequence (B·H rows) with shared weights; inside the host mixer every time '
            f'step is an independent set of {H} tokens (B·S rows). Online inference keeps, per episode, a '
            f'{st["last_event"][1]}×{st["last_event"][2]} last-event table and, per block, a convolution buffer and '
            f'an SSD state of shapes {per_block_state} '
            f'(over the {st["n_blocks"]} blocks: {n(st["floats"])} floats, {n(st["bytes_fp32"])} bytes in FP32, '
            f'without the integer last-event table).')
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'table.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n\n' + note + '\n')
    with open(os.path.join(OUT, 'summary.json'), 'w') as f:
        json.dump({'model_args': args, 'total_params': total, 'per_block': blk, 'other': other, 'layers': L,
                   'measured_shapes_B2': {k: list(map(list, v)) for k, v in shp.items()}, 'online_state': st},
                  f, indent=2)
    print('\n'.join(lines) + '\n\n' + note)


if __name__ == '__main__':
    main()

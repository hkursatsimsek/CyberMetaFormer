"""CyberMetaFormer (stage 3). Trained in episode mode (full history, many-to-many).

    o_t (78) → 13 host tokens × 6 features [+ alert-age channel] + host / subnet / time embedding
    L × block (pre-norm, residual):
        x = x + TemporalMixer(Norm(x))   # along time for each host, causal, weights shared across hosts
        x = x + HostMixer(Norm(x))       # across the 13 hosts at each step
        x = x + ChannelMLP(Norm(x))
    tokens of the 12 label hosts at each step → shared host head (3 classes) → (B, S, 12, 3)

Causality: output t sees only obs[0..t] (the same contract as RNNFull; train.py `mode: episode`). All temporal
mixers are causal; if `window` is given, attention, SSD and the time MLP see only the last `window` steps (windowed,
"no memory" ablation; convolution and pooling are local anyway, GRU cannot be windowed).

Temporal mixer branches (`temporal`; several are combined with a learned per-token softmax gate):
- `conv`: causal depthwise convolution (kernel `kernel`). A shifted sum instead of nn.Conv1d: the MPS Conv1d
  bug (batch > 2^16; tslib_adapter) does not affect this path.
- `pool`: causal average pooling − identity (PoolFormer). `mlp`: lower-triangular time MLP (shared across channels).
- `attn`: causal multi-head attention. `gru`: GRU (the winning mechanism of M1; full history).
- `ssd`: selective state space, Mamba-2 SSD form (scalar per head, input-dependent decay a_t ∈ (0, 1)):
  h_t = a_t h_{t−1} + dt_t B_t x_tᵀ, y_t = C_t h_t + D x_t, SiLU gate at the output. Computed in matrix form:
  y = (L ∘ C Bᵀ)(dt x), L[t, s] = Π_{r=s+1..t} a_r; as cheap as attention at S = 101, only matmul/exp/cumsum on MPS.

Host mixer (`host_mixer`): `none`, `attn` (full attention), `topo` (attention + a learned per-head bias by relation
type: self / attack-graph edge (`CONNECTED_HOSTS`) / same subnet / other; with `topo_mask` "other" is masked),
`gcn` (fixed normalised adjacency, edge ∪ same subnet ∪ self).

Memory: `age=True` is the explicit alert-age channel: for 4 events per host (export alert, security
flag, root `[1,1]`, removed `[1,0]`; the same events as features.temporal_features), the steps since the last event in
the full history, log1p(age)/log1p(100) and a "never seen" flag. Independent of `window` (this channel carries what
lies beyond the window).

Ablations: `tokens='flat'` (a single 78-dim token, no host mixer, head 36), `pos='none'`,
`action_input=True`: an embedding of the Blue action that produced obs[t] (`act_blue[t−1]`, "none" at
t = 0) is added to all tokens at every step; `forward_episode(x, act)`.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as nnf

from cybermetaformer.hidden_state.data import (BITS, LABEL_HOSTS, N_BLUE_ACTIONS, N_CLASSES, NUM_HOSTS, OBS_DIM,
                                             to_host_tokens)

MAX_STEPS = 101                                  # obs[0..T], T = 100
# MiniCAGE topology (CybORG_plus_plus/mini_CAGE/minimal.py; copied so that it also runs on machines without the
# CybORG++ clone; test_m3 compares it one-to-one when the clone exists)
HOSTS = ['def', 'ent0', 'ent1', 'ent2', 'ophost0', 'ophost1', 'ophost2', 'opserv', 'user0', 'user1', 'user2', 'user3',
         'user4']
CONNECTED_HOSTS = [None, None, None, 'opserv', None, None, None, None, None, 'ent0', 'ent0', 'ent1', 'ent1']
SUBNET = [0] * 4 + [1] * 4 + [2] * 5             # minimal.create_subnets: ent, op, user
REL_SELF, REL_EDGE, REL_SUBNET, REL_OTHER = range(4)
AGE_EVENTS = 4
TOKEN_FEATS = 6


def host_relations(connected=None):
    """(13, 13) relation type: self, attack-graph edge (symmetric), same subnet, other. connected: link table in HOSTS
    order (default: MiniCAGE's `CONNECTED_HOSTS`; for the updated prior of the topology OOD)."""
    rel = torch.full((NUM_HOSTS, NUM_HOSTS), REL_OTHER, dtype=torch.long)
    same = torch.tensor(SUBNET)[:, None] == torch.tensor(SUBNET)[None]
    rel[same] = REL_SUBNET
    for i, target in enumerate(CONNECTED_HOSTS if connected is None else connected):
        if target is not None:
            j = HOSTS.index(target)
            rel[i, j] = rel[j, i] = REL_EDGE
    rel[torch.arange(NUM_HOSTS), torch.arange(NUM_HOSTS)] = REL_SELF
    return rel


def alert_age_features(x):
    """Scaled observation (B, S, 78) → (B, S, 13, 8): 4 events × [log1p(age)/log1p(100), never seen]. Causal."""
    b = x[..., BITS].reshape(*x.shape[:-1], NUM_HOSTS, 4) > 0.5
    ev = torch.stack([b[..., 1] | b[..., 3], b[..., 3], b[..., 2] & b[..., 3], b[..., 2] & ~b[..., 3]], -1)
    S = x.shape[1]
    t = torch.arange(S, device=x.device).view(1, S, 1, 1)
    last = torch.where(ev, t, torch.full_like(t, -1)).cummax(dim=1).values
    never = last < 0
    age = torch.where(never, torch.full_like(t, MAX_STEPS - 1), t - last).float()
    scaled = torch.log1p(age) / math.log1p(MAX_STEPS - 1)
    return torch.cat([scaled, never.float()], -1)


def _band(S, window, device):
    """(S, S) bool: s ≤ t and (if window is given) t − s < window."""
    t = torch.arange(S, device=device)
    d = t[:, None] - t[None]
    return (d >= 0) & (d < window) if window else d >= 0


# ── temporal mixer branches: (N, S, d) → (N, S, d) ─────────────────────────

class CausalConv(nn.Module):
    def __init__(self, d, kernel=4, **_):
        super().__init__()
        self.kernel = kernel
        self.weight = nn.Parameter(torch.randn(kernel, d) / math.sqrt(kernel))
        self.bias = nn.Parameter(torch.zeros(d))
        self.out = nn.Linear(d, d)

    def forward(self, x):
        xp = nnf.pad(x, (0, 0, self.kernel - 1, 0))
        S = x.shape[1]
        y = sum(self.weight[j] * xp[:, self.kernel - 1 - j:self.kernel - 1 - j + S] for j in range(self.kernel))
        return self.out(nnf.gelu(y + self.bias))

    def init_state(self, N, d, device):
        return torch.zeros(N, self.kernel - 1, d, device=device)          # previous inputs (oldest first; zero padding)

    def step(self, x, buf):
        """x (N, d), buf (N, kernel−1, d) → (N, d), new buf."""
        win = torch.cat([buf, x[:, None]], 1)
        y = sum(self.weight[j] * win[:, -1 - j] for j in range(self.kernel))
        return self.out(nnf.gelu(y + self.bias)), win[:, 1:]


class CausalPool(nn.Module):
    def __init__(self, d, kernel=4, **_):
        super().__init__()
        self.kernel = kernel

    def forward(self, x):
        c = nnf.pad(x, (0, 0, 1, 0)).cumsum(1)                          # c[t+1] = Σ_{≤t}
        S = x.shape[1]
        lo = (torch.arange(S, device=x.device) + 1 - self.kernel).clamp(min=0)
        n = (torch.arange(S, device=x.device) + 1 - lo).view(1, S, 1)
        return (c[:, 1:] - c[:, lo]) / n - x


class TemporalMLP(nn.Module):
    def __init__(self, d, window=None, **_):
        super().__init__()
        self.window = window
        self.weight = nn.Parameter(torch.eye(MAX_STEPS) + 0.02 * torch.randn(MAX_STEPS, MAX_STEPS))
        self.out = nn.Linear(d, d)

    def forward(self, x):
        S = x.shape[1]
        w = self.weight[:S, :S] * _band(S, self.window, x.device)
        return self.out(nnf.gelu(torch.einsum('ts,nsd->ntd', w, x)))


class CausalAttention(nn.Module):
    def __init__(self, d, heads=4, window=None, **_):
        super().__init__()
        self.heads, self.window = heads, window
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x):
        N, S, d = x.shape
        q, k, v = self.qkv(x).view(N, S, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        y = nnf.scaled_dot_product_attention(q, k, v, attn_mask=_band(S, self.window, x.device))
        return self.out(y.transpose(1, 2).reshape(N, S, d))


class GRUMixer(nn.Module):
    def __init__(self, d, **_):
        super().__init__()
        self.rnn = nn.GRU(d, d, batch_first=True)

    def forward(self, x):
        return self.rnn(x)[0]

    def init_state(self, N, d, device):
        return torch.zeros(1, N, d, device=device)                       # nn.GRU hidden state (layer, N, d)

    def step(self, x, h):
        """x (N, d), h (1, N, d) → (N, d), new h (M4 item 4: online latency of the gru_host_* variants)."""
        y, h = self.rnn(x[:, None], h)
        return y[:, 0], h


class SSD(nn.Module):
    def __init__(self, d, heads=4, state=16, window=None, **_):
        super().__init__()
        self.heads, self.state, self.window, self.p = heads, state, window, d // heads
        self.inp = nn.Linear(d, 2 * d + 2 * heads * state + heads)       # x, z, B, C, dt
        self.A_log = nn.Parameter(torch.log(torch.linspace(1.0, 16.0, heads)))
        dt = torch.exp(torch.empty(heads).uniform_(math.log(1e-3), math.log(1e-1)))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))    # softplus⁻¹ (Mamba initialisation)
        self.D = nn.Parameter(torch.ones(heads))
        self.out = nn.Linear(d, d)

    def forward(self, u):
        N, S, d = u.shape
        H, P, Ns = self.heads, self.p, self.state
        x, z, B, C, dt = self.inp(u).split([d, d, H * Ns, H * Ns, H], -1)
        x, B, C = x.view(N, S, H, P), B.view(N, S, H, Ns), C.view(N, S, H, Ns)
        dt = nnf.softplus(dt + self.dt_bias)                             # (N, S, H)
        cum = (-dt * self.A_log.exp()).cumsum(1).transpose(1, 2)          # (N, H, S): Σ log a
        seg = cum[..., :, None] - cum[..., None, :]                       # (N, H, t, s) = Σ_{r=s+1..t} log a_r
        L = seg.masked_fill(~_band(S, self.window, u.device), float('-inf')).exp()
        scores = torch.einsum('nthk,nshk->nhts', C, B) * L
        y = torch.einsum('nhts,nshp->nthp', scores, x * dt[..., None]) + self.D.view(1, 1, H, 1) * x
        return self.out(y.reshape(N, S, d) * nnf.silu(z))

    def init_state(self, N, d, device):
        if self.window:
            raise NotImplementedError('no step mode for windowed SSD')
        return torch.zeros(N, self.heads, self.state, self.p, device=device)

    def step(self, u, h):
        """Recurrent form: h_t = a_t h_{t−1} + dt_t B_t x_tᵀ, y_t = C_t h_t + D x_t. u (N, d), h (N, H, Ns, P)."""
        N, d = u.shape
        H, P, Ns = self.heads, self.p, self.state
        x, z, B, C, dt = self.inp(u).split([d, d, H * Ns, H * Ns, H], -1)
        x, B, C = x.view(N, H, P), B.view(N, H, Ns), C.view(N, H, Ns)
        dt = nnf.softplus(dt + self.dt_bias)                             # (N, H)
        a = (-dt * self.A_log.exp()).exp()
        h = a[..., None, None] * h + (dt[..., None] * B)[..., None] * x[:, :, None]
        y = torch.einsum('nhk,nhkp->nhp', C, h) + self.D.view(1, H, 1) * x
        return self.out(y.reshape(N, d) * nnf.silu(z)), h


BRANCHES = {'conv': CausalConv, 'pool': CausalPool, 'mlp': TemporalMLP, 'attn': CausalAttention,
            'gru': GRUMixer, 'ssd': SSD}


class TemporalMixer(nn.Module):
    def __init__(self, d, kinds, **kw):
        super().__init__()
        if kw.get('window') and 'gru' in kinds:
            raise ValueError('GRU kolu pencerelenemez')
        self.branches = nn.ModuleList(BRANCHES[k](d, **kw) for k in kinds)
        self.gate = nn.Linear(d, len(kinds)) if len(kinds) > 1 else None

    def forward(self, x):
        """(B, S, H, d): each host a separate sequence, shared weights."""
        B, S, H, d = x.shape
        seq = x.transpose(1, 2).reshape(B * H, S, d)
        ys = [b(seq) for b in self.branches]
        if self.gate is None:
            y = ys[0]
        else:
            g = self.gate(seq).softmax(-1)
            y = sum(g[..., i:i + 1] * yi for i, yi in enumerate(ys))
        return y.view(B, H, S, d).transpose(1, 2)

    def init_state(self, N, d, device):
        for b in self.branches:
            if not hasattr(b, 'step'):
                raise NotImplementedError(f'no step mode for {type(b).__name__}')
        return [b.init_state(N, d, device) for b in self.branches]

    def step(self, x, states):
        """(B, H, d), branch states → (B, H, d), new states."""
        B, H, d = x.shape
        seq = x.reshape(B * H, d)
        outs = [b.step(seq, st) for b, st in zip(self.branches, states)]
        ys = [o[0] for o in outs]
        if self.gate is None:
            y = ys[0]
        else:
            g = self.gate(seq).softmax(-1)
            y = sum(g[..., i:i + 1] * yi for i, yi in enumerate(ys))
        return y.view(B, H, d), [o[1] for o in outs]


# ── host mixer: (B, S, 13, d) → same ─────────────────────────────────────

class HostMixer(nn.Module):
    def __init__(self, d, kind='topo', heads=4, topo_mask=False):
        super().__init__()
        self.kind, self.heads = kind, heads
        rel = host_relations()
        if kind in ('attn', 'topo'):
            self.qkv = nn.Linear(d, 3 * d)
            self.out = nn.Linear(d, d)
            self.register_buffer('rel', rel, persistent=False)
            if kind == 'topo':
                self.rel_bias = nn.Parameter(torch.zeros(heads, 4))
                self.register_buffer('mask', torch.where(rel == REL_OTHER, float('-inf'), 0.0) if topo_mask
                                     else torch.zeros(NUM_HOSTS, NUM_HOSTS), persistent=False)
        elif kind == 'gcn':
            a = (rel != REL_OTHER).float()
            dinv = a.sum(1).rsqrt()
            self.register_buffer('adj', dinv[:, None] * a * dinv[None], persistent=False)
            self.lin = nn.Linear(d, d)
        else:
            raise ValueError(f'host_mixer: {kind}')

    def forward(self, x):
        B, S, H, d = x.shape
        if self.kind == 'gcn':
            return self.lin(torch.einsum('ij,bsjd->bsid', self.adj, x))
        h = x.reshape(B * S, H, d)
        q, k, v = self.qkv(h).view(B * S, H, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        bias = None
        if self.kind == 'topo':
            bias = (self.rel_bias[:, self.rel] + self.mask)[None]           # (1, heads, 13, 13)
        y = nnf.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        return self.out(y.transpose(1, 2).reshape(B, S, H, d))


class Block(nn.Module):
    def __init__(self, d, temporal, host_mixer, mlp_ratio, dropout, heads, topo_mask, **kw):
        super().__init__()
        self.n1, self.temporal = nn.LayerNorm(d), TemporalMixer(d, temporal, heads=heads, **kw)
        self.host = None
        if host_mixer != 'none':
            self.n2, self.host = nn.LayerNorm(d), HostMixer(d, host_mixer, heads, topo_mask)
        self.n3 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, mlp_ratio * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(mlp_ratio * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.temporal(self.n1(x)))
        if self.host is not None:
            x = x + self.drop(self.host(self.n2(x)))
        return x + self.drop(self.mlp(self.n3(x)))

    def step(self, x, states):
        """(B, H, d) single step (eval; no dropout)."""
        y, states = self.temporal.step(self.n1(x), states)
        x = x + y
        if self.host is not None:
            x = x + self.host(self.n2(x)[:, None])[:, 0]
        return x + self.mlp(self.n3(x)), states


BUDGETS = {'b1': (147_000, 220_000), 'b2': (1_900_000, 2_900_000)}    # B1/B2 budget: full GRU / Transformer k=64 ± 20%


def n_params(**kw):
    return sum(p.numel() for p in CyberMetaFormer(**kw).parameters())


def fit_width(budget='b1', step=8, **kw):
    """Largest d that fits under the upper budget bound (a multiple of step; step must divide by the head count).
    ValueError if the lower bound is not reached. kw: CyberMetaFormer arguments (without d)."""
    lo, hi = BUDGETS[budget]
    best = None
    for d in range(step, 4096, step):
        n = n_params(d=d, **kw)
        if n > hi:
            break
        best = (d, n)
    if best is None or best[1] < lo:
        raise ValueError(f'{kw}: budget {budget} ({lo}–{hi}) cannot be met: {best}')
    return best[0]


class CyberMetaFormer(nn.Module):
    episode_mode = True

    def __init__(self, k=MAX_STEPS, d=64, layers=3, temporal=('conv', 'ssd'), host_mixer='topo', topo_mask=False,
                 heads=4, state=16, kernel=4, window=None, mlp_ratio=3, dropout=0.1, age=True, pos='learned',
                 tokens='host', action_input=False):
        super().__init__()
        temporal = [temporal] if isinstance(temporal, str) else list(temporal)
        if tokens not in ('host', 'flat'):
            raise ValueError(f'tokens: {tokens}')
        if tokens == 'flat' and host_mixer != 'none':
            raise ValueError("tokens='flat' only with host_mixer='none'")
        self.tokens, self.age, self.action_input = tokens, age, action_input
        self.act_emb = nn.Embedding(N_BLUE_ACTIONS + 1, d) if action_input else None
        n_in = (TOKEN_FEATS + 2 * AGE_EVENTS if age else TOKEN_FEATS) if tokens == 'host' else \
            OBS_DIM + (2 * AGE_EVENTS * NUM_HOSTS if age else 0)
        self.inp = nn.Linear(n_in, d)
        if tokens == 'host':
            self.host_emb = nn.Parameter(0.02 * torch.randn(NUM_HOSTS, d))
            self.subnet_emb = nn.Parameter(0.02 * torch.randn(3, d))
            self.register_buffer('subnet', torch.tensor(SUBNET), persistent=False)
        self.pos = nn.Parameter(0.02 * torch.randn(MAX_STEPS, d)) if pos == 'learned' else None
        if pos not in ('learned', 'none'):
            raise ValueError(f'pos: {pos}')
        self.blocks = nn.ModuleList(Block(d, temporal, host_mixer, mlp_ratio, dropout, heads, topo_mask,
                                          state=state, kernel=kernel, window=window) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, N_CLASSES if tokens == 'host' else len(LABEL_HOSTS) * N_CLASSES)
        self.register_buffer('label_hosts', torch.tensor(LABEL_HOSTS), persistent=False)

    def embed(self, x, act=None):
        """(B, S, 78) scaled observation [+ (B, S) previous action] → (B, S, 13 or 1, d)."""
        B, S, _ = x.shape
        if self.tokens == 'host':
            feats = to_host_tokens(x)                                        # (B, S, 13, 6)
            if self.age:
                feats = torch.cat([feats, alert_age_features(x)], -1)
            h = self.inp(feats) + self.host_emb + self.subnet_emb[self.subnet]
        else:
            feats = x if not self.age else torch.cat([x, alert_age_features(x).flatten(2)], -1)
            h = self.inp(feats)[:, :, None]
        if self.pos is not None:
            h = h + self.pos[:S, None]
        if self.act_emb is not None:
            if act is None:
                raise ValueError('action_input=True: forward_episode(x, act) gerekli')
            h = h + self.act_emb(act)[:, :, None]
        return h

    def forward_episode(self, x, act=None, return_features=False):
        """(B, S, 78) → (B, S, 12, 3). return_features: also the last normalised tokens (B, S, 13 or 1, d)
        (domain generalisation: CORAL, DANN)."""
        h = self.embed(x, act)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        if self.tokens == 'host':
            out = self.head(h[:, :, self.label_hosts])
        else:
            out = self.head(h[:, :, 0]).view(x.shape[0], x.shape[1], len(LABEL_HOSTS), N_CLASSES)
        return (out, h) if return_features else out

    def init_state(self, batch=1, device='cpu'):
        """Online inference state: step counter, last event step per host, temporal mixer state of each block."""
        if self.action_input:
            raise NotImplementedError('no step mode for a model with action input')
        n = NUM_HOSTS if self.tokens == 'host' else 1
        d = self.inp.out_features
        return {'t': 0, 'last': torch.full((batch, NUM_HOSTS, AGE_EVENTS), -1, device=device),
                'blocks': [blk.temporal.init_state(batch * n, d, device) for blk in self.blocks]}

    @torch.no_grad()
    def step(self, x, state):
        """Online single step (eval): scaled obs[t] (B, 78) → (B, 12, 3); same as forward_episode(x)[:, t]. Its cost
        does not depend on t (conv buffer + SSD state); the "online inference is cheap" claim is measured with this."""
        t = state['t']
        if t >= MAX_STEPS:
            raise ValueError(f'at most {MAX_STEPS} steps')
        B = x.shape[0]
        feats = to_host_tokens(x) if self.tokens == 'host' else x
        if self.age:
            b = x[:, BITS].reshape(B, NUM_HOSTS, 4) > 0.5
            ev = torch.stack([b[..., 1] | b[..., 3], b[..., 3], b[..., 2] & b[..., 3], b[..., 2] & ~b[..., 3]], -1)
            last = torch.where(ev, torch.full_like(state['last'], t), state['last'])
            never = last < 0
            age = torch.where(never, torch.full_like(last, MAX_STEPS - 1), t - last).float()
            af = torch.cat([torch.log1p(age) / math.log1p(MAX_STEPS - 1), never.float()], -1)
            feats = torch.cat([feats, af if self.tokens == 'host' else af.flatten(1)], -1)
            state['last'] = last
        if self.tokens == 'host':
            h = self.inp(feats) + self.host_emb + self.subnet_emb[self.subnet]
        else:
            h = self.inp(feats)[:, None]
        if self.pos is not None:
            h = h + self.pos[t]
        for i, blk in enumerate(self.blocks):
            h, state['blocks'][i] = blk.step(h, state['blocks'][i])
        h = self.norm(h)
        state['t'] = t + 1
        if self.tokens == 'host':
            return self.head(h[:, self.label_hosts])
        return self.head(h[:, 0]).view(B, len(LABEL_HOSTS), N_CLASSES)

    def forward(self, x, mask=None):
        """Window interface (shape checks; like RNNFull): the last step of the window. Padding is seen as zero input."""
        return self.forward_episode(x)[:, -1]

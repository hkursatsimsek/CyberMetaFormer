"""M3: test-only Blue policies and CyberMetaFormer.

Policies: the old indices are kept; the two mechanism policies differ from React_restore_minimal only at the
defined point. Model: causality, the window limit, equivalence of the SSD with the sequential recurrence, the
alert-age channel and the B1 budget, for every temporal/host mixer type."""
import numpy as np
import pytest
import torch
import yaml

from cybermetaformer.hidden_state import data as D
from cybermetaformer.hidden_state import features as F
from cybermetaformer.hidden_state.models import build_model
from cybermetaformer.hidden_state.models import cybermetaformer as C


try:                                                  # the policies need CybORG++ (no clone on the M4 Pro)
    from CybORG_plus_plus.mini_CAGE import React_restore_minimal
    from minicage.dataset import policies as P
except ImportError:
    P = None
needs_cyborg = pytest.mark.skipif(P is None, reason='CybORG_plus_plus not available')


@needs_cyborg
def test_blue_policy_indices_preserved():
    assert P.BLUE_POLICIES[:6] == ['sleep', 'react_restore', 'restore_decoys', 'random',
                                   'react_restore_eps0.1', 'react_restore_eps0.3']
    assert P.V1_BLUE_POLICIES == P.BLUE_POLICIES[:6]
    assert P.BLUE_POLICIES[6:] == ['react_restore_eps0.5', 'react_restore_eps0.7', 'react_restore_analyse', 'react_remove']
    eps = {b: P.blue_epsilon(b) for b in P.BLUE_POLICIES[6:]}
    assert eps == {'react_restore_eps0.5': 0.5, 'react_restore_eps0.7': 0.7,
                   'react_restore_analyse': 0.0, 'react_remove': 0.0}
    assert isinstance(P.make_blue('react_restore_analyse'), P.ReactRestoreAnalyse)
    assert isinstance(P.make_blue('react_remove'), P.ReactRemove)


def _react_restore_obs(raw):
    """Observation sequence from the react_restore episodes of minicage_v1: (T+1, n, 78)."""
    idx = np.where(raw['blue_policy'] == P.BLUE_POLICIES.index('react_restore'))[0][:64]
    return np.swapaxes(raw['obs_blue'][idx], 0, 1).astype(np.int64)


def _actions(agent, obs):
    return np.stack([agent.get_action(o)[:, 0] for o in obs])


@needs_cyborg
def test_react_remove_only_swaps_restore(raw):
    obs = _react_restore_obs(raw)
    ref, new = _actions(React_restore_minimal(), obs), _actions(P.ReactRemove(), obs)
    restore = ref >= P.RESTORE
    assert restore.any()
    user0 = ref == P.RESTORE + P.USER0
    np.testing.assert_array_equal(new[~restore | user0], ref[~restore | user0])
    np.testing.assert_array_equal(new[restore & ~user0], ref[restore & ~user0] - (P.RESTORE - P.REMOVE))
    assert ((new >= P.REMOVE) & (new < P.RESTORE)).any()


@needs_cyborg
def test_react_restore_analyse_fills_idle_steps(raw):
    obs = _react_restore_obs(raw)
    ref, new = _actions(React_restore_minimal(), obs), _actions(P.ReactRestoreAnalyse(), obs)
    idle = ref == 0
    np.testing.assert_array_equal(new[~idle], ref[~idle])
    assert (new != 0).all()
    hosts = new[idle] - P.ANALYSE
    assert ((hosts >= 0) & (hosts < P.NUM_HOSTS) & (hosts != P.USER0)).all()
    for e in range(obs.shape[1]):                     # in each environment the idle steps visit the 12 hosts in turn
        seq = new[idle[:, e], e] - P.ANALYSE
        expected = P.ReactRestoreAnalyse.HOSTS[np.arange(len(seq)) % 12]
        np.testing.assert_array_equal(seq, expected)


# ── CyberMetaFormer ─────────────────────────────────────────────────────

VARIANTS = [
    {},
    {'temporal': ['conv', 'attn']},
    {'temporal': 'gru', 'host_mixer': 'attn'},
    {'temporal': ['pool', 'mlp'], 'host_mixer': 'gcn', 'age': False, 'pos': 'none'},
    {'temporal': 'ssd', 'host_mixer': 'topo', 'topo_mask': True},
    {'temporal': ['conv', 'ssd'], 'host_mixer': 'none', 'tokens': 'flat'},
]


def _episodes(store, small_idx, n=3):
    wd = D.WindowData(store, small_idx[:n])
    obs, _ = wd.episodes(torch.arange(n))
    return wd, D.scale_obs(obs)


def _small(**kw):
    torch.manual_seed(0)
    return C.CyberMetaFormer(**{'d': 32, 'layers': 2, **kw}).eval()


def test_config_within_budget_b1():
    cfg = yaml.safe_load(open(f'{D.ROOT}/hidden_state/configs/cybermetaformer.yaml'))
    model = build_model(cfg['model'], cfg['k'], **cfg['model_args'])
    n = sum(p.numel() for p in model.parameters())
    assert 147_000 <= n <= 220_000, n                     # 183 588 ± 20%
    assert model.episode_mode


@pytest.mark.parametrize('kw', VARIANTS, ids=lambda kw: '-'.join(f'{k}={v}' for k, v in kw.items()) or 'main')
def test_causal(store, small_idx, kw):
    """Output t depends only on obs[0..t]: a change at t0 leaves earlier outputs unchanged and changes later ones."""
    _, x = _episodes(store, small_idx)
    model = _small(**kw)
    t0 = 40
    x2 = x.clone()
    x2[:, t0:] = torch.randint(0, 2, x2[:, t0:].shape).float()
    with torch.no_grad():
        a, b = model.forward_episode(x), model.forward_episode(x2)
    assert a.shape == (3, 101, 12, 3)
    torch.testing.assert_close(a[:, :t0], b[:, :t0], rtol=1e-5, atol=1e-5)
    assert (a[:, t0:] - b[:, t0:]).abs().max() > 1e-3


def test_episode_equals_unpadded_window(store, small_idx):
    wd, x = _episodes(store, small_idx, 2)
    model = _small()
    with torch.no_grad():
        seq = model.forward_episode(x)
        for t in [0, 9, 63, 100]:
            win, mask = wd.windows(torch.tensor([0, 1]), torch.tensor([t, t]), t + 1)
            torch.testing.assert_close(seq[:, t], model(D.scale_obs(win), mask), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize('temporal', [['conv', 'attn'], ['ssd'], ['mlp', 'pool']])
def test_window_limits_receptive_field(store, small_idx, temporal):
    """window=w, L layers, alert age off: a change at s reaches only t ≤ s + L·(w − 1)."""
    _, x = _episodes(store, small_idx)
    w, L, s = 6, 2, 30
    model = _small(temporal=temporal, window=w, kernel=w, layers=L, age=False)
    x2 = x.clone()
    x2[:, s] = 1 - x2[:, s]
    with torch.no_grad():
        a, b = model.forward_episode(x), model.forward_episode(x2)
    reach = s + L * (w - 1)
    torch.testing.assert_close(a[:, reach + 1:], b[:, reach + 1:], rtol=1e-5, atol=1e-5)
    assert (a[:, s:reach + 1] - b[:, s:reach + 1]).abs().max() > 1e-4


def test_window_rejects_gru():
    with pytest.raises(ValueError):
        C.CyberMetaFormer(temporal=['conv', 'gru'], window=32)


def test_ssd_matches_recurrence():
    torch.manual_seed(0)
    m = C.SSD(16, heads=2, state=4).eval()
    with torch.no_grad():
        m.dt_bias.add_(2.0)                               # clear decay: a_t really < 1
        u = torch.randn(3, 20, 16)
        y = m(u)
        N, S, d = u.shape
        H, Pd, Ns = m.heads, m.p, m.state
        x, z, B, Cm, dt = m.inp(u).split([d, d, H * Ns, H * Ns, H], -1)
        x, B, Cm = x.view(N, S, H, Pd), B.view(N, S, H, Ns), Cm.view(N, S, H, Ns)
        dt = torch.nn.functional.softplus(dt + m.dt_bias)
        a = torch.exp(-dt * m.A_log.exp())
        h = torch.zeros(N, H, Pd, Ns)
        out = []
        for t in range(S):
            h = a[:, t, :, None, None] * h + (dt[:, t, :, None] * x[:, t])[..., None] * B[:, t, :, None]
            out.append(torch.einsum('nhpk,nhk->nhp', h, Cm[:, t]) + m.D.view(1, H, 1) * x[:, t])
        ref = m.out(torch.stack(out, 1).reshape(N, S, d) * torch.nn.functional.silu(z))
    assert a.min() < 0.9
    torch.testing.assert_close(y, ref, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize('kw', [{}, {'tokens': 'flat', 'host_mixer': 'none'}, {'temporal': 'ssd', 'age': False},
                                {'temporal': 'conv', 'pos': 'none'},
                                {'temporal': ['gru']}, {'temporal': ['gru'], 'host_mixer': 'attn'},   # gru_host_*
                                {'temporal': ['conv', 'gru']}])                                         # konv_gru
def test_step_matches_episode(store, small_idx, kw):
    """The online step mode (CMF latency in efficiency.py) gives the same output as forward_episode."""
    torch.manual_seed(0)
    model = C.CyberMetaFormer(**kw).eval()
    with torch.no_grad():
        for b in model.modules():
            if isinstance(b, C.SSD):
                b.dt_bias.add_(2.0)                       # clear decay (as in test_ssd_matches_recurrence)
        x =D.scale_obs(torch.as_tensor(store.obs[small_idx[:3]]))
        ref = model.forward_episode(x)
    state = model.init_state(3)
    out = torch.stack([model.step(x[:, t], state) for t in range(x.shape[1])], 1)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)
    with pytest.raises(ValueError):
        model.step(x[:, 0], state)


def test_step_rejects_unsupported():
    for kw in [{'temporal': ['conv', 'attn']}, {'temporal': ['conv', 'ssd'], 'window': 16}]:
        with pytest.raises(NotImplementedError):
            C.CyberMetaFormer(**kw).init_state(1)
    with pytest.raises(NotImplementedError):
        C.CyberMetaFormer(action_input=True).init_state(1)


def test_alert_age_features_match_features_module(store, small_idx):
    obs = store.obs[small_idx[:8]]
    feats = C.alert_age_features(D.scale_obs(torch.from_numpy(obs)))      # (E, T+1, 13, 8)
    export = F.alerts(obs, 'export', label_hosts=False)                   # (E, T+1, 13)
    age = F.alert_age(export)
    never = age == F.NEVER
    np.testing.assert_array_equal(feats[..., 4].numpy() == 1, never)
    expect = np.log1p(np.where(never, 100, age)) / np.log1p(100)
    np.testing.assert_allclose(feats[..., 0].numpy(), expect, rtol=1e-6, atol=1e-6)


@needs_cyborg
def test_topology_matches_simulator():
    from CybORG_plus_plus.mini_CAGE import minimal
    assert C.HOSTS == minimal.HOSTS and C.CONNECTED_HOSTS == minimal.CONNECTED_HOSTS
    assert C.SUBNET == minimal.create_subnets(13).astype(int).tolist()


def test_host_relations():
    rel = C.host_relations()
    assert torch.equal(rel, rel.T)
    hosts = C.HOSTS
    for i, j in [('ent2', 'opserv'), ('user1', 'ent0'), ('user2', 'ent0'), ('user3', 'ent1'), ('user4', 'ent1')]:
        assert rel[hosts.index(i), hosts.index(j)] == C.REL_EDGE
    assert int((rel == C.REL_EDGE).sum()) == 10
    assert rel[hosts.index('ent0'), hosts.index('ent1')] == C.REL_SUBNET
    assert rel[hosts.index('def'), hosts.index('opserv')] == C.REL_OTHER


def test_cybermetaformer_training_runs(store):
    from cybermetaformer.hidden_state.train import train
    cfg = {'model': 'cybermetaformer', 'mode': 'episode', 'k': 101, 'train_episodes': 32, 'val_episodes': 16,
           'epochs': 2, 'episodes_per_batch': 8, 'device': 'cpu', 'grad_clip': 1.0,
           'model_args': {'d': 32, 'layers': 1}}
    model, history, info = train(cfg, store, log=lambda s: None)
    assert len(history) == 2 and history[1]['loss'] < history[0]['loss']
    with pytest.raises(ValueError):
        train({**cfg, 'mode': 'window'}, store, log=lambda s: None)


@pytest.mark.parametrize('params', [
    {'layers': 2, 'mlp_ratio': 2, 'state': 8, 'heads': 2, 'kernel': 3},
    {'layers': 4, 'mlp_ratio': 2, 'state': 8, 'heads': 2, 'kernel': 3},
    {'layers': 4, 'mlp_ratio': 4, 'state': 32, 'heads': 4, 'kernel': 6},
])
def test_hpo_space_stays_in_budget_b1(params):
    import optuna
    from cybermetaformer.hidden_state import hpo
    over = hpo.space_cybermetaformer(optuna.trial.FixedTrial({**params, 'dropout': 0.1, 'lr': 1e-3,
                                                              'weight_decay': 1e-2, 'class_weights': None}))
    args = {**hpo.load_base('cybermetaformer')['model_args'], **over['model_args']}
    lo, hi = C.BUDGETS['b1']
    assert lo <= C.n_params(**args) <= hi
    assert C.n_params(**{**args, 'd': args['d'] + 8}) > hi        # the widest width that fits


def test_variants_names_and_budgets():
    from cybermetaformer.hidden_state.run_grid import apply_variant, load_variants, planned_runs
    names = [r[3] for r in planned_runs(['cybermetaformer'], [8], [0, 1], ['host_yok', 'b2'])]
    assert names == ['cybermetaformer-host_yok_k101_s0', 'cybermetaformer-host_yok_k101_s1',
                     'cybermetaformer-b2_k101_s0', 'cybermetaformer-b2_k101_s1']
    assert [r[3] for r in planned_runs(['gru_full'], [8], [0])] == ['gru_full_k101_s0']
    base = {**yaml.safe_load(open(f'{D.ROOT}/hidden_state/configs/cybermetaformer.yaml'))}
    for v, spec in load_variants().items():
        cfg = apply_variant(base, 'cybermetaformer', v)
        lo, hi = C.BUDGETS[(spec or {}).get('fit', 'b1')]
        assert lo <= C.n_params(**cfg['model_args']) <= hi, v
        assert cfg['variant']['name'] == v
    assert apply_variant(base, 'cybermetaformer', 'ana_m4')['model_args'] == \
        {**base['model_args'], 'd': C.fit_width('b1', **{k: v for k, v in base['model_args'].items() if k != 'd'})}


# ── action input, domain generalisation, augmentation, (ii) ─────────────

def test_prev_action_alignment(store):
    from minicage.dataset import loaders as L
    raw, _ = L.load_raw(D.DEFAULT_DATA, fields=['act_blue'])
    assert (store.prev_action[:, 0] == D.NO_ACTION).all()
    np.testing.assert_array_equal(store.prev_action[:, 1:], raw['act_blue'])


@pytest.mark.parametrize('name', ['cybermetaformer', 'gru_full'])
def test_action_input_causal_and_used(store, small_idx, name):
    wd, x = _episodes(store, small_idx)
    act = wd.prev_actions(torch.arange(3))
    torch.manual_seed(0)
    model = build_model(name, 101, **({'d': 32, 'layers': 2} if name == 'cybermetaformer' else {}),
                        action_input=True).eval()
    act2 = act.clone()
    act2[:, 40:] = (act2[:, 40:] + 7) % D.N_BLUE_ACTIONS
    with torch.no_grad():
        a, b = model.forward_episode(x, act), model.forward_episode(x, act2)
    torch.testing.assert_close(a[:, :40], b[:, :40], rtol=1e-5, atol=1e-5)
    assert (a[:, 40:] - b[:, 40:]).abs().max() > 1e-4
    with pytest.raises(Exception):
        model.forward_episode(x)


def test_alert_drop_removes_whole_runs(store, small_idx):
    from cybermetaformer.hidden_state import dg
    _, x = _episodes(store, small_idx, 8)
    out = dg.alert_runs_drop(x, 0.5, generator=torch.Generator().manual_seed(0))
    bits = lambda z: z[..., D.BITS].reshape(*z.shape[:2], 13, 4)
    a0 = (bits(x)[..., 1] > 0.5) | (bits(x)[..., 3] > 0.5)
    a1 = (bits(out)[..., 1] > 0.5) | (bits(out)[..., 3] > 0.5)
    dropped = a0 & ~a1
    assert dropped.any() and (a1 <= a0).all()
    start = a0 & ~torch.cat([torch.zeros_like(a0[:, :1]), a0[:, :-1]], 1)
    run = start.long().cumsum(1)
    for e, h in zip(*torch.where(dropped.any(1))):           # every step of the dropped spell is dropped
        for r in torch.unique(run[e, :, h][dropped[e, :, h]]):
            m = (run[e, :, h] == r) & a0[e, :, h]
            assert dropped[e, :, h][m].all()
    torch.testing.assert_close(out[..., 4 * 13:], x[..., 4 * 13:])
    assert (bits(out)[..., 0] == bits(x)[..., 0]).all()


def test_time_mask_keeps_first_step():
    from cybermetaformer.hidden_state import dg
    x = torch.ones(4, 101, 78)
    out = dg.time_mask(x, 0.5, generator=torch.Generator().manual_seed(0))
    assert (out[:, 0] == 1).all() and (out.sum(-1) == 0).any()
    assert set(out.sum(-1).unique().tolist()) <= {0.0, 78.0}


@pytest.mark.parametrize('extra', [
    {'dg': {'method': 'groupdro'}}, {'dg': {'method': 'vrex', 'anneal_steps': 1}}, {'dg': {'method': 'coral'}},
    {'dg': {'method': 'dann'}}, {'augment': {'alert_drop': 0.1, 'time_mask': 0.1}},
    {'train_red': 'meander'}, {'model_args': {'d': 32, 'layers': 1, 'action_input': True}},
], ids=['groupdro', 'vrex', 'coral', 'dann', 'augment', 'train_red', 'aksiyon'])
def test_m3_training_options_run(store, extra):
    from cybermetaformer.hidden_state.train import train
    cfg = {'model': 'cybermetaformer', 'mode': 'episode', 'k': 101, 'train_episodes': 48, 'val_episodes': 32,
           'epochs': 1, 'episodes_per_batch': 8, 'device': 'cpu', 'grad_clip': 1.0,
           'model_args': {'d': 32, 'layers': 1}, **extra}
    model, history, info = train(cfg, store, log=lambda s: None)
    assert np.isfinite(history[0]['loss'])
    if 'train_red' in extra:
        assert info['n_train_episodes'] < 48 and info['n_val_episodes'] < 32


def test_domain_ids_cover_training_policies(store):
    from cybermetaformer.hidden_state import dg
    dom, names = dg.domain_ids(store, store.split('train'))
    assert len(names) == 10 and dom.max() == 9
    assert not any('eps0.3' in n for n in names)


def test_ood_suite_sets_and_pairs():
    """OOD package: sets depend on the base sets and policy names; paired test against the seed mean."""
    from cybermetaformer.hidden_state import metrics as M
    from cybermetaformer.hidden_state import ood_suite as O
    from minicage.dataset.policies import BLUE_POLICIES
    assert all(b in O.BASE for b, _ in O.SETS.values())
    assert all(p is None or p in BLUE_POLICIES for _, p in O.SETS.values())
    assert [s for s, _ in O.CURVE] == ['eps0.1', 'ood', 'eps0.5', 'eps0.7'] and set(O.MODELS) >= {m for p in O.PAIRS for m in p}
    rng = np.random.default_rng(0)
    y = rng.choice(3, size=(40, 20, 12), p=[0.7, 0.1, 0.2])

    def member(noise):
        pred = np.where(rng.random(y.shape) < 1 - noise, y, rng.choice(3, size=y.shape))
        silent = np.stack([(y > 0).sum((1, 2)), ((y > 0) & (pred > 0)).sum((1, 2))], 1)
        return M.episode_confusion(y, pred), silent
    boot = {m: {s: [member(0.05 if m == 'cybermetaformer' else 0.3) for _ in range(2)] for s in O.SETS} for m in O.MODELS}
    rows = O.pair_tests(boot, n_boot=200)
    assert len(rows) == len(O.PAIRS) * len(O.SETS) * 2
    r = next(r for r in rows if r['a'] == 'cybermetaformer' and r['set'] == 'ood' and r['criterion'] == 'host_macro_f1')
    assert r['diff'] > 0 and r['lo'] > 0 and all(x['p_holm'] >= x['p'] for x in rows)


@pytest.mark.parametrize('kind', ['topo', 'attn'])
def test_explain_m3_attention_matches_host_mixer(kind):
    """The attention recomputed by explain_m3 equals the output of HostMixer.forward (bias and mask included)."""
    from cybermetaformer.hidden_state.explain_m3 import attention_probs
    torch.manual_seed(0)
    hm = C.HostMixer(16, kind, heads=4, topo_mask=kind == 'topo').eval()
    with torch.no_grad():
        if kind == 'topo':
            hm.rel_bias.normal_()
        x = torch.randn(2, 5, D.NUM_HOSTS, 16)
        P = attention_probs(hm, x)                                           # (B, S, kafa, 13, 13)
        v = hm.qkv(x)[..., 32:].view(2, 5, D.NUM_HOSTS, 4, 4).transpose(2, 3)  # (B, S, kafa, 13, 4)
        y = hm.out((P @ v).transpose(2, 3).reshape(2, 5, D.NUM_HOSTS, 16))
        torch.testing.assert_close(y, hm(x), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(P.sum(-1), torch.ones(2, 5, 4, D.NUM_HOSTS))
    if kind == 'topo':
        assert P[..., C.host_relations() == C.REL_OTHER].max() == 0          # masked "other" relation

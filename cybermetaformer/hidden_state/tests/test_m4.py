"""M4 revision: corrected-observation environment (FixedObsCAGE) and paired resimulation with forced Blue actions.

The FixedObsCAGE unit tests call _process_state with the same RNG state and the same previous observation; only the
Blue action and blue_success change. The resimulation tests use the smoke data in the repository."""
import copy
import os

import numpy as np
import pytest

from cybermetaformer.hidden_state import data as D

try:                                                  # the environment needs CybORG++
    from minicage.dataset import resimulate as R
    from minicage.dataset.env import FixedCAGE, FixedObsCAGE
    from minicage.dataset.generate import run_shard
    from minicage.dataset.policies import NUM_HOSTS, REMOVE, RESTORE, USER0, make_red
except ImportError:
    R = None
needs_cyborg = pytest.mark.skipif(R is None, reason='CybORG_plus_plus not available')

SMOKE = os.path.join(D.DATA_ROOT, 'smoke')
HOST = 3                                              # target of the remove attempts (a label host)


def _safety(obs, h):
    return obs[:, 4 * h + 2:4 * h + 4]


def _safety_steps(obs_blue, h=HOST):
    return obs_blue[:, :, 4 * h + 2:4 * h + 4]


@pytest.fixture(scope='module')
def env():
    """FixedObsCAGE stepped a few times; HOST shows 'exploited' [0,1], user0 shows 'root' [1,1]."""
    np.random.seed(0)
    e = FixedObsCAGE(num_envs=4, remove_bugs=True)
    state, _ = e.reset()
    red = make_red('meander')
    for _ in range(6):
        state, *_ = e.step(red_action=red.get_action(state['Red']), blue_action=np.zeros((4, 1), int))
    blue = e.proc_states['Blue']
    blue[:, 4 * HOST + 2:4 * HOST + 4] = [0, 1]
    blue[:, 4 * USER0 + 2:4 * USER0 + 4] = [1, 1]
    return e


def _obs(env, cls, blue_action, success, seed=7):
    """cls._process_state on a copy of env; same seed, Red sleeps."""
    e = copy.deepcopy(env)
    e.blue_success = np.full((e.num_envs, 1), float(success))
    np.random.seed(seed)
    ob = cls._process_state(e, e.state, e.current_decoys, red_action=np.zeros((e.num_envs, 1), int),
                            blue_action=np.full((e.num_envs, 1), blue_action))
    return ob['Blue'], e


@needs_cyborg
def test_failed_remove_no_flag(env):
    buggy, _ = _obs(env, FixedCAGE, REMOVE + HOST, -1)
    fixed, e = _obs(env, FixedObsCAGE, REMOVE + HOST, -1)
    sleep, _ = _obs(env, FixedCAGE, 0, -1)
    assert (_safety(buggy, HOST) == [1, 0]).all()           # defect: a failed remove writes "removed"
    np.testing.assert_array_equal(fixed, sleep)             # fix: as if the action never happened
    assert (_safety(fixed, HOST) == [0, 1]).all()
    assert (e.blue_success == -1).all()                     # the recorded success does not change


@needs_cyborg
def test_successful_remove_unchanged(env):
    buggy, _ = _obs(env, FixedCAGE, REMOVE + HOST, 1)
    fixed, _ = _obs(env, FixedObsCAGE, REMOVE + HOST, 1)
    np.testing.assert_array_equal(fixed, buggy)
    assert (_safety(fixed, HOST) == [1, 0]).all()


@needs_cyborg
def test_invalid_user0_restore_keeps_flag(env):
    buggy, _ = _obs(env, FixedCAGE, RESTORE + USER0, -1)
    fixed, _ = _obs(env, FixedObsCAGE, RESTORE + USER0, -1)
    sleep, _ = _obs(env, FixedCAGE, 0, -1)
    assert (_safety(buggy, USER0) == [0, 0]).all()          # defect: a no-op restore clears the flag
    np.testing.assert_array_equal(fixed, sleep)
    assert (_safety(fixed, USER0) == [1, 1]).all()
    ok, _ = _obs(env, FixedObsCAGE, RESTORE + HOST, 1)      # a successful restore stays as is
    np.testing.assert_array_equal(ok, _obs(env, FixedCAGE, RESTORE + HOST, 1)[0])


def _smoke_shards():
    import json
    with open(os.path.join(SMOKE, 'meta.json')) as f:
        meta = json.load(f)
    for s in meta['shards']:
        with np.load(os.path.join(SMOKE, s['file'])) as z:
            yield meta, s, {k: z[k] for k in z.files}


def _resim(meta, s, src, cls):
    return run_shard(s['red'], s['blue'], s['episodes'], meta['steps_per_episode'], meta['remove_bugs'], s['seed'],
                     env_cls=cls, blue_actions=src['act_blue'])[0]


@needs_cyborg
def test_forced_actions_reproduce_bytes():
    """FixedCAGE + actions forced from the record = the record (all fields, byte for byte)."""
    for meta, s, src in _smoke_shards():
        new = _resim(meta, s, src, FixedCAGE)
        assert set(new) == set(src)
        for k in src:
            assert new[k].dtype == src[k].dtype and np.array_equal(new[k], src[k]), (s['file'], k)
        problems, counts = R.gate(src, new)
        assert problems == [] and counts['episodes_changed'] == 0


@needs_cyborg
def test_fixedobs_gate_and_attribution():
    seen = {}
    for meta, s, src in _smoke_shards():
        new = _resim(meta, s, src, FixedObsCAGE)
        problems, counts = R.gate(src, new)
        assert problems == [], (s['file'], problems)
        seen[s['blue']] = counts
    assert seen['react_restore']['episodes_changed'] == 0     # only the restoring policy: no failed action
    r = seen['random']
    assert r['chains_remove'] > 0 and r['host_steps_changed_remove'] >= r['chains_remove']
    assert r['chains_remove'] <= r['failed_remove_actions']


@needs_cyborg
def test_gate_rejects_violations():
    meta, s, src = next(x for x in _smoke_shards() if x[1]['blue'] == 'random')
    new = _resim(meta, s, src, FixedObsCAGE)
    assert R.gate(src, new)[0] == []

    bad = {k: v.copy() for k, v in new.items()}
    bad['obs_blue'][0, 5, 5 * NUM_HOSTS + HOST] ^= 1         # the decoy column must never change
    assert any('decoy' in q for q in R.gate(src, bad)[0])

    ph = (src['obs_blue'] != new['obs_blue'])[:, :, :4 * NUM_HOSTS].reshape(*src['obs_blue'].shape[:2], NUM_HOSTS, 4)
    coupled = np.zeros(ph.shape[:2], bool)
    coupled[:, 1:] = ph[:, :-1, :, 0].any(-1)                # steps allowed by the coupling between hosts
    d = ph[..., 2:].any(-1) & ~coupled[:, :, None]
    d[:, :, USER0] = False                                   # on label hosts only the remove chain occurs
    e, t, h = map(int, np.argwhere(d)[0])
    bad = {k: v.copy() for k, v in new.items()}
    bad['obs_blue'][e, t, 4 * h] ^= 1                        # activity difference in the remove chain
    assert any('activity/scan differences' in q for q in R.gate(src, bad)[0])

    own = np.zeros(ph.shape[:3], bool)                       # coupling allowed but no cause of the host itself
    a = src['act_blue'].astype(int)
    for lo in (REMOVE, RESTORE):
        ee, tt = np.nonzero((a >= lo) & (a < lo + NUM_HOSTS))
        own[ee, tt + 1, a[ee, tt] - lo] = True
    quiet = ~ph.any(-1)
    cand = coupled[:, 1:, None] & ~own[:, 1:] & quiet[:, :-1] & quiet[:, 1:]
    assert cand.any()                                        # smoke has a coupling step caused by a restore of user0
    e, t, h = map(int, np.argwhere(cand)[0])
    bad = {k: v.copy() for k, v in new.items()}
    bad['obs_blue'][e, t + 1, 4 * h + 2] ^= 1                # the coupling carries only activity, not the safety bit
    assert any('safety-bit differences' in q for q in R.gate(src, bad)[0])

    bad = {k: v.copy() for k, v in new.items()}
    same = (_safety_steps(src['obs_blue']) == _safety_steps(new['obs_blue'])).all(-1)    # (episode, step)
    aimed = np.isin(src['act_blue'], [REMOVE + HOST, RESTORE + HOST])
    e, t = map(int, np.argwhere(same[:, :-1] & same[:, 1:] & ~aimed)[0])   # no difference, no action on HOST
    bad['obs_blue'][e, t + 1, 4 * HOST + 2] ^= 1
    assert any('do not start' in q for q in R.gate(src, bad)[0])
    bad = {k: v.copy() for k, v in new.items()}
    bad['true_state'][0, 1, 0] ^= 1
    assert any('true_state' in q for q in R.gate(src, bad)[0])


def test_corrected_obs_stats_match_point_values():
    """corrected_obs: with all weights 1 the bootstrap statistic = mean of the members' point values."""
    from cybermetaformer.hidden_state import corrected_obs as G
    from cybermetaformer.hidden_state import metrics as M
    rng = np.random.default_rng(0)
    runs = []
    for _ in range(3):
        y, p = rng.integers(0, 3, (40, 100, 12)), rng.integers(0, 3, (40, 100, 12))
        n = rng.integers(0, 50, (40, 1))
        runs.append({'conf': M.episode_confusion(y, p), 'silent': np.concatenate([n, rng.integers(0, n + 1)], 1)})
    ones = {'s': np.ones((1, 40))}
    for key in ('conf', 'silent'):
        assert G._stat(runs, key, 's')(ones)[0] == pytest.approx(np.mean([G._value(r, key) for r in runs]))


@needs_cyborg
def test_rewired_topology_changes_only_user_links():
    from CybORG_plus_plus.mini_CAGE import minimal
    from minicage.dataset.env import REWIRED_HOSTS, RewiredCAGE
    orig, H = list(minimal.CONNECTED_HOSTS), minimal.HOSTS.index
    changed = {minimal.HOSTS[h]: b for h, (a, b) in enumerate(zip(orig, REWIRED_HOSTS)) if a != b}
    assert changed == {'user1': 'ent1', 'user2': 'ent1', 'user3': 'ent0', 'user4': 'ent0'}
    out = run_shard('bline', 'sleep', 20, 30, True, 5, env_cls=RewiredCAGE)[0]
    assert minimal.CONNECTED_HOSTS == orig                           # restored after the step
    ts = out['true_state'].reshape(20, 31, NUM_HOSTS, 3)
    for e in range(20):                                              # against Sleep: first discovered ent = new link
        users = [u for u in ('user1', 'user2', 'user3', 'user4') if (ts[e, :, H(u), 2] == 1).any()]
        known = {x: int(np.argmax(ts[e, :, H(x), 0] >= 0)) for x in ('ent0', 'ent1') if (ts[e, :, H(x), 0] >= 0).any()}
        assert len(users) == 1 and known, e
        assert min(known, key=known.get) == REWIRED_HOSTS[H(users[0])]


@needs_cyborg
def test_linked_bline_matches_bline_on_minicage_topology():
    """LinkedBline with MiniCAGE's own link table is byte-identical to B_line_minimal."""
    from CybORG_plus_plus.mini_CAGE import minimal

    class SameLinks(FixedCAGE):
        connected_hosts = list(minimal.CONNECTED_HOSTS)              # selects LinkedBline, same topology

    for blue in ('sleep', 'random', 'react_restore_eps0.3'):
        a = run_shard('bline', blue, 30, 100, True, 11, env_cls=FixedCAGE)[0]
        b = run_shard('bline', blue, 30, 100, True, 11, env_cls=SameLinks)[0]
        assert all(a[k].dtype == b[k].dtype and np.array_equal(a[k], b[k]) for k in a), blue


# ── host-permutation OOD (item 6 (a)) ──────────────────────────────────────

def test_permutation_columns_and_relations():
    import torch
    from cybermetaformer.hidden_state import ood_permutation as P
    from cybermetaformer.hidden_state.models.cybermetaformer import HOSTS, SUBNET, host_relations
    rel, rng = host_relations(), np.random.default_rng(0)
    obs = rng.integers(0, 5, (3, 6, D.OBS_DIM)).astype(np.int8)
    y = rng.integers(0, 3, (3, 5, len(D.LABEL_HOSTS))).astype(np.int8)
    for c, pi in P.PERMS.items():
        assert pi[HOSTS.index('def')] == HOSTS.index('def') and pi[HOSTS.index('user0')] == HOSTS.index('user0')
        assert [SUBNET[j] for j in pi] == SUBNET, c                      # subnets are preserved
        o, yy = P.permute(obs, y, pi)
        h = 9                                                            # user1 slot: all columns of π(user1)
        assert np.array_equal(D.to_host_tokens(torch.as_tensor(o))[..., h, :],
                              D.to_host_tokens(torch.as_tensor(obs))[..., pi[h], :]), c
        inv = np.argsort(pi)
        assert np.array_equal(P.permute(o, yy, inv)[0], obs) and np.array_equal(P.permute(o, yy, inv)[1], y), c
        os_ = P.permute(obs, y, pi, move_decoys=False)[0]                 # P1s/P2s: decoys stay, the rest moves
        assert np.array_equal(os_[..., D.DECOY], obs[..., D.DECOY]) and np.array_equal(os_[..., :D.DECOY.start],
                                                                                       o[..., :D.DECOY.start]), c
    p = torch.as_tensor(P.PERMS['P1'])
    assert torch.equal(rel[p][:, p], rel)                                # P1 preserves the edges
    p = torch.as_tensor(P.PERMS['P2'])
    assert not torch.equal(rel[p][:, p], rel)                            # P2 breaks the user → ent edges


@pytest.mark.parametrize('host_mixer,cond', [('topo', 'P1'), ('attn', 'P1'), ('attn', 'P2'), ('topo', 'P2')])
def test_data_permutation_equals_host_embedding_permutation(host_mixer, cond):
    """Data permutation ≡ permutation of the host embedding: f(Px)_i = f_{E∘π⁻¹}(x)_{π(i)} when the relation
    matrix is invariant under π. P1 preserves the edges (topo and attn); P2 only in the unbiased model (attn), where
    P2 ≡ P3 (π2 is its own inverse). In the topology-biased model P2 departs: the gap is the misaligned prior."""
    import torch
    from cybermetaformer.hidden_state import evaluate as E
    from cybermetaformer.hidden_state import ood_permutation as P
    from cybermetaformer.hidden_state.models import build_model
    torch.manual_seed(0)
    model = build_model('cybermetaformer', 101, d=16, layers=2, heads=2, host_mixer=host_mixer).eval()
    with torch.no_grad():                                                # zero bias at random init; make them differ
        for blk in model.blocks:
            if getattr(blk.host, 'rel_bias', None) is not None:
                blk.host.rel_bias.normal_(0, 1)
    st = D.EpisodeStore(SMOKE)
    obs, y, pa = st.obs[:4], st.labels[:4], st.prev_action[:4]
    pi, cols = P.PERMS[cond], P.label_columns(P.PERMS[cond])

    def logits(o, yy):
        return E.predict_logits(model, D.WindowData(P.ArrayStore(o, yy, pa), np.arange(len(o))), 101)

    lp = logits(*P.permute(obs, y, pi))
    with P.shuffled_host_emb(model, np.argsort(pi)):
        le = logits(obs, y)[..., cols, :]
    assert not np.allclose(lp, logits(obs, y)[..., cols, :], atol=1e-3)  # the embedding really has an effect
    if host_mixer == 'topo' and cond == 'P2':
        assert np.abs(lp - le).max() > 1e-3
        return
    np.testing.assert_allclose(lp, le, atol=1e-5)
    if cond == 'P2':
        with P.shuffled_host_emb(model, P.P3_PERM):
            np.testing.assert_allclose(lp, logits(obs, y)[..., cols, :], atol=1e-5)


def test_bootstrap_rows_point_values_paired_and_two_sample():
    from cybermetaformer.hidden_state import ood_permutation as P
    from cybermetaformer.hidden_state.corrected_obs import _value
    rng = np.random.default_rng(1)

    def member(E):
        y = rng.integers(0, 3, (E, 7, 12))
        p = np.where(rng.random(y.shape) < 0.8, y, rng.integers(0, 3, y.shape))
        alert = rng.random(y.shape) < 0.3
        return P.records(y, p, alert, P.F.alert_age(alert))

    runs = {k: [member(E) for _ in range(2)] for k, E in (('a', 9), ('b', 9), ('c', 6), ('d', 6))}
    mean = {k: {c: np.mean([_value(r, key) for r in v]) for c, (key, _) in P.CRITERIA.items()} for k, v in runs.items()}
    rows = P.bootstrap_rows(runs, {'x': 9}, 50, 'a', 'b')                                  # paired
    assert [r['diff'] for r in rows] == pytest.approx([mean['a'][c] - mean['b'][c] for c in P.CRITERIA])
    side = {'a': 'x', 'b': 'x', 'c': 'y', 'd': 'y'}
    rows = P.bootstrap_rows(runs, {'x': 9, 'y': 6}, 50, 'a', 'c', side={'a': 'x', 'c': 'y'})   # two-sample
    assert [r['diff'] for r in rows] == pytest.approx([mean['a'][c] - mean['c'][c] for c in P.CRITERIA])
    rows = P.bootstrap_rows(runs, {'x': 9, 'y': 6}, 50, 'a', 'b', base=('c', 'd'), side=side)
    assert [r['diff'] for r in rows] == pytest.approx(
        [(mean['a'][c] - mean['c'][c]) - (mean['b'][c] - mean['d'][c]) for c in P.CRITERIA])


def test_host_relations_rewired_links_and_updated_prior():
    """The relation matrix built with the new link table changes only at user ↔ ent edges; updated_prior is applied
    only to topology-biased mixers and is restored on exit."""
    import torch
    from cybermetaformer.hidden_state import ood_topology as T
    from cybermetaformer.hidden_state.models import build_model
    from cybermetaformer.hidden_state.models.cybermetaformer import (CONNECTED_HOSTS, HOSTS, REL_EDGE, REL_OTHER,
                                                                   host_relations)
    links = list(CONNECTED_HOSTS)
    for u, e in (('user1', 'ent1'), ('user2', 'ent1'), ('user3', 'ent0'), ('user4', 'ent0')):
        links[HOSTS.index(u)] = e
    old, new = host_relations(), host_relations(links)
    assert torch.equal(host_relations(CONNECTED_HOSTS), old)
    diff = {(HOSTS[i], HOSTS[j]) for i, j in zip(*torch.nonzero(old != new, as_tuple=True))}
    users, ents = {'user1', 'user2', 'user3', 'user4'}, {'ent0', 'ent1'}
    assert diff and all((a in users and b in ents) or (a in ents and b in users) for a, b in diff)
    assert new[HOSTS.index('user1'), HOSTS.index('ent1')] == REL_EDGE
    assert new[HOSTS.index('user1'), HOSTS.index('ent0')] == REL_OTHER
    torch.manual_seed(0)
    model = build_model('cybermetaformer', 101, d=16, layers=2, heads=2, host_mixer='topo').eval()
    with T.updated_prior(model, links):
        assert all(torch.equal(b.host.rel, new) for b in model.blocks)
    assert all(torch.equal(b.host.rel, old) for b in model.blocks)
    with pytest.raises(AssertionError):
        with T.updated_prior(build_model('cybermetaformer', 101, d=16, layers=1, heads=2, host_mixer='attn'), links):
            pass


# ── intervention experiments (item 7) ──────────────────────────────────────

def test_interventions_touch_only_the_host_set():
    from cybermetaformer.hidden_state import interventions as MU
    st = D.EpisodeStore(SMOKE)
    obs, red, blue = st.obs, st.red_policy, st.blue_policy
    hosts, target = [1, 9, 10], 11
    keep = np.setdiff1d(np.arange(D.OBS_DIM), np.concatenate([MU.host_columns(h) for h in hosts]))
    cols = np.concatenate([MU.host_columns(h) for h in hosts])
    assert set(MU.host_columns(target)) <= set(keep)
    for kind in MU.INTERVENTIONS:
        o = MU.intervene(obs, hosts, kind, red, blue, np.random.default_rng(0))
        assert np.array_equal(o[..., keep], obs[..., keep]), kind
        if kind == 'empty':
            assert np.array_equal(o[..., cols], np.broadcast_to(obs[:, :1, cols], o[..., cols].shape))
        elif kind == 'shuffle':
            assert np.array_equal(o[:, 0], obs[:, 0])
            for e in range(len(obs)):                                    # same rows in the episode, reordered
                a = np.sort(np.ascontiguousarray(obs[e][:, cols]).view(f'V{len(cols)}').ravel())
                b = np.sort(np.ascontiguousarray(o[e][:, cols]).view(f'V{len(cols)}').ravel())
                assert np.array_equal(a, b)
        else:                                                            # swap: another episode, same policy
            for e in range(len(obs)):
                src = [f for f in range(len(obs)) if np.array_equal(obs[f][:, cols], o[e][:, cols])]
                assert any(f != e and red[f] == red[e] and blue[f] == blue[e] for f in src) or \
                    np.array_equal(obs[e][:, cols], o[e][:, cols]), e
    sets = MU.host_sets()
    for t, d in sets.items():
        h = MU.HOSTS.index(t)
        assert h not in d['related'] and all(h not in s and len(s) == len(d['related']) for s in d['random'])
        assert not set(d['related']) & set(sum(d['random'], []))


def test_intervention_has_no_effect_without_host_mixer():
    """In the model without a host mixer the target's output does not depend on other hosts' observations (exactly);
    in the full-attention model it does."""
    import torch
    from cybermetaformer.hidden_state import evaluate as E
    from cybermetaformer.hidden_state import interventions as MU
    from cybermetaformer.hidden_state.models import build_model
    st = D.EpisodeStore(SMOKE)
    obs, y, pa = st.obs[:6], st.labels[:6], st.prev_action[:6]
    j = D.LABEL_HOSTS.index(MU.HOSTS.index('user1'))
    hosts = MU.host_sets()['user1']['related']
    o = MU.intervene(obs, hosts, 'swap', st.red_policy[:6], st.blue_policy[:6], np.random.default_rng(0))
    for mixer, same in (('none', True), ('attn', False)):
        torch.manual_seed(0)
        model = build_model('cybermetaformer', 101, d=16, layers=2, heads=2, host_mixer=mixer).eval()
        lg = [E.predict_logits(model, D.WindowData(MU.ArrayStore(x, y, pa), np.arange(len(x))), 101)[..., j, :]
              for x in (obs, o)]
        assert np.allclose(lg[0], lg[1], atol=1e-6) == same, mixer

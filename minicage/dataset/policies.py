"""Batched behaviour policies (MiniCAGE). A new instance is created for each shard."""
import numpy as np

from CybORG_plus_plus.mini_CAGE import (
    Meander_minimal, React_restore_minimal, Blue_sleep, Restore_decoys)
from CybORG_plus_plus.mini_CAGE.minimal import HOSTS
from CybORG_plus_plus.mini_CAGE.red_bline_agent import B_line_minimal

NUM_BLUE_ACTIONS = 53
NUM_HOSTS = 13
USER0 = 8                  # Red's foothold; restoring it is invalid (minimal.py check_blue_action, action 48)
ANALYSE, REMOVE, RESTORE = 1, 1 + 2 * NUM_HOSTS, 1 + 3 * NUM_HOSTS     # aksiyon = tip ofseti + host

RED_POLICIES = ['bline', 'meander']
# The order is written into the datasets as an index (blue_policy); new policies are only appended at the end.
BLUE_POLICIES = ['sleep', 'react_restore', 'restore_decoys', 'random',
                 'react_restore_eps0.1', 'react_restore_eps0.3',
                 # 2026-10-22, test only: severity curve and mechanism shift
                 'react_restore_eps0.5', 'react_restore_eps0.7', 'react_restore_analyse', 'react_remove']
V1_BLUE_POLICIES = BLUE_POLICIES[:6]   # policies of minicage_v1 and minicage_v1_bugged; default of generate.py


class LinkedBline(B_line_minimal):
    """B_line_minimal; the ent host targeted after a user host comes from the link table, not a hard-coded mapping.

    In state 4, B_line_minimal uses the fixed mapping user1/2 → ent0, user3/4 → ent1 (the MiniCAGE link table).
    The original CybORG B-line finds the next host from its observation; here the ent is the rooted user host's link.
    State 4 is reached only from a user host, so with the original table the behaviour is identical to B_line_minimal.
    """

    def __init__(self, links):
        super().__init__()
        self.links = list(links)

    def DiscoverNetworkServices_ent(self, host, obs, last_host):
        if host in (1, 2) and last_host in (9, 10, 11, 12):
            host = self.enterprise_host = HOSTS.index(self.links[last_host])
        return super().DiscoverNetworkServices_ent(host, obs, last_host)


class BatchedBline:
    """red_bline_agent.B_line_minimal handles a single environment; one instance is kept per environment.

    test_agent.B_line_minimal is not used: it gets stuck on network_ent1 after escalate_user1
    (B-line x Sleep ≈ -10, README -1141). If links is given (modified topology), LinkedBline is used.
    """

    def __init__(self, links=None):
        self.agents, self.links = None, links

    def get_action(self, observation):
        if self.agents is None:
            make = B_line_minimal if self.links is None else (lambda: LinkedBline(self.links))
            self.agents = [make() for _ in range(observation.shape[0])]
        return np.array([np.asarray(a.get_action(o)).reshape(-1)[0]
                         for a, o in zip(self.agents, observation)]).reshape(-1, 1).astype(int)


def make_red(name, links=None):
    """links: link table of the modified topology (env.RewiredCAGE); used only by B-line."""
    if name == 'bline':
        return BatchedBline(links)
    if name == 'meander':
        return Meander_minimal()
    raise ValueError(f'Unknown red policy: {name}')


class RandomBlue:
    def get_action(self, observation):
        return np.random.randint(0, NUM_BLUE_ACTIONS, size=(observation.shape[0], 1))


class EpsilonBlue:
    """Random action with probability ε; the base agent keeps processing the observation at every step."""

    def __init__(self, base, epsilon):
        self.base, self.epsilon = base, epsilon

    def get_action(self, observation):
        action = self.base.get_action(observation)
        explore = np.random.rand(observation.shape[0]) < self.epsilon
        action[explore, 0] = np.random.randint(0, NUM_BLUE_ACTIONS, size=explore.sum())
        return action


class ReactRestoreAnalyse(React_restore_minimal):
    """React_restore_minimal; at steps needing no restore (action 0), cyclic analyse over the 12 hosts except user0.

    analyse does not change the environment but rewrites the safety flag [1,1] on a rooted host (minimal.py:1288-1304);
    the flag triggers React_restore at the next step. In the training policies analyse appears only via random actions.
    """

    HOSTS = np.array([h for h in range(NUM_HOSTS) if h != USER0])

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self.cursor = None

    def get_action(self, observation, *args, **kwargs):
        action = super().get_action(observation, *args, **kwargs)
        if self.cursor is None:
            self.cursor = np.zeros(observation.shape[0], np.int64)
        idle = action[:, 0] == 0
        action[idle, 0] = ANALYSE + self.HOSTS[self.cursor[idle] % len(self.HOSTS)]
        self.cursor[idle] += 1
        return action


class ReactRemove(React_restore_minimal):
    """Trigger of React_restore_minimal; remove (27 + host) instead of restore (40 + host).

    remove always fails on a rooted host, but the observation shows [1,0] "removed". The user0
    restore is already invalid (no-op), so it stays as it is: a remove on user0 would be valid and cut Red's foothold.
    """

    def get_action(self, observation, *args, **kwargs):
        action = super().get_action(observation, *args, **kwargs)
        restore = (action[:, 0] >= RESTORE) & (action[:, 0] != RESTORE + USER0)
        action[restore, 0] -= RESTORE - REMOVE
        return action


def make_blue(name):
    if name == 'sleep':
        return Blue_sleep()
    if name == 'react_restore':
        return React_restore_minimal()
    if name == 'restore_decoys':
        return Restore_decoys()
    if name == 'random':
        return RandomBlue()
    if name == 'react_restore_analyse':
        return ReactRestoreAnalyse()
    if name == 'react_remove':
        return ReactRemove()
    if name.startswith('react_restore_eps'):
        return EpsilonBlue(React_restore_minimal(), float(name[len('react_restore_eps'):]))
    raise ValueError(f'Unknown blue policy: {name}')


def blue_epsilon(name):
    if name == 'random':
        return 1.0
    if name.startswith('react_restore_eps'):
        return float(name[len('react_restore_eps'):])
    return 0.0

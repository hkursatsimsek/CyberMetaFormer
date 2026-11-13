"""Subclass of SimplifiedCAGE with the reward bug fixed (without touching the repository files).

The repository's step() calls _process_reward first with the post-Red state and then with the true state.
Because _process_reward modifies the reward dictionary in place, the state penalty is subtracted twice
(e.g. Meander x Sleep: README -1067, repository step() ≈ -2120). Only the true state is used here.
"""
from CybORG_plus_plus.mini_CAGE import SimplifiedCAGE
from CybORG_plus_plus.mini_CAGE import minimal as _minimal


class FixedCAGE(SimplifiedCAGE):

    def step(self, red_action, blue_action, red_agent=None):
        err_msg = 'Ensure batch size is correct.'
        assert red_action.shape[0] == self.num_envs, err_msg
        assert blue_action.shape[0] == self.num_envs, err_msg

        true_state, _, reward = self._process_actions(
            self.state, red_action, blue_action, self.subnets, red_agent)
        self.state = true_state.copy()
        reward = self._process_reward(true_state, reward, self.impacted)
        done = (true_state[:, :1] * 0)

        next_state = self._process_state(
            state=true_state, logged_decoys=self.current_decoys,
            red_action=red_action, blue_action=blue_action)
        self.proc_states = next_state
        return next_state, reward, done, self._get_info()


class FixedObsCAGE(FixedCAGE):
    """FixedCAGE + fix of the success-flag flaw in the Blue observation (environment dynamics unchanged).

    update_blue returns success = -1 for a failed action. _process_state counts this value as true inside np.logical_and
    (minimal.py:1311 remove, :1321 restore). Consequences:
      - a failed remove writes [1,0] "removed" into the observation (remove always fails on a host with root);
      - the invalid user0 restore (action 48, no-op) clears user0's safety flag and activity bits to [0,0]
        (minimal.py:1322-1329); the scan history is fed from the activity bit, so it is affected as well.
    Here only successful actions (blue_success > 0) are passed to _process_state. True state, Red observation, reward,
    RNG consumption and the recorded blue_success are unchanged.
    """

    def _process_state(self, state, logged_decoys, red_action=None, blue_action=None):
        recorded = self.blue_success
        self.blue_success = (recorded > 0).astype(recorded.dtype)
        try:
            return super()._process_state(
                state=state, logged_decoys=logged_decoys, red_action=red_action, blue_action=blue_action)
        finally:
            self.blue_success = recorded


# user → ent links (minimal.CONNECTED_HOSTS, host order minimal.HOSTS). MiniCAGE: user1/2 → ent0, user3/4 → ent1.
# The rewired topology follows CAGE 2: user1/2 → ent1, user3/4 → ent0. ent2 → opserv is unchanged.
REWIRED_HOSTS = [None, None, None, 'opserv', None, None, None, None, None, 'ent1', 'ent1', 'ent0', 'ent0']


class RewiredCAGE(FixedCAGE):
    """FixedCAGE on a topology with modified user → ent links (subnets, hosts and observation layout unchanged).

    The link table is read only in the privilege-escalation step of update_red (minimal.py:537): Red, on gaining root
    on a user host, learns the linked ent host. The table is swapped at module level during the step and then restored.
    """

    connected_hosts = REWIRED_HOSTS

    def step(self, red_action, blue_action, red_agent=None):
        saved = _minimal.CONNECTED_HOSTS
        _minimal.CONNECTED_HOSTS = list(self.connected_hosts)
        try:
            return super().step(red_action, blue_action, red_agent=red_agent)
        finally:
            _minimal.CONNECTED_HOSTS = saved

"""SimplifiedCAGE'in ödül hatası düzeltilmiş alt sınıfı (depo dosyalarına dokunmadan).

Depodaki step(), _process_reward'ı önce Red sonrası durumla, sonra gerçek durumla çağırıyor.
_process_reward ödül sözlüğünü yerinde değiştirdiği için durum cezası iki kez düşülüyor
(ör. Meander x Sleep: README -1067, depo step() ≈ -2120). Burada yalnızca gerçek durum kullanılır.
"""
from CybORG_plus_plus.mini_CAGE import SimplifiedCAGE


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

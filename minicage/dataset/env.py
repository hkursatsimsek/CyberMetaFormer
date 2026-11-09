"""SimplifiedCAGE'in ödül hatası düzeltilmiş alt sınıfı (depo dosyalarına dokunmadan).

Depodaki step(), _process_reward'ı önce Red sonrası durumla, sonra gerçek durumla çağırıyor.
_process_reward ödül sözlüğünü yerinde değiştirdiği için durum cezası iki kez düşülüyor
(ör. Meander x Sleep: README -1067, depo step() ≈ -2120). Burada yalnızca gerçek durum kullanılır.
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
    """FixedCAGE + Blue gözlemindeki başarı bayrağı kusurunun düzeltmesi (ortam dinamiği aynı kalır).

    update_blue başarısız aksiyonda success = -1 döndürür. _process_state bu değeri np.logical_and içinde doğru sayar
    (minimal.py:1311 remove, :1321 restore). Sonuçları:
      - başarısız remove gözlemde [1,0] "kaldırıldı" basar (root'lu hostta remove her zaman başarısızdır);
      - geçersiz user0 restore'u (aksiyon 48, no-op) user0'ın güvenlik bayrağını ve aktivite bitlerini [0,0]'a siler
        (minimal.py:1322-1329); tarama geçmişi aktivite bitinden beslendiği için o da etkilenir.
    Burada _process_state'e yalnız başarılı aksiyonlar (blue_success > 0) geçirilir. Gerçek durum, Red gözlemi, ödül,
    RNG tüketimi ve kaydedilen blue_success değişmez.
    """

    def _process_state(self, state, logged_decoys, red_action=None, blue_action=None):
        recorded = self.blue_success
        self.blue_success = (recorded > 0).astype(recorded.dtype)
        try:
            return super()._process_state(
                state=state, logged_decoys=logged_decoys, red_action=red_action, blue_action=blue_action)
        finally:
            self.blue_success = recorded


# user → ent bağlantıları (minimal.CONNECTED_HOSTS, host sırası minimal.HOSTS). MiniCAGE: user1/2 → ent0, user3/4 → ent1.
# Yeniden bağlanmış topoloji CAGE 2 yönündedir: user1/2 → ent1, user3/4 → ent0. ent2 → opserv aynı kalır.
REWIRED_HOSTS = [None, None, None, 'opserv', None, None, None, None, None, 'ent1', 'ent1', 'ent0', 'ent0']


class RewiredCAGE(FixedCAGE):
    """FixedCAGE, user → ent bağlantıları değiştirilmiş topolojide (alt ağlar, hostlar ve gözlem düzeni aynı).

    Bağlantı tablosu yalnız update_red'deki yetki yükseltme adımında okunur (minimal.py:537): user hostunda root alan Red,
    bağlı ent hostunu öğrenir. Tablo adım sırasında modül düzeyinde değiştirilir ve adım bitince geri konur.
    """

    connected_hosts = REWIRED_HOSTS

    def step(self, red_action, blue_action, red_agent=None):
        saved = _minimal.CONNECTED_HOSTS
        _minimal.CONNECTED_HOSTS = list(self.connected_hosts)
        try:
            return super().step(red_action, blue_action, red_agent=red_agent)
        finally:
            _minimal.CONNECTED_HOSTS = saved

"""Batch'li davranış politikaları (MiniCAGE). Her shard için yeni örnek oluşturulur."""
import numpy as np

from CybORG_plus_plus.mini_CAGE import (
    Meander_minimal, React_restore_minimal, Blue_sleep, Restore_decoys)
from CybORG_plus_plus.mini_CAGE.minimal import HOSTS
from CybORG_plus_plus.mini_CAGE.red_bline_agent import B_line_minimal

NUM_BLUE_ACTIONS = 53
NUM_HOSTS = 13
USER0 = 8                  # Red'in dayanağı; restore'u geçersiz (minimal.py check_blue_action, aksiyon 48)
ANALYSE, REMOVE, RESTORE = 1, 1 + 2 * NUM_HOSTS, 1 + 3 * NUM_HOSTS     # aksiyon = tip ofseti + host

RED_POLICIES = ['bline', 'meander']
# Sıra veri setlerine indeks olarak yazılır (blue_policy); yeni politikalar yalnızca sona eklenir.
BLUE_POLICIES = ['sleep', 'react_restore', 'restore_decoys', 'random',
                 'react_restore_eps0.1', 'react_restore_eps0.3',
                 # 2026-10-22, yalnızca test: şiddet eğrisi ve mekanizma kayması
                 'react_restore_eps0.5', 'react_restore_eps0.7', 'react_restore_analyse', 'react_remove']
V1_BLUE_POLICIES = BLUE_POLICIES[:6]   # minicage_v1 ve minicage_v1_bugged'in politikaları; generate.py varsayılanı


class LinkedBline(B_line_minimal):
    """B_line_minimal; user hostundan sonra hedeflenen ent, koda gömülü eşleme yerine bağlantı tablosundan seçilir.

    B_line_minimal 4. durumda user1/2 → ent0, user3/4 → ent1 eşlemesini sabit kullanır (MiniCAGE'in bağlantı tablosu).
    Özgün CybORG B-line'ı sonraki hostu gözlemden bulur. Burada ent, root alınan user hostunun tablodaki bağlantısıdır.
    4. duruma yalnız user hostundan gelindiği için özgün tabloyla davranış B_line_minimal ile aynıdır.
    """

    def __init__(self, links):
        super().__init__()
        self.links = list(links)

    def DiscoverNetworkServices_ent(self, host, obs, last_host):
        if host in (1, 2) and last_host in (9, 10, 11, 12):
            host = self.enterprise_host = HOSTS.index(self.links[last_host])
        return super().DiscoverNetworkServices_ent(host, obs, last_host)


class BatchedBline:
    """red_bline_agent.B_line_minimal tek ortamlık; ortam başına bir örnek tutar.

    test_agent.B_line_minimal kullanılmaz: escalate_user1'den sonra network_ent1'de takılı kalıyor
    (B-line x Sleep ≈ -10, README -1141). links verilirse (değiştirilmiş topoloji) LinkedBline kullanılır.
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
    """links: değiştirilmiş topolojinin bağlantı tablosu (env.RewiredCAGE); yalnız B-line kullanır."""
    if name == 'bline':
        return BatchedBline(links)
    if name == 'meander':
        return Meander_minimal()
    raise ValueError(f'Bilinmeyen red politikası: {name}')


class RandomBlue:
    def get_action(self, observation):
        return np.random.randint(0, NUM_BLUE_ACTIONS, size=(observation.shape[0], 1))


class EpsilonBlue:
    """ε olasılıkla rastgele aksiyon; temel ajan her adımda gözlemi işlemeye devam eder."""

    def __init__(self, base, epsilon):
        self.base, self.epsilon = base, epsilon

    def get_action(self, observation):
        action = self.base.get_action(observation)
        explore = np.random.rand(observation.shape[0]) < self.epsilon
        action[explore, 0] = np.random.randint(0, NUM_BLUE_ACTIONS, size=explore.sum())
        return action


class ReactRestoreAnalyse(React_restore_minimal):
    """React_restore_minimal; restore gerekmeyen adımlarda (aksiyon 0) user0 dışındaki 12 host üzerinde döngüsel analyse.

    analyse ortamı değiştirmez ama root'lu hostta güvenlik bayrağını [1,1] yeniden basar (minimal.py:1288-1304); bayrak bir
    sonraki adımda React_restore'un tetiğine girer. Eğitim politikalarında analyse yalnızca rastgele aksiyonlarla görülür.
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
    """React_restore_minimal'ın tetiği; restore (40 + host) yerine remove (27 + host).

    Root'lu hostta remove her zaman başarısız olur ama gözlem [1,0] "kaldırıldı" gösterir. user0'ın
    restore'u zaten geçersiz (no-op) olduğu için olduğu gibi kalır: user0'da remove geçerli ve Red'in dayanağını keserdi.
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
    raise ValueError(f'Bilinmeyen blue politikası: {name}')


def blue_epsilon(name):
    if name == 'random':
        return 1.0
    if name.startswith('react_restore_eps'):
        return float(name[len('react_restore_eps'):])
    return 0.0

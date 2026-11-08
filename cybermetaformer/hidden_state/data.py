"""Gizli durum kestirimi için tembel (lazy) pencere veri katmanı.

Bölümler cihazda (n, T+1, 78) int8 olarak tutulur. (bölüm, adım) indekslerinden gather ile
(B, k, 78) pencere ve dolgu maskesi üretilir; k=32'de bile tüm pencereler bellekte açılmaz.

Hizalama `loaders.detection_view` ile birebir aynıdır: t ∈ {1..T} hedef adımı için pencere
obs[t-k+1 .. t]'dir. t-k+1 < 0 olan konumlar sıfırla doldurulur ve maskede False olur. Etiket
`host_labels(true_state[t])`'dir (user0 hariç 12 host). Düz satır sırası bölüm-ana:
f = e * T + (t - 1), yani detection_view'ün satır sırası.

    from cybermetaformer.hidden_state import data as D
    store = D.EpisodeStore()
    train = D.WindowData(store, store.split('train'), device='mps')
    for ep, t in D.batch_indices(train.n, train.T, 1024):
        x, mask = train.windows(ep, t, k=8)        # (B, 8, 78) int8, (B, 8) bool
        y = train.targets(ep, t)                   # (B, 12) int64
"""
import contextlib
import hashlib
import os

import numpy as np
import torch

from minicage.dataset import loaders as L

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # cybermetaformer/ (ders klasörü; sonuçlar burada)
REPO = os.path.dirname(ROOT)                                          # <repo>
DATA_ROOT = os.path.join(REPO, 'minicage', 'data')                       # dersler arası ortak veri
DEFAULT_DATA = os.path.join(DATA_ROOT, 'minicage_v1')
CAUSE_FILE = os.path.join(DATA_ROOT, 'minicage_v1_aux', 'silent_cause.npz')   # sessiz ihlal neden etiketleri (silent_cause.py)

OBS_DIM = 78
NUM_HOSTS = L.NUM_HOSTS                     # 13 (gözlemdeki host sırası = meta['hosts'])
LABEL_HOSTS = L.LABEL_HOSTS                 # user0 hariç 12 host
N_CLASSES = 3                               # 0 temiz, 1 user, 2 root
HOST_FEATS = 6                              # host token: 4 bit + tarama geçmişi + decoy sayısı
BITS = slice(0, 4 * NUM_HOSTS)
SCAN = slice(4 * NUM_HOSTS, 5 * NUM_HOSTS)
DECOY = slice(5 * NUM_HOSTS, 6 * NUM_HOSTS)
N_BLUE_ACTIONS = 53
NO_ACTION = N_BLUE_ACTIONS                  # obs[0]'dan önce aksiyon yok (aksiyon girdisi ablasyonu)


class OODAccessError(RuntimeError):
    """ood_test bölümlerine nihai değerlendirme dışında erişildi."""


class EpisodeStore:
    """Veri setinin tamamı (numpy, CPU) ve split koruması.

    `split('ood_test')` yalnızca `with store.final_evaluation():` bloğu içinde çalışır. Model seçimi,
    HPO ve eğitim OOD bölümlerini hiçbir aşamada göremez.
    """

    def __init__(self, data_dir=DEFAULT_DATA):
        raw, meta = L.load_raw(data_dir, fields=['obs_blue', 'true_state', 'red_policy', 'blue_policy', 'act_blue'])
        self.data_dir = data_dir
        self.meta = meta
        self.obs = raw['obs_blue']                                                # (E, T+1, 78) int8
        self.labels = L.host_labels(raw['true_state'][:, 1:])[..., LABEL_HOSTS]   # (E, T, 12) int8
        self.red_policy = raw['red_policy'].astype(np.int64)
        self.blue_policy = raw['blue_policy'].astype(np.int64)
        E = len(raw['act_blue'])                                                  # obs ile hizalı önceki aksiyon
        self.prev_action = np.concatenate([np.full((E, 1), NO_ACTION), raw['act_blue']], 1).astype(np.int8)   # (E, T+1)
        self.E, self.T = self.labels.shape[:2]
        self.cause, self.cause_names = None, None                                # sessiz ihlal nedeni (E, T, 12)
        if data_dir == DEFAULT_DATA and os.path.exists(CAUSE_FILE):
            with np.load(CAUSE_FILE) as z:
                self.cause, self.cause_names = z['cause'], [str(c) for c in z['causes']]
        self._splits = L.load_splits(data_dir)
        self._ood = set(self._splits['ood_test'].tolist())
        self._ood_unlocked = False

    def split(self, name):
        if name == 'ood_test' and not self._ood_unlocked:
            raise OODAccessError("ood_test yalnızca nihai değerlendirmede kullanılır: "
                                 "`with store.final_evaluation(): ...`")
        return self._splits[name].copy()

    @contextlib.contextmanager
    def final_evaluation(self):
        self._ood_unlocked = True
        try:
            yield self
        finally:
            self._ood_unlocked = False

    def assert_no_ood(self, idx):
        """Eğitim/HPO bölümleri OOD ile kesişmemeli."""
        overlap = self._ood.intersection(np.asarray(idx).tolist())
        if overlap:
            raise OODAccessError(f'{len(overlap)} OOD bölümü eğitim/seçim kümesine sızmış')


def episode_hash(idx):
    """Bölüm kümesinin kısa özeti; koşu kayıtlarına yazılır."""
    return hashlib.sha256(np.sort(np.asarray(idx, np.int64)).tobytes()).hexdigest()[:16]


def subsample_episodes(idx, max_n, seed=0):
    """baselines.subsample ile aynı: sıralı alt küme (orijinal protokol 4000 bölüm, seed 0)."""
    idx = np.asarray(idx)
    if max_n is None or len(idx) <= max_n:
        return idx
    return idx[np.sort(np.random.default_rng(seed).choice(len(idx), max_n, replace=False))]


class WindowData:
    """Bir bölüm alt kümesinin cihazdaki kopyası: obs (n, T+1, 78) int8, etiket (n, T, 12) int8."""

    def __init__(self, store, idx, device='cpu'):
        self.idx = np.asarray(idx, np.int64)
        self.device = torch.device(device)
        self.obs = torch.as_tensor(store.obs[self.idx], device=self.device)
        self.labels = torch.as_tensor(store.labels[self.idx], device=self.device)
        self.prev_action = torch.as_tensor(store.prev_action[self.idx], device=self.device)
        self.n, self.T = self.labels.shape[:2]

    def windows(self, ep, t, k):
        """ep: yerel bölüm indeksi, t ∈ 1..T (B,) → x (B, k, 78) int8, mask (B, k) bool."""
        ep, t = ep.to(self.device), t.to(self.device)
        pos = t[:, None] + torch.arange(-k + 1, 1, device=self.device)[None]
        mask = pos >= 0
        x = self.obs[ep[:, None], pos.clamp(min=0)]
        return torch.where(mask[..., None], x, torch.zeros((), dtype=x.dtype, device=x.device)), mask

    def targets(self, ep, t):
        return self.labels[ep.to(self.device), t.to(self.device) - 1].long()

    def episodes(self, ep):
        """Bölüm modu (tam geçmiş, çoktan-çoğa): obs[0..T] (B, T+1, 78) ve etiketler (B, T, 12).
        Model t. konumdaki çıktısıyla y[t]'yi kestirir (t = 1..T); obs[0] başlangıç gözlemidir."""
        ep = ep.to(self.device)
        return self.obs[ep], self.labels[ep].long()

    def prev_actions(self, ep):
        """Bölüm modu: obs[t]'yi üreten Blue aksiyonu (t = 0'da NO_ACTION), (B, T+1) int64."""
        return self.prev_action[ep.to(self.device)].long()


def batch_indices(n_ep, T, batch_size, shuffle=True, steps_per_episode=None, generator=None):
    """(ep, t) mini-batch'leri üretir (CPU tensörleri).

    steps_per_episode=None: tüm n_ep*T çift; karıştırma `torch.randperm(n_ep*T)` ile yapılır.
    Bu, baselines.fit'in permütasyonuyla aynıdır (aynı torch seed'i → aynı batch'ler).
    steps_per_episode=s: her epoch'ta her bölümden yerine koymadan s adım (epoch alt örneklemesi).
    """
    if steps_per_episode is None or steps_per_episode >= T:
        N = n_ep * T
        order = torch.randperm(N, generator=generator) if shuffle else torch.arange(N)
    else:
        pick = torch.rand(n_ep, T, generator=generator).argsort(dim=1)[:, :steps_per_episode]
        order = (torch.arange(n_ep)[:, None] * T + pick).reshape(-1)
        if shuffle:
            order = order[torch.randperm(len(order), generator=generator)]
    for i in range(0, len(order), batch_size):
        f = order[i:i + batch_size]
        yield f // T, f % T + 1


def eval_indices(n_ep, T, batch_size):
    """Değerlendirme için sıralı (ep, t) batch'leri (bölüm-ana)."""
    return batch_indices(n_ep, T, batch_size, shuffle=False)


# ── girdi dönüşümleri ───────────────────────────────────────────────────

def scale_obs(x, mode='global'):
    """int8 gözlem → float32. 'none': ham değerler (orijinal MLP protokolü).
    'global': bitler olduğu gibi, tarama geçmişi /2, decoy sayısı /4 (örnek normalizasyonu yok, T1)."""
    x = x.float()
    if mode == 'none':
        return x
    if mode != 'global':
        raise ValueError(f'Bilinmeyen ölçekleme: {mode}')
    scale = torch.ones(OBS_DIM, device=x.device)
    scale[SCAN] = 0.5
    scale[DECOY] = 0.25
    return x * scale


def to_host_tokens(x):
    """(..., 78) → (..., 13, 6): host başına [aktivite tarama, aktivite istismar, güvenlik×2, tarama geçmişi, decoy]."""
    bits = x[..., BITS].reshape(*x.shape[:-1], NUM_HOSTS, 4)
    return torch.cat([bits, x[..., SCAN, None], x[..., DECOY, None]], dim=-1)

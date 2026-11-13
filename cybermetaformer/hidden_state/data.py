"""Lazy window data layer for hidden-state estimation.

Episodes are kept on the device as (n, T+1, 78) int8. Windows (B, k, 78) and padding masks are produced from
(episode, step) indices with gather; even at k=32 the windows are never all materialized in memory.

The alignment is exactly that of `loaders.detection_view`: for target step t ∈ {1..T} the window is
obs[t-k+1 .. t]. Positions with t-k+1 < 0 are zero-padded and False in the mask. The label is
`host_labels(true_state[t])` (12 hosts, excluding user0). The flat row order is episode-major:
f = e * T + (t - 1), i.e. the row order of detection_view.

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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # cybermetaformer/ (course folder; results live here)
REPO = os.path.dirname(ROOT)                                          # <repo>
DATA_ROOT = os.path.join(REPO, 'minicage', 'data')                       # data shared across courses
DEFAULT_DATA = os.path.join(DATA_ROOT, 'minicage_v1')
CAUSE_FILE = os.path.join(DATA_ROOT, 'minicage_v1_aux', 'silent_cause.npz')   # silent-compromise causes (silent_cause.py)

OBS_DIM = 78
NUM_HOSTS = L.NUM_HOSTS                     # 13 (host order in the observation = meta['hosts'])
LABEL_HOSTS = L.LABEL_HOSTS                 # 12 hosts excluding user0
N_CLASSES = 3                               # 0 temiz, 1 user, 2 root
HOST_FEATS = 6                              # host token: 4 bits + scan history + decoy count
BITS = slice(0, 4 * NUM_HOSTS)
SCAN = slice(4 * NUM_HOSTS, 5 * NUM_HOSTS)
DECOY = slice(5 * NUM_HOSTS, 6 * NUM_HOSTS)
N_BLUE_ACTIONS = 53
NO_ACTION = N_BLUE_ACTIONS                  # no action before obs[0] (action-input ablation)


class OODAccessError(RuntimeError):
    """ood_test episodes were accessed outside the final evaluation."""


class EpisodeStore:
    """The whole dataset (numpy, CPU) and split protection.

    `split('ood_test')` works only inside a `with store.final_evaluation():` block. Model selection,
    HPO and training never see the OOD episodes at any stage.
    """

    def __init__(self, data_dir=DEFAULT_DATA):
        raw, meta = L.load_raw(data_dir, fields=['obs_blue', 'true_state', 'red_policy', 'blue_policy', 'act_blue'])
        self.data_dir = data_dir
        self.meta = meta
        self.obs = raw['obs_blue']                                                # (E, T+1, 78) int8
        self.labels = L.host_labels(raw['true_state'][:, 1:])[..., LABEL_HOSTS]   # (E, T, 12) int8
        self.red_policy = raw['red_policy'].astype(np.int64)
        self.blue_policy = raw['blue_policy'].astype(np.int64)
        E = len(raw['act_blue'])                                                  # previous action aligned with obs
        self.prev_action = np.concatenate([np.full((E, 1), NO_ACTION), raw['act_blue']], 1).astype(np.int8)   # (E, T+1)
        self.E, self.T = self.labels.shape[:2]
        self.cause, self.cause_names = None, None                                # silent-compromise cause (E, T, 12)
        if data_dir == DEFAULT_DATA and os.path.exists(CAUSE_FILE):
            with np.load(CAUSE_FILE) as z:
                self.cause, self.cause_names = z['cause'], [str(c) for c in z['causes']]
        self._splits = L.load_splits(data_dir)
        self._ood = set(self._splits['ood_test'].tolist())
        self._ood_unlocked = False

    def split(self, name):
        if name == 'ood_test' and not self._ood_unlocked:
            raise OODAccessError("ood_test is used only in the final evaluation: "
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
        """Training/HPO episodes must not intersect the OOD set."""
        overlap = self._ood.intersection(np.asarray(idx).tolist())
        if overlap:
            raise OODAccessError(f'{len(overlap)} OOD episodes leaked into the training/selection set')


def episode_hash(idx):
    """Short summary of an episode set; written to the run records."""
    return hashlib.sha256(np.sort(np.asarray(idx, np.int64)).tobytes()).hexdigest()[:16]


def subsample_episodes(idx, max_n, seed=0):
    """Same as baselines.subsample: ordered subset (original protocol 4000 episodes, seed 0)."""
    idx = np.asarray(idx)
    if max_n is None or len(idx) <= max_n:
        return idx
    return idx[np.sort(np.random.default_rng(seed).choice(len(idx), max_n, replace=False))]


class WindowData:
    """On-device copy of an episode subset: obs (n, T+1, 78) int8, labels (n, T, 12) int8."""

    def __init__(self, store, idx, device='cpu'):
        self.idx = np.asarray(idx, np.int64)
        self.device = torch.device(device)
        self.obs = torch.as_tensor(store.obs[self.idx], device=self.device)
        self.labels = torch.as_tensor(store.labels[self.idx], device=self.device)
        self.prev_action = torch.as_tensor(store.prev_action[self.idx], device=self.device)
        self.n, self.T = self.labels.shape[:2]

    def windows(self, ep, t, k):
        """ep: local episode index, t ∈ 1..T (B,) → x (B, k, 78) int8, mask (B, k) bool."""
        ep, t = ep.to(self.device), t.to(self.device)
        pos = t[:, None] + torch.arange(-k + 1, 1, device=self.device)[None]
        mask = pos >= 0
        x = self.obs[ep[:, None], pos.clamp(min=0)]
        return torch.where(mask[..., None], x, torch.zeros((), dtype=x.dtype, device=x.device)), mask

    def targets(self, ep, t):
        return self.labels[ep.to(self.device), t.to(self.device) - 1].long()

    def episodes(self, ep):
        """Episode mode (full history, many-to-many): obs[0..T] (B, T+1, 78) and labels (B, T, 12).
        The model estimates y[t] with its output at position t (t = 1..T); obs[0] is the initial observation."""
        ep = ep.to(self.device)
        return self.obs[ep], self.labels[ep].long()

    def prev_actions(self, ep):
        """Episode mode: the Blue action that produced obs[t] (NO_ACTION at t = 0), (B, T+1) int64."""
        return self.prev_action[ep.to(self.device)].long()


def batch_indices(n_ep, T, batch_size, shuffle=True, steps_per_episode=None, generator=None):
    """Produces (ep, t) mini-batches (CPU tensors).

    steps_per_episode=None: all n_ep*T pairs; shuffling uses `torch.randperm(n_ep*T)`.
    This is the same permutation as baselines.fit (same torch seed → same batches).
    steps_per_episode=s: s steps per episode without replacement in every epoch (epoch subsampling).
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
    """Ordered (ep, t) batches for evaluation (episode-major)."""
    return batch_indices(n_ep, T, batch_size, shuffle=False)


# ── input transforms ────────────────────────────────────────────────────

def scale_obs(x, mode='global'):
    """int8 observation → float32. 'none': raw values (original MLP protocol).
    'global': bits as they are, scan history /2, decoy count /4 (no per-sample normalization, T1)."""
    x = x.float()
    if mode == 'none':
        return x
    if mode != 'global':
        raise ValueError(f'Unknown scaling: {mode}')
    scale = torch.ones(OBS_DIM, device=x.device)
    scale[SCAN] = 0.5
    scale[DECOY] = 0.25
    return x * scale


def to_host_tokens(x):
    """(..., 78) → (..., 13, 6): per host [activity scan, activity exploit, safety×2, scan history, decoy]."""
    bits = x[..., BITS].reshape(*x.shape[:-1], NUM_HOSTS, 4)
    return torch.cat([bits, x[..., SCAN, None], x[..., DECOY, None]], dim=-1)

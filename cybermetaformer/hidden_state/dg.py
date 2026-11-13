"""Training-time OOD methods: domain generalization and mechanism augmentation. Episode mode.

Domain = the (Red, Blue) policy pair of the training episode: 2 Red × 5 Blue = 10 domains (the ood_test policy is not in training).
Model selection with the val of the training domains (train.py; OOD enters no selection). DomainBed adaptations:

- `groupdro` (Sagawa et al. 2020): domain weights q, q_g ← q_g · exp(η L_g); loss Σ q_g L_g (domains in the batch).
- `vrex` (Krueger et al. 2021): mean(L_g) + λ var(L_g); λ = 1 in the first `anneal_steps` steps.
- `coral` (Sun & Saenko 2016): difference in mean and covariance of the last-token features across domains, weighted by λ.
- `dann` (Ganin et al. 2016): an MLP distinguishing the domain from the host-mean feature per (episode, step), gradient
  reversal; coefficient 2/(1 + e^(−10p)) − 1 (p training progress) × λ.

Augmentation (training only; labels unchanged):
- `alert_drop: p`: for each (episode, host) a whole alert run (consecutive alerted steps; alert = exploit activity
  or safety g1, features 'export') is deleted with probability p: the activity exploit bit and both safety bits set to 0.
  The real miss (5%) also never shows a compromise; deleting step by step would create flicker absent from the simulator.
- `time_mask: p`: the observation of each step (t ≥ 1) is zeroed completely with probability p (missing observation).
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf

from cybermetaformer.hidden_state.data import BITS, N_CLASSES, NUM_HOSTS

METHODS = ('groupdro', 'vrex', 'coral', 'dann')


def domain_ids(store, idx):
    """Domain id per episode (0..G−1) and domain names; ordered by the (red, blue) ids."""
    combo = store.red_policy[idx] * 100 + store.blue_policy[idx]
    uniq, dom = np.unique(combo, return_inverse=True)
    names = [f"{store.meta['red_policies'][c // 100]}×{store.meta['blue_policies'][c % 100]}" for c in uniq]
    return dom.astype(np.int64), names


# ── augmentation ────────────────────────────────────────────────────────

def alert_runs_drop(x, p, generator=None):
    """Scaled observation (B, S, 78) → copy with alert runs deleted with probability p."""
    B, S, _ = x.shape
    bits = x[..., BITS].reshape(B, S, NUM_HOSTS, 4)
    alert = (bits[..., 1] > 0.5) | (bits[..., 3] > 0.5)
    start = alert & ~torch.cat([torch.zeros_like(alert[:, :1]), alert[:, :-1]], 1)
    run = start.long().cumsum(1)                                                  # (B, S, 13), run number
    table = torch.rand(B, S + 1, NUM_HOSTS, device=x.device, generator=generator) < p
    drop = alert & table.gather(1, run)
    keep = torch.ones_like(bits)
    keep[..., 1:] = (~drop).unsqueeze(-1).float()
    out = x.clone()
    out[..., BITS] = (bits * keep).reshape(B, S, -1)
    return out


def time_mask(x, p, generator=None):
    m = torch.rand(x.shape[:2], device=x.device, generator=generator) < p
    m[:, 0] = False
    return x.masked_fill(m[..., None], 0.0)


def augment(x, spec):
    if not spec:
        return x
    unknown = set(spec) - {'alert_drop', 'time_mask'}
    if unknown:
        raise ValueError(f'augment: bilinmeyen {unknown}')
    if spec.get('alert_drop'):
        x = alert_runs_drop(x, spec['alert_drop'])
    if spec.get('time_mask'):
        x = time_mask(x, spec['time_mask'])
    return x


# ── alan genelleme ──────────────────────────────────────────────────────

class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coef):
        ctx.coef = coef
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.coef * g, None


def episode_losses(out, y, weight):
    """out (B, T, 12, 3), y (B, T, 12) → per-episode (weighted) CE sum and weight sum, (B,) each."""
    ce = nnf.cross_entropy(out.reshape(-1, N_CLASSES), y.reshape(-1), weight=weight, reduction='none')
    w = torch.ones_like(ce) if weight is None else weight[y.reshape(-1)]
    B = y.shape[0]
    return ce.view(B, -1).sum(1), w.view(B, -1).sum(1)


def group_losses(num, den, dom):
    """Mean loss of the domains present in the batch: (domains, losses)."""
    present = torch.unique(dom)
    L = torch.stack([num[dom == g].sum() / den[dom == g].sum() for g in present])
    return present, L


class DomainGeneralization(nn.Module):
    """Replaces the episode-mode loss of train.py: loss(out, feats, y, dom, weight, progress) → scalar."""

    def __init__(self, spec, n_domains, d):
        super().__init__()
        self.method = spec['method']
        if self.method not in METHODS:
            raise ValueError(f'dg: {self.method}')
        self.spec, self.G = spec, n_domains
        if self.method == 'groupdro':
            self.register_buffer('q', torch.full((n_domains,), 1.0 / n_domains))
        if self.method == 'dann':
            self.disc = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU(), nn.Linear(d, n_domains))
        self.step = 0

    @property
    def needs_features(self):
        return self.method in ('coral', 'dann')

    def forward(self, out, feats, y, dom, weight, progress):
        num, den = episode_losses(out, y, weight)
        erm = num.sum() / den.sum()
        self.step += 1
        if self.method == 'groupdro':
            present, L = group_losses(num, den, dom)
            with torch.no_grad():
                self.q[present] *= torch.exp(self.spec.get('eta', 0.01) * L)
                self.q /= self.q.sum()
            q = self.q[present]
            return (q * L).sum() / q.sum()
        if self.method == 'vrex':
            present, L = group_losses(num, den, dom)
            lam = self.spec.get('lam', 10.0) if self.step > self.spec.get('anneal_steps', 500) else 1.0
            return L.mean() + (lam * L.var(unbiased=False) if len(L) > 1 else 0.0)
        f = feats[:, 1:]                                                       # target steps (t = 1..T)
        if self.method == 'coral':
            present = torch.unique(dom)
            stats = []
            for g in present:
                z = f[dom == g].reshape(-1, f.shape[-1])
                mu = z.mean(0)
                zc = z - mu
                stats.append((mu, zc.T @ zc / max(len(z) - 1, 1)))
            pen, n = 0.0, 0
            for i in range(len(stats)):
                for j in range(i + 1, len(stats)):
                    pen = pen + (stats[i][0] - stats[j][0]).pow(2).mean() + (stats[i][1] - stats[j][1]).pow(2).mean()
                    n += 1
            return erm + self.spec.get('lam', 1.0) * (pen / max(n, 1))
        # dann
        coef = 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0
        z = _GradReverse.apply(f.mean(2), coef)                                 # (B, T, d)
        logits = self.disc(z)
        target = dom[:, None].expand(-1, z.shape[1])
        return erm + self.spec.get('lam', 1.0) * nnf.cross_entropy(logits.reshape(-1, self.G), target.reshape(-1))

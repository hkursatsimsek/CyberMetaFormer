"""Eğitim zamanı OOD yöntemleri: alan genelleme ve mekanizma artırması. Bölüm kipi.

Alan = eğitim bölümünün (Red, Blue) politika çifti: 2 Red × 5 Blue = 10 alan (ood_test'in politikası eğitimde yok).
Model seçimi eğitim alanlarının val'i ile (train.py; OOD hiçbir seçime girmez). DomainBed uyarlamaları:

- `groupdro` (Sagawa vd. 2020): alan ağırlıkları q, q_g ← q_g · exp(η L_g); kayıp Σ q_g L_g (batch'te olan alanlar).
- `vrex` (Krueger vd. 2021): ort(L_g) + λ var(L_g); ilk `anneal_steps` adımda λ = 1.
- `coral` (Sun & Saenko 2016): son token özniteliklerinin alanlar arası ortalama ve kovaryans farkı, λ ağırlıkla.
- `dann` (Ganin vd. 2016): (bölüm, adım) başına host ortalaması özniteliğinden alanı ayırt eden MLP, gradyan ters
  çevirme; katsayı 2/(1 + e^(−10p)) − 1 (p eğitim ilerlemesi) × λ.

Artırma (yalnızca eğitimde; etiketler değişmez):
- `alert_drop: p`: her (bölüm, host) için bütün bir uyarı koşusu (ardışık uyarılı adımlar; uyarı = istismar aktivitesi
  ya da güvenlik g1, features 'export') p olasılıkla silinir: aktivite istismar biti ve iki güvenlik biti 0. Gerçek
  kaçak (%5) da bir ele geçirmeyi hiç göstermez; adım adım silmek simülatörde olmayan titreme üretirdi.
- `time_mask: p`: her adımın gözlemi (t ≥ 1) p olasılıkla tamamen sıfırlanır (eksik gözlem).
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf

from cybermetaformer.hidden_state.data import BITS, N_CLASSES, NUM_HOSTS

METHODS = ('groupdro', 'vrex', 'coral', 'dann')


def domain_ids(store, idx):
    """Bölüm başına alan kimliği (0..G−1) ve alan adları; sıra (red, blue) kimliğine göre."""
    combo = store.red_policy[idx] * 100 + store.blue_policy[idx]
    uniq, dom = np.unique(combo, return_inverse=True)
    names = [f"{store.meta['red_policies'][c // 100]}×{store.meta['blue_policies'][c % 100]}" for c in uniq]
    return dom.astype(np.int64), names


# ── artırma ─────────────────────────────────────────────────────────────

def alert_runs_drop(x, p, generator=None):
    """Ölçekli gözlem (B, S, 78) → uyarı koşuları p olasılıkla silinmiş kopya."""
    B, S, _ = x.shape
    bits = x[..., BITS].reshape(B, S, NUM_HOSTS, 4)
    alert = (bits[..., 1] > 0.5) | (bits[..., 3] > 0.5)
    start = alert & ~torch.cat([torch.zeros_like(alert[:, :1]), alert[:, :-1]], 1)
    run = start.long().cumsum(1)                                                  # (B, S, 13), koşu numarası
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
    """out (B, T, 12, 3), y (B, T, 12) → bölüm başına (ağırlıklı) CE toplamı ve ağırlık toplamı, (B,) ikişer."""
    ce = nnf.cross_entropy(out.reshape(-1, N_CLASSES), y.reshape(-1), weight=weight, reduction='none')
    w = torch.ones_like(ce) if weight is None else weight[y.reshape(-1)]
    B = y.shape[0]
    return ce.view(B, -1).sum(1), w.view(B, -1).sum(1)


def group_losses(num, den, dom):
    """Batch'te bulunan alanların ortalama kaybı: (alanlar, kayıplar)."""
    present = torch.unique(dom)
    L = torch.stack([num[dom == g].sum() / den[dom == g].sum() for g in present])
    return present, L


class DomainGeneralization(nn.Module):
    """train.py'nin bölüm kipi kaybı yerine: loss(out, feats, y, dom, weight, progress) → skaler."""

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
        f = feats[:, 1:]                                                       # hedef adımlar (t = 1..T)
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

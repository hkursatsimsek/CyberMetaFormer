"""M2 analizleri: dikkat rollout'u, iTransformer'ın host dikkati, Integrated Gradients.

Hepsi test bölümlerinin rastgele bir alt kümesinde (`N_EP` bölüm, `subsample_episodes` seed 0; test bölümleri politikaya
göre sıralı, ilk bölümler tek kırmızı politikadır), kaydedilmiş ağırlıklarla (`model.pt`), eager.
Sessiz ihlal: host ele geçirilmiş, o adımda uyarı yok (`features.alerts`, 'export'); uyarı yaşı a = son uyarıdan bu
yana geçen adım. Pencerenin son konumu şimdiki gözlem; a adım önceki uyarı k-1-a konumundadır. "Yakalandı" =
kestirim > 0 (sessiz recall ile aynı).

1. **Rollout (vanilla Transformer, k=32 ve 64; Abnar & Zuidema 2020).** Katmanlar elle yürütülür (çıktının modelin
   kendi ileri geçişiyle aynı olduğu ilk batch'te denetlenir); başlık ortalamalı dikkat 0.5·A + 0.5·I ile birleştirilir.
   Okuma son adım olduğu için son satır girdiye düşen payı verir. Okuma token'ı 12 host için ortaktır: rollout host'a
   özgü değildir, "dikkat uyarılı adımlara gidiyor mu"yu ölçer. Ölçü, bir konuma düşen payın o konumun bütün
   pencerelerdeki ortalamasına oranı (konum önseli; model zaten son adımlara daha çok bakıyor). İki oran: herhangi bir
   host'ta uyarı olan adımlar ve sessiz ihlalde ilgili host'un son uyarısının görüldüğü adım. PatchTST'de okuma token'ı
   yok (bütün yamalar düzleştirilip başa giriyor), rollout çıktıya düşen payı vermez; ona yalnız IG ile bakılır.
2. **iTransformer host dikkati (k=32 ve 64).** Token = 78 değişken; host başına 6 token (4 bit, tarama, decoy). Dikkat
   (katman ve başlık ortalaması) host bloklarına toplanır: satır = sorgu host'u, sütun = anahtar host'u, satır toplamı
   1. Tekdüze dikkatte her hücre 6/78. Kontrol: aynı ayarla eğitilmemiş model. Sorgu bağımlılığı: satırların ortalama
   satırdan toplam varyasyon uzaklığının ortalaması (0 = her host aynı hostlara bakıyor, ilişki değil önem öğrenilmiş).
3. **Integrated Gradients (captum; M2'nin altı modeli, k=32, seed 0).** Sessiz ihlal host-adımları: yarısı uyarı
   pencere içinde (1 ≤ a < k), yarısı bölüm başından beri uyarısız. Çıktı ilgili host'un "ele geçirilmiş" log-olasılık
   oranı log(p_user + p_root) − log(p_temiz), host'a özgü. Taban sıfır gözlem (uyarı, tarama, decoy yok). |IG| zaman
   adımlarına ve özellik gruplarına (host'un kendisi / diğerleri; bit / tarama / decoy) toplanır; uyarı adımı oranı
   uyarısız gruptaki konum profiline göre. IG CPU'da: MPS'te (torch 2.14.1) permute edilmiş girdide AvgPool1d'nin geri
   geçişi yanlış (TimeMixer'ın parametresiz alt örneklemesi; yalnız girdi gradyanını bozar, eğitimi etkilemez).
   TimesNet'in TSLib uygulaması periyotları batch'in ortalama genliğinden seçer, çıktısı batch'e bağlıdır; IG ona örnek
   örnek uygulanır (yol noktaları aynı pencerenin ölçekli kopyaları, periyotlar pencerenin kendi periyotları). Adım
   sayısı 32'den başlar; tamlık hatasının medyanı %5'in üstündeyse 128'e, sonra 256'ya çıkar (iTransformer'da yol
   boyunca keskin geçişler var).
4. **TimesNet'in batch bağımlılığı (k=8 ve 32, 3 seed; bütün test ve OOD).** Üç değerlendirme: resmi sıra (bölüm-ana,
   batch 8192; test/OOD bölümleri politikaya göre sıralı olduğu için bir batch çoğunlukla tek politikadan), karışık
   batch'ler (8192, eğitimdeki gibi politikalar karışık) ve tek tek (bs=1, çevrimiçi kullanım; her 19. pencere, T=100
   ile aralarında asal olduğu için bütün zaman adımlarından). Üç kip ayrıca aynı alt kümede karşılaştırılır.

    python -m cybermetaformer.hidden_state.explain                  # asama2/analiz/ ve figürler
    python -m cybermetaformer.hidden_state.explain --only timesnet  # yalnız 4. bölüm, diğerleri analiz.json'dan
"""
import argparse
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt                     # noqa: E402
import numpy as np                                  # noqa: E402
import torch                                        # noqa: E402
import yaml                                         # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from cybermetaformer.hidden_state import data as D                                   # noqa: E402
from cybermetaformer.hidden_state import features as F                               # noqa: E402
from cybermetaformer.hidden_state.aggregate import (BLUES, INK, INK2, LABEL, M2_MODELS, MARKER, MUTED,  # noqa: E402
                                                  SLOT, SURFACE, _grid)
from cybermetaformer.hidden_state.models import build_model                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE_DIR = os.path.join(os.path.dirname(HERE), 'results', 'hidden_state', 'asama2')
OUT_DIR, FIG_DIR = os.path.join(STAGE_DIR, 'analiz'), os.path.join(STAGE_DIR, 'figures')
N_EP = 300                    # test alt kümesi (30 000 pencere)
N_IG = 1000                   # IG örneği (sessiz ihlal host-adımı)
SEEDS = (0, 1, 2)
TOKEN_HOST = np.r_[np.arange(4 * D.NUM_HOSTS) // 4, np.arange(D.NUM_HOSTS), np.arange(D.NUM_HOSTS)]   # 78 → host
TOKEN_KIND = np.r_[np.zeros(4 * D.NUM_HOSTS), np.ones(D.NUM_HOSTS), np.full(D.NUM_HOSTS, 2)]   # bit, tarama, decoy


def load_model(name, k, seed, device, trained=True):
    run = os.path.join(STAGE_DIR, f'{name}_k{k}_s{seed}')
    cfg = yaml.safe_load(open(os.path.join(run, 'config.yaml')))
    torch.manual_seed(seed)
    model = build_model(name, k, **(cfg.get('model_args') or {}))
    if trained:
        model.load_state_dict(torch.load(os.path.join(run, 'model.pt'), map_location='cpu'))
    return model.to(device).eval()


class Subset:
    """Test alt kümesi: etiketler, uyarılar, uyarı yaşı (hepsi (E, T, 12), hedef adımla hizalı)."""

    def __init__(self, store, device):
        self.idx = D.subsample_episodes(store.split('test'), N_EP, seed=0)
        self.wd = D.WindowData(store, self.idx, device=device)
        self.y = store.labels[self.idx].astype(np.int64)
        self.alert = F.episode_alerts(store.obs[self.idx])
        self.age = F.alert_age(self.alert)
        self.silent = (self.y > 0) & ~self.alert
        self.any_alert = np.c_[np.zeros((len(self.idx), 1), bool), self.alert.any(-1)]   # obs[0..T]

    def batches(self, k, bs=4096):
        """(ep, t, x, mask): bölüm-ana sırayla bütün pencereler; ep yerel, t ∈ 1..T (numpy)."""
        for ep, t in D.eval_indices(self.wd.n, self.wd.T, bs):
            x, mask = self.wd.windows(ep, t, k)
            yield ep.numpy(), t.numpy(), D.scale_obs(x), mask

    def window_alerts(self, ep, t, k):
        """(B, k): pencerenin her konumunda herhangi bir host'ta uyarı var mı (dolgu hariç)."""
        pos = t[:, None] + np.arange(-k + 1, 1)[None]
        return self.any_alert[ep[:, None], pos.clip(0)] & (pos >= 0)


# ── 1. rollout ──────────────────────────────────────────────────────────

@torch.no_grad()
def rollout(model, x, mask, check=False):
    """Vanilla Transformer (okuma 'last') → ((B, k) son adımın girdiye düşen rollout payı, logitler (B, 12, 3))."""
    assert model.readout == 'last' and not model.causal
    h, pad = model.inp(x) + model.pos[:, -x.shape[1]:], ~mask
    L = h.shape[1]
    eye = torch.eye(L, device=x.device)
    R = eye.expand(h.shape[0], L, L)
    for layer in model.enc.layers:                       # norm_first=True, eval: dropout yok
        z = layer.norm1(h)
        a, w = layer.self_attn(z, z, z, key_padding_mask=pad, need_weights=True, average_attn_weights=True)
        h = h + a
        h = h + layer._ff_block(layer.norm2(h))
        W = 0.5 * w + 0.5 * eye
        R = (W / W.sum(-1, keepdim=True)) @ R
    out = model.head(model.norm(h[:, -1])).view(-1, len(D.LABEL_HOSTS), D.N_CLASSES)
    if check:
        torch.testing.assert_close(out, model(x, mask), atol=1e-4, rtol=1e-4)
    return R[:, -1], out


def rollout_stats(model, sub, k):
    """Konum önseli, uyarılı adımların oranı, sessiz ihlallerde uyarı adımının oranı."""
    pos_sum, pos_n, al_sum, al_n = (np.zeros(k) for _ in range(4))
    pairs = []                                             # (a, pay, yakalandı mı)
    for i, (ep, t, x, mask) in enumerate(sub.batches(k)):
        r, out = rollout(model, x, mask, check=i == 0)
        r, pred, valid = r.cpu().numpy(), out.argmax(-1).cpu().numpy(), mask.cpu().numpy()
        al = sub.window_alerts(ep, t, k)
        pos_sum += (r * valid).sum(0)
        pos_n += valid.sum(0)
        al_sum += (r * al).sum(0)
        al_n += al.sum(0)
        s, age = sub.silent[ep, t - 1], sub.age[ep, t - 1]
        b, hh = np.nonzero(s & (age >= 1) & (age < k))
        a = age[b, hh]
        pairs.append(np.c_[a, r[b, k - 1 - a], pred[b, hh] > 0])
    prior = pos_sum / np.maximum(pos_n, 1)
    noal_mean = (pos_sum - al_sum) / np.maximum(pos_n - al_n, 1)
    P = np.concatenate(pairs)
    a, w, hit = P[:, 0].astype(int), P[:, 1], P[:, 2].astype(bool)
    ratio = w / prior[k - 1 - a]
    return {'prior_by_steps_back': prior[::-1].tolist(),
            'alert_any_ratio': float((al_sum / prior).sum() / al_n.sum()),
            'no_alert_ratio': float(((pos_sum - al_sum) / prior).sum() / (pos_n - al_n).sum()),
            'alert_any_ratio_by_steps_back': (al_sum / np.maximum(al_n, 1) / prior)[::-1].tolist(),
            'no_alert_ratio_by_steps_back': (noal_mean / prior)[::-1].tolist(),
            'n_pairs': int(len(P)),
            'alert_weight_by_age': [float(w[a == j].mean()) if (a == j).any() else None for j in range(k)],
            'ratio_mean': float(ratio.mean()), 'ratio_median': float(np.median(ratio)),
            'share_above_prior': float((ratio > 1).mean()),
            'ratio_mean_hit': float(ratio[hit].mean()), 'ratio_mean_missed': float(ratio[~hit].mean()),
            'recall': float(hit.mean())}


# ── 2. iTransformer host dikkati ────────────────────────────────────────

@torch.no_grad()
def itransformer_host_attention(model, sub, k, max_windows=8192):
    """(13, 13): sorgu host'u × anahtar host'u, satır toplamı 1 (katman, başlık ve pencere ortalaması)."""
    net = model.net
    for layer in net.encoder.attn_layers:
        layer.attention.inner_attention.output_attention = True
    A, n = torch.zeros(D.OBS_DIM, D.OBS_DIM, device=sub.wd.device), 0
    for _, _, x, _ in sub.batches(k, bs=1024):
        _, attns = net.encoder(net.enc_embedding(x, None), attn_mask=None)
        A += torch.stack([a.mean(1) for a in attns]).mean(0).sum(0)
        n += x.shape[0]
        if n >= max_windows:
            break
    for layer in net.encoder.attn_layers:
        layer.attention.inner_attention.output_attention = False
    A = (A / n).cpu().numpy()
    return np.array([[A[TOKEN_HOST == q][:, TOKEN_HOST == kk].sum(1).mean() for kk in range(D.NUM_HOSTS)]
                     for q in range(D.NUM_HOSTS)])


def host_summary(H, hosts):
    off = H.copy()
    np.fill_diagonal(off, -1)
    top = np.argsort(off.ravel())[::-1][:5]
    return {'self_mean': float(np.diag(H).mean()), 'uniform': 6 / D.OBS_DIM, 'max_cell': float(H.max()),
            'query_tv': float(0.5 * np.abs(H - H.mean(0)).sum(1).mean()),
            'col_mass': H.mean(0).tolist(),
            'top_cross': [[hosts[i // len(H)], hosts[i % len(H)], float(H.flat[i])] for i in top]}


# ── 3. Integrated Gradients ─────────────────────────────────────────────

def ig_samples(sub, k, rng):
    """N_IG sessiz ihlal host-adımı (ep, t ∈ 1..T, host): yarısı 1 ≤ a < k, yarısı a = hiç."""
    out = []
    for cond in [(sub.age >= 1) & (sub.age < k), sub.age == F.NEVER]:
        e, t0, h = np.nonzero(sub.silent & cond)
        pick = rng.choice(len(e), min(N_IG // 2, len(e)), replace=False)
        out.append(np.c_[e[pick], t0[pick] + 1, h[pick]])
    return np.concatenate(out)


class _Score(torch.nn.Module):
    """Örnek başına ilgili host'un log(p_user + p_root) − log(p_temiz)'i."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, mask, host):
        lp = torch.log_softmax(self.model(x, mask), -1)[torch.arange(x.shape[0], device=x.device), host]
        return torch.logsumexp(lp[:, 1:], -1) - lp[:, 0]


def integrated_gradients(model, sub, k, samples, bs=100, n_steps=32):
    """→ (|IG| (N, k, 78), |Σ IG − (f(x) − f(0))| / |f(x) − f(0)| medyanı)."""
    from captum.attr import IntegratedGradients
    score = _Score(model)
    ig = IntegratedGradients(score)
    attrs, rel = [], []
    dev = sub.wd.device
    for i in range(0, len(samples), bs):
        e, t, h = (torch.as_tensor(samples[i:i + bs, j], device=dev) for j in range(3))
        x, mask = sub.wd.windows(e, t, k)
        x = D.scale_obs(x)
        a = ig.attribute(x, baselines=torch.zeros_like(x), additional_forward_args=(mask, h), n_steps=n_steps)
        with torch.no_grad():
            f = score(x, mask, h) - score(torch.zeros_like(x), mask, h)
        rel.append(((a.sum((1, 2)) - f).abs() / f.abs().clamp_min(1e-6)).detach().cpu().numpy())
        attrs.append(a.abs().detach().cpu().numpy())
    return np.concatenate(attrs), float(np.median(np.concatenate(rel)))


def ig_summary(A, samples, sub, k):
    """Zaman profili (adım geri), uyarı adımı oranı, özellik grubu payları."""
    An = A / np.maximum(A.sum((1, 2), keepdims=True), 1e-12)
    time_prof, feat = An.sum(2), An.sum(1)                              # (N, k), (N, 78)
    own = TOKEN_HOST[None, :] == np.asarray(D.LABEL_HOSTS)[samples[:, 2]][:, None]   # etiket → gözlem host'u
    own_share = (feat * own).sum(1)
    age = sub.age[samples[:, 0], samples[:, 1] - 1, samples[:, 2]]
    inside = (age >= 1) & (age < k)
    prior = time_prof[~inside].mean(0)                                 # uyarısız grubun konum profili
    pos = k - 1 - age[inside]
    return {'time_by_steps_back': time_prof.mean(0)[::-1].tolist(),
            'time_by_steps_back_inside': time_prof[inside].mean(0)[::-1].tolist(),
            'time_by_steps_back_never': prior[::-1].tolist(),
            'alert_step_ratio': float((time_prof[inside, pos] / prior[pos]).mean()),
            'last_step_share': float(time_prof[:, -1].mean()), 'last4_share': float(time_prof[:, -4:].sum(1).mean()),
            'own_host_share': float(own_share.mean()), 'own_host_share_inside': float(own_share[inside].mean()),
            'own_host_share_never': float(own_share[~inside].mean()),
            'bits_share': float(feat[:, TOKEN_KIND == 0].sum(1).mean()),
            'scan_share': float(feat[:, TOKEN_KIND == 1].sum(1).mean()),
            'decoy_share': float(feat[:, TOKEN_KIND == 2].sum(1).mean()), 'uniform_own_share': 6 / D.OBS_DIM}


# ── 4. TimesNet'in batch bağımlılığı ─────────────────────────────────────

TN_MODES = ('resmi', 'karisik', 'tek')
TN_EVERY = 19                 # tek tek alt kümesi: her 19. pencere (T=100 ile aralarında asal → her zaman adımı)


@torch.no_grad()
def timesnet_batch_preds(model, wd, k, mode, bs=8192):
    """→ kestirimler (N, 12); 'tek' kipinde yalnız her `TN_EVERY`. pencere, diğerleri -1."""
    N = wd.n * wd.T
    order = {'resmi': torch.arange(N), 'karisik': torch.randperm(N, generator=torch.Generator().manual_seed(0)),
             'tek': torch.arange(0, N, TN_EVERY)}[mode]
    step = bs if mode != 'tek' else 256
    pred = np.full((N, len(D.LABEL_HOSTS)), -1, np.int64)
    for i in range(0, len(order), step):
        f = order[i:i + step]
        x, m = wd.windows(f // wd.T, f % wd.T + 1, k)
        x = D.scale_obs(x)
        out = model(x, m) if mode != 'tek' else torch.cat([model(x[j:j + 1], m[j:j + 1]) for j in range(len(f))])
        pred[f.numpy()] = out.argmax(-1).cpu().numpy()
    return pred


def timesnet_batch(store, device):
    """Host makro-F1: 'tum' = bütün pencereler (resmi, karışık); 'alt' = tek tek alt kümesi, üç kip aynı pencerelerde."""
    from cybermetaformer.hidden_state import metrics as M
    out = {}
    with store.final_evaluation():
        wds = {split: D.WindowData(store, store.split(split), device=device) for split in ('test', 'ood_test')}
    for k in (8, 32):
        out[k] = {}
        for seed in SEEDS:
            model = load_model('timesnet', k, seed, device)
            out[k][seed] = {}
            for split, wd in wds.items():
                y = wd.labels.reshape(wd.n * wd.T, -1).cpu().numpy()
                sub = np.arange(0, len(y), TN_EVERY)
                P = {mode: timesnet_batch_preds(model, wd, k, mode) for mode in TN_MODES}
                out[k][seed][split] = {
                    'tum': {mode: float(M.host_macro_f1(y, P[mode])) for mode in ('resmi', 'karisik')},
                    'alt': {mode: float(M.host_macro_f1(y[sub], P[mode][sub])) for mode in TN_MODES}}
            print(f'timesnet k={k} s{seed}: {out[k][seed]}', flush=True)
    return out


# ── figürler ────────────────────────────────────────────────────────────

def fig_rollout(res, path):
    fig, axes = plt.subplots(1, len(res), figsize=(6.2 * len(res), 4.2))
    for ax, (k, r) in zip(np.atleast_1d(axes), res.items()):
        s0 = r['seeds'][0]
        x = np.arange(int(k))
        ax.plot(x, s0['prior_by_steps_back'], color=MUTED, lw=2, label='bütün pencereler (konum önseli)')
        w = np.array([np.nan if v is None else v for v in s0['alert_weight_by_age']])
        ax.plot(x, w, color=SLOT['transformer'], lw=0, marker='o', ms=4,
                label='sessiz ihlal: host\'un son uyarısının adımı')
        ax.set_yscale('log')
        ax.set_xlabel('adım geri (0 = şimdiki gözlem)')
        ax.set_ylabel('rollout payı (s0)')
        rs = [s['ratio_mean'] for s in r['seeds']]
        ax.set_title(f"Transformer k={k}: uyarı adımı / önsel = {min(rs):.2f}–{max(rs):.2f} ({len(rs)} seed)",
                     loc='left', fontsize=10)
        _grid(ax)
        ax.legend(fontsize=8)
    fig.suptitle('Sessiz ihlallerde dikkat son uyarının görüldüğü adıma gidiyor mu?', x=0.01, ha='left',
                 fontsize=11, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fig_hosts(mats, hosts, path):
    cmap = LinearSegmentedColormap.from_list('mavi', BLUES)
    vmax = max(H.max() for H in mats.values())
    fig, axes = plt.subplots(1, len(mats), figsize=(5.4 * len(mats), 5.6))
    for ax, (title, H) in zip(axes, mats.items()):
        ax.imshow(H, cmap=cmap, vmin=0, vmax=vmax, aspect='auto')
        for i in range(len(H)):
            for j in range(len(H)):
                ax.text(j, i, f'{100 * H[i, j]:.0f}', ha='center', va='center', fontsize=6.5,
                        color=SURFACE if H[i, j] > 0.55 * vmax else INK)
        ax.set_xticks(range(len(hosts)), hosts, fontsize=7.5, rotation=90)
        ax.set_yticks(range(len(hosts)), hosts, fontsize=7.5)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, loc='left', fontsize=10)
        ax.set_xlabel('anahtar host (dikkat edilen)')
    axes[0].set_ylabel('sorgu host')
    fig.suptitle(f'iTransformer: değişken-token dikkati host bloklarına toplandı (hücre = satır payı, %; tekdüze '
                 f'{600 / D.OBS_DIM:.1f})', x=0.01, ha='left', fontsize=11, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fig_ig(res, path):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    for m, r in res.items():
        prof = r['time_by_steps_back']
        axes[0].plot(range(len(prof)), prof, color=SLOT[m], marker=MARKER[m], ms=3, lw=1.5, label=LABEL[m])
    axes[0].set_yscale('log')
    axes[0].set_xlabel('adım geri (0 = şimdiki gözlem)')
    axes[0].set_ylabel('|IG| payı')
    axes[0].set_title('Zaman adımlarının payı (k=32)', loc='left', fontsize=10)
    _grid(axes[0])
    axes[0].legend(fontsize=8)
    groups = [('own_host_share', 'host\'un\nkendisi'), ('bits_share', 'aktivite/\ngüvenlik bitleri'),
              ('scan_share', 'tarama\ngeçmişi'), ('decoy_share', 'decoy')]
    xs = np.arange(len(groups))
    w = 0.8 / len(res)
    for i, (m, r) in enumerate(res.items()):
        axes[1].bar(xs + i * w, [r[g] for g, _ in groups], w, color=SLOT[m], label=LABEL[m])
    axes[1].plot([-0.1, 0.8], [6 / D.OBS_DIM] * 2, color=INK2, lw=1, ls=':')
    axes[1].text(0.85, 6 / D.OBS_DIM, 'tekdüze (6/78)', va='center', fontsize=8, color=INK2)
    axes[1].set_xticks(xs + 0.4 - w / 2, [lab for _, lab in groups])
    axes[1].set_ylabel('|IG| payı')
    axes[1].set_title('Özellik grupları', loc='left', fontsize=10)
    _grid(axes[1])
    fig.suptitle('Integrated Gradients: sessiz ihlallerde "ele geçirilmiş" kestirimini ne taşıyor?', x=0.01,
                 ha='left', fontsize=11, fontweight='bold', color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def summary_md(out):
    L = ['# M2 analizleri (dikkat, IG)', '',
         f"Test alt kümesi: rastgele {out['n_episodes']} bölüm (seed 0), {out['n_silent']} sessiz ihlal host-adımı. Yöntem: "
         '`hidden_state/explain.py` başlığı. Bütün sayılar `analiz.json`.', '',
         '## Rollout (Transformer)', '',
         'Oran = konuma düşen rollout payı / o konumun bütün pencerelerdeki ortalaması (1 = konum önseli).', '',
         '| k | seed | uyarılı adımlar | uyarısız adımlar | sessiz ihlal: uyarı adımı (ort. / medyan) | önselin '
         'üstünde | yakalanan | kaçan | çift |', '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for k, r in out['rollout'].items():
        for s, x in zip(SEEDS, r['seeds']):
            L.append(f"| {k} | {s} | {x['alert_any_ratio']:.2f} | {x['no_alert_ratio']:.2f} | {x['ratio_mean']:.2f} / "
                     f"{x['ratio_median']:.2f} | {100 * x['share_above_prior']:.0f}% | {x['ratio_mean_hit']:.2f} | "
                     f"{x['ratio_mean_missed']:.2f} | {x['n_pairs']} |")
    L += ['', '## iTransformer host dikkati', '',
          f"Tekdüze hücre {6 / D.OBS_DIM:.3f}. Kendi host = köşegen ortalaması.", '',
          '| k | model | kendi host | sorgu bağımlılığı (TV) | en büyük hücre | en güçlü çapraz (sorgu → anahtar) |',
          '|---:|---|---:|---:|---:|---|']
    for k, r in out['itransformer'].items():
        rows = [(f's{s}', x) for s, x in zip(SEEDS, r['seeds'])] + [('eğitilmemiş', r['untrained'])]
        for name, x in rows:
            cross = ', '.join(f'{q}→{kk} {v:.2f}' for q, kk, v in x['top_cross'][:3])
            L.append(f"| {k} | {name} | {x['self_mean']:.3f} | {x['query_tv']:.3f} | {x['max_cell']:.3f} | {cross} |")
    ig = out['ig']
    L += ['', f"## Integrated Gradients (k={ig['k']}, seed 0, {ig['n']} örnek)", '',
          '| Model | kendi host (uyarı pencerede / hiç) | bit | tarama | decoy | son adım | son 4 adım | uyarı adımı '
          'oranı | yakınsama (medyan göreli; adım) |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m, x in ig['models'].items():
        L.append(f"| {LABEL[m]} | {x['own_host_share']:.3f} ({x['own_host_share_inside']:.3f} / "
                 f"{x['own_host_share_never']:.3f}) | {x['bits_share']:.3f} | {x['scan_share']:.3f} | "
                 f"{x['decoy_share']:.3f} | {x['last_step_share']:.3f} | {x['last4_share']:.3f} | "
                 f"{x['alert_step_ratio']:.2f} | {x['convergence_rel_median']:.3f}; {x['n_steps']} |")
    if 'timesnet_batch' in out:
        L += ['', '## TimesNet: değerlendirme batch\'ine bağımlılık', '',
              'Host makro-F1, 3 seed ortalaması (aralık). Resmi = raporlanan değerlendirme sırası (bölüm-ana, batch '
              f'8192); karışık = rastgele sıra, batch 8192; tek tek = bs=1 (çevrimiçi kullanım), yalnız her {TN_EVERY}. '
              'pencerede. Alt küme sütunlarında üç kip aynı pencerelerde.', '',
              '| k | küme | resmi | karışık | alt küme: resmi | alt küme: karışık | alt küme: tek tek |',
              '|---:|---|---:|---:|---:|---:|---:|']
        for k, r in out['timesnet_batch'].items():
            for split in ('test', 'ood_test'):
                cells = []
                for part, modes in (('tum', ('resmi', 'karisik')), ('alt', TN_MODES)):
                    for mode in modes:
                        v = [r[s][split][part][mode] for s in r]
                        cells.append(f'{np.mean(v):.3f} ({min(v):.3f}–{max(v):.3f})')
                L.append(f"| {k} | {'test' if split == 'test' else 'OOD'} | " + ' | '.join(cells) + ' |')
    return '\n'.join(L) + '\n'


def write_outputs(out):
    with open(os.path.join(OUT_DIR, 'analiz.json'), 'w') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    with open(os.path.join(OUT_DIR, 'ozet.md'), 'w') as f:
        f.write(summary_md(out))
    print('→', OUT_DIR)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--only', choices=['timesnet'], help="yalnız bu bölüm yeniden hesaplanır; diğerleri analiz.json'dan")
    args = p.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    store = D.EpisodeStore()
    if args.only == 'timesnet':
        with open(os.path.join(OUT_DIR, 'analiz.json')) as f:
            out = json.load(f)
        out['timesnet_batch'] = timesnet_batch(store, device)
        write_outputs(out)
        return
    sub = Subset(store, device)
    hosts = [str(h) for h in store.meta['hosts']]
    out = {'n_episodes': N_EP, 'n_silent': int(sub.silent.sum()), 'hosts': hosts}

    out['rollout'] = {}
    for k in (32, 64):
        out['rollout'][k] = {'seeds': [rollout_stats(load_model('transformer', k, s, device), sub, k) for s in SEEDS]}
        print(f"rollout k={k}: " + ', '.join(f"{x['alert_any_ratio']:.2f}/{x['ratio_mean']:.2f}"
                                             for x in out['rollout'][k]['seeds']), flush=True)
    fig_rollout(out['rollout'], os.path.join(FIG_DIR, 'fig_dikkat_rollout.png'))

    out['itransformer'], mats = {}, {}
    for k in (32, 64):
        res = []
        for seed in SEEDS:
            H = itransformer_host_attention(load_model('itransformer', k, seed, device), sub, k)
            res.append({'matrix': H.tolist(), **host_summary(H, hosts)})
            if seed == 0:
                mats[f'eğitilmiş, k={k} (s0)'] = H
        H0 = itransformer_host_attention(load_model('itransformer', k, 0, device, trained=False), sub, k)
        out['itransformer'][k] = {'seeds': res, 'untrained': {'matrix': H0.tolist(), **host_summary(H0, hosts)}}
        if k == 64:
            mats['eğitilmemiş, k=64'] = H0
        print(f"itransformer k={k}: kendi host {[round(r['self_mean'], 3) for r in res]}, eğitilmemiş "
              f"{out['itransformer'][k]['untrained']['self_mean']:.3f}", flush=True)
    fig_hosts(mats, hosts, os.path.join(FIG_DIR, 'fig_itransformer_host.png'))

    k = 32
    samples = ig_samples(sub, k, np.random.default_rng(0))
    sub_cpu = Subset(store, 'cpu')
    out['ig'] = {'k': k, 'n': int(len(samples)), 'models': {}}
    for m in M2_MODELS:
        if not os.path.exists(os.path.join(STAGE_DIR, f'{m}_k{k}_s0', 'model.pt')):
            print(f'IG {m}: koşu yok, atlandı', flush=True)
            continue
        model = load_model(m, k, 0, 'cpu')
        for n_steps in (32, 128, 256):
            A, rel = integrated_gradients(model, sub_cpu, k, samples, bs=1 if m == 'timesnet' else max(1, 3200 // n_steps),
                                          n_steps=n_steps)
            if rel < 0.05:
                break
        r = out['ig']['models'][m] = {**ig_summary(A, samples, sub, k), 'convergence_rel_median': rel, 'n_steps': n_steps}
        print(f"IG {m}: kendi host {r['own_host_share']:.3f}, son adım {r['last_step_share']:.3f}, "
              f"uyarı adımı oranı {r['alert_step_ratio']:.2f}, yakınsama {rel:.3f} ({n_steps} adım)", flush=True)
    fig_ig(out['ig']['models'], os.path.join(FIG_DIR, 'fig_ig.png'))
    out['timesnet_batch'] = timesnet_batch(store, device)
    write_outputs(out)


if __name__ == '__main__':
    main()

"""TSLib modelleri. Ortak arayüz: forward(x, mask) → logits (B, 12, 3).

Kaynak dosyalar `third_party/tslib/`'de (sabit commit, değiştirilmez). Uyarlamalar:
- Sınıflandırma kipi: `seq_len = k`, `enc_in = 78`, `num_class = 36` → (B, 12, 3).
- `x_mark_enc` = dolgu maskesi. TSLib'in sınıflandırma başlığı dolgu adımlarının gömmesini sıfırlar (TimesNet,
  TimeMixer); iTransformer ve PatchTST maskeyi kullanmaz, dolgu adımları sıfır girdi olarak görülür.
- Örnek normalizasyonu yok (T1). PatchTST'nin `classification`'ı normalizasyon içerdiği için ileri geçiş burada aynı
  sırayla, normalizasyonsuz yazıldı; diğerlerinin sınıflandırma yolunda normalizasyon zaten yok.
- TSMixer TSLib'de yalnızca tahmin kipinde: ResBlock'lar alınıp TSLib'in sınıflandırma başlığı kalıbı (GELU →
  dropout → dolgu sıfırlama → düzleştirme → doğrusal) eklendi.
- k'ya bağlı ayarlar geçerlilik matrisinden: PatchTST yama/adım, TimesNet top_k, TimeMixer alt örnekleme
  katmanı sayısı. N/A hücreler `models.NOT_APPLICABLE`'da.
- MPS hatası (torch 2.14.1, 2026-10-21): Conv1d'nin ileri geçişi ve gradyanları batch > 2^16'da yanlış. PatchTST
  kanal-bağımsız olduğundan kodlayıcının çekirdek-1 Conv1d'leri B·78 satır görür (batch 1024'te 79 872); bu
  konvolüsyonlar `_ChunkedConv1d` ile ≤ 2^16'lık parçalarda çalışır (sonuç aynı, state_dict anahtarları değişmez).
  Bu düzeltmeden önceki PatchTST koşuları (HPO ve finaller) bozuk gradyanla eğitildi.

Varsayılan argümanlar `configs/<model>.yaml`'daki temel konfigürasyonla aynı (bench.py bunları ölçer).
"""
import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as nnf

from cybermetaformer.hidden_state.data import OBS_DIM, LABEL_HOSTS, N_CLASSES
from cybermetaformer.hidden_state.third_party.tslib.models import PatchTST as _patchtst
from cybermetaformer.hidden_state.third_party.tslib.models import TimeMixer as _timemixer
from cybermetaformer.hidden_state.third_party.tslib.models import TimesNet as _timesnet
from cybermetaformer.hidden_state.third_party.tslib.models import TSMixer as _tsmixer
from cybermetaformer.hidden_state.third_party.tslib.models import iTransformer as _itransformer

N_OUT = len(LABEL_HOSTS) * N_CLASSES
MPS_CONV_MAX_BATCH = 2 ** 16


def _configs(k, **kw):
    """TSLib'in argparse Namespace'inin karşılığı: sınıflandırma yolunun okuduğu alanlar."""
    return SimpleNamespace(task_name='classification', seq_len=k, label_len=0, pred_len=0, enc_in=OBS_DIM,
                           c_out=OBS_DIM, num_class=N_OUT, embed='fixed', freq='h', factor=1, activation='gelu', **kw)


def _out(logits):
    return logits.view(-1, len(LABEL_HOSTS), N_CLASSES)


class _ChunkedConv1d(nn.Conv1d):
    """Batch boyutunda ≤ 2^16'lık parçalarla Conv1d (MPS hatası; örnekler bağımsız, sonuç aynı)."""

    def forward(self, x):
        if x.shape[0] <= MPS_CONV_MAX_BATCH:
            return super().forward(x)
        return torch.cat([super().forward(c) for c in x.split(MPS_CONV_MAX_BATCH)])


class _Classification(nn.Module):
    """TSLib `Model.classification(x_enc, x_mark_enc)` olduğu gibi."""

    def forward(self, x, mask):
        return _out(self.net.classification(x, mask.to(x.dtype)))


class ITransformer(_Classification):
    """Token = değişken (78 özellik); her token k adımlık dizisinin gömmesi."""

    def __init__(self, k, d_model=128, heads=8, layers=2, ff=256, dropout=0.1):
        super().__init__()
        self.net = _itransformer.Model(_configs(k, d_model=d_model, n_heads=heads, e_layers=layers, d_ff=ff,
                                                dropout=dropout))


class TimesNet(_Classification):
    """k=1 N/A. top_k k=4'te 1 (matris), k ≥ 8'de en fazla k // 2 (sıfır frekans dışındaki rfft kutusu sayısı)."""

    def __init__(self, k, d_model=32, ff=64, layers=2, top_k=3, kernels=6, dropout=0.1):
        super().__init__()
        self.top_k = 1 if k <= 4 else min(top_k, k // 2)
        self.net = _timesnet.Model(_configs(k, d_model=d_model, d_ff=ff, e_layers=layers, top_k=self.top_k,
                                            num_kernels=kernels, dropout=dropout))


class _SingleScale(nn.Module):
    """Tek ölçekte (alt örnekleme yok) TimeMixer'ın ölçekler arası karıştırması birimdir; TSLib'in sınıfı en az iki
    ölçek bekliyor (`season_list[1]`)."""

    def forward(self, xs):
        return [xs[0].permute(0, 2, 1)]


class TimeMixer(_Classification):
    """Alt örnekleme (pencere 2, ortalama havuzlama) katman sayısı max(0, min(3, log2 k − 1)): k=2 → 0 (tek ölçek),
    4 → 1, 8 → 2, 16 → 3, 32 → 3. k=1 N/A (MPS çekirdek hatası; `models.NOT_APPLICABLE`). Sınıflandırma yolu
    kanal-bağımsız gömmeyle çalışmadığı için channel_independence=0."""

    def __init__(self, k, d_model=32, ff=64, layers=2, moving_avg=5, dropout=0.1):
        super().__init__()
        self.down_sampling_layers = max(0, min(3, int(math.log2(k)) - 1))
        self.net = _timemixer.Model(_configs(
            k, d_model=d_model, d_ff=ff, e_layers=layers, dropout=dropout, channel_independence=0,
            decomp_method='moving_avg', moving_avg=moving_avg, top_k=5, use_norm=0,
            down_sampling_layers=self.down_sampling_layers, down_sampling_window=2, down_sampling_method='avg'))
        if self.down_sampling_layers == 0:
            for block in self.net.pdm_blocks:
                block.mixing_multi_scale_season = _SingleScale()
                block.mixing_multi_scale_trend = _SingleScale()


class PatchTST(nn.Module):
    """k=1 N/A. Kanal-bağımsız: her özellik ayrı dizi. Yama = max(2, round(k · patch_ratio)), adım = yama // 2
    (patch_ratio 0.5 → matris: k=4 → 2/1, 8 → 4/2, 16 → 8/4, 32 → 16/8; 0.25 → k=32'de 8/4)."""

    def __init__(self, k, d_model=128, heads=8, layers=2, ff=256, dropout=0.1, patch_ratio=0.5):
        super().__init__()
        self.patch_len = max(2, round(k * patch_ratio))
        self.stride = max(1, self.patch_len // 2)
        self.net = _patchtst.Model(_configs(k, d_model=d_model, n_heads=heads, e_layers=layers, d_ff=ff,
                                            dropout=dropout), patch_len=self.patch_len, stride=self.stride)
        for layer in self.net.encoder.attn_layers:                   # sınıf değişimi: parametre ve RNG aynı
            layer.conv1.__class__ = layer.conv2.__class__ = _ChunkedConv1d

    def forward(self, x, mask):
        """TSLib `PatchTST.classification` ile aynı sıra, örnek normalizasyonu olmadan."""
        net = self.net
        enc, n_vars = net.patch_embedding(x.permute(0, 2, 1))                # (B·78, yama sayısı, d_model)
        enc, _ = net.encoder(enc)
        enc = enc.reshape(-1, n_vars, enc.shape[-2], enc.shape[-1]).permute(0, 1, 3, 2)
        return _out(net.projection(net.dropout(net.flatten(enc)).reshape(x.shape[0], -1)))


class TSMixer(nn.Module):
    """TSLib ResBlock'ları (zaman MLP'si + kanal MLP'si, artık bağlantılı) + sınıflandırma başlığı."""

    def __init__(self, k, d_model=128, layers=2, dropout=0.1):
        super().__init__()
        cfg = _configs(k, d_model=d_model, e_layers=layers, dropout=dropout)
        self.blocks = nn.ModuleList([_tsmixer.ResBlock(cfg) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(k * OBS_DIM, N_OUT)

    def forward(self, x, mask):
        for block in self.blocks:
            x = block(x)
        out = self.dropout(nnf.gelu(x)) * mask[..., None].to(x.dtype)
        return _out(self.projection(out.reshape(x.shape[0], -1)))

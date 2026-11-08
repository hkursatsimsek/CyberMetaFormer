"""Dizi modelleri (Aşama 1–2 temsilcileri). Ortak arayüz: forward(x, mask) → logits (B, 12, 3).

M0'da hesap bütçesi ölçümü (bench.py) için yazıldı; HPO uzayları M1/M2'de kesinleşir.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as nnf

from cybermetaformer.hidden_state.data import N_BLUE_ACTIONS, OBS_DIM, LABEL_HOSTS, N_CLASSES

N_OUT = len(LABEL_HOSTS) * N_CLASSES


def _head(d):
    return nn.Linear(d, N_OUT)


class CNN1D(nn.Module):
    """Nedensel 1D konvolüsyon blokları (k ≥ 8'de dilate, TCN tarzı) + maskeli ortalama havuzlama.
    k=1'de 1×1 konvolüsyona iner (MLP'ye denk)."""

    def __init__(self, k, channels=128, layers=3, kernel=3, dropout=0.1, dilated=None):
        super().__init__()
        dilated = k >= 8 if dilated is None else dilated
        self.inp = nn.Linear(OBS_DIM, channels)            # 1×1 konvolüsyonun eşdeğeri (CPU'da çok daha hızlı)
        self.blocks = nn.ModuleList()
        self.pads = []
        for i in range(layers):
            d = 2 ** i if dilated else 1
            self.pads.append((kernel - 1) * d)
            self.blocks.append(nn.Conv1d(channels, channels, kernel, dilation=d))
        self.drop = nn.Dropout(dropout)
        self.head = _head(channels)

    def forward(self, x, mask):
        h = self.inp(x).transpose(1, 2).contiguous()         # (B, C, k)
        for pad, conv in zip(self.pads, self.blocks):
            h = h + self.drop(nnf.relu(conv(nnf.pad(h, (pad, 0)))))
        m = mask[:, None].float()
        pooled = (h * m).sum(-1) / m.sum(-1).clamp(min=1)
        return self.head(pooled).view(-1, len(LABEL_HOSTS), N_CLASSES)


class RNN(nn.Module):
    """LSTM / GRU; son gizli durum → 12×3 başlık. Baştaki dolgu adımları sıfırdır."""

    def __init__(self, k, cell='lstm', hidden=128, layers=1, dropout=0.0):
        super().__init__()
        cls = {'lstm': nn.LSTM, 'gru': nn.GRU}[cell]
        self.rnn = cls(OBS_DIM, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = _head(hidden)

    def forward(self, x, mask=None):
        out, _ = self.rnn(x)
        return self.head(out[:, -1]).view(-1, len(LABEL_HOSTS), N_CLASSES)


class RNNFull(RNN):
    """Tam geçmiş varyantı: bölüm boyunca durumlu, her adımda tahmin (çoktan-çoğa).

    train.py'de `mode: episode` ile eğitilir; t. konumdaki çıktı obs[0..t]'yi görür, yani k = t+1 dolgusuz pencereyle
    aynı bilgiyi kullanır. Pencere arayüzü (forward) RNN ile aynıdır (test ve şekil kontrolleri için).
    action_input (M3 aksiyon ablasyonu): önceki Blue aksiyonunun 16 boyutlu gömmesi gözleme eklenir."""
    episode_mode = True
    ACT_DIM = 16

    def __init__(self, k, cell='lstm', hidden=128, layers=1, dropout=0.0, action_input=False):
        super().__init__(k, cell, hidden, layers, dropout)
        self.action_input = action_input
        if action_input:
            cls = {'lstm': nn.LSTM, 'gru': nn.GRU}[cell]
            self.act_emb = nn.Embedding(N_BLUE_ACTIONS + 1, self.ACT_DIM)
            self.rnn = cls(OBS_DIM + self.ACT_DIM, hidden, layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)

    def forward_episode(self, x, act=None):
        """(B, S, 78) [+ (B, S) önceki aksiyon] → (B, S, 12, 3)."""
        if self.action_input:
            x = torch.cat([x, self.act_emb(act)], -1)
        out, _ = self.rnn(x)
        return self.head(out).view(x.shape[0], x.shape[1], len(LABEL_HOSTS), N_CLASSES)


class Transformer(nn.Module):
    """Yalnızca kodlayıcı; token = gözlem adımı; öğrenilen ya da sinüzoidal konum (vanilla Transformer).
    Okuma: son adım (`last`) ya da dizinin sonuna eklenen öğrenilen CLS token'ı (`cls`; nedensel maskede de bütün
    adımları görür). Dolgu adımları key_padding_mask ile dışlanır. causal=True nedensel maske ekler (ablasyon)."""

    def __init__(self, k, d_model=128, heads=4, layers=2, ff=256, dropout=0.1, pos='learned', causal=False,
                 readout='last'):
        super().__init__()
        self.k, self.causal, self.readout, self.heads = k, causal, readout, heads
        if readout == 'cls':
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls, std=0.02)
        elif readout != 'last':
            raise ValueError(f'readout: {readout}')
        self.inp = nn.Linear(OBS_DIM, d_model)
        if pos == 'learned':
            self.pos = nn.Parameter(torch.zeros(1, k, d_model))
            nn.init.normal_(self.pos, std=0.02)
        else:
            p = torch.arange(k)[:, None] * torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe = torch.zeros(1, k, d_model)
            pe[0, :, 0::2], pe[0, :, 1::2] = torch.sin(p), torch.cos(p)
            self.register_buffer('pos', pe)
        layer = nn.TransformerEncoderLayer(d_model, heads, ff, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = _head(d_model)

    def forward(self, x, mask):
        h, pad = self.inp(x) + self.pos[:, -x.shape[1]:], ~mask
        if self.readout == 'cls':
            h = torch.cat([h, self.cls.expand(h.shape[0], -1, -1)], 1)
            pad = torch.cat([pad, pad.new_zeros(pad.shape[0], 1)], 1)
        attn = None
        if self.causal:
            # Dolgu ile nedensel maske birleşince bölüm başındaki dolgu sorgularının bütün anahtarları maskeli kalır ve
            # softmax NaN verir; NaN sonraki katmanda değerler üzerinden geçerli adımlara taşınır. Her adım kendine
            # bakabilir; geçerli adımlar dolguyu yine görmez.
            L = h.shape[1]
            eye = torch.eye(L, dtype=torch.bool, device=x.device)
            attn = (torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), 1)[None] | pad[:, None]) & ~eye
            attn, pad = attn.repeat_interleave(self.heads, 0), None
        h = self.enc(h, mask=attn, src_key_padding_mask=pad)
        return self.head(self.norm(h[:, -1])).view(-1, len(LABEL_HOSTS), N_CLASSES)

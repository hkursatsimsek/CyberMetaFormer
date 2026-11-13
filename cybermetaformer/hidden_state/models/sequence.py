"""Sequence models (representatives of stages 1–2). Common interface: forward(x, mask) → logits (B, 12, 3).

Written in M0 for the compute budget measurement (bench.py); the HPO spaces are fixed in M1/M2.
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
    """Causal 1D convolution blocks (dilated for k ≥ 8, TCN style) + masked mean pooling.
    Reduces to a 1×1 convolution at k=1 (equivalent to an MLP)."""

    def __init__(self, k, channels=128, layers=3, kernel=3, dropout=0.1, dilated=None):
        super().__init__()
        dilated = k >= 8 if dilated is None else dilated
        self.inp = nn.Linear(OBS_DIM, channels)            # equivalent of a 1×1 convolution (much faster on CPU)
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
    """LSTM / GRU; last hidden state → 12×3 head. Leading padding steps are zero."""

    def __init__(self, k, cell='lstm', hidden=128, layers=1, dropout=0.0):
        super().__init__()
        cls = {'lstm': nn.LSTM, 'gru': nn.GRU}[cell]
        self.rnn = cls(OBS_DIM, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = _head(hidden)

    def forward(self, x, mask=None):
        out, _ = self.rnn(x)
        return self.head(out[:, -1]).view(-1, len(LABEL_HOSTS), N_CLASSES)


class RNNFull(RNN):
    """Full-history variant: stateful over the episode, a prediction at every step (many-to-many).

    Trained in train.py with `mode: episode`; the output at position t sees obs[0..t], i.e. it uses the same
    information as an unpadded window with k = t+1. The window interface (forward) is the same as RNN (for tests and
    shape checks). action_input (M3 action ablation): a 16-dim embedding of the previous Blue action is added
    to the observation."""
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
        """(B, S, 78) [+ (B, S) previous action] → (B, S, 12, 3)."""
        if self.action_input:
            x = torch.cat([x, self.act_emb(act)], -1)
        out, _ = self.rnn(x)
        return self.head(out).view(x.shape[0], x.shape[1], len(LABEL_HOSTS), N_CLASSES)


class Transformer(nn.Module):
    """Encoder only; token = observation step; learned or sinusoidal positions (vanilla Transformer).
    Readout: the last step (`last`) or a learned CLS token appended at the end of the sequence (`cls`; it sees all
    steps even under the causal mask). Padding steps are excluded with key_padding_mask. causal=True adds a causal
    mask (ablation)."""

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
            # When padding and the causal mask are combined, all keys of the padding queries at the episode start stay
            # masked and softmax gives NaN; the NaN is carried to valid steps through the values in the next layer. Every
            # step may attend to itself; valid steps still do not see the padding.
            L = h.shape[1]
            eye = torch.eye(L, dtype=torch.bool, device=x.device)
            attn = (torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), 1)[None] | pad[:, None]) & ~eye
            attn, pad = attn.repeat_interleave(self.heads, 0), None
        h = self.enc(h, mask=attn, src_key_padding_mask=pad)
        return self.head(self.norm(h[:, -1])).view(-1, len(LABEL_HOSTS), N_CLASSES)

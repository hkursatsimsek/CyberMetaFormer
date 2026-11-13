"""TSLib models. Common interface: forward(x, mask) → logits (B, 12, 3).

The source files are in `third_party/tslib/` (pinned commit, not modified). Adaptations:
- Classification mode: `seq_len = k`, `enc_in = 78`, `num_class = 36` → (B, 12, 3).
- `x_mark_enc` = padding mask. The TSLib classification head zeroes the embedding of padding steps (TimesNet,
  TimeMixer); iTransformer and PatchTST do not use the mask, padding steps are seen as zero input.
- No instance normalisation (T1). PatchTST's `classification` includes normalisation, so the forward pass is written
  here in the same order without it; the classification paths of the others have no normalisation anyway.
- In TSLib, TSMixer exists only in forecasting mode: its ResBlocks are taken and the TSLib classification head pattern
  (GELU → dropout → padding zeroing → flatten → linear) is added.
- k-dependent settings come from the validity matrix: PatchTST patch/stride, TimesNet top_k, number of
  TimeMixer downsampling layers. The N/A cells are in `models.NOT_APPLICABLE`.
- MPS bug (torch 2.14.1, 2026-10-21): the forward pass and gradients of Conv1d are wrong for batch > 2^16. PatchTST
  is channel-independent, so the kernel-1 Conv1d layers of the encoder see B·78 rows (79,872 at batch 1024); these
  convolutions run in chunks of ≤ 2^16 via `_ChunkedConv1d` (same result, state_dict keys unchanged).
  PatchTST runs before this fix (HPO and finals) were trained with corrupted gradients.

Default arguments match the base configuration in `configs/<model>.yaml` (bench.py measures these).
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
    """Counterpart of the TSLib argparse Namespace: the fields read by the classification path."""
    return SimpleNamespace(task_name='classification', seq_len=k, label_len=0, pred_len=0, enc_in=OBS_DIM,
                           c_out=OBS_DIM, num_class=N_OUT, embed='fixed', freq='h', factor=1, activation='gelu', **kw)


def _out(logits):
    return logits.view(-1, len(LABEL_HOSTS), N_CLASSES)


class _ChunkedConv1d(nn.Conv1d):
    """Conv1d in chunks of ≤ 2^16 along the batch dimension (MPS bug; samples are independent, same result)."""

    def forward(self, x):
        if x.shape[0] <= MPS_CONV_MAX_BATCH:
            return super().forward(x)
        return torch.cat([super().forward(c) for c in x.split(MPS_CONV_MAX_BATCH)])


class _Classification(nn.Module):
    """TSLib `Model.classification(x_enc, x_mark_enc)` as is."""

    def forward(self, x, mask):
        return _out(self.net.classification(x, mask.to(x.dtype)))


class ITransformer(_Classification):
    """Token = variable (78 features); each token is the embedding of its k-step sequence."""

    def __init__(self, k, d_model=128, heads=8, layers=2, ff=256, dropout=0.1):
        super().__init__()
        self.net = _itransformer.Model(_configs(k, d_model=d_model, n_heads=heads, e_layers=layers, d_ff=ff,
                                                dropout=dropout))


class TimesNet(_Classification):
    """k=1 N/A. top_k is 1 at k=4 (matrix), at most k // 2 for k ≥ 8 (number of rfft bins other than zero frequency)."""

    def __init__(self, k, d_model=32, ff=64, layers=2, top_k=3, kernels=6, dropout=0.1):
        super().__init__()
        self.top_k = 1 if k <= 4 else min(top_k, k // 2)
        self.net = _timesnet.Model(_configs(k, d_model=d_model, d_ff=ff, e_layers=layers, top_k=self.top_k,
                                            num_kernels=kernels, dropout=dropout))


class _SingleScale(nn.Module):
    """At a single scale (no downsampling) TimeMixer's cross-scale mixing is the identity; the TSLib class expects at
    least two scales (`season_list[1]`)."""

    def forward(self, xs):
        return [xs[0].permute(0, 2, 1)]


class TimeMixer(_Classification):
    """Number of downsampling layers (window 2, average pooling) max(0, min(3, log2 k − 1)): k=2 → 0 (single scale),
    4 → 1, 8 → 2, 16 → 3, 32 → 3. k=1 N/A (MPS kernel error; `models.NOT_APPLICABLE`). The classification path does
    not work with channel-independent embeddings, hence channel_independence=0."""

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
    """k=1 N/A. Channel-independent: each feature is a separate sequence. Patch = max(2, round(k · patch_ratio)),
    stride = patch // 2 (patch_ratio 0.5 → matrix: k=4 → 2/1, 8 → 4/2, 16 → 8/4, 32 → 16/8; 0.25 → 8/4 at k=32)."""

    def __init__(self, k, d_model=128, heads=8, layers=2, ff=256, dropout=0.1, patch_ratio=0.5):
        super().__init__()
        self.patch_len = max(2, round(k * patch_ratio))
        self.stride = max(1, self.patch_len // 2)
        self.net = _patchtst.Model(_configs(k, d_model=d_model, n_heads=heads, e_layers=layers, d_ff=ff,
                                            dropout=dropout), patch_len=self.patch_len, stride=self.stride)
        for layer in self.net.encoder.attn_layers:                   # class swap: parameters and RNG unchanged
            layer.conv1.__class__ = layer.conv2.__class__ = _ChunkedConv1d

    def forward(self, x, mask):
        """Same order as TSLib `PatchTST.classification`, without instance normalisation."""
        net = self.net
        enc, n_vars = net.patch_embedding(x.permute(0, 2, 1))                # (B·78, number of patches, d_model)
        enc, _ = net.encoder(enc)
        enc = enc.reshape(-1, n_vars, enc.shape[-2], enc.shape[-1]).permute(0, 1, 3, 2)
        return _out(net.projection(net.dropout(net.flatten(enc)).reshape(x.shape[0], -1)))


class TSMixer(nn.Module):
    """TSLib ResBlocks (time MLP + channel MLP, with residual connections) + classification head."""

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

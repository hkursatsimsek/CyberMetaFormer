"""Model registry: build_model(name, k, **arguments). Torch models only; XGBoost in `xgb.py`, rules in `rules.py`."""
from functools import partial

from cybermetaformer.hidden_state.models.baselines import MLP
from cybermetaformer.hidden_state.models.cybermetaformer import CyberMetaFormer
from cybermetaformer.hidden_state.models.sequence import CNN1D, RNN, RNNFull, Transformer
from cybermetaformer.hidden_state.models.tslib_adapter import ITransformer, PatchTST, TimeMixer, TimesNet, TSMixer

REGISTRY = {
    'mlp': MLP,
    'cnn1d': CNN1D,
    'lstm': partial(RNN, cell='lstm'),
    'gru': partial(RNN, cell='gru'),
    'lstm_full': partial(RNNFull, cell='lstm'),
    'gru_full': partial(RNNFull, cell='gru'),
    'transformer': Transformer,
    'patchtst': PatchTST,
    'itransformer': ITransformer,
    'tsmixer': TSMixer,
    'timesnet': TimesNet,
    'timemixer': TimeMixer,
    'cybermetaformer': CyberMetaFormer,
}

# N/A cells of the validity matrix: (model, k) → reason. Shown as "—" in the tables.
NOT_APPLICABLE = {
    ('patchtst', 1): 'single patch: a single channel-independent scalar',
    ('timesnet', 1): 'no period can be found by FFT in a single step',
    # The matrix said "single scale (reduces to TSMixer)"; the MPS convolution kernel kills the process with an assert
    # in this cell (eager and compile; k=2 and 4 work), decided 2026-10-19.
    ('timemixer', 1): "MPS kernel error (reduces to TSMixer at a single scale)",
}


def build_model(name, k, **kwargs):
    if name not in REGISTRY:
        raise ValueError(f'Unknown model: {name} (available: {sorted(REGISTRY)})')
    if (name, k) in NOT_APPLICABLE:
        raise ValueError(f'{name} k={k} is N/A in the validity matrix: {NOT_APPLICABLE[name, k]}')
    return REGISTRY[name](k=k, **kwargs)

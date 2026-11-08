"""Model kaydı: build_model(ad, k, **argümanlar). Yalnızca torch modelleri; XGBoost `xgb.py`, kurallar `rules.py`."""
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

# Geçerlilik matrisinin N/A hücreleri: (model, k) → neden. Tablolarda "—" olarak gösterilir.
NOT_APPLICABLE = {
    ('patchtst', 1): 'tek yama: kanal-bağımsız tek bir skaler',
    ('timesnet', 1): 'tek adımda FFT ile periyot bulunamaz',
    # Matris "tek ölçek (TSMixer'a iner)" diyordu; MPS'in konvolüsyon çekirdeği bu hücrede assert ile süreci düşürüyor
    # (eager ve compile; k=2 ve 4 sorunsuz). README Kararlar 2026-10-19.
    ('timemixer', 1): "MPS çekirdek hatası (tek ölçekte TSMixer'a iner)",
}


def build_model(name, k, **kwargs):
    if name not in REGISTRY:
        raise ValueError(f'Bilinmeyen model: {name} (mevcut: {sorted(REGISTRY)})')
    if (name, k) in NOT_APPLICABLE:
        raise ValueError(f'{name} k={k} geçerlilik matrisinde N/A: {NOT_APPLICABLE[name, k]}')
    return REGISTRY[name](k=k, **kwargs)

"""Hiperparametre araması: Optuna TPE + MedianPruner, yalnızca val.

Adil bütçe: her model için k ∈ {8, 32}'de 25'er deneme, train'in %25'i (4000 bölüm, seed 0), en fazla 10 epoch
(sabır 3). XGBoost aynı veri bütçesiyle aranır (budama yok; val mlogloss ile erken durdurma). Sonuç
`results/hidden_state/<asama>/hpo/<model>_k<k>/best.yaml`; run_grid.py k ≤ 8 için k=8'in, k ≥ 16 (ve uzatmalar)
için k=32'nin konfigürasyonunu kullanır. Çalışma sqlite'ta tutulur; kesilirse aynı komut kaldığı yerden sürer.

    python -m cybermetaformer.hidden_state.hpo --models mlp cnn1d lstm gru --device mps
    python -m cybermetaformer.hidden_state.hpo --models xgb_tf xgb --nthread 8
    python -m cybermetaformer.hidden_state.hpo --models mlp --smoke          # 2 deneme, 200 bölüm, 2 epoch
    python -m cybermetaformer.hidden_state.hpo --models transformer --stage asama2 --device mps       # Aşama 2

Aşama 2 uzayları M1'dekilerle aynı ölçekte: model başına 3–5 mimari parametre + ortak (lr, weight decay,
sınıf ağırlığı). k'ya bağlı ayarlar (PatchTST yaması, TimesNet top_k üst sınırı, TimeMixer ölçek sayısı) aranmaz,
geçerlilik matrisinden gelir (`models/tslib_adapter.py`).

M2 bütçesi (README Kararlar 2026-10-19): Aşama 2'de 15 deneme. PatchTST ve TimesNet'in genişliği sabit
(d_model 32 ve 16, en fazla 2 katman; PatchTST yama oranı 0.5), parametre sayıları M1 modelleriyle aynı ölçekte.
Geniş ayarları epoch başına 26–156 dk sürüyor.
"""
import argparse
import csv
import json
import os
import time

import optuna
import yaml

from cybermetaformer.hidden_state import data as D
from cybermetaformer.hidden_state.train import RESULTS, release_device_memory, train

CONFIGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs')
PROTOCOL = {'n_trials': 25, 'ks': [8, 32], 'train_episodes': 4000, 'epochs': 10, 'patience': 3, 'seed': 0,
            'sampler_seed': 0, 'pruner': {'n_startup_trials': 5, 'n_warmup_steps': 2}}
STAGE_PROTOCOL = {'asama2': {'n_trials': 15}, 'asama3': {'n_trials': 15}}   # M2 bütçesi; M3 aynı
SMOKE = {'n_trials': 2, 'train_episodes': 200, 'epochs': 2, 'val_episodes': 200}
TORCH_MODELS = ['mlp', 'cnn1d', 'lstm', 'gru', 'transformer', 'patchtst', 'itransformer', 'tsmixer', 'timesnet',
                'timemixer', 'cybermetaformer']
XGB_MODELS = ['xgb', 'xgb_tf']


def _common(t):
    return {'lr': t.suggest_float('lr', 1e-4, 3e-3, log=True),
            'weight_decay': t.suggest_float('weight_decay', 1e-5, 1e-1, log=True),
            'class_weights': t.suggest_categorical('class_weights', [None, 'balanced'])}


def space_mlp(t):
    n, w = t.suggest_int('layers', 2, 4), t.suggest_categorical('width', [128, 256, 512, 1024])
    return {'model_args': {'hidden': [w] * n, 'dropout': t.suggest_float('dropout', 0.0, 0.3)}, **_common(t)}


def space_cnn1d(t):
    return {'model_args': {'channels': t.suggest_categorical('channels', [64, 128, 256]),
                           'layers': t.suggest_int('layers', 2, 4),
                           'kernel': t.suggest_categorical('kernel', [3, 5]),
                           'dropout': t.suggest_float('dropout', 0.0, 0.3)}, **_common(t)}


def space_rnn(t):
    layers = t.suggest_int('layers', 1, 2)
    dropout = t.suggest_float('dropout', 0.0, 0.3) if layers > 1 else 0.0
    return {'model_args': {'hidden': t.suggest_categorical('hidden', [64, 128, 256]), 'layers': layers,
                           'dropout': dropout}, **_common(t)}


def _attention(t, widths, heads, max_layers):
    d = t.suggest_categorical('d_model', widths)
    return {'d_model': d, 'heads': t.suggest_categorical('heads', heads),
            'layers': t.suggest_int('layers', 1, max_layers), 'ff': d * t.suggest_categorical('ff_mult', [2, 4]), 'dropout': t.suggest_float('dropout', 0.0, 0.3)}


def space_transformer(t):
    return {'model_args': {**_attention(t, [64, 128, 256], [2, 4, 8], 4),
                           'pos': t.suggest_categorical('pos', ['learned', 'sinusoidal']),
                           'readout': t.suggest_categorical('readout', ['last', 'cls'])}, **_common(t)}


def space_itransformer(t):
    return {'model_args': _attention(t, [64, 128, 256], [2, 4, 8], 4), **_common(t)}


def space_patchtst(t):
    # kanal-bağımsız: her örnek 78 ayrı dizi (batch başına 1024 × 78 × 4 yama). Bütçe: d 32, ≤ 2 katman, yama oranı
    # temeldeki 0.5 (0.25 yamaları ikiye katlıyor); genişlik PatchTST makalesindeki aralıkta (16–128)
    return {'model_args': _attention(t, [32], [4, 8], 2), **_common(t)}


def space_tsmixer(t):
    return {'model_args': {'d_model': t.suggest_categorical('d_model', [64, 128, 256, 512]),
                           'layers': t.suggest_int('layers', 1, 4),
                           'dropout': t.suggest_float('dropout', 0.0, 0.3)}, **_common(t)}


def space_timesnet(t):
    d = t.suggest_categorical('d_model', [16])          # bütçe: d 32'de epoch ×2, d 64'te ×4–6.6
    return {'model_args': {'d_model': d, 'ff': d * t.suggest_categorical('ff_mult', [1, 2]),
                           'layers': t.suggest_int('layers', 1, 2), 'top_k': t.suggest_int('top_k', 2, 3),
                           'dropout': t.suggest_float('dropout', 0.0, 0.3)}, **_common(t)}


def space_timemixer(t):
    d = t.suggest_categorical('d_model', [16, 32, 64, 128])
    return {'model_args': {'d_model': d, 'ff': d * t.suggest_categorical('ff_mult', [2, 4]),
                           'layers': t.suggest_int('layers', 1, 3),
                           'moving_avg': t.suggest_categorical('moving_avg', [3, 5, 9]),
                           'dropout': t.suggest_float('dropout', 0.0, 0.3)}, **_common(t)}


def space_cybermetaformer(t):
    # Bileşenler (zaman kolları, host karıştırıcı, hafıza) config'ten ve ablasyondan gelir; burada derinlik ve
    # düzenlileştirme aranır. Genişlik aranmaz: d, B1'in üst sınırına sığan en büyük 8'in katı,
    # böylece her deneme aynı bütçede. Bölüm kipi, k = 101 (`--ks 101`).
    from cybermetaformer.hidden_state.models.cybermetaformer import fit_width
    arch = {'layers': t.suggest_int('layers', 2, 4), 'mlp_ratio': t.suggest_categorical('mlp_ratio', [2, 3, 4]),
            'state': t.suggest_categorical('state', [8, 16, 32]), 'heads': t.suggest_categorical('heads', [2, 4]),
            'kernel': t.suggest_categorical('kernel', [3, 4, 6])}
    base = {k: v for k, v in load_base('cybermetaformer')['model_args'].items() if k != 'd'}
    d = fit_width('b1', **{**base, **arch})
    return {'model_args': {**arch, 'd': d, 'dropout': t.suggest_float('dropout', 0.0, 0.3)}, **_common(t)}


def space_xgb(t):
    return {'params': {'max_depth': t.suggest_int('max_depth', 4, 10),
                       'eta': t.suggest_float('eta', 0.02, 0.3, log=True),
                       'subsample': t.suggest_float('subsample', 0.5, 1.0),
                       'colsample_bytree': t.suggest_float('colsample_bytree', 0.3, 1.0),
                       'min_child_weight': t.suggest_float('min_child_weight', 1.0, 20.0, log=True),
                       'reg_lambda': t.suggest_float('reg_lambda', 0.1, 10.0, log=True)}}


SPACES = {'mlp': space_mlp, 'cnn1d': space_cnn1d, 'lstm': space_rnn, 'gru': space_rnn,
          'transformer': space_transformer, 'patchtst': space_patchtst, 'itransformer': space_itransformer,
          'tsmixer': space_tsmixer, 'timesnet': space_timesnet, 'timemixer': space_timemixer,
          'cybermetaformer': space_cybermetaformer, 'xgb': space_xgb, 'xgb_tf': space_xgb}


def deep_merge(base, over):
    out = dict(base)
    for key, val in over.items():
        out[key] = deep_merge(out[key], val) if isinstance(val, dict) and isinstance(out.get(key), dict) else val
    return out


def load_base(model):
    with open(os.path.join(CONFIGS, f'{model}.yaml')) as f:
        return yaml.safe_load(f)


def overrides_from_params(model, params):
    """En iyi denemenin düz parametreleri → konfigürasyon yaması (aynı uzay fonksiyonuyla)."""
    return SPACES[model](optuna.trial.FixedTrial(params))


def make_objective(model, k, store, device, nthread, smoke):
    base = load_base(model)
    prot = {**PROTOCOL, **(SMOKE if smoke else {})}

    def objective(trial):
        over = SPACES[model](trial)
        cfg = deep_merge(base, over)
        cfg.update(k=k, seed=PROTOCOL['seed'], train_episodes=prot['train_episodes'], train_subsample_seed=0,
                   val_episodes=prot.get('val_episodes'))
        t0 = time.time()
        if model in XGB_MODELS:
            from cybermetaformer.hidden_state.xgb import train_xgb
            cfg['nthread'] = nthread
            if smoke:
                cfg['num_boost_round'] = 20
            _, info = train_xgb(cfg, store, log=lambda s: None)
            trial.set_user_attr('rounds', info['rounds_per_host'])
        else:
            cfg.update(device=device, epochs=prot['epochs'], early_stopping={'patience': PROTOCOL['patience']})

            def prune(epoch, row):
                trial.report(row['val_host_macro_f1'], epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            _, history, info = train(cfg, store, log=lambda s: None, callback=prune)
            trial.set_user_attr('epochs', info['epochs_run'])
            trial.set_user_attr('n_params', info['n_params'])
        trial.set_user_attr('seconds', time.time() - t0)
        return info['best_val_host_macro_f1']
    return objective


def protocol(stage, smoke):
    return {**PROTOCOL, **STAGE_PROTOCOL.get(stage.removesuffix('_smoke'), {}), **(SMOKE if smoke else {})}


def export(study, out_dir, model, k, prot):
    rows = []
    for t in study.trials:
        rows.append({'number': t.number, 'state': t.state.name, 'value': t.value,
                     'seconds': t.user_attrs.get('seconds'), 'epochs': t.user_attrs.get('epochs'),
                     **{f'p_{key}': v for key, v in t.params.items()}})
    keys = list(dict.fromkeys(key for r in rows for key in r))
    with open(os.path.join(out_dir, 'trials.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not done:
        return None
    best = study.best_trial
    out = {'model': model, 'k': k, 'value': best.value, 'trial': best.number, 'params': best.params,
           'overrides': overrides_from_params(model, best.params),
           'protocol': prot,
           'n_complete': len(done), 'n_pruned': sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
           'hours': sum(t.user_attrs.get('seconds', 0) for t in study.trials) / 3600}
    with open(os.path.join(out_dir, 'best.yaml'), 'w') as f:
        yaml.safe_dump(out, f, allow_unicode=True, sort_keys=False)
    return out


def run_study(model, k, store, stage, device, nthread, smoke, log=lambda s: print(s, flush=True)):
    out_dir = os.path.join(RESULTS, stage, 'hpo', f'{model}_k{k}')
    os.makedirs(out_dir, exist_ok=True)
    prot = protocol(stage, smoke)
    study = optuna.create_study(
        study_name=f'{model}_k{k}', storage=f"sqlite:///{os.path.join(out_dir, 'study.db')}", load_if_exists=True,
        direction='maximize', sampler=optuna.samplers.TPESampler(seed=PROTOCOL['sampler_seed']),
        pruner=optuna.pruners.MedianPruner(**PROTOCOL['pruner']))
    finished = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    remaining = prot['n_trials'] - sum(t.state in finished for t in study.trials)

    def report(study, trial):
        log(f"[{model} k={k}] deneme {trial.number}: {trial.state.name} değer {trial.value} "
            f"({trial.user_attrs.get('seconds', 0):.0f} s)")
        export(study, out_dir, model, k, prot)
        release_device_memory()
    if remaining > 0:
        study.optimize(make_objective(model, k, store, device, nthread, smoke), n_trials=remaining,
                       callbacks=[report], catch=(RuntimeError,))
    best = export(study, out_dir, model, k, prot)
    if best:
        log(f"[{model} k={k}] en iyi val host makro-F1 {best['value']:.4f} (deneme {best['trial']}): {best['params']}")
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models', nargs='+', required=True, choices=sorted(SPACES))
    p.add_argument('--ks', type=int, nargs='+', default=PROTOCOL['ks'])
    p.add_argument('--device', default='auto')
    p.add_argument('--nthread', type=int, default=0, help='XGBoost iş parçacığı (0 = hepsi)')
    p.add_argument('--stage', default='asama1')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    stage = args.stage + ('_smoke' if args.smoke else '')
    store = D.EpisodeStore()
    summary = {}
    for model in args.models:
        for k in args.ks:
            best = run_study(model, k, store, stage, args.device, args.nthread, args.smoke)
            summary[f'{model}_k{k}'] = None if best is None else {'value': best['value'], 'params': best['params']}
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()

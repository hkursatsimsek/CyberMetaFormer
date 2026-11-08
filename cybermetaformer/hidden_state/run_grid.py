"""Izgara koşucusu: model × k × seed için eğitim + nihai değerlendirme.

Her koşu `results/hidden_state/<asama>/<model>_k<k>_s<seed>/` altına yazılır (kurallarda seed yok). `metrics.json`
varsa koşu atlanır, yani kesilen ızgara aynı komutla kaldığı yerden sürer. Konfigürasyon = configs/<model>.yaml +
HPO yaması (k ≤ 8 → k=8 araması, k ≥ 16 → k=32 araması; lstm_full/gru_full → lstm/gru). Geçerlilik matrisinin N/A
hücreleri (`models.NOT_APPLICABLE`) planlanmaz.

    # MPS kuyruğu (sinir ağları)
    python -m cybermetaformer.hidden_state.run_grid --models mlp cnn1d lstm gru --device mps
    # CPU kuyruğu (paralel): XGBoost ve kurallar
    python -m cybermetaformer.hidden_state.run_grid --models majority persist xgb_tf xgb --nthread 8
    # Uzatma ızgarası: k=64 ve tam geçmiş (k=101)
    python -m cybermetaformer.hidden_state.run_grid --models mlp cnn1d lstm gru --ks 64 101 --device mps
    python -m cybermetaformer.hidden_state.run_grid --models lstm_full gru_full --ks 101 --device mps
    python -m cybermetaformer.hidden_state.run_grid --models persist xgb_tf --ks 64 101 --nthread 8
    # Aşama 2 (iki şerit aynı anda koşabilir; modeller ayrık, log ayrı: scripts/m2_serit.sh)
    python -m cybermetaformer.hidden_state.run_grid --models transformer --stage asama2 --device mps --log run_grid_A.log
    # Aşama 3 çıtası: tam geçmiş GRU, M1'in HPO sonucuyla, ağırlıkları saklanarak
    python -m cybermetaformer.hidden_state.run_grid --models gru_full --ks 101 --seeds 0 1 2 --stage asama3 --hpo-stage asama1
    # Aşama 3 varyantları (configs/m3_varyantlar.yaml; koşu adı <model>-<varyant>_k101_s<seed>)
    python -m cybermetaformer.hidden_state.run_grid --models cybermetaformer --variants host_yok hafiza_yok --seeds 0 1 2 --stage asama3
    # Durum: python -m cybermetaformer.hidden_state.run_grid --models ... --dry-run
"""
import argparse
import datetime
import os
import time

import optuna
import yaml

from cybermetaformer.hidden_state import data as D
from cybermetaformer.hidden_state.evaluate import final_reports
from cybermetaformer.hidden_state.hpo import deep_merge, load_base, overrides_from_params
from cybermetaformer.hidden_state.models import NOT_APPLICABLE
from cybermetaformer.hidden_state.train import RESULTS, release_device_memory, save_run, torch_predictor, train

CORE_KS = [1, 4, 8, 16, 32]
EXT_KS = [64, 101]                      # uzatma: 101 = tam geçmiş (obs[0..t], T = 100)
SEEDS = [0, 1, 2, 3, 4]
RULES = ['majority', 'persist']
XGB = ['xgb', 'xgb_tf']
TORCH = ['mlp', 'cnn1d', 'lstm', 'gru', 'lstm_full', 'gru_full',
         'transformer', 'patchtst', 'itransformer', 'tsmixer', 'timesnet', 'timemixer', 'cybermetaformer']
HPO_SOURCE = {'lstm_full': 'lstm', 'gru_full': 'gru'}
FULL_ONLY = ['lstm_full', 'gru_full', 'cybermetaformer']   # yalnızca k=101 (bölüm kipi)
VARIANTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'm3_varyantlar.yaml')


def load_variants():
    with open(VARIANTS_FILE) as f:
        return yaml.safe_load(f)


def run_name(model, k, seed, variant=None):
    if model == 'majority':
        return 'majority'
    if model in RULES:
        return f'{model}_k{k}'
    return f"{model}{'-' + variant if variant else ''}_k{k}_s{seed}"


def planned_runs(models, ks, seeds, variants=(None,)):
    out, seen = [], set()
    for variant in variants:
        for model in models:
            for k in ([101] if model in FULL_ONLY else ks):
                if (model, k) in NOT_APPLICABLE:
                    continue
                for seed in (seeds[:1] if model in RULES else seeds):
                    name = run_name(model, k, seed, variant)
                    if name not in seen:
                        seen.add(name)
                        out.append((model, k, seed, name, variant))
    return out


def apply_variant(cfg, model, variant):
    """Varyant yaması (configs/m3_varyantlar.yaml): model_args ve train yamaları; CyberMetaFormer'da genişlik bütçeye
    yeniden sığdırılır (fit, varsayılan b1)."""
    spec = load_variants()[variant] or {}
    unknown = set(spec) - {'model_args', 'train', 'fit'}
    if unknown:
        raise SystemExit(f'{variant}: bilinmeyen anahtar {unknown}')
    cfg = deep_merge(cfg, {'model_args': spec.get('model_args', {}), **spec.get('train', {})})
    budget = spec.get('fit', 'b1')
    if model == 'cybermetaformer' and budget:
        from cybermetaformer.hidden_state.models.cybermetaformer import fit_width
        cfg['model_args']['d'] = fit_width(budget, **{k: v for k, v in cfg['model_args'].items() if k != 'd'})
    cfg['variant'] = {'name': variant, **spec}
    return cfg


def resolve_config(model, k, seed, stage, allow_default, hpo_trial=None):
    cfg = load_base(model)
    src = HPO_SOURCE.get(model, model)
    if hpo_trial:                       # 'çalışma:deneme' (ör. xgb_k8:0): en iyi yerine bu deneme, bütün k'larda aynı
        study_name, number = hpo_trial.split(':')
        db = os.path.join(RESULTS, stage, 'hpo', study_name, 'study.db')
        study = optuna.load_study(study_name=study_name, storage=f'sqlite:///{db}')
        t = next(t for t in study.trials if t.number == int(number))
        assert t.state == optuna.trial.TrialState.COMPLETE, f'{hpo_trial}: {t.state.name}'
        cfg = deep_merge(cfg, overrides_from_params(model, t.params))
        cfg['hpo'] = {'study': study_name, 'trial': t.number, 'val_host_macro_f1': t.value, 'fixed_trial': True}
        cfg.update(k=k, seed=seed)
        return cfg
    hk = 101 if model in FULL_ONLY and model not in HPO_SOURCE else 8 if k <= 8 else 32   # bölüm kipi: kendi HPO'su
    path = os.path.join(RESULTS, stage, 'hpo', f'{src}_k{hk}', 'best.yaml')
    if os.path.exists(path):
        with open(path) as f:
            best = yaml.safe_load(f)
        cfg = deep_merge(cfg, best['overrides'])
        cfg['hpo'] = {'study': f'{src}_k{hk}', 'trial': best['trial'], 'val_host_macro_f1': best['value']}
    elif not allow_default:
        raise SystemExit(f'HPO sonucu yok: {path}\nÖnce hpo.py koşulmalı (ya da --allow-default).')
    cfg.update(k=k, seed=seed)
    return cfg


def attacker_shift_splits(store, train_red):
    """(ii) saldırgan kayması: val yalnızca eğitim Red'i (sıcaklık); test ve ood_test tamamı ile ayrıca
    _in (eğitim Red'i) ve _shift (öbür Red) parçaları."""
    red = store.meta['red_policies'].index(train_red)
    parts = {'val': ('val', 'in'), 'test': ('test', None), 'ood_test': ('ood_test', None),
             'test_in': ('test', 'in'), 'test_shift': ('test', 'shift'),
             'ood_test_in': ('ood_test', 'in'), 'ood_test_shift': ('ood_test', 'shift')}

    def resolve(name):
        base, part = parts[name]
        idx = store.split(base)
        if part is None:
            return idx
        same = store.red_policy[idx] == red
        return idx[same if part == 'in' else ~same]
    return tuple(parts), resolve


def run_one(model, k, seed, store, args, log, variant=None):
    stage_dir = os.path.join(RESULTS, args.stage)
    history, save_probs = None, seed == 0 and not args.smoke
    if model in RULES:
        from cybermetaformer.hidden_state.rules import rule_predictor
        cfg = {'model': model, 'k': k}
        predict = rule_predictor(model, k, store, store.split('train'))
        info = {'device': 'cpu', 'train_seconds': 0.0}
        save_probs = False
    else:
        cfg = resolve_config(model, k, seed, args.hpo_stage or args.stage.removesuffix('_smoke'),
                             args.allow_default or args.smoke, args.hpo_trial)
        if variant:
            cfg = apply_variant(cfg, model, variant)
        if model in XGB:
            from cybermetaformer.hidden_state.xgb import train_xgb, xgb_predictor
            cfg['nthread'] = args.nthread
            if args.smoke:
                cfg.update(train_episodes=200, val_episodes=100, num_boost_round=20)
            xmodel, info = train_xgb(cfg, store, log=log)
            predict = xgb_predictor(xmodel, store)
            if seed == 0:                                           # açıklama val üzerinde (test'e dokunmadan)
                va = store.split('val')[:200 if args.smoke else None]
                info['shap'] = xmodel.shap_summary(store, va, store.meta['hosts'], rows=2000 if args.smoke else 20000)
        elif model in TORCH:
            cfg['device'] = args.device
            if args.smoke:
                cfg.update(train_episodes=200, epochs=1, val_episodes=200)
            tmodel, history, info = train(cfg, store, log=log)
            predict = torch_predictor(tmodel, cfg, store, info['device'])
        else:
            raise SystemExit(f'Bilinmeyen model: {model}')
    t0 = time.time()
    splits, resolve = ('val', 'test', 'ood_test'), None
    if model not in RULES and cfg.get('train_red'):                  # (ii): seçim eğitim Red'inin val'i ile
        splits, resolve = attacker_shift_splits(store, cfg['train_red'])
    reports, outs = final_reports(store, predict, splits=splits, max_episodes=200 if args.smoke else None, log=log,
                                  resolve=resolve)
    info['eval_seconds'] = time.time() - t0
    save_run(os.path.join(stage_dir, run_name(model, k, seed, variant)), cfg, history, info, reports, outs, save_probs,
             model=tmodel if model in TORCH else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models', nargs='+', required=True, choices=RULES + XGB + TORCH)
    p.add_argument('--ks', type=int, nargs='+', default=CORE_KS)
    p.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    p.add_argument('--device', default='auto')
    p.add_argument('--nthread', type=int, default=0, help='XGBoost iş parçacığı (0 = hepsi)')
    p.add_argument('--stage', default='asama1')
    p.add_argument('--allow-default', action='store_true', help='HPO sonucu yoksa temel konfigürasyonu kullan')
    p.add_argument('--variants', nargs='+', help='configs/m3_varyantlar.yaml adları (Aşama 3)')
    p.add_argument('--hpo-stage', help='HPO sonucunun okunacağı aşama (varsayılan --stage; ör. M3 çıtası: asama1)')
    p.add_argument('--hpo-trial', help="'çalışma:deneme' (ör. xgb_k8:0): best.yaml yerine bu denemenin konfigürasyonu")
    p.add_argument('--smoke', action='store_true', help='200 bölüm, 1 epoch; <asama>_smoke klasörüne')
    p.add_argument('--dry-run', action='store_true', help='yalnızca planı ve durumu yazdır')
    p.add_argument('--log', default='run_grid.log', help='aşama klasöründeki log dosyası (paralel şeritlerde ayrı)')
    args = p.parse_args()
    if args.smoke:
        args.stage += '_smoke'
    stage_dir = os.path.join(RESULTS, args.stage)
    os.makedirs(stage_dir, exist_ok=True)
    log_path = os.path.join(stage_dir, args.log)

    def log(msg):
        line = f"{datetime.datetime.now():%m-%d %H:%M:%S} {msg}"
        print(line, flush=True)
        with open(log_path, 'a') as f:
            f.write(line + '\n')

    if args.variants:
        missing = set(args.variants) - set(load_variants())
        if missing:
            raise SystemExit(f'Bilinmeyen varyant: {sorted(missing)}')
    runs = planned_runs(args.models, args.ks, args.seeds, args.variants or (None,))
    todo = [r for r in runs if not os.path.exists(os.path.join(stage_dir, r[3], 'metrics.json'))]
    print(f'{len(runs)} koşu planlandı, {len(runs) - len(todo)} tamam, {len(todo)} kaldı.')
    for model, k in sorted(NOT_APPLICABLE):
        if model in args.models and k in args.ks:
            print(f'  N/A: {model} k={k} ({NOT_APPLICABLE[model, k]})')
    if args.dry_run:
        for r in todo:
            print('  bekliyor:', r[3])
        return
    store = D.EpisodeStore()
    for i, (model, k, seed, name, variant) in enumerate(todo):
        log(f'── [{i + 1}/{len(todo)}] {name} başladı')
        t0 = time.time()
        run_one(model, k, seed, store, args, log, variant)
        release_device_memory()
        log(f'── {name} bitti ({(time.time() - t0) / 60:.1f} dk)')


if __name__ == '__main__':
    main()

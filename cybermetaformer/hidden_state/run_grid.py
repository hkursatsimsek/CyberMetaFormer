"""Grid runner: training + final evaluation for model × k × seed.

Each run is written under `results/hidden_state/<asama>/<model>_k<k>_s<seed>/` (rules have no seed). If `metrics.json`
exists, the run is skipped, so an interrupted grid resumes with the same command. Configuration = configs/<model>.yaml +
HPO patch (k ≤ 8 → k=8 search, k ≥ 16 → k=32 search; lstm_full/gru_full → lstm/gru). N/A cells of the validity
matrix (`models.NOT_APPLICABLE`) are not planned.

    # MPS queue (neural networks)
    python -m cybermetaformer.hidden_state.run_grid --models mlp cnn1d lstm gru --device mps
    # CPU queue (parallel): XGBoost and rules
    python -m cybermetaformer.hidden_state.run_grid --models majority persist xgb_tf xgb --nthread 8
    # Extension grid: k=64 and full history (k=101)
    python -m cybermetaformer.hidden_state.run_grid --models mlp cnn1d lstm gru --ks 64 101 --device mps
    python -m cybermetaformer.hidden_state.run_grid --models lstm_full gru_full --ks 101 --device mps
    python -m cybermetaformer.hidden_state.run_grid --models persist xgb_tf --ks 64 101 --nthread 8
    # Stage 2 (two lanes can run at once; disjoint models, separate logs: scripts/m2_serit.sh)
    python -m cybermetaformer.hidden_state.run_grid --models transformer --stage asama2 --device mps --log run_grid_A.log
    # Stage 3 baseline: full-history GRU, with the M1 HPO result, keeping the weights
    python -m cybermetaformer.hidden_state.run_grid --models gru_full --ks 101 --seeds 0 1 2 --stage asama3 --hpo-stage asama1
    # Stage 3 variants (configs/m3_variants.yaml; run name <model>-<variant>_k101_s<seed>)
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
EXT_KS = [64, 101]                      # extension: 101 = full history (obs[0..t], T = 100)
SEEDS = [0, 1, 2, 3, 4]
RULES = ['majority', 'persist']
XGB = ['xgb', 'xgb_tf']
TORCH = ['mlp', 'cnn1d', 'lstm', 'gru', 'lstm_full', 'gru_full',
         'transformer', 'patchtst', 'itransformer', 'tsmixer', 'timesnet', 'timemixer', 'cybermetaformer']
HPO_SOURCE = {'lstm_full': 'lstm', 'gru_full': 'gru'}
FULL_ONLY = ['lstm_full', 'gru_full', 'cybermetaformer']   # k=101 only (episode mode)
VARIANTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'm3_variants.yaml')


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
    """Variant patch (configs/m3_variants.yaml): model_args and train patches; for CyberMetaFormer the width is refitted
    to the budget (fit, default b1)."""
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
    if hpo_trial:                       # 'study:trial' (e.g. xgb_k8:0): this trial instead of the best, same for all k
        study_name, number = hpo_trial.split(':')
        db = os.path.join(RESULTS, stage, 'hpo', study_name, 'study.db')
        study = optuna.load_study(study_name=study_name, storage=f'sqlite:///{db}')
        t = next(t for t in study.trials if t.number == int(number))
        assert t.state == optuna.trial.TrialState.COMPLETE, f'{hpo_trial}: {t.state.name}'
        cfg = deep_merge(cfg, overrides_from_params(model, t.params))
        cfg['hpo'] = {'study': study_name, 'trial': t.number, 'val_host_macro_f1': t.value, 'fixed_trial': True}
        cfg.update(k=k, seed=seed)
        return cfg
    hk = 101 if model in FULL_ONLY and model not in HPO_SOURCE else 8 if k <= 8 else 32   # episode mode: its own HPO
    path = os.path.join(RESULTS, stage, 'hpo', f'{src}_k{hk}', 'best.yaml')
    if os.path.exists(path):
        with open(path) as f:
            best = yaml.safe_load(f)
        cfg = deep_merge(cfg, best['overrides'])
        cfg['hpo'] = {'study': f'{src}_k{hk}', 'trial': best['trial'], 'val_host_macro_f1': best['value']}
    elif not allow_default:
        raise SystemExit(f'No HPO result: {path}\nRun hpo.py first (or use --allow-default).')
    cfg.update(k=k, seed=seed)
    return cfg


def attacker_shift_splits(store, train_red):
    """(ii) attacker shift: val only with the training Red (temperature); test and ood_test in full and also
    as _in (training Red) and _shift (the other Red) parts."""
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
            if seed == 0:                                           # explanation on val (without touching test)
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
    if model not in RULES and cfg.get('train_red'):                  # (ii): selection on the val of the training Red
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
    p.add_argument('--nthread', type=int, default=0, help='XGBoost threads (0 = all)')
    p.add_argument('--stage', default='asama1')
    p.add_argument('--allow-default', action='store_true', help='use the base configuration if there is no HPO result')
    p.add_argument('--variants', nargs='+', help='names in configs/m3_variants.yaml (Stage 3)')
    p.add_argument('--hpo-stage', help='stage to read the HPO result from (default --stage; e.g. M3 baseline: asama1)')
    p.add_argument('--hpo-trial', help="'study:trial' (e.g. xgb_k8:0): configuration of this trial instead of best.yaml")
    p.add_argument('--smoke', action='store_true', help='200 episodes, 1 epoch; into the <asama>_smoke folder')
    p.add_argument('--dry-run', action='store_true', help='print only the plan and the status')
    p.add_argument('--log', default='run_grid.log', help='log file in the stage folder (separate for parallel lanes)')
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
    print(f'{len(runs)} runs planned, {len(runs) - len(todo)} done, {len(todo)} left.')
    for model, k in sorted(NOT_APPLICABLE):
        if model in args.models and k in args.ks:
            print(f'  N/A: {model} k={k} ({NOT_APPLICABLE[model, k]})')
    if args.dry_run:
        for r in todo:
            print('  bekliyor:', r[3])
        return
    store = D.EpisodeStore()
    for i, (model, k, seed, name, variant) in enumerate(todo):
        log(f'── [{i + 1}/{len(todo)}] {name} started')
        t0 = time.time()
        run_one(model, k, seed, store, args, log, variant)
        release_device_memory()
        log(f'── {name} bitti ({(time.time() - t0) / 60:.1f} dk)')


if __name__ == '__main__':
    main()

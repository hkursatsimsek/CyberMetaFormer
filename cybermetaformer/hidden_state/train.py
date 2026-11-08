"""Tek eğitici. Cihaz: cuda / mps / cpu.

    venv_cyborg/bin/python -m cybermetaformer.hidden_state.train --config cybermetaformer/hidden_state/configs/mlp.yaml --k 8 --seed 0
    venv_cyborg/bin/python -m cybermetaformer.hidden_state.train --config ... --smoke      # 200 bölüm, 1 epoch

Model seçimi yalnızca val ile yapılır; ood_test eğitimde hiç açılmaz. `--final` verilirse eğitimden sonra val
(sıcaklık), test ve (`store.final_evaluation()` içinde) ood_test değerlendirilir. Izgara koşuları için `run_grid.py`.

İki eğitim kipi: `mode: window` (son k gözlem → y[t], varsayılan) ve `mode: episode` (tam geçmiş, bölüm boyunca
çoktan-çoğa; `lstm_full`, `gru_full`; batch = `episodes_per_batch` bölüm).

M3 anahtarları (yalnızca bölüm kipi; kapalıyken eğitim yolu ve RNG akışı aynı):
- `train_red: bline|meander`: (ii) saldırgan kayması; train ve val (seçim) yalnızca bu Red'in bölümleri.
- `dg: {method: groupdro|vrex|coral|dann, ...}`, `augment: {alert_drop: p, time_mask: p}`: `dg.py`.
- `model_args.action_input: true`: önceki Blue aksiyonu girdi (aksiyon ablasyonu).

`compile: true` (M2 modelleri): pencere kipinde eğitim ileri geçişi `torch.compile` ile, değerlendirme eager. Model,
veri ve optimizasyon aynı; sonuç kendi içinde belirlenimci, eager'dan farkı dropout'un rastgele akışı (README
Kararlar 2026-10-19).
"""
import argparse
import copy
import csv
import gc
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as nnf
import yaml

from cybermetaformer.hidden_state import data as D
from cybermetaformer.hidden_state import metrics as M
from cybermetaformer.hidden_state.evaluate import final_reports, predict_logits
from cybermetaformer.hidden_state.models import build_model

RESULTS = os.path.join(D.ROOT, 'results', 'hidden_state')

DEFAULTS = {
    'model': 'mlp', 'model_args': {}, 'k': 8, 'scale': 'global',
    'optimizer': 'adamw', 'lr': 1e-3, 'weight_decay': 0.01, 'betas': [0.9, 0.999],
    'epochs': 30, 'batch_size': 1024, 'schedule': 'warmup_cosine', 'warmup_frac': 0.05,
    'train_episodes': None, 'train_subsample_seed': 0, 'steps_per_episode': None,
    'early_stopping': {'patience': 5}, 'val_episodes': None,
    'loss': 'ce', 'class_weights': None, 'focal_gamma': 2.0, 'grad_clip': None,
    'seed': 0, 'device': 'auto', 'eval_batch_size': 65536,
    'mode': 'window', 'episodes_per_batch': 16, 'compile': False,
    'train_red': None, 'dg': None, 'augment': None,
}


def resolve_device(name):
    if name != 'auto':
        return name
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def release_device_memory():
    """Ayırıcı önbelleğini boşaltır. Koşular/HPO denemeleri arasında çağrılır: MPS önbelleği farklı boyutlu modeller
    arasında geri verilmeden büyür (LSTM k=32 HPO'sunda ~46 GB → OOM, 2026-10-16). Su işaretleri bunu önlemiyor."""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def make_optimizer(cfg, params):
    if cfg['optimizer'] == 'adam':
        return torch.optim.Adam(params, lr=cfg['lr'], betas=tuple(cfg['betas']))
    if cfg['optimizer'] == 'adamw':
        return torch.optim.AdamW(params, lr=cfg['lr'], betas=tuple(cfg['betas']), weight_decay=cfg['weight_decay'])
    raise ValueError(cfg['optimizer'])


def make_schedule(cfg, opt, total_steps):
    if cfg['schedule'] == 'none':
        return None
    warm = max(1, int(cfg['warmup_frac'] * total_steps))

    def f(step):
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(1, total_steps - warm))))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def class_weights(store, idx, mode, device):
    if mode is None:
        return None
    if mode != 'balanced':
        raise ValueError(mode)
    counts = np.bincount(store.labels[idx].reshape(-1), minlength=D.N_CLASSES).astype(np.float64)
    w = counts.sum() / (D.N_CLASSES * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32, device=device)


def loss_fn(cfg, logits, y, weight):
    z, t = logits.reshape(-1, D.N_CLASSES), y.reshape(-1)
    if cfg['loss'] == 'ce':
        return nnf.cross_entropy(z, t, weight=weight)
    if cfg['loss'] == 'focal':
        logp = nnf.log_softmax(z, -1).gather(1, t[:, None])[:, 0]
        w = 1.0 if weight is None else weight[t]
        return (-(1 - logp.exp()) ** cfg['focal_gamma'] * logp * w).mean()
    raise ValueError(cfg['loss'])


def _episode_batches(n_ep, per_batch, generator=None):
    order = torch.randperm(n_ep, generator=generator)
    for i in range(0, n_ep, per_batch):
        yield order[i:i + per_batch]


def train(cfg, store, log=print, callback=None):
    """Modeli eğitir. cfg['seed'] None ise global RNG yeniden tohumlanmaz (orijinal protokol zinciri için).

    callback(epoch, row): her epoch'un val değerlendirmesinden sonra çağrılır (HPO budaması; istisna fırlatabilir)."""
    cfg = {**DEFAULTS, **cfg}
    device = resolve_device(cfg['device'])
    if cfg['seed'] is not None:
        torch.manual_seed(cfg['seed'])
        np.random.seed(cfg['seed'])

    tr_idx = D.subsample_episodes(store.split('train'), cfg['train_episodes'], cfg['train_subsample_seed'])
    red = None if cfg['train_red'] is None else store.meta['red_policies'].index(cfg['train_red'])
    if red is not None:
        tr_idx = tr_idx[store.red_policy[tr_idx] == red]
    store.assert_no_ood(tr_idx)
    train_wd = D.WindowData(store, tr_idx, device=device)
    use_val = cfg['early_stopping'] is not None
    if use_val:
        va_idx = D.subsample_episodes(store.split('val'), cfg['val_episodes'], 1)
        if red is not None:
            va_idx = va_idx[store.red_policy[va_idx] == red]
        store.assert_no_ood(va_idx)
        val_wd = D.WindowData(store, va_idx, device=device)
        y_val = store.labels[va_idx].reshape(-1, len(D.LABEL_HOSTS))

    model = build_model(cfg['model'], cfg['k'], **cfg['model_args']).to(device)
    episode_mode = cfg['mode'] == 'episode'
    if episode_mode != getattr(model, 'episode_mode', False):
        raise ValueError(f"{cfg['model']}: mode={cfg['mode']} modelle uyumsuz")
    if (cfg['dg'] or cfg['augment']) and not episode_mode:
        raise ValueError('dg / augment yalnızca bölüm kipinde')
    use_act = getattr(model, 'action_input', False)
    dgm, dom = None, None
    if cfg['dg']:
        from cybermetaformer.hidden_state.dg import DomainGeneralization, domain_ids
        dom_np, dom_names = domain_ids(store, tr_idx)
        dom = torch.as_tensor(dom_np, device=device)
        dgm = DomainGeneralization(cfg['dg'], len(dom_names), cfg['model_args'].get('d', 64)).to(device)
    opt = make_optimizer(cfg, list(model.parameters()) + ([] if dgm is None else list(dgm.parameters())))
    fwd = model
    if cfg['compile'] and not episode_mode:
        torch._dynamo.reset()          # HPO ve ızgara aynı süreçte onlarca model derler; önbellek sınırına takılmasın
        fwd = torch.compile(model, dynamic=False)
    if episode_mode:
        steps_per_epoch = math.ceil(train_wd.n / cfg['episodes_per_batch'])
    else:
        spe = cfg['steps_per_episode'] or train_wd.T
        steps_per_epoch = math.ceil(train_wd.n * min(spe, train_wd.T) / cfg['batch_size'])
    sched = make_schedule(cfg, opt, steps_per_epoch * cfg['epochs'])
    weight = class_weights(store, tr_idx, cfg['class_weights'], device)

    history, best, best_state, bad = [], -1.0, None, 0
    for epoch in range(cfg['epochs']):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        if episode_mode:
            batches = ((ep, None) for ep in _episode_batches(train_wd.n, cfg['episodes_per_batch']))
        else:
            batches = D.batch_indices(train_wd.n, train_wd.T, cfg['batch_size'],
                                      steps_per_episode=cfg['steps_per_episode'])
        for ep, t in batches:
            opt.zero_grad()
            if episode_mode:
                obs, y = train_wd.episodes(ep)
                x = D.scale_obs(obs, cfg['scale'])
                if cfg['augment']:
                    from cybermetaformer.hidden_state.dg import augment
                    x = augment(x, cfg['augment'])
                kw = {'act': train_wd.prev_actions(ep)} if use_act else {}
                if dgm is not None and dgm.needs_features:
                    out, feats = model.forward_episode(x, return_features=True, **kw)
                else:
                    out, feats = model.forward_episode(x, **kw), None
                out = out[:, 1:]
            else:
                x, mask = train_wd.windows(ep, t, cfg['k'])
                y = train_wd.targets(ep, t)
                out = fwd(D.scale_obs(x, cfg['scale']), mask)
            if dgm is not None:
                progress = (epoch * steps_per_epoch + nb) / (steps_per_epoch * cfg['epochs'])
                loss = dgm(out, feats, y, dom[ep.to(device)], weight, progress)
            else:
                loss = loss_fn(cfg, out, y, weight)
            loss.backward()
            if cfg['grad_clip']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])   # DANN ayırt edicisi kırpılmaz
            opt.step()
            if sched is not None:
                sched.step()
            tot, nb = tot + loss.item(), nb + 1
        row = {'epoch': epoch + 1, 'loss': tot / nb, 'epoch_seconds': time.time() - t0}
        if use_val:
            pred = predict_logits(model, val_wd, cfg['k'], cfg['scale'], cfg['eval_batch_size']).argmax(-1)
            row['val_host_macro_f1'] = M.host_macro_f1(y_val, pred.reshape(-1, len(D.LABEL_HOSTS)))
            if row['val_host_macro_f1'] > best:
                best, best_state, bad = row['val_host_macro_f1'], copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
        history.append(row)
        log(' '.join(f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}' for k, v in row.items()))
        if callback is not None:
            callback(epoch, row)
        if use_val and bad >= cfg['early_stopping']['patience']:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    info = {'device': device, 'train_episodes_hash': D.episode_hash(tr_idx), 'n_train_episodes': len(tr_idx),
            'n_val_episodes': len(va_idx) if use_val else None,
            'n_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'best_val_host_macro_f1': best if use_val else None,
            'epochs_run': len(history), 'train_seconds': sum(r['epoch_seconds'] for r in history)}
    return model, history, info


def torch_predictor(model, cfg, store, device):
    """final_reports için predict(idx) → (logits, pred)."""
    full = {**DEFAULTS, **cfg}

    def predict(idx):
        logits = predict_logits(model, D.WindowData(store, idx, device=device), full['k'], full['scale'],
                                full['eval_batch_size'])
        return logits, logits.argmax(-1)
    return predict


def save_run(run_dir, cfg, history, info, reports, outs=None, save_probs=False, model=None):
    """config.yaml, metrics.json, log.csv; episode_confusion.npz (uint8, git dışı); episode_slices.npz (bölüm başına
    sessiz dilim sayımları, eşli bölüm bootstrap'ı için; git dışı); save_probs: test/ood olasılıkları float16 (git dışı;
    protokol gereği yalnızca seed 0); model: ağırlıklar `model.pt` (state_dict, CPU; git dışı; M2'den beri, dikkat ve
    IG analizleri ile sonradan eklenen metrikler yeniden eğitim istemesin)."""
    os.makedirs(run_dir, exist_ok=True)
    if model is not None:
        torch.save({key: v.detach().cpu() for key, v in model.state_dict().items()}, os.path.join(run_dir, 'model.pt'))
    with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    if history:
        with open(os.path.join(run_dir, 'log.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(history[-1]))
            w.writeheader()
            w.writerows(history)
    if outs:
        np.savez_compressed(os.path.join(run_dir, 'episode_confusion.npz'),
                            **{s: o['ep_conf'].astype(np.uint8) for s, o in outs.items()})
        sl = {s: o['ep_slices'] for s, o in outs.items() if o.get('ep_slices') is not None}
        if sl:
            np.savez_compressed(os.path.join(run_dir, 'episode_slices.npz'), names=np.array(next(iter(sl.values()))[1]),
                                **{s: c.astype(np.uint16) for s, (c, _) in sl.items()})
        if save_probs:
            T = reports.get('temperature') or 1.0
            np.savez_compressed(os.path.join(run_dir, 'probs.npz'),
                                **{s: M.softmax(o['logits'], T).astype(np.float16)
                                   for s, o in outs.items() if s != 'val' and o['logits'] is not None})
    with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:          # en son: varlığı "koşu tamam" demek
        json.dump({'info': info, 'reports': reports}, f, indent=2, ensure_ascii=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--k', type=int)
    p.add_argument('--seed', type=int)
    p.add_argument('--device')
    p.add_argument('--stage', default='asama1')
    p.add_argument('--final', action='store_true', help='eğitimden sonra test + ood_test değerlendir')
    p.add_argument('--smoke', action='store_true', help='200 bölüm, 1 epoch, val 200 bölüm')
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for key in ['k', 'seed', 'device']:
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    if args.smoke:
        cfg.update(train_episodes=200, epochs=1, val_episodes=200)
    store = D.EpisodeStore()
    model, history, info = train(cfg, store)
    full = {**DEFAULTS, **cfg}
    reports, outs = {}, None
    if args.final:
        reports, outs = final_reports(store, torch_predictor(model, cfg, store, info['device']),
                                      max_episodes=200 if args.smoke else None)
    name = f"{full['model']}_k{full['k']}_s{full['seed']}" + ('_smoke' if args.smoke else '')
    run_dir = os.path.join(RESULTS, args.stage, name)
    save_run(run_dir, full, history, info, reports, outs, model=model)
    print(f'Kaydedildi: {run_dir}')


if __name__ == '__main__':
    main()

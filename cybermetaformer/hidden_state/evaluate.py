"""Model → predictions and metrics (episode layout).

    rep, out = evaluate_split(model, store, 'test', k=8, scale='global')
    # OOD only in the final evaluation:
    with store.final_evaluation():
        rep, out = evaluate_split(model, store, 'ood_test', k=8)

Final evaluation independent of the model family (torch, XGBoost, rules):

    reports, outs = final_reports(store, predict)     # predict(idx) → (logits (E,T,12,3) or None, pred (E,T,12))

The temperature is found on val; raw and scaled calibration are reported for test and ood_test.
"""
import numpy as np
import torch

from cybermetaformer.hidden_state import data as D
from cybermetaformer.hidden_state import features as F
from cybermetaformer.hidden_state import metrics as M

TEMP_FIT_ROWS = 500_000          # number of random host-steps from val for the temperature search (speed)


@torch.no_grad()
def predict_logits(model, wd, k, scale='global', batch_size=65536):
    """WindowData → logits (n, T, 12, 3) float32 numpy, in episode-major order.

    Episode-mode models (`episode_mode = True`, full history) process the episode in one pass; position t estimates y[t]."""
    model.eval()
    out = []
    if getattr(model, 'episode_mode', False):
        n_ep = max(1, batch_size // (wd.T + 1))
        for i in range(0, wd.n, n_ep):
            ep = torch.arange(i, min(i + n_ep, wd.n))
            obs, _ = wd.episodes(ep)
            kw = {'act': wd.prev_actions(ep)} if getattr(model, 'action_input', False) else {}
            out.append(model.forward_episode(D.scale_obs(obs, scale), **kw)[:, 1:].float().cpu())
        return torch.cat(out).numpy()
    batch_size = min(batch_size, max(2048, batch_size * 8 // max(k, 8)))    # so that intermediate tensors fit at large k
    for ep, t in D.eval_indices(wd.n, wd.T, batch_size):
        x, mask = wd.windows(ep, t, k)
        out.append(model(D.scale_obs(x, scale), mask).float().cpu())
    return torch.cat(out).view(wd.n, wd.T, len(D.LABEL_HOSTS), D.N_CLASSES).numpy()


def split_report(store, idx, pred, logits=None, temperature=None, full=True, bootstrap=False):
    """Report for an episode set. Returns: (report, episode confusion (E, 12, 3, 3), silent slice counts).

    Counts only with full=True: (episode × slice × [n, hit], slice names), M.episode_slice_counts."""
    y = store.labels[idx].astype(np.int64)
    slices = None
    if full:
        alert = F.episode_alerts(store.obs[idx])
        age = F.alert_age(alert)
        cause = None if store.cause is None else store.cause[idx]
        rep = M.full_report(y, pred, alert, age, logits, cause, store.cause_names, temperature)
        slices = M.episode_slice_counts(y, pred, alert, age, cause, store.cause_names)
        rep['policy'] = M.policy_breakdown(
            y, pred, store.blue_policy[idx], store.red_policy[idx],
            store.meta['blue_policies'], store.meta['red_policies'])
    else:
        rep = {'host_macro_f1': M.host_macro_f1(y.reshape(-1, 12), pred.reshape(-1, 12)),
               'breach_f1': M.breach_f1(y.reshape(-1, 12), pred.reshape(-1, 12))}
    ep_conf = M.episode_confusion(y, pred)
    if bootstrap:
        rep['bootstrap'] = M.bootstrap_ci(ep_conf)
    rep['n_episodes'] = int(len(idx))
    rep['episodes_hash'] = D.episode_hash(idx)
    return rep, ep_conf, slices


def evaluate_split(model, store, split, k, scale='global', device='cpu', batch_size=65536, full=True):
    """Prediction + metric report on a split. Returns: (report, {'logits', 'pred', 'y', 'ep_conf', 'ep_slices'})."""
    idx = store.split(split)
    wd = D.WindowData(store, idx, device=device)
    logits = predict_logits(model, wd, k, scale, batch_size)
    pred = logits.argmax(-1)
    rep, ep_conf, slices = split_report(store, idx, pred, logits, full=full)
    return rep, {'logits': logits, 'pred': pred, 'y': store.labels[idx].astype(np.int64), 'ep_conf': ep_conf,
                 'ep_slices': slices}


def fit_temperature(logits, y, rows=TEMP_FIT_ROWS, seed=0):
    L = np.asarray(logits).reshape(-1, D.N_CLASSES)
    Y = np.asarray(y).reshape(-1)
    if len(Y) > rows:
        pick = np.random.default_rng(seed).choice(len(Y), rows, replace=False)
        L, Y = L[pick], Y[pick]
    return M.fit_temperature(L, Y)


def final_reports(store, predict, splits=('val', 'test', 'ood_test'), max_episodes=None, log=print, resolve=None):
    """Final evaluation: val (temperature + selection metrics), test and ood_test (with bootstrap CIs).

    predict(idx) → (logits or None, pred). max_episodes: upper bound per split, for smoke only.
    resolve(name) → episode indices: extra or narrowed sets ((ii) attacker shift; called while the OOD lock is open).
    Returns: (reports {'val', 'test', 'ood_test', 'temperature'}, outputs {split: {'logits', 'ep_conf', 'ep_slices'}})."""
    reports, outs, temp = {}, {}, None
    with store.final_evaluation():
        for split in splits:
            idx = store.split(split) if resolve is None else resolve(split)
            if max_episodes:
                idx = idx[:max_episodes]
            logits, pred = predict(idx)
            if split == 'val' and logits is not None:
                temp = fit_temperature(logits, store.labels[idx])
            rep, ep_conf, slices = split_report(store, idx, pred, logits, temp, bootstrap=split != 'val')
            reports[split], outs[split] = rep, {'logits': logits, 'ep_conf': ep_conf, 'ep_slices': slices}
            log(f"{split:8s} host makro-F1 {rep['host_macro_f1']:.4f}  ihlal F1 {rep['breach_f1']:.4f}  "
                f"sessiz recall {rep['silent_recall']:.4f}")
    reports['temperature'] = temp
    return reports, outs

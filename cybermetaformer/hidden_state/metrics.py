"""Shared metrics. numpy only; independent of CybORG (T7).

Array layouts:
- flat: y, p (N, H) int, H = 12 hosts, classes {0 clean, 1 user, 2 root}
- episode: y, p (E, T, H); probabilities (E, T, H, 3)
`host_macro_f1` is exactly the definition in baselines.py: macro-F1 per host over the classes present (tp+fn > 0),
then the mean over hosts with a positive label (effectively 11 hosts, since def is never compromised).
"""
import numpy as np

N_CLASSES = 3

ALERT_AGE_BINS = [            # (name, low, high) — steps since the last alert in silent compromises
    ('never', -1, -1),
    ('1–3', 1, 3),
    ('4–7', 4, 7),
    ('8–15', 8, 15),
    ('16–31', 16, 31),
    ('32–63', 32, 63),
    ('64+', 64, 10**9),
]


# ── confusion-matrix-based F1 ───────────────────────────────────────────

def confusion(y, p, mask=None):
    """(N, H) → (H, 3, 3) counts [true, predicted]. mask (N, H) bool: only the selected host-steps."""
    y, p = np.asarray(y, np.int64), np.asarray(p, np.int64)
    H = y.shape[-1]
    code = (y * N_CLASSES + p).reshape(-1, H)
    m = None if mask is None else np.asarray(mask, bool).reshape(-1, H)
    out = np.zeros((H, N_CLASSES, N_CLASSES), np.int64)
    for h in range(H):
        c = code[:, h] if m is None else code[m[:, h], h]
        out[h] = np.bincount(c, minlength=N_CLASSES ** 2).reshape(N_CLASSES, N_CLASSES)
    return out


def _f1_classes(C, classes):
    f1s = []
    for c in classes:
        tp = C[c, c]
        fn, fp = C[c].sum() - tp, C[:, c].sum() - tp
        if tp + fn > 0:
            f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    return f1s


def macro_f1_from_confusion(C):
    """Single host (3, 3) → macro-F1 (same as baselines.macro_f1(y, p, [0, 1, 2]))."""
    return float(np.mean(_f1_classes(C, range(N_CLASSES))))


def host_macro_f1_from_confusion(C, return_per_host=False):
    """(H, 3, 3) → mean host macro-F1. Hosts without a positive (user/root) true label are skipped."""
    hosts = [h for h in range(C.shape[0]) if C[h, 1:].sum() > 0]
    per = {h: macro_f1_from_confusion(C[h]) for h in hosts}
    val = float(np.mean([per[h] for h in hosts])) if hosts else float('nan')
    return (val, per) if return_per_host else val


def host_macro_f1(y, p, mask=None, return_per_host=False):
    return host_macro_f1_from_confusion(confusion(y, p, mask), return_per_host)


def binary_f1(y, p):
    """Binary F1 (positive class 1) = baselines.macro_f1(y, p, [1])."""
    y, p = np.asarray(y).astype(bool), np.asarray(p).astype(bool)
    tp, fp, fn = (p & y).sum(), (p & ~y).sum(), (~p & y).sum()
    return float(2 * tp / max(2 * tp + fp + fn, 1)) if tp + fn > 0 else float('nan')


def breach_f1(y, p):
    """Network level: is there a compromise on at least one host."""
    y, p = np.asarray(y).reshape(-1, np.shape(y)[-1]), np.asarray(p).reshape(-1, np.shape(p)[-1])
    return binary_f1((y > 0).any(1), (p > 0).any(1))


# ── silent-compromise metrics ───────────────────────────────────────────

def silent_metrics(y, p, alert):
    """alert: (…, H) bool, the alert in the observation of the target step (features.alerts, 'export').

    - silent_recall: P(prediction > 0) on compromised, alert-free host-steps
    - silent_exact: P(prediction = true class) on the same subset
    - noalert_macro_f1: host macro-F1 on all alert-free host-steps (including clean ones)
    - alert_recall: P(prediction > 0) on compromised host-steps with an alert (for comparison)
    """
    y, p, a = np.asarray(y), np.asarray(p), np.asarray(alert, bool)
    H = y.shape[-1]
    y2, p2, a2 = y.reshape(-1, H), p.reshape(-1, H), a.reshape(-1, H)
    silent = (y2 > 0) & ~a2
    loud = (y2 > 0) & a2
    return {
        'silent_recall': float((p2[silent] > 0).mean()) if silent.any() else float('nan'),
        'silent_exact': float((p2[silent] == y2[silent]).mean()) if silent.any() else float('nan'),
        'noalert_macro_f1': host_macro_f1(y2, p2, mask=~a2),
        'alert_recall': float((p2[loud] > 0).mean()) if loud.any() else float('nan'),
        'n_silent': int(silent.sum()),
        'silent_share': float(silent.sum() / max((y2 > 0).sum(), 1)),
    }


def alert_age_recall(y, p, alert, age):
    """Recall in silent compromises by alert-age bin. age: features.alert_age."""
    y, p, a, g = (np.asarray(v) for v in (y, p, alert, age))
    silent = (y > 0) & ~a.astype(bool)
    out = {}
    for name, lo, hi in ALERT_AGE_BINS:
        m = silent & (g >= lo) & (g <= hi)
        out[name] = {'n': int(m.sum()), 'recall': float((p[m] > 0).mean()) if m.any() else float('nan')}
    return out


def cause_recall(p, cause, names):
    """Recall in silent compromises by cause. cause: (…, H) code, 0 = not silent; names[code] = name."""
    p, c = np.asarray(p), np.asarray(cause)
    out = {}
    for code, name in enumerate(names):
        m = c == code
        if code == 0 or not m.any():
            continue
        out[name] = {'n': int(m.sum()), 'recall': float((p[m] > 0).mean())}
    return out


# ── temporal metrics (episode layout) ───────────────────────────────────

def detection_latency(y, p):
    """For every compromise period (uninterrupted run of y > 0), the steps until the first correct (p > 0) prediction.

    y, p: (E, T, H). Returns: mean/median delay (caught periods), share of caught periods, number of periods.
    """
    y, p = np.asarray(y) > 0, np.asarray(p) > 0
    E, T, H = y.shape
    in_spell = np.zeros((E, H), bool)
    start = np.zeros((E, H), np.int64)
    found = np.zeros((E, H), bool)
    lat, n_spells, n_found = [], 0, 0
    for t in range(T + 1):
        cur = y[:, t] if t < T else np.zeros((E, H), bool)
        ended = in_spell & ~cur
        n_spells += int(ended.sum())
        n_found += int((ended & found).sum())
        begun = cur & ~in_spell
        start[begun], found[begun] = t, False
        in_spell = cur
        if t < T:
            hit = in_spell & ~found & p[:, t]
            lat.extend((t - start[hit]).tolist())
            found |= hit
    lat = np.asarray(lat)
    return {'mean_latency': float(lat.mean()) if len(lat) else float('nan'),
            'median_latency': float(np.median(lat)) if len(lat) else float('nan'),
            'detected_spells': n_found / max(n_spells, 1), 'n_spells': n_spells}


def silent_persistence(y, p, alert):
    """How many steps a correct prediction (p > 0) persists without interruption after the alert disappears.

    Silent run: consecutive compromised and alert-free steps of the same host. The number of steps from the start of
    the run to the first miss, and its ratio to the run length, are reported.
    """
    y, p, a = np.asarray(y) > 0, np.asarray(p) > 0, np.asarray(alert, bool)
    silent = y & ~a
    E, T, H = silent.shape
    run_len = np.zeros((E, H), np.int64)
    held = np.zeros((E, H), np.int64)
    alive = np.zeros((E, H), bool)
    lens, holds = [], []
    for t in range(T + 1):
        cur = silent[:, t] if t < T else np.zeros((E, H), bool)
        ended = (run_len > 0) & ~cur
        lens.extend(run_len[ended].tolist())
        holds.extend(held[ended].tolist())
        run_len[ended], held[ended] = 0, 0
        if t < T:
            new = cur & (run_len == 0)
            alive[new] = True
            alive &= ~(cur & ~p[:, t])
            held += (cur & alive).astype(np.int64)
            run_len += cur
    lens, holds = np.asarray(lens), np.asarray(holds)
    return {'mean_run': float(lens.mean()) if len(lens) else float('nan'),
            'mean_held': float(holds.mean()) if len(holds) else float('nan'),
            'held_fraction': float(holds.sum() / max(lens.sum(), 1)), 'n_runs': int(len(lens))}


# ── kalibrasyon ─────────────────────────────────────────────────────────

def softmax(logits, T=1.0):
    z = np.asarray(logits, np.float64) / T
    z -= z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def calibration(probs, y, n_bins=15):
    """probs (…, 3), y (…) → ECE (most probable class, equal-width bins), Brier (multi-class), NLL."""
    P = np.asarray(probs, np.float64).reshape(-1, N_CLASSES)
    Y = np.asarray(y, np.int64).reshape(-1)
    conf, pred = P.max(1), P.argmax(1)
    correct = (pred == Y).astype(np.float64)
    bins = np.minimum((conf * n_bins).astype(np.int64), n_bins - 1)
    cnt = np.bincount(bins, minlength=n_bins)
    acc_sum = np.bincount(bins, correct, minlength=n_bins)
    conf_sum = np.bincount(bins, conf, minlength=n_bins)
    ece = float(np.abs(acc_sum - conf_sum).sum() / len(Y))
    onehot = np.eye(N_CLASSES)[Y]
    brier = float(((P - onehot) ** 2).sum(1).mean())
    nll = float(-np.log(np.clip(P[np.arange(len(Y)), Y], 1e-12, None)).mean())
    return {'ece': ece, 'brier': brier, 'nll': nll}


def fit_temperature(logits, y, lo=0.05, hi=20.0, iters=60):
    """Temperature minimizing NLL on val (golden-section search in log space)."""
    L = np.asarray(logits, np.float64).reshape(-1, N_CLASSES)
    Y = np.asarray(y, np.int64).reshape(-1)

    def nll(log_t):
        P = softmax(L, np.exp(log_t))
        return -np.log(np.clip(P[np.arange(len(Y)), Y], 1e-12, None)).mean()

    a, b = np.log(lo), np.log(hi)
    g = (np.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    for _ in range(iters):
        if nll(c) < nll(d):
            b, d = d, c
            c = b - g * (b - a)
        else:
            a, c = c, d
            d = a + g * (b - a)
    return float(np.exp((a + b) / 2))


# ── episode-level aggregation and breakdowns ─────────────────────────────

def episode_confusion(y, p):
    """(E, T, H) → (E, H, 3, 3): confusion per episode (for the episode bootstrap, T5)."""
    y, p = np.asarray(y, np.int64), np.asarray(p, np.int64)
    E, T, H = y.shape
    code = y * N_CLASSES + p                                       # (E, T, H)
    flat = (np.arange(E)[:, None, None] * H + np.arange(H)[None, None]) * N_CLASSES ** 2 + code
    cnt = np.bincount(flat.reshape(-1), minlength=E * H * N_CLASSES ** 2)
    return cnt.reshape(E, H, N_CLASSES, N_CLASSES)


def episode_slice_counts(y, p, alert, age, cause=None, cause_names=None):
    """(E, T, H) → (E, S, 2) [silent host-step, P(prediction > 0) hit] and S slice names (for the paired episode bootstrap).

    Slices: 'sessiz' (same subset as silent_recall), alert-age bins, and causes if given (cause_recall)."""
    y, p, a, g = np.asarray(y), np.asarray(p), np.asarray(alert, bool), np.asarray(age)
    silent, hit = (y > 0) & ~a, p > 0
    names, masks = ['sessiz'], [silent]
    for name, lo, hi in ALERT_AGE_BINS:
        names.append(name)
        masks.append(silent & (g >= lo) & (g <= hi))
    if cause is not None:
        for code, name in enumerate(cause_names[1:], start=1):
            names.append(name)
            masks.append(np.asarray(cause) == code)
    out = np.stack([np.stack([m.sum((1, 2)), (m & hit).sum((1, 2))], -1) for m in masks], 1)
    return out, names


def bootstrap_ci(ep_conf, n_boot=1000, seed=0, alpha=0.05):
    """95% confidence interval of host macro-F1 by episode-level resampling."""
    rng = np.random.default_rng(seed)
    E = ep_conf.shape[0]
    stats = np.empty(n_boot)
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, E, E), minlength=E)
        stats[b] = host_macro_f1_from_confusion(np.tensordot(w, ep_conf, axes=1))
    point = host_macro_f1_from_confusion(ep_conf.sum(0))
    return {'point': point, 'lo': float(np.quantile(stats, alpha / 2)), 'hi': float(np.quantile(stats, 1 - alpha / 2))}


# ── paired episode-level comparison of two models ──

def host_macro_f1_batch(C):
    """(B, H, 3, 3) → (B,): vectorized host_macro_f1_from_confusion (for bootstrap samples)."""
    C = np.asarray(C, np.float64)
    tp = np.diagonal(C, axis1=-2, axis2=-1)
    true, pred = C.sum(-1), C.sum(-2)
    f1 = 2 * tp / np.maximum(true + pred, 1)                     # 2tp + fp + fn = true + pred
    present = true > 0
    per_host = (f1 * present).sum(-1) / np.maximum(present.sum(-1), 1)
    hosts = true[..., 1:].sum(-1) > 0
    return np.where(hosts.any(-1), (per_host * hosts).sum(-1) / np.maximum(hosts.sum(-1), 1), np.nan)


def _boot_weights(rng, E, n):
    """(n, E): in each row, how many times each episode is drawn in an E-episode resample."""
    flat = rng.integers(0, E, (n, E)) + (np.arange(n) * E)[:, None]
    return np.bincount(flat.reshape(-1), minlength=n * E).reshape(n, E).astype(np.float64)


def paired_bootstrap(stat_a, stat_b, sizes, n_boot=10_000, seed=0, alpha=0.05, chunk=500):
    """Episode-level paired bootstrap for the difference a − b.

    sizes: {split: number of episodes}. stat_x(W) → (n,) value; W = {split: (n, E) episode weights}. Both models see the
    same weights in every sample (paired); splits are sampled independently. p is two-sided: the share of the bootstrap
    distribution of the difference on the other side of 0, × 2 (+1 corrected, minimum 2 / (n_boot + 1))."""
    rng = np.random.default_rng(seed)
    ones = {s: np.ones((1, E)) for s, E in sizes.items()}
    point = float(stat_a(ones)[0] - stat_b(ones)[0])
    diffs = []
    for i in range(0, n_boot, chunk):
        W = {s: _boot_weights(rng, E, min(chunk, n_boot - i)) for s, E in sizes.items()}
        diffs.append(stat_a(W) - stat_b(W))
    d = np.concatenate(diffs)
    p = min(1.0, 2 * (min((d <= 0).sum(), (d >= 0).sum()) + 1) / (n_boot + 1))
    return {'diff': point, 'lo': float(np.quantile(d, alpha / 2)), 'hi': float(np.quantile(d, 1 - alpha / 2)),
            'p': float(p), 'n_boot': n_boot}


def holm(pvals):
    """Holm–Bonferroni corrected p-values (in input order)."""
    p = np.asarray(pvals, np.float64)
    adj, run = np.empty(len(p)), 0.0
    for r, i in enumerate(np.argsort(p)):
        run = max(run, min(1.0, (len(p) - r) * p[i]))
        adj[i] = run
    return adj.tolist()


def ood_drop(f1_test, f1_ood):
    return {'delta': f1_test - f1_ood, 'relative': (f1_test - f1_ood) / f1_test}


def policy_breakdown(y, p, blue_policy, red_policy, blue_names, red_names):
    """(E, T, H) + policy ids per episode → {'red×blue': host macro-F1}."""
    out = {}
    for r in np.unique(red_policy):
        for b in np.unique(blue_policy):
            m = (red_policy == r) & (blue_policy == b)
            if m.any():
                H = y.shape[-1]
                out[f'{red_names[r]}×{blue_names[b]}'] = host_macro_f1(
                    y[m].reshape(-1, H), p[m].reshape(-1, H))
    return out


def full_report(y, p, alert, age, logits=None, cause=None, cause_names=None, temperature=None):
    """All main metrics for an evaluation set (episode layout, (E, T, H)).

    temperature: the temperature found on val; if given, the scaled calibration is reported as well ('calibration_T')."""
    H = y.shape[-1]
    y2, p2 = y.reshape(-1, H), p.reshape(-1, H)
    rep = {'host_macro_f1': host_macro_f1(y2, p2), 'breach_f1': breach_f1(y2, p2)}
    rep.update(silent_metrics(y, p, alert))
    rep['alert_age_recall'] = alert_age_recall(y, p, alert, age)
    if cause is not None:
        rep['cause_recall'] = cause_recall(p, cause, cause_names)
    rep['latency'] = detection_latency(y, p)
    rep['persistence'] = silent_persistence(y, p, alert)
    if logits is not None:
        rep['calibration'] = calibration(softmax(logits), y)
        if temperature is not None:
            rep['calibration_T'] = calibration(softmax(logits, temperature), y)
    return rep

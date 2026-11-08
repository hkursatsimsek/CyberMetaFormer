"""M3 genişletilmiş OOD paketi: (i) görülmemiş savunmacı, (iii) şiddet eğrisi ve mekanizma
politikaları, (iv) simülatör sürümü. Kaydedilmiş ağırlıklarla (`model.pt`), yeniden eğitim yok.

Bölüm kipindeki asama3 modelleri, 3 seed. İki yapılandırma:
    uye         üye başına, kendi val sıcaklığıyla; seed ortalaması
    topluluk_T  üyelerin ham softmax ortalaması + topluluğun kendi val sıcaklığı (belirsizlik ön çalışmasının ana ayarı)

Kümeler (hepsi `minicage_v1`'de eğitilmiş modellerle; sıcaklık yalnız v1 val'inde):
    test, ood                      v1 test ve `ood_test` (ε=0.3)
    eps0.1                         v1 test'in `react_restore_eps0.1` bölümleri (eğitim politikası; eğrinin ilk noktası)
    eps0.5, eps0.7, analyse, remove  `minicage_v1_shift`, politika başına bütün bölümler (2 Red × 2000)
    bug_test, bug_ood              `minicage_v1_bugged` (iv)
Şiddet eğrisi: ε = 0.1, 0.3, 0.5, 0.7; `react_restore_analyse` ve `react_remove` ayrı satırlar (mekanizma politikaları).
Neden dilimleri (kaçak, Green maskesi, remove maskesi) v1 ve shift kümelerinde (`*_aux/silent_cause.npz`).

Eşli bölüm bootstrap'ı (compare.py ile aynı: her model seed ortalamasıyla, kümenin bölümleri örneklenir): her kümede
host makro-F1 ve sessiz recall; Holm bütün satırlar üzerinden. Çiftler `PAIRS`.

    python -m cybermetaformer.hidden_state.ood_paketi collect      # asama3/ood_paketi/logits/ (git dışı)
    python -m cybermetaformer.hidden_state.ood_paketi collect --models cybermetaformer gru_full --seeds 3 4
                                                                 # M4: başka üye ve seed'lerin logitleri; analyze
                                                                 # yine MODELS × SEEDS ile çalışır (M3 tabloları)
    python -m cybermetaformer.hidden_state.ood_paketi analyze      # ozet.json, tablolar.md, figür
    python -m cybermetaformer.hidden_state.ood_paketi members --models cybermetaformer gru_full --seeds 0 1 2 3 4
                                                                 # M4 madde 5: yalnız üye ölçütleri → ozet_5seed.json
"""
import argparse
import json
import os
import time

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt                     # noqa: E402
import numpy as np                                  # noqa: E402
import torch                                        # noqa: E402
import yaml                                         # noqa: E402

from cybermetaformer.hidden_state import data as D          # noqa: E402
from cybermetaformer.hidden_state import evaluate as E      # noqa: E402
from cybermetaformer.hidden_state import features as F      # noqa: E402
from cybermetaformer.hidden_state import metrics as M       # noqa: E402
from cybermetaformer.hidden_state import train as T         # noqa: E402
from cybermetaformer.hidden_state import uncertainty as U   # noqa: E402
from cybermetaformer.hidden_state.models import build_model  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE_DIR = os.path.join(os.path.dirname(HERE), 'results', 'hidden_state', 'asama3')
OUT_DIR = os.path.join(STAGE_DIR, 'ood_paketi')
LOGIT_DIR = os.path.join(OUT_DIR, 'logits')
DATA = {'v1': D.DEFAULT_DATA, 'shift': os.path.join(D.DATA_ROOT, 'minicage_v1_shift'),
        'bug': os.path.join(D.DATA_ROOT, 'minicage_v1_bugged')}
CAUSES = {'v1': D.CAUSE_FILE, 'shift': os.path.join(D.DATA_ROOT, 'minicage_v1_shift_aux', 'silent_cause.npz')}
# toplanan temel kümeler: (veri, split; None = bütün bölümler)
BASE = {'val': ('v1', 'val'), 'test': ('v1', 'test'), 'ood': ('v1', 'ood_test'), 'shift': ('shift', None),
        'bug_test': ('bug', 'test'), 'bug_ood': ('bug', 'ood_test')}
# değerlendirme kümesi → (temel küme, Blue politikası süzgeci ya da None)
SETS = {'test': ('test', None), 'eps0.1': ('test', 'react_restore_eps0.1'), 'ood': ('ood', None),
        'eps0.5': ('shift', 'react_restore_eps0.5'), 'eps0.7': ('shift', 'react_restore_eps0.7'),
        'analyse': ('shift', 'react_restore_analyse'), 'remove': ('shift', 'react_remove'),
        'bug_test': ('bug_test', None), 'bug_ood': ('bug_ood', None)}
CURVE = [('eps0.1', 0.1), ('ood', 0.3), ('eps0.5', 0.5), ('eps0.7', 0.7)]
MODELS = ['cybermetaformer', 'gru_full', 'cybermetaformer-ana_m4', 'cybermetaformer-aksiyon', 'gru_full-aksiyon']
LABEL = {'cybermetaformer': 'CyberMetaFormer', 'gru_full': 'GRU tam', 'cybermetaformer-ana_m4': 'CMF (M4 kopyası)',
         'cybermetaformer-aksiyon': 'CMF + aksiyon', 'gru_full-aksiyon': 'GRU tam + aksiyon'}
PAIRS = [('cybermetaformer', 'gru_full'), ('cybermetaformer-aksiyon', 'cybermetaformer-ana_m4'),
         ('gru_full-aksiyon', 'gru_full')]
CONFIGS = ['uye', 'topluluk_T']
K, SEEDS = 101, (0, 1, 2)
CAUSE_SLICES = ['kacak_5', 'green_maske', 'remove_maske']


def run_dir(name, seed):
    return os.path.join(STAGE_DIR, f'{name}_k{K}_s{seed}')


def open_stores():
    return {src: D.EpisodeStore(path) for src, path in DATA.items()}


def base_indices(stores):
    with stores['v1'].final_evaluation(), stores['bug'].final_evaluation():
        return {b: (np.arange(stores[src].E) if split is None else stores[src].split(split))
                for b, (src, split) in BASE.items()}


def load_model(name, seed, device):
    cfg = {**T.DEFAULTS, **yaml.safe_load(open(os.path.join(run_dir(name, seed), 'config.yaml')))}
    torch.manual_seed(seed)
    model = build_model(cfg['model'], K, **(cfg.get('model_args') or {}))
    model.load_state_dict(torch.load(os.path.join(run_dir(name, seed), 'model.pt'), map_location='cpu'))
    return model.to(device).eval(), cfg


# ── toplama ──────────────────────────────────────────────────────────────

def collect(device, models=MODELS, seeds=SEEDS):
    """Her üyenin temel kümelerdeki logitleri (float16) → ood_paketi/logits/<koşu>.npz. Test ve val, koşunun
    metrics.json'undaki host makro-F1 ve sıcaklıkla denetlenir (aynı ağırlık, aynı tahmin). Var olan dosya atlanır."""
    stores = open_stores()
    idx = base_indices(stores)
    os.makedirs(LOGIT_DIR, exist_ok=True)
    for name in models:
        for seed in seeds:
            path = os.path.join(LOGIT_DIR, f'{name}_k{K}_s{seed}.npz')
            if os.path.exists(path):
                continue
            model, cfg = load_model(name, seed, device)
            t0, out = time.time(), {}
            for b, (src, _) in BASE.items():
                wd = D.WindowData(stores[src], idx[b], device=device)
                out[b] = E.predict_logits(model, wd, K, cfg['scale'], cfg['eval_batch_size'])
            rep = json.load(open(os.path.join(run_dir(name, seed), 'metrics.json')))['reports']
            y = stores['v1'].labels[idx['test']]
            f1 = M.host_macro_f1(y.reshape(-1, 12), out['test'].argmax(-1).reshape(-1, 12))
            temp = E.fit_temperature(out['val'], stores['v1'].labels[idx['val']])
            assert abs(f1 - rep['test']['host_macro_f1']) < 1e-4, f'{name} s{seed}: test F1 {f1} ≠ {rep["test"]}'
            np.savez_compressed(path, **{b: v.astype(np.float16) for b, v in out.items()})
            print(f'{name} s{seed}: {time.time() - t0:.0f} s; test F1 {f1:.4f} (kayıt '
                  f'{rep["test"]["host_macro_f1"]:.4f}), T {temp:.3f} (kayıt {rep["temperature"]:.3f})', flush=True)
            del model
            if device == 'mps':
                torch.mps.empty_cache()


# ── analiz ───────────────────────────────────────────────────────────────

class EvalSet:
    """Bir değerlendirme kümesi: etiketler, uyarılar, yaş, neden (hepsi (E, T, 12)) ve temel kümedeki konumları."""

    def __init__(self, store, idx, pos, cause=None, cause_names=None):
        self.pos = pos
        self.y = store.labels[idx].astype(np.int64)
        self.alert = F.episode_alerts(store.obs[idx])
        self.age = F.alert_age(self.alert)
        self.cause = None if cause is None else cause[idx]
        self.cause_names = cause_names
        self.E, self.n = len(idx), self.y.size


def eval_sets(stores, idx):
    cause = {}
    for src, path in CAUSES.items():
        with np.load(path) as z:
            cause[src] = (z['cause'], [str(c) for c in z['causes']])
    out = {}
    for s, (b, pol) in SETS.items():
        src = BASE[b][0]
        st, base = stores[src], idx[b]
        pos = np.arange(len(base)) if pol is None else \
            np.flatnonzero(st.blue_policy[base] == st.meta['blue_policies'].index(pol))
        c, names = cause.get(src, (None, None))
        out[s] = EvalSet(st, base[pos], pos, c, names)
    return out


def load_member(name, seed):
    with np.load(os.path.join(LOGIT_DIR, f'{name}_k{K}_s{seed}.npz')) as z:
        return {b: z[b] for b in z.files}


def evaluate(P, es):
    """Olasılık → rapor (F1, ihlal F1, sessiz recall, kalibrasyon, neden recall'ları) ve bootstrap girdileri."""
    rep, ep_conf = U.evaluate_probs(P, es)
    pred = P.argmax(-1)
    counts, names = M.episode_slice_counts(es.y, pred, es.alert, es.age, es.cause, es.cause_names)
    for c in CAUSE_SLICES:
        if c in names:
            n = counts[:, names.index(c)].sum(0)
            rep[f'recall_{c}'] = float(n[1] / n[0]) if n[0] else None
    return rep, ep_conf, counts[:, names.index('sessiz')]


def analyze_model(name, stores, idx, sets, val_y, log=print, seeds=SEEDS):
    """Modelin üyeleri ve topluluğu, bütün kümelerde. Dönüş: (raporlar {config: {küme: rapor}}, bootstrap girdileri
    {küme: [(ep_conf, sessiz sayımları)] üye başına})."""
    members = [load_member(name, s) for s in seeds]
    temps = [E.fit_temperature(m['val'].astype(np.float32), val_y) for m in members]
    P_val = np.mean([U.probs(m['val']) for m in members], 0)
    T_ens = E.fit_temperature(np.log(np.clip(P_val, 1e-12, None)), val_y)
    reps = {c: {} for c in CONFIGS}
    boot = {}
    for s, es in sets.items():
        b = SETS[s][0]
        member_reps, boot[s] = [], []
        P_raw = []
        for m, t in zip(members, temps):
            lg = m[b][es.pos]
            P_raw.append(U.probs(lg))
            rep, ep_conf, silent = evaluate(U.probs(lg, t), es)
            member_reps.append(rep)
            boot[s].append((ep_conf, silent))
        reps['uye'][s] = {k: float(np.mean([r[k] for r in member_reps])) for k in member_reps[0]
                          if isinstance(member_reps[0][k], float) and all(r[k] is not None for r in member_reps)}
        reps['uye'][s]['seeds'] = {k: [r[k] for r in member_reps] for k in ('host_macro_f1', 'silent_recall')}
        reps['topluluk_T'][s] = evaluate(U.rescale(np.mean(P_raw, 0), T_ens), es)[0]
        log(f"{name:24s} {s:9s} üye F1 {reps['uye'][s]['host_macro_f1']:.4f} sessiz {reps['uye'][s]['silent_recall']:.4f}"
            f" | topluluk F1 {reps['topluluk_T'][s]['host_macro_f1']:.4f} sessiz "
            f"{reps['topluluk_T'][s]['silent_recall']:.4f}")
    reps['temperature'] = {'uye': temps, 'topluluk_T': T_ens}
    return reps, boot


def pair_tests(boot, n_boot, seed=0):
    """PAIRS × kümeler × {host makro-F1, sessiz recall}: a − b, seed ortalaması, eşli bölüm bootstrap'ı, Holm (hepsi)."""
    rows = []
    for a, b in PAIRS:
        for s in SETS:
            E_ = len(boot[a][s][0][0])
            for crit in ('host_macro_f1', 'silent_recall'):
                if crit == 'host_macro_f1':
                    stat = lambda runs: lambda W: np.mean(                                  # noqa: E731
                        [M.host_macro_f1_batch(np.tensordot(W[s], c, axes=1)) for c, _ in runs], 0)
                else:
                    stat = lambda runs: lambda W: np.mean(                                  # noqa: E731
                        [(W[s] @ n[:, 1].astype(np.float64)) / np.maximum(W[s] @ n[:, 0].astype(np.float64), 1)
                         for _, n in runs], 0)
                res = M.paired_bootstrap(stat(boot[a][s]), stat(boot[b][s]), {s: E_}, n_boot=n_boot, seed=seed)
                rows.append({'a': a, 'b': b, 'set': s, 'criterion': crit, **res})
    for r, p in zip(rows, M.holm([r['p'] for r in rows])):
        r['p_holm'] = p
    return rows


def analyze(n_boot=4000, log=print):
    stores = open_stores()
    idx = base_indices(stores)
    sets = eval_sets(stores, idx)
    val_y = stores['v1'].labels[idx['val']]
    res, boot = {'sets': {s: {'n_episodes': es.E, 'source': SETS[s]} for s, es in sets.items()}, 'models': {}}, {}
    for name in MODELS:
        res['models'][name], boot[name] = analyze_model(name, stores, idx, sets, val_y, log)
    res['pairs'] = pair_tests(boot, n_boot)
    res['n_boot'] = n_boot
    with open(os.path.join(OUT_DIR, 'ozet.json'), 'w') as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    with open(os.path.join(OUT_DIR, 'tablolar.md'), 'w') as f:
        f.write(tables_md(res))
    fig_curve(res)
    log('→', OUT_DIR)
    return res


def analyze_members(models, seeds, log=print):
    """M4 (madde 5): yalnız verilen modellerin üye ölçütleri, verilen seed'lerle (kayıtlı logitler; bootstrap yok).
    M3 kaydı (ozet.json) değişmez; çıktı ozet_<n>seed.json. Makalenin 5 seed'li Tablo 9'u buradan okunur."""
    stores = open_stores()
    idx = base_indices(stores)
    sets = eval_sets(stores, idx)
    val_y = stores['v1'].labels[idx['val']]
    res = {'sets': {s: {'n_episodes': es.E, 'source': SETS[s]} for s, es in sets.items()}, 'seeds': list(seeds),
           'models': {}}
    for name in models:
        reps, _ = analyze_model(name, stores, idx, sets, val_y, log, seeds=tuple(seeds))
        res['models'][name] = {'uye': reps['uye'], 'temperature': reps['temperature']['uye']}
    path = os.path.join(OUT_DIR, f'ozet_{len(seeds)}seed.json')
    with open(path, 'w') as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    log('→', path)
    return res


# ── çıktılar ─────────────────────────────────────────────────────────────

def _c(v, d=4):
    return '' if v is None else f'{v:.{d}f}'


def tables_md(res):
    sets = list(SETS)
    L = ['# M3 OOD paketi (i), (iii), (iv)', '',
         'Kaydedilmiş ağırlıklar; sıcaklık yalnız v1 val. **Üye**: seed ortalaması (kendi sıcaklığı); **topluluk**: 3 seed, '
         'topluluk_T. ε=0.1 v1 test\'in eğitim politikası, ε=0.3 `ood_test`, 0.5 / 0.7 / analyse / remove '
         '`minicage_v1_shift` (politika başına 4000 bölüm), bug `minicage_v1_bugged`. '
         '`python -m cybermetaformer.hidden_state.ood_paketi analyze`.', '',
         '| Küme | Bölüm |', '|---|---:|']
    L += [f"| {s} | {res['sets'][s]['n_episodes']} |" for s in sets]
    for metric, title in [('host_macro_f1', 'Host makro-F1'), ('silent_recall', 'Sessiz recall'), ('ece', 'ECE'),
                          ('recall_remove_maske', 'Remove maskesi recall\'u'), ('recall_green_maske', 'Green maskesi recall\'u'),
                          ('recall_kacak_5', 'Kaçak (%5) recall\'u')]:
        L += ['', f'## {title}', '', '| Model | Yapılandırma | ' + ' | '.join(sets) + ' |',
              '|---|---|' + '---:|' * len(sets)]
        for name in MODELS:
            for c in CONFIGS:
                r = res['models'][name][c]
                L.append(f'| {LABEL[name]} | {c} | ' + ' | '.join(_c(r[s].get(metric)) for s in sets) + ' |')
    L += ['', '## Eşli sınamalar (üye, seed ortalaması; a − b)', '',
          f"Eşli bölüm bootstrap'ı, {res['n_boot']} örnek; p Holm bütün satırlar üzerinden.", '',
          '| a | b | küme | ölçüt | a − b | %95 GA | p Holm |', '|---|---|---|---|---:|---|---:|']
    for r in res['pairs']:
        L.append(f"| {LABEL[r['a']]} | {LABEL[r['b']]} | {r['set']} | {r['criterion']} | {r['diff']:+.4f} | "
                 f"[{r['lo']:+.4f}, {r['hi']:+.4f}] | {r['p_holm']:.4f} |")
    return '\n'.join(L) + '\n'


def fig_curve(res):
    """Şiddet eğrisi: host makro-F1 ve sessiz recall, ε'a karşı; mekanizma politikaları ayrı noktalar."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    xs = [e for _, e in CURVE]
    mech = [('analyse', 0.9, 's'), ('remove', 1.05, 'X')]
    colors = {'cybermetaformer': '#1f77b4', 'gru_full': '#d62728', 'cybermetaformer-aksiyon': '#2ca02c',
              'gru_full-aksiyon': '#ff7f0e', 'cybermetaformer-ana_m4': '#7f7f7f'}
    for ax, metric, title in [(axes[0], 'host_macro_f1', 'Host makro-F1'), (axes[1], 'silent_recall', 'Sessiz recall')]:
        for name in MODELS:
            for c, ls in [('uye', '-'), ('topluluk_T', '--')]:
                r = res['models'][name][c]
                ax.plot(xs, [r[s][metric] for s, _ in CURVE], ls, marker='o', ms=4, color=colors[name],
                        label=f'{LABEL[name]}' if c == 'uye' else None, alpha=0.9 if c == 'uye' else 0.6)
                for s, x, mk in mech:
                    ax.plot([x], [r[s][metric]], mk, color=colors[name], ms=6 if c == 'uye' else 4,
                            alpha=0.9 if c == 'uye' else 0.5)
        ax.set_xticks(xs + [m[1] for m in mech])
        ax.set_xticklabels(['0.1', '0.3\n(OOD)', '0.5', '0.7', 'analyse', 'remove'])
        ax.set_xlabel('react_restore ε  |  mekanizma politikaları')
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc='lower left')
    fig.suptitle('Şiddet eğrisi (düz: üye ort., kesikli: 3 seed topluluk)', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig_siddet_egrisi.png'), dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['collect', 'analyze', 'members'])
    p.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    p.add_argument('--n-boot', type=int, default=4000)
    p.add_argument('--models', nargs='+', default=MODELS, help='collect ve members')
    p.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS), help='collect ve members')
    args = p.parse_args()
    if args.cmd == 'collect':
        collect(args.device, args.models, args.seeds)
    elif args.cmd == 'members':
        analyze_members(args.models, args.seeds)
    else:
        analyze(args.n_boot)


if __name__ == '__main__':
    main()

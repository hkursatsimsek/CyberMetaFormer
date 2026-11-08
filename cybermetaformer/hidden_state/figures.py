"""Aşama 0 figürleri (eda.json → PNG). Yalnızca json + numpy + matplotlib; torch/CybORG gerekmez.

    venv_mini/bin/python -m cybermetaformer.hidden_state.figures      # VM (matplotlib venv_mini'de)
    python -m cybermetaformer.hidden_state.figures                    # Mac venv_ts

Renkler: dataviz yönergesinin doğrulanmış varsayılan paletinden (kategorik ilk üç slot; sıralı mavi rampa).
"""
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt                     # noqa: E402
import numpy as np                                  # noqa: E402
from matplotlib.colors import ListedColormap        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE0 = os.path.join(os.path.dirname(HERE), 'results', 'hidden_state', 'asama0')
FIG_DIR = os.path.join(STAGE0, 'figures')

SURFACE, INK, INK2, MUTED, GRID, BASE = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a']           # kategorik slot 1–3
USER, ROOT, CLEAN = '#86b6ef', '#1c5cab', '#f0efec'  # sıralı mavi rampa 250 / 550, nötr gri
CAUSE_LABEL = {'kacak_5': '%5 tespit kaçağı', 'green_maske': 'Green taraması maskesi', 'remove_maske': 'remove maskesi'}
POLICY_LABEL = {'sleep': 'sleep', 'react_restore': 'react_restore', 'restore_decoys': 'restore_decoys',
                'random': 'random', 'react_restore_eps0.1': 'react_restore\nε=0.1', 'react_restore_eps0.3': 'react_restore\nε=0.3 (OOD)'}
HOST_ORDER = ['user1', 'user2', 'user3', 'user4', 'ent0', 'ent1', 'ent2', 'ophost0', 'ophost1', 'ophost2', 'opserv', 'def']

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.family': 'sans-serif', 'font.size': 10, 'text.color': INK, 'axes.labelcolor': INK2,
    'axes.edgecolor': BASE, 'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.grid': False,
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
})


def _grid(ax, axis='y'):
    ax.grid(axis=axis, color=GRID, linewidth=1, linestyle='-')
    ax.set_axisbelow(True)


def _title(fig, title, subtitle):
    fig.text(0.01, 0.985, title, ha='left', va='top', fontsize=12, fontweight='bold', color=INK)
    fig.text(0.01, 0.93, subtitle, ha='left', va='top', fontsize=9, color=INK2)


def fig_label_dist(eda):
    d = eda['label_dist_train']
    hosts = [h for h in HOST_ORDER if h in d][::-1]
    tot = {h: sum(d[h]) for h in hosts}
    u = np.array([d[h][1] / tot[h] for h in hosts])
    r = np.array([d[h][2] / tot[h] for h in hosts])
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    y = np.arange(len(hosts))
    ax.barh(y, u, height=0.55, color=USER, label='user erişimi')
    ax.barh(y, r, left=u, height=0.55, color=ROOT, label='root erişimi', edgecolor=SURFACE, linewidth=2)
    for yi, (a, b) in enumerate(zip(u, r)):
        ax.text(a + b + 0.006, yi, f'{(a + b) * 100:.1f}%', va='center', fontsize=8.5, color=INK2)
    ax.set_yticks(y, hosts)
    ax.set_xlim(0, 0.42)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.set_xlabel('host-adım oranı (train)')
    _grid(ax, 'x')
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2, fontsize=9)
    _title(fig, 'Ele geçirilmiş host-adım oranı, host başına',
           "Gerisi temiz. def hiç ele geçirilmiyor; ophost'lar nadiren. Sınıflar dengesiz.")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    return fig


def fig_silent_cause_by_policy(eda):
    # sütun yüksekliği = sessiz / ele geçirilmiş (iki Red birleşik); dilimler = neden payları
    names = eda['cause_names']
    pol = eda['blue_policies']
    silent_by = {b: np.array(eda['cause_by_blue'][b], float) for b in pol}
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    x = np.arange(len(pol))
    keys = ['kacak_5', 'green_maske', 'remove_maske']
    bottoms = np.zeros(len(pol))
    totals = np.array([eda['silent_share_blue'][b] for b in pol])
    for i, key in enumerate(keys):
        j = names.index(key)
        part = np.array([silent_by[b][j] / max(silent_by[b][1:].sum(), 1) for b in pol]) * totals
        ax.bar(x, part, bottom=bottoms, width=0.5, color=SERIES[i], label=CAUSE_LABEL[key],
               edgecolor=SURFACE, linewidth=2)
        bottoms += part
    for xi, tv in zip(x, totals):
        ax.text(xi, tv + 0.015, f'{tv * 100:.0f}%', ha='center', fontsize=9, color=INK)
    ax.set_xticks(x, [POLICY_LABEL[b] for b in pol], fontsize=8.5)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.set_ylabel('sessiz / ele geçirilmiş host-adım')
    _grid(ax)
    ax.legend(loc='upper left', fontsize=9, ncol=3)
    _title(fig, 'Sessiz ihlal oranı ve nedeni, Blue politikasına göre',
           'Restore kaynaklı sessizlik yok (%0). remove maskesi yalnızca remove kullanan politikalarda.')
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def fig_window_coverage(eda):
    cov = eda['window_coverage_split']
    ks = np.arange(1, 101)
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for i, (split, label) in enumerate([('test', 'test'), ('ood_test', 'OOD test')]):
        c = np.array(cov[split]['coverage'])
        ax.plot(ks, c, color=SERIES[i], linewidth=2, solid_capstyle='round', label=label)
        for k in [8, 32]:
            ax.plot([k], [c[k - 1]], 'o', color=SERIES[i], markersize=6, markeredgecolor=SURFACE, markeredgewidth=2)
    c_te = np.array(cov['test']['coverage'])
    for k in [8, 32]:
        ax.annotate(f'k={k}: {c_te[k - 1] * 100:.0f}%', (k, c_te[k - 1]), xytext=(k + 3, c_te[k - 1] - 0.09),
                    fontsize=8.5, color=INK2, arrowprops=dict(arrowstyle='-', color=MUTED, linewidth=1))
    ceil = 1 - cov['test']['never']
    ax.axhline(ceil, color=BASE, linewidth=1)
    ax.text(2, ceil + 0.015, f'tavan: o hostta daha önce uyarı görülmüş sessizler ({ceil * 100:.0f}%, test)',
            fontsize=8.5, color=INK2)
    ax.set_xscale('log', base=2)
    ax.set_xticks([1, 4, 8, 16, 32, 64, 100], ['1', '4', '8', '16', '32', '64', '100'])
    ax.set_xlim(1, 100)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.set_xlabel('pencere uzunluğu k (son k gözlem)')
    ax.set_ylabel('son uyarısı pencereye düşen sessizler')
    _grid(ax)
    ax.legend(loc='upper left', fontsize=9)
    _title(fig, 'Pencere k, sessiz ihlallerin ne kadarında son uyarıyı içeriyor?',
           'k=8 sessizlerin yalnızca ~%10\'unu görüyor; k=32 ~%42, k=64 ~%67. Kalanların bir kısmında hiç uyarı yok.')
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def fig_example_episode(eda):
    ex = eda['example_episode']
    hosts = eda['hosts']
    order = [hosts.index(h) for h in HOST_ORDER if h in hosts]
    lab = np.array(ex['labels'])[order]
    al = np.array(ex['alerts'])[order]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.imshow(lab, aspect='auto', interpolation='nearest', cmap=ListedColormap([CLEAN, USER, ROOT]), vmin=0, vmax=2,
              extent=(0.5, lab.shape[1] + 0.5, len(order) - 0.5, -0.5))
    yy, xx = np.nonzero(al)
    ax.scatter(xx + 1, yy, s=14, color=INK, marker='o', linewidths=0, label='uyarı (gözlemde)')
    ax.set_yticks(range(len(order)), [hosts[i] for i in order])
    ax.set_xlabel('adım t')
    for v, name in [(USER, 'user erişimi'), (ROOT, 'root erişimi')]:
        ax.bar([0], [0], color=v, label=name)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=9)
    ax.spines[['left', 'bottom']].set_visible(False)
    ax.tick_params(length=0)
    _title(fig, f"Örnek bölüm #{ex['episode']}: {ex['red']} × {ex['blue']} (test)",
           'Koyu mavi ve noktasız hücreler sessiz root: host ele geçirilmiş, gözlemde uyarı yok. '
           'Gerçek durum modelin göremediği etikettir.')
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def main():
    with open(os.path.join(STAGE0, 'eda.json')) as f:
        eda = json.load(f)
    # Blue politikası başına sessiz oranı (iki Red birleşik, ele geçirilmiş host-adım ağırlıklı)
    eda['silent_share_blue'] = {}
    for b in eda['blue_policies']:
        rows = [eda['silent_by_policy'][f'{r}×{b}'] for r in eda['red_policies']]
        w = np.array([x['ele_gecirilmis_orani'] for x in rows])
        eda['silent_share_blue'][b] = float((np.array([x['sessiz_orani'] for x in rows]) * w).sum() / w.sum())
    os.makedirs(FIG_DIR, exist_ok=True)
    for name, fn in [('fig_label_dist', fig_label_dist), ('fig_silent_cause_by_policy', fig_silent_cause_by_policy),
                     ('fig_window_coverage', fig_window_coverage), ('fig_example_episode', fig_example_episode)]:
        fig = fn(eda)
        fig.savefig(os.path.join(FIG_DIR, f'{name}.png'), dpi=160)
        plt.close(fig)
        print('yazıldı:', os.path.join(FIG_DIR, f'{name}.png'))


if __name__ == '__main__':
    main()

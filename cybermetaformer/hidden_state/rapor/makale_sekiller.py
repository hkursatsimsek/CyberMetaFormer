"""Makale (M4) şekilleri, İngilizce. Şekiller son baskı genişliğinde (6.3 inç) çizilir, en küçük yazı 8 pt.
Çıktılar 300 dpi PNG ve SVG'dir; SVG vektör sürümüdür, PDF git'e girmediği için SVG kullanılır.

    venv_ts/bin/python -m cybermetaformer.hidden_state.rapor.makale_sekiller

- fig_cmf_architecture: CyberMetaFormer (madde 2). Şekil, §3.5'teki denklemlerle ve models/cybermetaformer.py ile
  birebir uyumludur:
  - girdi: host token'ları ve uyarı yaşı (Eş. 1–2), gömme (Eş. 3);
  - blok: üç pre-LN artık alt katman. Zaman karıştırıcısında konv ve SSD kolları ile softmax kapısı (Eş. 4–8);
    host karıştırıcısında ilişki sapmalı dikkat (Eş. 9); kanal MLP'si (Eş. 10);
  - çıkış: son LN ve paylaşılan başlık (Eş. 11).
- fig_confusion: CMF ve tam geçmiş GRU'nun test ve OOD'de havuzlanmış, satır normalize karışıklık matrisi (madde 3;
  seed 0–4 ortalaması ± SD, makale/sinif_duzeyi/ozet.json).

Taslaktan gelen şekillerin İngilizce, betikten yeniden çizimleri (2026-11-08; hiçbir şekilde Türkçe sözcük yok). Sayılar
tabloların kullandığı kayıtlardan gelir; tablo hücreleriyle aynı değerler `makale_tablolar` üzerinden okunur:
- fig_network (Şekil 1): ağ, saldırı grafı kenarları (models/cybermetaformer.py CONNECTED_HOSTS) ve kestirim akışı.
- fig_window_coverage (Şekil 2): asama0/eda.json (figures.fig_window_coverage'ın verisi).
- fig_memory_stage1 / fig_memory_stage2 (Şekil 4–5): asama1/2 ozet/ozet.json (aggregate.fig_memory_curve'ün verisi).
- fig_selected (Şekil 6): seçili modellerin test ve OOD F1'i; CMF ve GRU tam seed 0–4 (Tablo 8 ile aynı kayıtlar).
- fig_ablation (Şekil 8): Tablo 15'in farkları ve %95 GA'ları.
- fig_attention (Şekil 9): asama3/aciklama/dikkat.json (explain_m3.fig_attention'ın verisi; seed 0, ε = 0.1 testi).
- fig_severity (Şekil 10): asama3/ood_paketi/ozet_5seed.json (Tablo 16 ile aynı; ortalama ± seed SD).
- fig_pareto (Şekil 11): Tablo 18'in hücreleri (önceki gecikme ölçümü).
"""
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

from cybermetaformer.hidden_state.figures import BASE, GRID, INK, INK2, MUTED, ROOT, SERIES, SURFACE
from cybermetaformer.hidden_state.rapor.sekiller import PALE, PALE2, RESULTS, _arrow, _box

OUT_DIR = os.path.join(RESULTS, 'makale', 'sekiller')
WIDTH_IN = 6.3
FS, FS_SMALL, FS_TITLE = 8, 8, 9
WHITE = '#ffffff'
RC = {'figure.facecolor': WHITE, 'axes.facecolor': WHITE, 'savefig.facecolor': WHITE, 'font.size': FS,
      'axes.labelsize': FS, 'xtick.labelsize': FS, 'ytick.labelsize': FS, 'legend.fontsize': FS,
      'axes.titlesize': FS_TITLE, 'svg.fonttype': 'none', 'xtick.color': INK2, 'ytick.color': INK2,
      'axes.labelcolor': INK2}
plt.rcParams.update(RC)     # aggregate.py içe aktarılınca kendi rcParams'ını yazar; main() bunu yeniden uygular
CMF_C, GRU_C, OTHER_C = SERIES[0], SERIES[1], MUTED       # CMF mavi, tam geçmiş GRU turuncu, ötekiler nötr


def _save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.patch.set_facecolor(WHITE)
    for ax in fig.axes:
        ax.set_facecolor(WHITE)
    fig.savefig(os.path.join(OUT_DIR, f'{name}.png'), dpi=300, facecolor=WHITE)
    fig.savefig(os.path.join(OUT_DIR, f'{name}.svg'), facecolor=WHITE)
    plt.close(fig)


def _ygrid(ax, axis='y'):
    ax.grid(axis=axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def _line(ax, xs, ys, color=INK2, lw=1.1):
    ax.plot(xs, ys, color=color, lw=lw, solid_capstyle='round', zorder=1)


def _plus(ax, x, y, r=0.22):
    ax.add_patch(Circle((x, y), r, fc=SURFACE, ec=INK, lw=1.1, zorder=3))
    ax.text(x, y - 0.02, '+', ha='center', va='center', fontsize=FS + 1, color=INK, zorder=4)


def fig_cmf_architecture():
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 4.1))
    fig.subplots_adjust(0.003, 0.003, 0.997, 0.997)
    ax.set_xlim(0, 20)
    ax.set_ylim(0.55, 10.0)
    ax.axis('off')
    y_res, y_ln, y_mod, y_drop = 8.55, 7.45, 5.45, 2.55

    # ── girdi (aşağıdan yukarı) ──
    _box(ax, 1.75, 2.55, 3.2, 1.0, 'Observation $\\hat{o}_t$\n(78 values)', fc=PALE2, fs=FS)
    _box(ax, 1.75, 5.0, 3.2, 1.95, '13 host tokens\n$x_{t,h}$ (6) + age\n$\\phi, \\nu$ (8)\nEqs. 1–2',
         fc=PALE, ec=SERIES[0], fs=FS)
    _box(ax, 1.75, 8.55, 3.2, 1.0, 'Embedding\nEq. 3', fc=PALE2, fs=FS)
    _arrow(ax, (1.75, 3.05), (1.75, 4.02))
    _arrow(ax, (1.75, 5.98), (1.75, 8.05))

    # ── blok ──
    ax.add_patch(FancyBboxPatch((3.75, 0.75), 12.75, 8.55, boxstyle='round,pad=0,rounding_size=0.25', fc=SURFACE,
                                ec=INK2, lw=1.1, ls=(0, (4, 3)), zorder=0))
    ax.text(10.02, 9.62, 'Block $\\ell$ × 3: three pre-LN residual sub-layers', ha='center', va='center',
            fontsize=FS_TITLE, fontweight='bold', color=INK)
    _arrow(ax, (3.35, y_res), (3.95, y_res), lw=1.3)
    _line(ax, [3.95, 16.15], [y_res, y_res], color=INK, lw=1.4)
    cols = [  # (LN x, ⊕ x, LN etiketi, alt yazı)
        (6.3, 9.0, '$\\mathrm{LN}_1$', 'per host, along time\n(weights shared)'),
        (10.85, 12.75, '$\\mathrm{LN}_2$', 'per step, across\nthe 13 hosts'),
        (14.45, 16.15, '$\\mathrm{LN}_3$', 'per token'),
    ]
    for x_ln, x_add, ln, cap in cols:
        _arrow(ax, (x_ln, y_res), (x_ln, y_ln + 0.33))
        _box(ax, x_ln, y_ln, 1.25, 0.6, ln, fc=SURFACE, ec=BASE, fs=FS)
        _box(ax, x_ln, y_drop, 1.75, 0.6, 'Dropout', fc=SURFACE, ec=BASE, fs=FS_SMALL, color=INK2)
        _line(ax, [x_ln + 0.9, x_add], [y_drop, y_drop])
        _arrow(ax, (x_add, y_drop), (x_add, y_res - 0.24))
        _plus(ax, x_add, y_res)
        ax.text((x_ln + x_add) / 2 - (0.35 if x_ln == 6.3 else 0.1), 1.35, cap, ha='center', va='center',
                fontsize=FS_SMALL, color=MUTED, style='italic', linespacing=1.2)

    # zaman karıştırıcısı: konv ve SSD kolları → softmax kapısı
    ax.text(6.45, 9.05, 'Temporal mixer', ha='center', va='center', fontsize=FS, fontweight='bold', color=INK2)
    _box(ax, 5.1, y_mod + 0.45, 2.1, 1.75, 'Causal\nconv\n$K$ = 4\nEq. 4', fc=SURFACE, ec=SERIES[0], fs=FS)
    _box(ax, 7.5, y_mod + 0.45, 2.1, 1.75, 'SSD\n4 heads\nEqs. 5–7', fc=SURFACE, ec=SERIES[0], fs=FS)
    _box(ax, 6.3, 3.7, 3.9, 0.95, 'Softmax gate\n$g^c c + g^s s$, Eq. 8', fc=SURFACE, ec=BASE, fs=FS)
    _arrow(ax, (6.05, y_ln - 0.3), (5.3, y_mod + 1.32))
    _arrow(ax, (6.55, y_ln - 0.3), (7.3, y_mod + 1.32))
    _arrow(ax, (5.1, y_mod - 0.43), (5.6, 4.17))
    _arrow(ax, (7.5, y_mod - 0.43), (7.0, 4.17))
    _arrow(ax, (6.3, 3.22), (6.3, y_drop + 0.3))

    # host karıştırıcısı
    ax.text(11.4, 9.05, 'Host mixer', ha='center', va='center', fontsize=FS, fontweight='bold', color=INK2)
    _box(ax, 10.85, 5.05, 2.75, 3.1, 'Multi-head\nattention\nacross hosts\n+ relation\nbias $\\beta_{m,\\rho}$\nEq. 9',
         fc=SURFACE, ec=SERIES[0], fs=FS)
    _arrow(ax, (10.85, y_ln - 0.3), (10.85, 6.6))
    _arrow(ax, (10.85, 3.5), (10.85, y_drop + 0.3))

    # kanal MLP'si
    ax.text(14.85, 9.05, 'Channel MLP', ha='center', va='center', fontsize=FS, fontweight='bold', color=INK2)
    _box(ax, 14.45, 5.05, 2.6, 3.1, 'Linear $d$→2$d$\nGELU\nDropout\nLinear 2$d$→$d$\nEq. 10', fc=SURFACE,
         ec=SERIES[0], fs=FS)
    _arrow(ax, (14.45, y_ln - 0.3), (14.45, 6.6))
    _arrow(ax, (14.45, 3.5), (14.45, y_drop + 0.3))

    # ── çıkış (yukarıdan aşağı) ──
    _arrow(ax, (16.15, y_res), (16.95, y_res), lw=1.3)
    _box(ax, 18.3, y_res, 2.6, 0.62, '$\\mathrm{LN}_f$', fc=SURFACE, ec=BASE, fs=FS)
    _box(ax, 18.3, 6.5, 2.95, 1.55, 'Shared head\n12 label hosts\nEq. 11', fc='#e8f0fb', ec=ROOT, fs=FS, lw=1.4)
    _box(ax, 18.3, 4.2, 2.95, 1.55, 'Logits\n$z_{t,h} \\in \\mathbb{R}^3$\nclean, user, root', fc='#e8f0fb',
         ec=ROOT, fs=FS)
    _arrow(ax, (18.3, y_res - 0.33), (18.3, 7.3))
    _arrow(ax, (18.3, 5.7), (18.3, 5.0))
    _save(fig, 'fig_cmf_architecture')


# sıralı mavi rampa (dataviz paleti, adım 100 → 700); hücre değeri satır normalize oran
SEQ_BLUE = ['#f4f8fd', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']


def fig_confusion():
    """Madde 3: havuzlanmış, satır normalize 3×3 karışıklık; satırlar CMF / tam geçmiş GRU, sütunlar test / OOD; seed
    0–4 ortalaması (sinif_duzeyi/ozet.json). Renk PowerNorm(0.5) ile: küçük hata oranları (%4 – %20) ayırt edilsin
    diye; her hücrede değer ve seed SD'si yazılıdır."""
    import json

    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap, PowerNorm, to_rgb

    with open(os.path.join(RESULTS, 'makale', 'sinif_duzeyi', 'ozet.json')) as f:
        S = json.load(f)
    classes = S['classes']
    models = [('CyberMetaFormer', 'CyberMetaFormer'), ('Full-history GRU', 'Full-history GRU')]
    splits = [('test', 'standard test'), ('ood', 'OOD test')]
    cmap = LinearSegmentedColormap.from_list('seq_blue', SEQ_BLUE)
    norm = PowerNorm(0.5, 0, 1)
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH_IN, 5.0))
    fig.subplots_adjust(left=0.13, right=0.86, bottom=0.09, top=0.93, wspace=0.28, hspace=0.42)
    for i, (key, mlab) in enumerate(models):
        for j, (split, slab) in enumerate(splits):
            ax = axes[i, j]
            r = S[key][split]['row_normalized']
            assert r['n'] == len(S[key]['seeds']) == 5
            M, SD = np.asarray(r['mean']), np.asarray(r['sd'])
            assert np.allclose(M.sum(1), 1)
            ax.pcolormesh(M, cmap=cmap, norm=norm, edgecolors=SURFACE, linewidth=2)
            for a in range(3):
                for b in range(3):
                    rgb = to_rgb(cmap(norm(M[a, b])))
                    lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
                    c = SURFACE if lum < 0.45 else INK
                    ax.text(b + 0.5, a + 0.42, f'{100 * M[a, b]:.1f}%', ha='center', va='center', fontsize=FS + 1,
                            color=c, fontweight='bold' if a == b else 'normal')
                    ax.text(b + 0.5, a + 0.72, f'± {100 * SD[a, b]:.1f}', ha='center', va='center', fontsize=FS - 1,
                            color=c)
            ax.set_xlim(0, 3)
            ax.set_ylim(3, 0)
            ax.set_xticks([0.5, 1.5, 2.5], classes, fontsize=FS)
            ax.set_yticks([0.5, 1.5, 2.5], classes, fontsize=FS)
            ax.tick_params(length=0, pad=3)
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_title(f'{mlab}, {slab}', fontsize=FS_TITLE, color=INK, pad=6, loc='left')
            ax.set_xlabel('Predicted class', fontsize=FS, color=INK2)
            if j == 0:
                ax.set_ylabel('True class', fontsize=FS, color=INK2)
    cax = fig.add_axes([0.89, 0.18, 0.018, 0.64])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    ticks = [0, 0.01, 0.05, 0.2, 0.5, 1.0]
    cb.set_ticks(ticks, labels=[f'{100 * t:g}%' for t in ticks])
    cb.ax.tick_params(labelsize=FS, length=0, colors=INK2)
    cb.outline.set_visible(False)
    cb.set_label('Share of the true class (row)', fontsize=FS, color=INK2)
    _save(fig, 'fig_confusion')


# ── taslaktan gelen şekiller, İngilizce ─────────────────────────────────────

def _region(ax, x0, y0, x1, y1, title):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle='round,pad=0,rounding_size=0.12', fc=PALE2,
                                ec=BASE, lw=1.0))
    ax.text((x0 + x1) / 2, y1 - 0.3, title, ha='center', va='center', fontsize=FS_TITLE, fontweight='bold', color=INK)


def fig_network():
    """Şekil 1: üç alt ağ, saldırı grafı kenarları (CONNECTED_HOSTS: user1/2 → ent0, user3/4 → ent1, ent2 → opserv)
    ve gözlem → kestirici → gizli durum akışı."""
    from cybermetaformer.hidden_state.models.cybermetaformer import CONNECTED_HOSTS, HOSTS
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 3.35))
    fig.subplots_adjust(0.005, 0.005, 0.995, 0.995)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.4)
    ax.axis('off')
    _region(ax, 0.1, 1.95, 4.05, 6.3, 'User subnet')
    _region(ax, 4.2, 1.95, 7.55, 6.3, 'Enterprise subnet')
    _region(ax, 7.7, 1.95, 11.9, 6.3, 'Operational subnet')
    w, h = 1.15, 0.5
    pos = {'user0': (1.05, 4.1), 'user1': (2.85, 5.45), 'user2': (2.85, 4.65), 'user3': (2.85, 3.55),
           'user4': (2.85, 2.75), 'ent0': (5.0, 5.05), 'ent1': (5.0, 3.15), 'ent2': (6.65, 4.1),
           'def': (5.0, 2.35), 'opserv': (8.55, 4.1), 'ophost0': (10.85, 5.3), 'ophost1': (10.85, 4.1),
           'ophost2': (10.85, 2.9)}
    assert set(pos) == set(HOSTS)
    for name, (x, y) in pos.items():
        if name == 'user0':
            _box(ax, x, y, w, h, name, fc='#fde3d7', ec=SERIES[1], lw=1.3, weight='bold', fs=FS)
            ax.text(x, y - 0.48, 'attacker entry', ha='center', va='center', fontsize=FS, color=INK2)
        elif name == 'opserv':
            _box(ax, x, y, w, h, name, fc=PALE, ec=ROOT, lw=1.6, weight='bold', fs=FS)
            ax.text(x, y - 0.48, 'critical server', ha='center', va='center', fontsize=FS, color=INK2)
        elif name == 'def':
            _box(ax, x, y, w, h, name, fc='#ecebe7', ec=MUTED, color=INK2, fs=FS)
        else:
            _box(ax, x, y, w, h, name, fs=FS)
    hw = w / 2 + 0.03
    for src, dst in zip(HOSTS, CONNECTED_HOSTS):
        if dst is None:
            continue
        (x0, y0), (x1, y1) = pos[src], pos[dst]
        _arrow(ax, (x0 + hw, y0), (x1 - hw, y1), color=ROOT, lw=1.5)
    flow = [(1.95, 'Defender observation\n13 hosts × 6 features', PALE2, BASE),
            (6.0, 'Causal temporal\nstate estimator', PALE, SERIES[0]),
            (10.05, '12 hosts × 3 states\n(clean / user / root)', '#e8f0fb', ROOT)]
    for x, text, fc, ec in flow:
        _box(ax, x, 0.85, 3.4, 1.05, text, fc=fc, ec=ec, lw=1.2, fs=FS + 1)
    for xa, xb in ((3.7, 4.25), (7.75, 8.3)):
        _arrow(ax, (xa, 0.85), (xb, 0.85), lw=1.5, ms=12)
    _save(fig, 'fig_network')


def fig_window_coverage():
    """Şekil 2: sessiz host-adımlar içinde son uyarısı k adımlık pencereye düşenlerin payı (asama0/eda.json)."""
    import json

    import numpy as np
    from matplotlib.ticker import PercentFormatter
    with open(os.path.join(RESULTS, 'asama0', 'eda.json')) as f:
        cov = json.load(f)['window_coverage_split']
    ks = np.arange(1, 101)
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 2.9))
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.16, top=0.97)
    for (split, label), color in zip([('test', 'Standard test'), ('ood_test', 'OOD test')], (SERIES[0], SERIES[1])):
        c = np.array(cov[split]['coverage'])
        ax.plot(ks, c, color=color, lw=2, solid_capstyle='round', label=label)
        for k in (8, 32, 64):
            ax.plot([k], [c[k - 1]], 'o', color=color, ms=5, mec=WHITE, mew=1.5, zorder=3)
    c = np.array(cov['test']['coverage'])
    for k, xt, yt in ((8, 3.2, 0.24), (32, 13, 0.5), (64, 30, 0.67)):          # etiketler eğrilerin sol üstünde
        ax.annotate(f'k = {k}: {100 * c[k - 1]:.0f}%', (k, c[k - 1]), xytext=(xt, yt), fontsize=FS, color=INK2,
                    ha='right', arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8, shrinkB=3))
    ceil = 1 - cov['test']['never']
    ax.axhline(ceil, color=BASE, lw=1)
    ax.text(1.1, ceil + 0.025, f'Ceiling: silent host-steps whose host had an earlier alert ({100 * ceil:.0f}%, test)',
            fontsize=FS, color=INK2)
    ax.set_xscale('log', base=2)
    ax.set_xticks([1, 4, 8, 16, 32, 64, 100], ['1', '4', '8', '16', '32', '64', '100'])
    ax.minorticks_off()
    ax.set_xlim(1, 100)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlabel('Window length k (last k observations)')
    ax.set_ylabel('Last alert inside the window')
    _ygrid(ax)
    ax.legend(loc='upper left')
    _save(fig, 'fig_window_coverage')


STAGE_LABEL = {'persist': 'Alert-persistence rule', 'xgb': 'XGBoost', 'xgb_tf': 'XGB-TF', 'mlp': 'MLP',
               'cnn1d': '1D-CNN', 'lstm': 'LSTM', 'gru': 'GRU', 'lstm_full': 'LSTM (full history)',
               'gru_full': 'GRU (full history)', 'transformer': 'Transformer', 'patchtst': 'PatchTST',
               'itransformer': 'iTransformer', 'tsmixer': 'TSMixer', 'timesnet': 'TimesNet', 'timemixer': 'TimeMixer'}


def _memory_curve(stage, name):
    """aggregate.fig_memory_curve'ün İngilizce, başlıksız makale sürümü; veri aynı (ozet/ozet.json)."""
    import json

    from matplotlib.lines import Line2D

    from cybermetaformer.hidden_state import aggregate as A

    def load(st):
        with open(os.path.join(RESULTS, st, 'ozet', 'ozet.json')) as f:
            return json.load(f)
    summary = load(stage)
    R = A.m1_refs(load('asama1')) if stage == 'asama2' else None
    panels = [('test_host_macro_f1', 'Test host macro-F1', 'test'), ('ood_host_macro_f1', 'OOD host macro-F1', 'ood_test'),
              ('test_silent_recall', 'Test silent recall', None)]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_IN, 3.05))
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.4, top=0.93, wspace=0.32)
    allk = sorted({g['k'] for g in summary.values() if g['k'] > 0})
    FULL = 128                      # tam geçmiş (k = 101) log ekseninde 64'ün bir adım sağında, "full" etiketiyle
    X = lambda k: FULL if k == 101 else k          # noqa: E731
    ms, lw = 4, 1.4
    for ax, (col, title, orig) in zip(axes, panels):
        if R:
            bk = [k for k in allk if R['best'].get(k) and R['best'][k].get(col)]
            ax.plot([X(k) for k in bk], [R['best'][k][col]['mean'] for k in bk], color=INK2, lw=1, marker='x', ms=4, zorder=1)
            if R['gru_full'] and R['gru_full'].get(col):
                ax.axhline(R['gru_full'][col]['mean'], color=MUTED, lw=1, ls='--', zorder=1)
        rule = sorted([g for g in summary.values() if g['model'] == 'persist' and g.get(col)], key=lambda g: g['k'])
        if rule:
            ax.plot([X(g['k']) for g in rule], [g[col]['mean'] for g in rule], color=MUTED, lw=1.2, ls='--')
        for m in A.LEARNED:
            gs = sorted([g for g in summary.values() if g['model'] == m and g.get(col)], key=lambda g: g['k'])
            if not gs:
                continue
            ax.plot([X(g['k']) for g in gs], [g[col]['mean'] for g in gs], color=A.SLOT[m], lw=lw, zorder=2)
            multi = [g for g in gs if not A._single(g)]
            if multi:
                ax.errorbar([X(g['k']) for g in multi], [g[col]['mean'] for g in multi],
                            yerr=[g[col]['std'] for g in multi], color=A.SLOT[m], marker=A.MARKER[m], ms=ms,
                            ls='none', capsize=0, elinewidth=0.8, mec=WHITE, mew=0.6, zorder=3)
            single = [g for g in gs if A._single(g)]
            if single:
                ax.plot([X(g['k']) for g in single], [g[col]['mean'] for g in single], marker=A.MARKER[m], ms=ms,
                        mfc=WHITE, mec=A.SLOT[m], mew=1.1, ls='none', zorder=3)
        for full, base in A.FULL_OF.items():
            g = summary.get(f'{full}_k101')
            if g and g.get(col):
                if A._single(g):
                    ax.plot([FULL], [g[col]['mean']], marker=A.MARKER[base], ms=6, mfc=WHITE, mec=A.SLOT[base],
                            mew=1.4, ls='none', zorder=3)
                else:
                    ax.errorbar([FULL], [g[col]['mean']], yerr=[g[col]['std']], color=A.SLOT[base],
                                marker=A.MARKER[base], ms=6, ls='none', capsize=0, elinewidth=0.8, mec=WHITE, mew=0.6,
                                zorder=3)
        if orig:
            ax.axhline(A.ORIGINAL_MLP[orig], color=INK2, lw=0.9, ls=':')
        ax.set_xscale('log', base=2)
        ax.set_xticks([X(k) for k in allk], ['full' if k == 101 else str(k) for k in allk])
        ax.minorticks_off()
        ax.set_xlabel('History length k')
        ax.set_title(title, loc='left', fontsize=FS_TITLE, color=INK)
        _ygrid(ax)
    present = {g['model'] for g in summary.values()}
    handles = []
    for m in [m for m in A.LEARNED if m in present]:
        filled = any(not A._single(g) for g in summary.values() if g['model'] == m)
        handles.append(Line2D([], [], color=A.SLOT[m], lw=lw, marker=A.MARKER[m], ms=ms,
                              mfc=A.SLOT[m] if filled else WHITE, mec=WHITE if filled else A.SLOT[m],
                              mew=0.6 if filled else 1.1, label=STAGE_LABEL[m] + ('' if filled else ' (one seed)')))
    for full, base in A.FULL_OF.items():
        if full in present:
            single = A._single(summary[f'{full}_k101'])
            handles.append(Line2D([], [], ls='none', marker=A.MARKER[base], ms=6,
                                  mfc=WHITE if single else A.SLOT[base], mec=A.SLOT[base] if single else WHITE,
                                  mew=1.4 if single else 0.6,
                                  label=STAGE_LABEL[full] + (' (one seed)' if single else '')))
    if 'persist' in present:
        handles.append(Line2D([], [], color=MUTED, lw=1.2, ls='--', label=STAGE_LABEL['persist']))
    handles.append(Line2D([], [], color=INK2, lw=0.9, ls=':', label='Original MLP (k = 8)'))
    if R:
        handles += [Line2D([], [], color=INK2, lw=1, marker='x', ms=4, label='Stage 1: best model at the same k'),
                    Line2D([], [], color=MUTED, lw=1, ls='--', label='Stage 1: GRU (full history)')]
    fig.legend(handles=handles, loc='lower center', ncol=3 if R else 4, bbox_to_anchor=(0.5, 0.0),
               handlelength=2.2, columnspacing=1.2, fontsize=FS)
    _save(fig, name)
    return A._n_seeds(summary)


def _cmf_gru_runs():
    from cybermetaformer.hidden_state.rapor import makale_tablolar as MT
    out = {}
    for key in ('cmf', 'gru_full'):
        stage, prefix, _ = MT.RUNS[key]
        out[key] = [MT.seed_metrics(stage, prefix, s) for s in MT.run_seeds(stage, prefix)]
        assert len(out[key]) == 5, key
    return out


def fig_selected():
    """Şekil 6: seçili modellerin standart test ve OOD host makro-F1'i (nokta: seed ortalaması, çubuk: seed SD)."""
    import json

    import numpy as np

    from cybermetaformer.hidden_state.aggregate import ORIGINAL_MLP

    def stage(st):
        with open(os.path.join(RESULTS, st, 'ozet', 'ozet.json')) as f:
            return json.load(f)
    s1, s2 = stage('asama1'), stage('asama2')
    runs = _cmf_gru_runs()

    def from_summary(g):
        return [(g[c]['mean'], g[c]['std']) for c in ('test_host_macro_f1', 'ood_host_macro_f1')]

    def from_runs(rows):
        return [(float(np.mean([r[m] for r in rows])), float(np.std([r[m] for r in rows], ddof=1)))
                for m in ('test_f1', 'ood_f1')]
    rows = [('Original MLP (k = 8)', [(ORIGINAL_MLP['test'], 0), (ORIGINAL_MLP['ood_test'], 0)], OTHER_C),
            ('GRU (k = 32)', from_summary(s1['gru_k32']), OTHER_C),
            ('Transformer (k = 64)', from_summary(s2['transformer_k64']), OTHER_C),
            ('TimeMixer (k = 64)', from_summary(s2['timemixer_k64']), OTHER_C),
            ('Full-history GRU', from_runs(runs['gru_full']), GRU_C),
            ('CyberMetaFormer', from_runs(runs['cmf']), CMF_C)]
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 2.6))
    fig.subplots_adjust(left=0.2, right=0.97, bottom=0.17, top=0.9)
    for i, (label, ((t, ts), (o, os_)), color) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([o, t], [y, y], color=BASE, lw=2, zorder=1, solid_capstyle='round')
        ax.errorbar([t], [y], xerr=[ts], fmt='o', ms=6, color=color, mfc=WHITE, mew=1.5, elinewidth=0.9, zorder=3)
        ax.errorbar([o], [y], xerr=[os_], fmt='o', ms=6, color=color, mec=WHITE, mew=0.8, elinewidth=0.9, zorder=3)
        ax.text(o - os_ - 0.005, y, f'{o:.3f}', ha='right', va='center', fontsize=FS, color=INK2)
        ax.text(t + ts + 0.005, y, f'{t:.3f}', ha='left', va='center', fontsize=FS, color=INK2)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
    ax.set_xlim(0.78, 1.0)
    ax.set_xlabel('Host macro-F1')
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    _ygrid(ax, 'x')
    ax.plot([], [], 'o', color=INK2, mfc=WHITE, mew=1.5, ms=6, label='Standard test')
    ax.plot([], [], 'o', color=INK2, mec=WHITE, ms=6, label='OOD test')
    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=2, borderaxespad=0.2)
    _save(fig, 'fig_selected')


def fig_ablation(T8):
    """Şekil 8: Tablo 15'in farkları (varyant − ana model) ve %95 GA'ları; GA sıfırı dışlıyorsa koyu."""
    import numpy as np
    short = {'duz78': 'No host tokens (one 78-d token)', 'host_yok': 'No host mixer',
             'host_tam': 'Full attention, no topology bias', 'host_gcn': 'GCN-like host mixer',
             'kol_konv': 'Short convolution only', 'kol_ssd': 'SSD branch only',
             'konv_dikkat': 'Causal attention instead of SSD', 'konv_gru': 'GRU instead of SSD',
             'hafiza_yok': 'Window 32, no alert-age channel', 'b2': 'B2 scale (2.8 M parameters)'}
    keys = [k for k, _ in T8.rows]
    assert set(keys) == set(short), keys
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 3.2), sharey=True)
    fig.subplots_adjust(left=0.335, right=0.985, bottom=0.14, top=0.82, wspace=0.1)
    y = np.arange(len(keys))[::-1]
    for ax, col, title in ((axes[0], 'ood', 'Δ OOD host macro-F1'), (axes[1], 'silent', 'Δ OOD silent recall')):
        for yi, k in zip(y, keys):
            d, lo, hi = T8.cell(k, col).values
            sig = lo > 0 or hi < 0
            ax.barh(yi, d, height=0.62, color=CMF_C if sig else BASE, zorder=2)
            ax.plot([lo, hi], [yi, yi], color=INK, lw=0.9, zorder=3, solid_capstyle='butt')
        ax.axvline(0, color=INK2, lw=0.9, zorder=1)
        ax.set_title(title, loc='left', fontsize=FS_TITLE, color=INK)
        ax.set_xlabel('Variant − main model')
        _ygrid(ax, 'x')
        ax.tick_params(axis='y', length=0)
    axes[0].set_yticks(y, [short[k] for k in keys])
    axes[1].spines['left'].set_visible(False)
    axes[0].spines['left'].set_visible(False)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=CMF_C, label='95% CI excludes 0'), Patch(color=BASE, label='95% CI includes 0')],
               loc='upper left', bbox_to_anchor=(0.325, 1.0), ncol=2, frameon=False)
    _save(fig, 'fig_ablation')


def fig_attention():
    """Şekil 9: host karıştırıcısının katman başına ortalama dikkati (seed 0, ε = 0.1 test bölümleri)."""
    import json

    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    from cybermetaformer.hidden_state.models.cybermetaformer import HOSTS
    with open(os.path.join(RESULTS, 'asama3', 'aciklama', 'dikkat.json')) as f:
        d = json.load(f)
    models = [('cybermetaformer', 'CyberMetaFormer'), ('cybermetaformer-host_tam', 'No topology bias')]
    mats = {n: [np.asarray(L['matrix']) for L in d['models'][n]['0']['eps0.1']['layers']] for n, _ in models}
    vmax = max(m.max() for v in mats.values() for m in v)
    cmap = LinearSegmentedColormap.from_list('seq_blue', SEQ_BLUE)
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH_IN, 4.4))
    fig.subplots_adjust(left=0.09, right=0.88, bottom=0.08, top=0.95, wspace=0.12, hspace=0.2)
    ticks = range(len(HOSTS))
    for r, (n, label) in enumerate(models):
        for li, ax in enumerate(axes[r]):
            m = mats[n][li]
            assert m.shape == (len(HOSTS), len(HOSTS)) and np.allclose(m.sum(1), 1, atol=1e-3)
            im = ax.imshow(m, cmap=cmap, vmin=0, vmax=vmax)
            for b in (3.5, 7.5):
                ax.axhline(b, color=INK2, lw=0.6)
                ax.axvline(b, color=INK2, lw=0.6)
            ax.set_xticks(ticks, HOSTS if r == 1 else [''] * len(HOSTS), rotation=90, fontsize=6)
            ax.set_yticks(ticks, HOSTS if li == 0 else [''] * len(HOSTS), fontsize=6)
            ax.tick_params(length=0, pad=1.5)
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_title(f'{label}, layer {li + 1}', loc='left', fontsize=FS, color=INK, pad=3)
    cax = fig.add_axes([0.9, 0.22, 0.016, 0.6])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=FS, length=0, colors=INK2)
    cb.outline.set_visible(False)
    cb.set_label('Attention weight (each query row sums to 1)', fontsize=FS, color=INK2)
    _save(fig, 'fig_attention')


def fig_severity():
    """Şekil 10: react_restore'un ε'u ve mekanizma değişimleri (5 seed; ortalama ± seed SD; ozet_5seed.json)."""
    import json

    import numpy as np
    with open(os.path.join(RESULTS, 'asama3', 'ood_paketi', 'ozet_5seed.json')) as f:
        M = json.load(f)['models']
    curve = [('eps0.1', 0.0), ('ood', 1.0), ('eps0.5', 2.0), ('eps0.7', 3.0)]
    mech = [('analyse', 4.3), ('remove', 5.8)]
    models = [('cybermetaformer', 'CyberMetaFormer', CMF_C, 'o'), ('gru_full', 'Full-history GRU', GRU_C, 's')]
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH_IN, 3.9), sharex=True)
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.16, top=0.9, hspace=0.32)
    for ax, metric, title in ((axes[0], 'host_macro_f1', 'Host macro-F1'), (axes[1], 'silent_recall', 'Silent recall')):
        for j, (name, label, color, mk) in enumerate(models):
            r = M[name]['uye']
            dx = (j - 0.5) * 0.08
            def ms_(key):
                v = np.asarray(r[key]['seeds'][metric])
                assert len(v) == 5 and abs(v.mean() - r[key][metric]) < 1e-9
                return v.mean(), v.std(ddof=1)
            cm = [ms_(k) for k, _ in curve]
            ax.plot([x + dx for _, x in curve], [m for m, _ in cm], '-', color=color, lw=1.5, zorder=2)
            ax.errorbar([x + dx for _, x in curve], [m for m, _ in cm], yerr=[s for _, s in cm], fmt=mk, ms=5,
                        color=color, mec=WHITE, mew=0.6, elinewidth=0.9, zorder=3, label=label)
            mm = [ms_(k) for k, _ in mech]
            ax.errorbar([x + dx for _, x in mech], [m for m, _ in mm], yerr=[s for _, s in mm], fmt=mk, ms=5,
                        color=color, mec=WHITE, mew=0.6, elinewidth=0.9, zorder=3)
        ax.axvline(3.6, color=BASE, lw=1)
        ax.set_title(title, loc='left', fontsize=FS_TITLE, color=INK)
        _ygrid(ax)
    axes[1].set_xticks([x for _, x in curve] + [x for _, x in mech],
                       ['0.1\n(training)', '0.3\n(main OOD)', '0.5', '0.7', 'Periodic analyse\non idle steps',
                        'Remove instead\nof restore'])
    axes[1].set_xlim(-0.45, 6.5)
    axes[1].text(1.5, -0.4, 'Random-action probability ε of the restore-based defender',
                 transform=axes[1].get_xaxis_transform(), ha='center', fontsize=FS, color=INK2)
    axes[1].text(5.05, -0.4, 'Mechanism change', transform=axes[1].get_xaxis_transform(), ha='center',
                 fontsize=FS, color=INK2)
    axes[0].legend(loc='lower left', bbox_to_anchor=(0.0, 1.12), ncol=2, frameon=False)
    _save(fig, 'fig_severity')


def fig_pareto(T11):
    """Şekil 11: Tablo 18'in hücreleri: OOD host makro-F1'e karşı CPU gecikmesi (sol) ve parametre sayısı (sağ)."""
    name = {'b1': 'CyberMetaFormer (B1)', 'b2': 'CyberMetaFormer (B2)', 'gru_full': 'Full-history GRU',
            'timemixer': 'TimeMixer (k = 64)', 'transformer': 'Transformer (k = 64)'}
    rows = [(k, name[k], T11.cell(k, 'lat').values[0], T11.cell(k, 'params').values[0], T11.cell(k, 'ood').values[0])
            for k, _ in T11.rows]
    color = {'b1': CMF_C, 'gru_full': GRU_C}
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 2.6), sharey=True)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.17, top=0.96, wspace=0.08)
    offs = [{'b1': (6, 4), 'b2': (-6, -13), 'gru_full': (6, 4), 'timemixer': (-6, 5), 'transformer': (-6, -14)},
            {'b1': (6, 4), 'b2': (-6, 5), 'gru_full': (6, 4), 'timemixer': (6, -10), 'transformer': (-6, 5)}]
    for ax, xi, xlabel in ((axes[0], 2, 'CPU latency (ms per step)'), (axes[1], 3, 'Parameters (millions)')):
        for r in rows:
            k, label, x = r[0], r[1], r[xi] / (1e6 if xi == 3 else 1)
            ax.plot([x], [r[4]], 'D' if k == 'b1' else 'o', ms=6, color=color.get(k, OTHER_C), mec=WHITE, mew=0.6,
                    zorder=3)
            dx, dy = offs[xi - 2][k]
            ax.annotate(label, (x, r[4]), xytext=(dx, dy), textcoords='offset points', fontsize=FS, color=INK2,
                        ha='left' if dx > 0 else 'right')
        ax.set_xlabel(xlabel)
        _ygrid(ax)
        _ygrid(ax, 'x')
    axes[0].set_ylabel('OOD host macro-F1')
    axes[0].set_xlim(-0.05, 1.6)
    axes[1].set_xlim(0, 3.1)
    axes[0].set_ylim(0.879, 0.948)
    _save(fig, 'fig_pareto')


def main():
    from cybermetaformer.hidden_state import aggregate  # noqa: F401  (rcParams'ını önce yazsın)
    from cybermetaformer.hidden_state.rapor import makale_tablolar as MT
    plt.rcParams.update(RC)
    fig_cmf_architecture()
    fig_confusion()
    fig_network()
    fig_window_coverage()
    n1 = _memory_curve('asama1', 'fig_memory_stage1')
    n2 = _memory_curve('asama2', 'fig_memory_stage2')
    fig_selected()
    T = MT.build('m4')
    fig_ablation(T['t08_ablations'])
    fig_attention()
    fig_severity()
    fig_pareto(T['t11_efficiency'])
    print(f'şekiller → {OUT_DIR} (aşama 1: {n1} seed, aşama 2: {n2} seed)')


if __name__ == '__main__':
    main()

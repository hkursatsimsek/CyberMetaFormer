"""Ders raporu şekilleri; `makale_sekiller.py` çizim yardımcılarını buradan alır. Yeni hesap yok: şemalar elle çizilir, şiddet eğrisi kayıtlı özetten okunur.

    venv_ts/bin/python -m cybermetaformer.hidden_state.rapor.sekiller

Çıktılar `results/hidden_state/rapor/`:
- fig_ag_yapisi.png: ağ, saldırı yolu ve kestirim görevi (topoloji: models/cybermetaformer.py CONNECTED_HOSTS, SUBNET;
  saldırı yolu: veri setinin datasheet'i).
- fig_cmf_mimari.png: CyberMetaFormer'ın blok şeması.
- fig_siddet_egrisi.png: CMF ve GRU tam, üye (seed ortalaması); asama3/ood_paketi/ozet.json. Aşama 3'teki şeklin
  rapor sürümü: makine kopyası ve aksiyon varyantları yok (aksiyon raporda tabloyla veriliyor).
"""
import json
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from cybermetaformer.hidden_state.figures import BASE, GRID, INK, INK2, MUTED, ROOT, SERIES, SURFACE, USER

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'results', 'hidden_state')
OUT_DIR = os.path.join(RESULTS, 'rapor')
OOD_SUMMARY = os.path.join(RESULTS, 'asama3', 'ood_paketi', 'ozet.json')

RED, PALE, PALE2 = SERIES[1], '#eef4fc', '#f6f5f2'


def _box(ax, x, y, w, h, text, fc=SURFACE, ec=BASE, lw=1.2, fs=9, weight='normal', color=INK, ls='-', pad=0.02,
         ha='center'):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=f'round,pad={pad},rounding_size=0.12',
                                fc=fc, ec=ec, lw=lw, ls=ls))
    tx = x if ha == 'center' else x - w / 2 + 0.15
    ax.text(tx, y, text, ha=ha, va='center', fontsize=fs, fontweight=weight, color=color, linespacing=1.35)


def _arrow(ax, p, q, ls='-', color=INK2, lw=1.3, rad=0.0, style='-|>', ms=11):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, color=color, lw=lw, ls=ls,
                                 connectionstyle=f'arc3,rad={rad}', shrinkA=2, shrinkB=2))


def _region(ax, x0, y0, x1, y1, title):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle='round,pad=0,rounding_size=0.25',
                                fc=PALE2, ec=GRID, lw=1.2))
    ax.text((x0 + x1) / 2, y1 - 0.32, title, ha='center', va='center', fontsize=10, fontweight='bold', color=INK2)


def fig_ag_yapisi():
    fig = plt.figure(figsize=(10, 6.6))
    ax = fig.add_axes([0.01, 0.30, 0.98, 0.69])
    ax.set_xlim(0, 12)
    ax.set_ylim(0.4, 7.6)
    ax.axis('off')

    _region(ax, 0.2, 0.6, 3.6, 7.4, 'Kullanıcı alt ağı')
    _region(ax, 4.2, 0.6, 7.7, 7.4, 'Kurumsal alt ağ')
    _region(ax, 8.3, 0.6, 11.8, 7.4, 'Operasyonel alt ağ')

    w, h = 1.15, 0.62
    pos = {'user0': (1.0, 4.2), 'user1': (2.75, 6.0), 'user2': (2.75, 4.9), 'user3': (2.75, 3.5),
           'user4': (2.75, 2.4), 'ent0': (5.1, 5.45), 'ent1': (5.1, 2.95), 'ent2': (6.85, 4.2),
           'def': (5.95, 1.35), 'opserv': (9.25, 4.2), 'ophost0': (10.9, 5.6), 'ophost1': (10.9, 4.2),
           'ophost2': (10.9, 2.8)}
    for name, (x, y) in pos.items():
        if name == 'user0':
            _box(ax, x, y, w, h, name, fc='#fde3d7', ec=RED, lw=1.6, weight='bold')
            ax.text(x, y - 0.62, 'Red\'in dayanağı', ha='center', va='center', fontsize=8, color=RED)
        elif name == 'def':
            _box(ax, x, y, w, h, name, fc='#ecebe7', ec=MUTED, color=INK2)
            ax.text(x + 0.95, y, 'savunmacının\nmakinesi', ha='left', va='center', fontsize=8, color=INK2)
        elif name == 'opserv':
            _box(ax, x, y, w, h, name, fc=PALE, ec=ROOT, lw=2.0, weight='bold')
            ax.text(x, y - 0.62, 'kritik hedef', ha='center', va='center', fontsize=8, color=ROOT)
        else:
            _box(ax, x, y, w, h, name)

    hw = w / 2
    for u in ('user1', 'user2', 'user3', 'user4'):
        (x0, y0), (x1, y1) = pos['user0'], pos[u]
        _arrow(ax, (x0 + hw, y0), (x1 - hw, y1), ls=(0, (3, 2)), color=MUTED, lw=1.1)
    for u, e in (('user1', 'ent0'), ('user2', 'ent0'), ('user3', 'ent1'), ('user4', 'ent1'), ('ent2', 'opserv')):
        (x0, y0), (x1, y1) = pos[u], pos[e]
        _arrow(ax, (x0 + hw, y0), (x1 - hw, y1), color=ROOT, lw=1.8)
    for e in ('ent0', 'ent1'):
        (x0, y0), (x1, y1) = pos[e], pos['ent2']
        _arrow(ax, (x0 + hw, y0), (x1 - hw, y1), ls=(0, (3, 2)), color=MUTED, lw=1.1)

    ax.plot([0.35, 0.95], [0.25, 0.25], color=ROOT, lw=1.8, clip_on=False)
    ax.text(1.05, 0.25, 'host bağlantısı (saldırı grafı kenarı)', va='center', fontsize=8.5, color=INK2)
    ax.plot([4.6, 5.2], [0.25, 0.25], color=MUTED, lw=1.1, ls=(0, (3, 2)), clip_on=False)
    ax.text(5.3, 0.25, 'Red\'in keşifle ilerleyişi', va='center', fontsize=8.5, color=INK2)

    bx = fig.add_axes([0.01, 0.01, 0.98, 0.25])
    bx.set_xlim(0, 12)
    bx.set_ylim(0, 2.6)
    bx.axis('off')
    _box(bx, 2.15, 1.3, 4.0, 2.05,
         'Savunmacının gözlemi (her adım)\n13 host × 6 öznitelik = 78 sayı\n'
         'istismar ve tarama aktivitesi, güvenlik bayrağı,\ntarama geçmişi, kalan tuzak sayısı',
         fc=PALE2, ec=BASE, fs=8.8)
    _box(bx, 6.0, 1.3, 2.4, 1.2, 'Kestirici model\n(son k gözlem ya da\ntam geçmiş)', fc=PALE, ec=SERIES[0], fs=8.8,
         lw=1.5)
    _box(bx, 9.85, 1.3, 4.0, 2.05,
         'Gizli durum (simülatör bilir, savunmacı bilmez)\n12 host × {temiz, user, root}\n'
         'user: saldırgan kullanıcı yetkisiyle içeride\nroot: saldırgan yönetici yetkisiyle içeride',
         fc='#e8f0fb', ec=ROOT, fs=8.8)
    _arrow(bx, (4.2, 1.3), (4.75, 1.3), lw=1.6, ms=13)
    _arrow(bx, (7.25, 1.3), (7.8, 1.3), lw=1.6, ms=13)
    fig.savefig(os.path.join(OUT_DIR, 'fig_ag_yapisi.png'), dpi=200)
    plt.close(fig)


def fig_cmf_mimari():
    fig, ax = plt.subplots(figsize=(11, 5.0))
    fig.subplots_adjust(0.005, 0.005, 0.995, 0.995)
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # girdi
    _box(ax, 1.9, 8.2, 3.3, 1.05, 'Gözlem oₜ\n78 sayı', fc=PALE2, fs=9)
    _box(ax, 1.9, 5.4, 3.3, 2.7, '13 host token\'ı\n\nher biri: hostun 6\nözniteliği + uyarı yaşı\n'
         '(4 olay türü için son\nolaydan bu yana geçen\nadım, "hiç" bayrağı)', fc=PALE, ec=SERIES[0], fs=8.5)
    _box(ax, 1.9, 2.55, 3.3, 1.35, '+ host kimliği, alt ağ\nve adım (konum)\ngömmeleri', fc=PALE2, fs=8.5)
    _arrow(ax, (1.9, 7.65), (1.9, 6.8))
    _arrow(ax, (1.9, 4.0), (1.9, 3.27))

    # blok
    ax.add_patch(FancyBboxPatch((4.3, 0.55), 13.1, 8.9, boxstyle='round,pad=0,rounding_size=0.3', fc=SURFACE,
                                ec=INK2, lw=1.3, ls=(0, (4, 3))))
    ax.text(10.85, 9.05, 'MetaFormer bloğu  × 3   (her alt katmanda önce normalizasyon, sonra artık bağlantı)',
            ha='center', va='center', fontsize=9.5, fontweight='bold', color=INK)
    _arrow(ax, (3.6, 2.55), (4.85, 4.9), lw=1.5, rad=-0.25)

    # zaman karıştırıcı
    ax.add_patch(FancyBboxPatch((4.75, 1.05), 5.0, 7.5, boxstyle='round,pad=0,rounding_size=0.2', fc=PALE2,
                                ec=BASE, lw=1.1))
    ax.text(7.25, 8.05, 'Zaman karıştırıcı', ha='center', fontsize=9.5, fontweight='bold', color=INK)
    ax.text(7.25, 7.5, 'her host kendi geçmişinde;\nhostlar arasında ortak ağırlık',
            ha='center', va='center', fontsize=8, color=INK2)
    _box(ax, 6.05, 5.2, 2.2, 1.9, 'Kısa vade\nnedensel konv.\n(son 4 adım)', fc=SURFACE, ec=SERIES[0], fs=8.3)
    _box(ax, 8.45, 5.2, 2.2, 1.9, 'Uzun vade\nseçici durum\nuzayı (SSD,\ntam geçmiş)', fc=SURFACE, ec=SERIES[0],
         fs=8.3)
    _box(ax, 7.25, 2.7, 3.6, 1.0, 'Kapı: iki kolu her adımda\nhost başına karıştırır', fc=SURFACE, ec=BASE, fs=8.3)
    _arrow(ax, (6.05, 4.2), (6.8, 3.25))
    _arrow(ax, (8.45, 4.2), (7.7, 3.25))
    ax.text(7.25, 1.45, 'zaman ekseni', ha='center', fontsize=8, color=MUTED, style='italic')

    # host karıştırıcı
    ax.add_patch(FancyBboxPatch((10.15, 1.05), 4.0, 7.5, boxstyle='round,pad=0,rounding_size=0.2', fc=PALE2,
                                ec=BASE, lw=1.1))
    ax.text(12.15, 8.05, 'Host karıştırıcı', ha='center', fontsize=9.5, fontweight='bold', color=INK)
    ax.text(12.15, 7.5, 'her adımda 13 host\nbirbirine bakar', ha='center', va='center', fontsize=8, color=INK2)
    _box(ax, 12.15, 5.2, 3.3, 1.9, 'Çok başlıklı dikkat\n+ ilişki sapması', fc=SURFACE, ec=SERIES[0], fs=8.3)
    _box(ax, 12.15, 2.7, 3.3, 1.55, 'ilişki türleri:\nkendisi · saldırı kenarı\naynı alt ağ · diğer', fc=SURFACE,
         ec=BASE, fs=8.0)
    _arrow(ax, (12.15, 3.5), (12.15, 4.2), style='-|>')
    ax.text(12.15, 1.45, 'host ekseni', ha='center', fontsize=8, color=MUTED, style='italic')

    # kanal MLP
    ax.add_patch(FancyBboxPatch((14.55, 1.05), 2.45, 7.5, boxstyle='round,pad=0,rounding_size=0.2', fc=PALE2,
                                ec=BASE, lw=1.1))
    ax.text(15.78, 8.05, 'Kanal MLP\'si', ha='center', fontsize=9.5, fontweight='bold', color=INK)
    ax.text(15.78, 7.5, 'her token\ntek başına', ha='center', va='center', fontsize=8, color=INK2)
    _box(ax, 15.78, 5.2, 1.95, 1.9, 'iki katmanlı\nküçük ağ', fc=SURFACE, ec=SERIES[0], fs=8.3)
    ax.text(15.78, 1.45, 'öznitelik ekseni', ha='center', fontsize=8, color=MUTED, style='italic')

    _arrow(ax, (9.75, 5.2), (10.15, 5.2), lw=1.5)
    _arrow(ax, (14.15, 5.2), (14.55, 5.2), lw=1.5)

    # çıkış
    _arrow(ax, (17.4, 5.2), (18.05, 5.2), lw=1.5)
    _box(ax, 19.9, 5.2, 3.5, 2.3, 'Paylaşılan host başlığı\n\nher adımda 12 host ×\n{temiz, user, root}',
         fc='#e8f0fb', ec=ROOT, fs=8.8, lw=1.5)
    ax.text(19.9, 2.35, 'B1: 186 665 parametre\n(genişlik 64, 4 başlık)\n\nçevrimiçi kullanımda\nadım başına sabit maliyet',
            ha='center', va='center', fontsize=8, color=INK2, linespacing=1.35)
    fig.savefig(os.path.join(OUT_DIR, 'fig_cmf_mimari.png'), dpi=200)
    plt.close(fig)


def fig_siddet_egrisi():
    res = json.load(open(OOD_SUMMARY))['models']
    curve = [('eps0.1', 0.1), ('ood', 0.3), ('eps0.5', 0.5), ('eps0.7', 0.7)]
    mech = [('analyse', 0.93, 's'), ('remove', 1.08, 'D')]
    models = [('cybermetaformer', 'CyberMetaFormer', SERIES[0]), ('gru_full', 'GRU (tam geçmiş)', RED)]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for ax, metric, title in ((axes[0], 'host_macro_f1', 'Host makro-F1'), (axes[1], 'silent_recall', 'Sessiz recall')):
        for name, label, color in models:
            r = res[name]['uye']
            ax.plot([x for _, x in curve], [r[s][metric] for s, _ in curve], '-o', color=color, ms=5, lw=2,
                    label=label)
            for s, x, mk in mech:
                ax.plot([x], [r[s][metric]], mk, color=color, ms=7)
        ax.axvline(0.815, color=GRID, lw=1.2)
        ax.set_xticks([x for _, x in curve] + [x for _, x, _ in mech])
        ax.set_xticklabels(['0.1\n(eğitim)', '0.3\n(OOD)', '0.5', '0.7', 'analyse', 'remove'])
        ax.set_xlim(0.03, 1.15)
        ax.set_title(title, loc='left', fontsize=10.5, color=INK)
        ax.set_xlabel('react_restore ε  |  mekanizma politikaları', fontsize=9)
        ax.grid(axis='y', color=GRID, lw=1)
        ax.set_axisbelow(True)
    axes[0].legend(loc='lower left', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig_siddet_egrisi.png'), dpi=200)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    fig_ag_yapisi()
    fig_cmf_mimari()
    fig_siddet_egrisi()
    print('şekiller:', OUT_DIR)


if __name__ == '__main__':
    main()

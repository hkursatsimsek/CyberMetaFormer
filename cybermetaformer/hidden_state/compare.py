"""İki modelin bölüm düzeyinde eşli karşılaştırması.

Sınama birimi bölümdür: iki model aynı test/OOD bölümlerinde değerlendirildiği için her bootstrap örneğinde aynı
bölüm ağırlıklarını görür (eşli). Her model kendi seed'lerinin ortalamasıyla girer. 5 eşli seed üzerinden Wilcoxon
0.05'e inemediği için (en küçük iki yönlü p 0.0625) bu yöntem seçildi (README → Kararlar 2026-10-18).

Ölçütler (RQ3): OOD host makro-F1, göreli OOD düşüşü (test ve OOD bölümleri bağımsız örneklenir), OOD sessiz recall.
p iki yönlü, üç ölçüt üzerinden Holm düzeltmeli. 4 000 OOD bölümüyle küçük farklar da anlamlı çıkar; karar farkın
büyüklüğüne ve GA'sına dayanır.

Aşama 2'de ayrıca test host makro-F1 ve test sessiz recall ölçütleri (`--criteria`) ve aşamalar arası
karşılaştırma var: ad `<aşama>/<koşu öneki>` biçiminde verilirse o aşamadan okunur. `--plan m2` Aşama 2'nin ikili
sınamalarını topluca koşar (`m2_plan`) ve `asama2/karsilastirma/ozet.md` yazar.

Aşama 3'te RQ4'ün "hiç uyarı görülmemiş" dilimi (`*_never_recall`) ve (ii) saldırgan kaymasının
bölümleri (`test_shift_*`, `ood_test_shift_*`; yalnızca `train_red`'li koşularda) ölçüt olarak eklendi. `--plan m3`
`m3_plan`'ı koşar ve `asama3/karsilastirma/ozet.md` yazar. Bir ölçüt yalnızca kendi bölümlerini örnekler; test/OOD
ölçütlerinin bootstrap akışı M2'dekiyle aynıdır.

M4 revizyonu: ad sonundaki `@0,1,2` seed kümesini sabitler (`asama3/cybermetaformer_k101@0,1,2`).
CMF, GRU tam (asama3) ve ana_m4 artık 5 seed'li olduğundan `--plan m3` M3'teki 3 seed'i sabitleyerek M3 kaydını aynen
üretir. `--plan m3_5seed` aynı planı 5 seed kuralıyla koşar: ana çiftler bütün seed'lerle; ablasyonlar ve 3 seed'li
varyantlar CMF'nin s0–s2'sine karşı. `--plan m4` madde 4'ün host-token GRU çiftleridir (ayrı Holm ailesi). İkisi
`asama3/karsilastirma_m4/<plan>/` altına yazar; M3 kaydı değişmez.

Girdi: koşu klasörlerindeki `episode_confusion.npz` ve `episode_slices.npz` (git dışı, yalnızca Mac'te). Sessiz
recall için iki modelin de bütün seed'lerinde `episode_slices.npz` olmalı (2026-10-18'den sonraki koşular).

    python -m cybermetaformer.hidden_state.compare --a gru_k32 --b gru_full_k101      # b = referans
    python -m cybermetaformer.hidden_state.compare --stage asama2 --a transformer_k64 --b asama1/gru_k32 \
        --criteria test_host_macro_f1 ood_host_macro_f1
    python -m cybermetaformer.hidden_state.compare --plan m2
    python -m cybermetaformer.hidden_state.compare --plan m3
    python -m cybermetaformer.hidden_state.compare --plan m3_5seed --n-boot 100000
    python -m cybermetaformer.hidden_state.compare --plan m4 --n-boot 100000
"""
import argparse
import functools
import json
import os
import re

import numpy as np

from cybermetaformer.hidden_state import metrics as M

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), 'results', 'hidden_state')
SPLITS = ('test', 'ood_test')
SLICES = ('sessiz', 'hiç')                 # sessiz ihlal; hiç uyarı görülmemiş (RQ4, H4)


def load_runs(stage_dir, name, seeds=None):
    """<name>_s<seed> (ya da seed'siz <name>) klasörleri → seed başına karışıklık, dilim sayımları ve bölüm özeti.
    Karışıklık dosyasındaki bütün bölümler okunur (test, ood_test; (ii) koşularında ayrıca test_in/_shift, ood_test_in/_shift).
    seeds verilirse yalnız o seed'ler okunur ve hepsinin bulunması gerekir."""
    pat = re.compile(rf'^{re.escape(name)}(_s(\d+))?$')
    runs = []
    for d in sorted(os.listdir(stage_dir)):
        m, path = pat.match(d), os.path.join(stage_dir, d)
        if not m or not os.path.exists(os.path.join(path, 'metrics.json')):
            continue
        if seeds is not None and (m.group(2) is None or int(m.group(2)) not in seeds):
            continue
        reports = json.load(open(os.path.join(path, 'metrics.json')))['reports']
        conf = np.load(os.path.join(path, 'episode_confusion.npz'))
        splits = [s for s in conf.files if s != 'val']
        sl_path = os.path.join(path, 'episode_slices.npz')
        sl = np.load(sl_path) if os.path.exists(sl_path) else None
        slices = None if sl is None else {n: {s: sl[s][:, list(sl['names']).index(n)] for s in splits if s in sl.files}
                                          for n in SLICES}
        runs.append({'seed': int(m.group(2)) if m.group(2) else None,
                     'conf': {s: conf[s] for s in splits},
                     'silent': None if slices is None else slices['sessiz'],
                     'slices': slices,
                     'hash': {s: reports[s]['episodes_hash'] for s in splits}})
    if not runs:
        raise SystemExit(f'Koşu bulunamadı: {stage_dir}/{name}[_s<seed>]')
    if seeds is not None and sorted(r['seed'] for r in runs) != sorted(seeds):
        raise SystemExit(f'{stage_dir}/{name}: istenen seed\'ler {sorted(seeds)}, bulunan {[r["seed"] for r in runs]}')
    return runs


def _f1(conf, w):
    return M.host_macro_f1_batch((w @ conf.reshape(len(conf), -1)).reshape(-1, *conf.shape[1:]))


def _recall(counts, w):
    n_hit = w @ counts.astype(np.float64)
    return n_hit[:, 1] / np.maximum(n_hit[:, 0], 1)


def _slice(r, name, split):
    return r['silent'][split] if name == 'sessiz' else r['slices'][name][split]


# ölçüt → (iyi yön: +1 büyük daha iyi, -1 küçük daha iyi; seed başına istatistik)
CRITERIA = {
    'test_host_macro_f1': (+1, lambda r, W: _f1(r['conf']['test'], W['test'])),
    'ood_host_macro_f1': (+1, lambda r, W: _f1(r['conf']['ood_test'], W['ood_test'])),
    'ood_rel_drop': (-1, lambda r, W: 1 - _f1(r['conf']['ood_test'], W['ood_test']) / _f1(r['conf']['test'], W['test'])),
    'ood_silent_recall': (+1, lambda r, W: _recall(r['silent']['ood_test'], W['ood_test'])),
    'test_silent_recall': (+1, lambda r, W: _recall(r['silent']['test'], W['test'])),
}
for _c, _s in (('test', 'test'), ('ood', 'ood_test')):
    CRITERIA[f'{_c}_never_recall'] = (+1, lambda r, W, s=_s: _recall(_slice(r, 'hiç', s), W[s]))
SHIFT_SPLITS = {}                          # ölçüt → örneklediği bölümler (varsayılan SPLITS)
for _s in ('test_shift', 'ood_test_shift'):
    CRITERIA[f'{_s}_host_macro_f1'] = (+1, lambda r, W, s=_s: _f1(r['conf'][s], W[s]))
    CRITERIA[f'{_s}_silent_recall'] = (+1, lambda r, W, s=_s: _recall(_slice(r, 'sessiz', s), W[s]))
    SHIFT_SPLITS.update({f'{_s}_host_macro_f1': (_s,), f'{_s}_silent_recall': (_s,)})
RQ3 = ['ood_host_macro_f1', 'ood_rel_drop', 'ood_silent_recall']
SILENT = {'ood_silent_recall', 'test_silent_recall'}
# ölçütün dilimi (dilim dosyası ister)
SLICE_OF = {**{c: 'sessiz' for c in CRITERIA if c.endswith('silent_recall')},
            **{c: 'hiç' for c in CRITERIA if c.endswith('never_recall')}}


def _has_slice(r, name):
    return r['silent'] is not None if name == 'sessiz' else bool(r.get('slices'))


def compare(runs_a, runs_b, n_boot=10_000, seed=0, criteria=RQ3):
    for s in dict.fromkeys([*SPLITS, *(s for c in criteria for s in SHIFT_SPLITS.get(c, ()))]):
        hashes = {r['hash'][s] for r in runs_a + runs_b}
        assert len(hashes) == 1, f'{s}: iki model aynı bölümlerde değerlendirilmemiş'
    has_silent = all(r['silent'] is not None for r in runs_a + runs_b)
    out = {}
    for name in criteria:
        better, stat = CRITERIA[name]
        if name in SLICE_OF and not all(_has_slice(r, SLICE_OF[name]) for r in runs_a + runs_b):
            continue
        sizes = {s: len(runs_a[0]['conf'][s]) for s in SHIFT_SPLITS.get(name, SPLITS)}
        ones = {s: np.ones((1, E)) for s, E in sizes.items()}
        sa = lambda W, stat=stat: np.mean([stat(r, W) for r in runs_a], 0)     # noqa: E731
        sb = lambda W, stat=stat: np.mean([stat(r, W) for r in runs_b], 0)     # noqa: E731
        res = M.paired_bootstrap(sa, sb, sizes, n_boot=n_boot, seed=seed)
        out[name] = {'a': float(sa(ones)[0]), 'b': float(sb(ones)[0]), **res, 'better': better}
    for name, p in zip(out, M.holm([v['p'] for v in out.values()])):
        v = out[name]
        v['p_holm'] = p
        v['a_better'] = bool(p < 0.05 and np.sign(v['diff']) == v['better'])
        v['b_better'] = bool(p < 0.05 and np.sign(v['diff']) == -v['better'])
    return out, has_silent


def _resolve(stage, name):
    """`<aşama>/<önek>[@s,s,…]` ya da yalnız önek (varsayılan aşama) → (aşama, önek, seed'ler ya da None)."""
    name, _, pin = name.partition('@')
    seeds = tuple(int(x) for x in pin.split(',')) if pin else None
    st, n = tuple(name.split('/', 1)) if '/' in name else (stage, name)
    return st, n, seeds


def run_pair(stage, a, b, criteria=RQ3, n_boot=10_000, seed=0, cache=None, verbose=True, out_dir=None):
    """a − b sınaması; sonuç `<out_dir>/<a>__<b>.json`'a yazılır (varsayılan `<stage>/karsilastirma`; başka aşamadaki
    ad `<aşama>-<önek>`). Sabit seed'ler (`@…`) dosya adına ve a/b alanına girmez; seeds_a/seeds_b'de kayıtlıdır."""
    cache = {} if cache is None else cache
    (sa, na, pa), (sb, nb, pb) = _resolve(stage, a), _resolve(stage, b)
    for key in [(sa, na, pa), (sb, nb, pb)]:
        if key not in cache:
            cache[key] = load_runs(os.path.join(RESULTS, key[0]), key[1], key[2])
    runs_a, runs_b = cache[sa, na, pa], cache[sb, nb, pb]
    res, has_silent = compare(runs_a, runs_b, n_boot, seed, criteria)
    if verbose:
        print(f"{a} ({len(runs_a)} seed) − {b} ({len(runs_b)} seed), eşli bölüm bootstrap'ı, {n_boot} örnek, Holm")
        print(f"{'ölçüt':20s} {'a':>7s} {'b':>7s} {'a − b':>8s}  {'%95 GA':19s} {'p':>7s} {'p Holm':>7s}")
        for name, v in res.items():
            print(f"{name:20s} {v['a']:7.4f} {v['b']:7.4f} {v['diff']:+8.4f}  [{v['lo']:+.4f}, {v['hi']:+.4f}] "
                  f"{v['p']:7.4f} {v['p_holm']:7.4f}")
        if not has_silent and SILENT & set(criteria):
            print('Sessiz recall atlandı: en az bir koşuda episode_slices.npz yok.')
    out_dir = out_dir or os.path.join(RESULTS, stage, 'karsilastirma')
    os.makedirs(out_dir, exist_ok=True)
    tag = lambda st, n: n if st == stage else f'{st}-{n}'      # noqa: E731
    path = os.path.join(out_dir, f'{tag(sa, na)}__{tag(sb, nb)}.json')
    rec = {'a': a.partition('@')[0], 'b': b.partition('@')[0], 'seeds_a': [r['seed'] for r in runs_a], 'seeds_b': [r['seed'] for r in runs_b],
           'n_boot': n_boot, 'bootstrap_seed': seed, 'results': res}
    with open(path, 'w') as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)
    if verbose:
        print('→', path)
    return rec


F1 = ['test_host_macro_f1', 'ood_host_macro_f1']


def m2_plan():
    """Aşama 2'nin ikili sınamaları: [(aile, a, b, ölçütler)], adlar `<aşama>/<önek>`. Referanslar README Kararlar
    2026-10-18'e göre val ile seçilir (`aggregate.m1_refs`; `asama1/ozet/ozet.json` ve `asama2/ozet/ozet.json`).
    - k=8 ve k=32 (altı modelin hepsinin olduğu k'lar): her M2 modeli − aynı k'daki en iyi M1 modeli;
    - k=64: hafif dörtlü − en iyi pencereli M1 modeli (M1'de k=64'te yalnız XGB-TF var);
    - RQ3: her M2 modeli val'e göre en iyi k'sında − tam geçmiş GRU (OOD F1, göreli düşüş, OOD sessiz recall)."""
    from cybermetaformer.hidden_state.aggregate import M2_MODELS, _best, m1_refs
    s1, s2 = (json.load(open(os.path.join(RESULTS, st, 'ozet', 'ozet.json'))) for st in ['asama1', 'asama2'])
    R = m1_refs(s1)
    plan = []
    for k in (8, 32):
        b = R['best'][k]
        plan += [(f'k={k}: aynı k\'daki en iyi M1', f'asama2/{m}_k{k}', f"asama1/{b['model']}_k{k}", F1)
                 for m in M2_MODELS]
    bw = R['best_window']
    plan += [('k=64: en iyi pencereli M1', f'asama2/{m}_k64', f"asama1/{bw['model']}_k{bw['k']}", F1)
             for m in M2_MODELS if f'{m}_k64' in s2]
    plan += [('RQ3: tam geçmiş GRU', f"asama2/{m}_k{_best(s2, m)['k']}", 'asama1/gru_full_k101', RQ3)
             for m in M2_MODELS]
    return plan


CMF = 'cybermetaformer'
M3_CORE = ['duz78', 'host_yok', 'host_tam', 'host_gcn', 'konv_dikkat', 'hafiza_yok', 'yalniz_yas', 'yalniz_ssd',
           'konum_yok']
M3_EXTRA = ['kol_konv', 'kol_ssd', 'konv_gru', 'havuz_mlp']
M3_DG = ['dg_groupdro', 'dg_vrex', 'dg_coral', 'dg_dann', 'art_uyari', 'art_zaman']
ABL = ['test_host_macro_f1', 'ood_host_macro_f1', 'ood_silent_recall']
NEVER = ['test_never_recall', 'ood_never_recall']
SHIFT = ['test_shift_host_macro_f1', 'ood_test_shift_host_macro_f1', 'ood_test_shift_silent_recall']


def m3_plan(five=False):
    """Aşama 3'ün ikili sınamaları: [(aile, a, b, ölçütler)].
    five=False: M3 kaydı; sonradan 5 seed'e tamamlanan CMF, ana_m4 ve asama3 GRU tamı M3'teki s0–s2'ye sabitlenir.
    five=True (M4 revizyonu, §A5): RQ3 ve makineler arası tekrar bütün seed'lerle; 3 seed'li varyantların karşısındaki
    referans s0–s2'ye sabitlenir (aynı seed kümesi). Karşılaştırmalar aynı makinede: ablasyonlar
    M3 Pro'daki ana modele, alan genelleme / artırma / aksiyon / B2 M4 Pro'daki `ana_m4`'e karşı; gru_full aksiyonu
    asama3'teki 3 seed GRU tamına (M3 Pro; makineler arası).
    - RQ3: ana model − çıtalar (M2 modelleri val'e göre en iyi k'larında);
    - RQ4 (H4): host karıştırıcılı modeller − karıştırıcısız / düz model, "hiç uyarı görülmemiş" dilimi recall'u;
    - (ii): aynı Red'le eğitilmiş modeller, görülmemiş Red'in test ve OOD bölümlerinde."""
    from cybermetaformer.hidden_state.aggregate import _best, m1_refs
    s1, s2 = (json.load(open(os.path.join(RESULTS, st, 'ozet', 'ozet.json'))) for st in ['asama1', 'asama2'])
    R = m1_refs(s1)
    S3 = '@0,1,2'
    ana3, m43, gru3 = f'asama3/{CMF}_k101{S3}', f'asama3/{CMF}-ana_m4_k101{S3}', f'asama3/gru_full_k101{S3}'
    ana, m4 = (f'asama3/{CMF}_k101', f'asama3/{CMF}-ana_m4_k101') if five else (ana3, m43)
    refs = ['asama1/gru_full_k101', f"asama1/{R['best'][32]['model']}_k32",
            *(f"asama2/{m}_k{_best(s2, m)['k']}" for m in ('timemixer', 'transformer', 'patchtst'))]
    plan = [('RQ3: ana model − çıtalar', ana, b, ['test_host_macro_f1', *RQ3]) for b in refs]
    ana, m4 = ana3, m43                                        # bundan sonrası 3 seed'li varyantlara karşı
    plan += [('Ablasyon çekirdeği (M3 Pro, ana modele karşı)', f'asama3/{CMF}-{v}_k101', ana, ABL) for v in M3_CORE]
    plan += [('Ablasyon eki (M3 Pro, ana modele karşı)', f'asama3/{CMF}-{v}_k101', ana, ABL) for v in M3_EXTRA]
    plan += [('RQ4 (H4): hiç uyarı görülmemiş dilim', a, f'asama3/{CMF}-{b}_k101', NEVER)
             for a, b in [(ana, 'host_yok'), (ana, 'duz78'), (f'asama3/{CMF}-host_tam_k101', 'host_yok'),
                          (f'asama3/{CMF}-host_gcn_k101', 'host_yok')]]
    plan += [('Alan genelleme ve artırma (M4 Pro, ana_m4\'e karşı)', f'asama3/{CMF}-{v}_k101', m4,
              ['test_host_macro_f1', *RQ3]) for v in M3_DG]
    plan += [('Aksiyon girdisi', f'asama3/{CMF}-aksiyon_k101', m4, [*ABL, 'test_silent_recall']),
             ('Aksiyon girdisi', 'asama3/gru_full-aksiyon_k101', gru3, [*ABL, 'test_silent_recall'])]
    plan += [('Ölçek ve makineler arası tekrar', f'asama3/{CMF}-b2_k101', m4, ABL),
             ('Ölçek ve makineler arası tekrar', *((f'asama3/{CMF}-ana_m4_k101', f'asama3/{CMF}_k101') if five
                                                   else (m4, ana)), ABL)]
    for red in ('bline', 'meander'):
        a = f'asama3/{CMF}-saldirgan_{red}_k101'
        plan += [(f'(ii) saldırgan kayması: yalnız {red} ile eğitim', a, b, SHIFT)
                 for b in (f'asama3/gru_full-saldirgan_{red}_k101', f'asama3/timemixer-saldirgan_{red}_k64')]
    return plan


M4_ALL = ['test_host_macro_f1', 'ood_host_macro_f1', 'ood_rel_drop', 'test_silent_recall', 'ood_silent_recall',
          'test_never_recall', 'ood_never_recall']


def m4_plan():
    """M4 revizyonu, madde 4: host-token GRU + hostlar arası dikkat referansları, 5 seed. Aynı makine
    kuralıyla gru_host_* (M4 Pro) M4 Pro'daki ana_m4'e karşı; GRU tamı (M3 Pro) ile kıyas makineler arası
    (ana_m4 − ana tekrarı m3_5seed'de)."""
    m4, gru = f'asama3/{CMF}-ana_m4_k101', 'asama1/gru_full_k101'
    topo, attn = f'asama3/{CMF}-gru_host_topo_k101', f'asama3/{CMF}-gru_host_attn_k101'
    return [('A4: host-token GRU − CMF (M4 Pro, ana_m4\'e karşı)', topo, m4, M4_ALL),
            ('A4: host-token GRU − CMF (M4 Pro, ana_m4\'e karşı)', attn, m4, M4_ALL),
            ('A4: topoloji sapması, host-token GRU içinde (M4 Pro)', topo, attn, M4_ALL),
            ('A4: host-token GRU − tam geçmiş GRU (makineler arası)', topo, gru, M4_ALL),
            ('A4: host-token GRU − tam geçmiş GRU (makineler arası)', attn, gru, M4_ALL)]


# plan → (aşama, plan üreticisi, başlık, çıktı klasörü (aşamaya göre), tabloda seed sütunu)
PLANS = {'m2': ('asama2', m2_plan, 'M2', 'karsilastirma', False),
         'm3': ('asama3', m3_plan, 'M3', 'karsilastirma', False),
         'm3_5seed': ('asama3', functools.partial(m3_plan, five=True), 'M3, 5 seed (M4 revizyonu, madde 5)',
                      os.path.join('karsilastirma_m4', 'm3_5seed'), True),
         'm4': ('asama3', m4_plan, 'M4 revizyonu, madde 4', os.path.join('karsilastirma_m4', 'm4'), True)}


def run_plan(name, n_boot=10_000, seed=0, out_dir=None):
    """Planı koşar; her çift için JSON, ayrıca `<çıktı klasörü>/ozet.md` (çift içi ve plan geneli Holm)."""
    stage, make, title, sub, show_seeds = PLANS[name]
    out_dir = out_dir or os.path.join(RESULTS, stage, sub)
    cache, rows = {}, []
    plan = make()
    for i, (fam, a, b, crit) in enumerate(plan, 1):
        print(f'[{i}/{len(plan)}] {a} − {b}', flush=True)
        rec = run_pair(stage, a, b, crit, n_boot, seed, cache, verbose=False, out_dir=out_dir)
        rows += [(fam, a, b, name, v, rec) for name, v in rec['results'].items()]
    for row, p in zip(rows, M.holm([r[4]['p'] for r in rows])):
        row[4]['p_holm_plan'] = p
    lab = lambda n: n.partition('@')[0].split('/', 1)[1]        # noqa: E731
    seed_note = ("Her model belirtilen seed'lerinin ortalamasıyla girer (sütun: seed sayısı). "
                 if show_seeds else "Her model kendi seed'lerinin ortalamasıyla girer. ")
    lines = [f'# {title} ikili sınamaları (eşli bölüm bootstrap\'ı)', '',
             f"{n_boot} örnek, bootstrap seed {seed}. {seed_note}a − b; GA %95. "
             "p Holm: çift içi ölçütler üzerinden; plan: bütün satırlar üzerinden. En küçük p 2 / (n + 1). "
             f"Göreli düşüşte küçük daha iyi. `python -m cybermetaformer.hidden_state.compare --plan {name}`.", '']
    if show_seeds:
        lines[-1:-1] = [f'Plan geneli Holm ailesi: {len(rows)} satır; en küçük ham p {2 / (n_boot + 1):.2g}, plan '
                        f'geneli en küçük düzeltilmiş p {min(1.0, len(rows) * 2 / (n_boot + 1)):.2g}.']
    for fam in dict.fromkeys(r[0] for r in rows):
        if show_seeds:
            lines += [f'## {fam}', '', '| a | b | seed a / b | ölçüt | a | b | a − b | %95 GA | p Holm (çift) | '
                      'p Holm (plan) |', '|---|---|---|---|---:|---:|---:|---|---:|---:|']
        else:
            lines += [f'## {fam}', '', '| a | b | ölçüt | a | b | a − b | %95 GA | p Holm (çift) | p Holm (plan) |',
                      '|---|---|---|---:|---:|---:|---|---:|---:|']
        for _, a, b, name, v, rec in [r for r in rows if r[0] == fam]:
            sd = (f" {len(rec['seeds_a'])} / {len(rec['seeds_b'])} |" if show_seeds else '')
            lines.append(f"| {lab(a)} | {lab(b)} |{sd} {name} | {v['a']:.4f} | {v['b']:.4f} | {v['diff']:+.4f} | "
                         f"[{v['lo']:+.4f}, {v['hi']:+.4f}] | {v['p_holm']:.4f} | {v['p_holm_plan']:.4f} |")
        lines.append('')
    path = os.path.join(out_dir, 'ozet.md')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    # plan geneli Holm'lu satırlar tek dosyada (makale tabloları M3 kaydını da buradan okur)
    with open(os.path.join(out_dir, 'ozet.json'), 'w') as f:
        json.dump({'plan': name, 'n_boot': n_boot, 'bootstrap_seed': seed, 'n_rows': len(rows),
                   'rows': [{'family': fam, 'model_a': rec['a'], 'model_b': rec['b'], 'seeds_a': rec['seeds_a'],
                             'seeds_b': rec['seeds_b'], 'criterion': c, **v}
                            for fam, _, _, c, v, rec in rows]}, f, indent=1, ensure_ascii=False)
    print('→', path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--a', help='karşılaştırılan model, koşu adı öneki (ör. gru_k32; başka aşama: asama1/gru_k32)')
    p.add_argument('--b', help='referans model (ör. gru_full_k101)')
    p.add_argument('--stage', default='asama1')
    p.add_argument('--criteria', nargs='+', default=RQ3, choices=list(CRITERIA))
    p.add_argument('--plan', choices=list(PLANS), help='Aşama 2 / Aşama 3 / M4 revizyonu sınamalarını topluca koş')
    p.add_argument('--n-boot', type=int, default=10_000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out-dir', help='yalnız --plan: çıktı klasörü (varsayılan plan tanımındaki)')
    args = p.parse_args()
    if args.plan:
        run_plan(args.plan, args.n_boot, args.seed, args.out_dir)
    elif args.a and args.b:
        run_pair(args.stage, args.a, args.b, args.criteria, args.n_boot, args.seed)
    else:
        p.error('--a ve --b ya da --plan gerekli')


if __name__ == '__main__':
    main()

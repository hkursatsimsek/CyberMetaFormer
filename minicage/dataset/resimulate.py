"""Kayıtlı bir veri sürümünü aynı seed'lerle yeniden simüle eder; Blue aksiyonları kayıttan zorlanır.

Ortam dinamiğine dokunmayan, yalnız Blue gözlemini değiştiren bir düzeltmenin etkisini aynı bölümlerde (eşli) ölçmek
için kullanılır. Varsayılan ortam FixedObsCAGE'dir: başarısız remove/restore gözleme yansımaz (env.py).

Politikalar yine çağrılır ama çıktıları atılır. Blue politikalarının RNG tüketimi gözleme bağlı değildir:
React_restore, Restore_decoys, Sleep, ReactRestoreAnalyse ve ReactRemove deterministiktir; RandomBlue ile EpsilonBlue
her adımda sabit boyutta çeker. Red gözlemi Blue gözleminden bağımsızdır. Bu yüzden rastgele sayı akışı ve gerçek durum
kayıttakiyle aynı kalmalıdır; kapı bunu shard başına sınar.

Kapı (biri bozulursa çıkış kodu 1):
  1. obs_blue dışındaki bütün alanlar kayıtla bayt bayt aynı (dtype ve biçim dahil).
  2. Her fark zinciri, yani bir hostta farkın başladığı adım, bir nedene bağlanır: bir önceki adımda o hosta yapılmış
     başarısız (blue_success = -1) bir remove ya da restore, ya da aynı bölümde bir önceki adımda herhangi bir hostta
     tarama biti farkı (hostlar arası bağ: _process_state toplu aktivite sıfırlaması, minimal.py:1186-1194).
  3. Fark türleri nedenle uyumlu: güvenlik bitlerini (host başına 4 bitin son ikisi) yalnız remove ve restore,
     aktivite bitlerini ve tarama geçmişini yalnız restore ve hostlar arası bağ değiştirir; decoy sütunları hiç
     değişmez.
--env FixedCAGE ile hiç fark çıkmamalıdır (zorlanan aksiyonlarla yeniden üretim sınaması).

Kullanım (depo kökünden):
    venv_ts/bin/python -m minicage.dataset.resimulate --src minicage/data/minicage_v1 --out minicage/data/minicage_v1_fixedobs
    venv_ts/bin/python -m minicage.dataset.resimulate --src minicage/data/minicage_v1 --out <boş_klasör> --env FixedCAGE

Var olan veri sürümlerinin üzerine yazılmaz: --out boş değilse başlamaz. splits.npz kaynaktan kopyalanır (bölüm
kimlikleri aynıdır).
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time

import numpy as np

from minicage.dataset.env import FixedCAGE, FixedObsCAGE
from minicage.dataset.generate import run_shard
from minicage.dataset.policies import NUM_HOSTS, REMOVE, RESTORE, USER0

ENVS = {'FixedObsCAGE': FixedObsCAGE, 'FixedCAGE': FixedCAGE}
FIX_NOTE = ('_process_state yalnız başarılı Blue aksiyonlarını görür (blue_success > 0): başarısız remove [1,0] '
            '"kaldırıldı" basmaz; geçersiz user0 restore user0\'ın güvenlik bayrağını ve aktivite bitlerini (dolayısıyla '
            'tarama geçmişini) silmez. Gerçek durum, Red gözlemi, ödül ve kaydedilen blue_success değişmez.')
NONE, BY_REMOVE, BY_RESTORE, BY_COUPLING = 0, 1, 2, 4     # fark zincirinin nedeni (bit maskesi)


def gate(src, new):
    """Kapının üç koşulunu sınar. src, new: alan adı → dizi. Döndürür: (sorunlar, sayımlar)."""
    problems = []
    if set(src) != set(new):
        problems.append(f'alan kümesi farklı: {sorted(set(src) ^ set(new))}')
    for k in sorted((set(src) & set(new)) - {'obs_blue'}):
        a, b = src[k], new[k]
        if a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a, b):
            problems.append(f'{k} kayıttan farklı')
    ob0, ob1 = src['obs_blue'], new['obs_blue']
    if ob0.shape != ob1.shape or ob0.dtype != ob1.dtype:
        return problems + ['obs_blue biçimi farklı'], {}

    N, (E, T1) = NUM_HOSTS, ob0.shape[:2]
    diff = ob0 != ob1
    per_host = diff[:, :, :4 * N].reshape(E, T1, N, 4)
    d_act, d_safe = per_host[..., :2].any(-1), per_host[..., 2:].any(-1)        # (bölüm, gözlem adımı, host)
    d_scan, d_decoy = diff[:, :, 4 * N:5 * N], diff[:, :, 5 * N:6 * N]
    if d_decoy.any():
        problems.append(f'obs_blue: decoy sütunlarında {int(d_decoy.sum())} fark')
    host = d_act | d_safe | d_scan

    # Başarısız aksiyonun nedeni, ürettiği gözlem adımına (t + 1) ve hedef hostuna yazılır.
    a, s = src['act_blue'].astype(int), src['blue_success']
    cause = np.zeros(host.shape, np.int8)
    for lo, hi, bit in ((REMOVE, RESTORE, BY_REMOVE), (RESTORE, RESTORE + N, BY_RESTORE)):
        e, t = np.nonzero((a >= lo) & (a < hi) & (s == -1))
        cause[e, t + 1, a[e, t] - lo] |= bit
    # Hostlar arası bağ: önceki adımda herhangi bir hostta tarama biti (aktivite biti 0) varsa _process_state o
    # bölümdeki bütün hostların aktivitesini sıfırlar, bu adımın yeşil taramasını da (minimal.py:1186-1194). Bir hosttaki
    # tarama biti farkı bu yolla bir sonraki adımda başka hostların aktivite ve tarama sütunlarına geçebilir.
    coupled = per_host[:, :-1, :, 0].any(-1)                       # (bölüm, gözlem adımı - 1)
    cause[:, 1:] |= np.where(coupled, BY_COUPLING, NONE)[:, :, None].astype(np.int8)

    # Zincir: bir hostta ardışık farklı adımlar. Etiketi, zincir boyunca görülen nedenlerin birleşimidir.
    start = host.copy()
    start[:, 1:] &= ~host[:, :-1]
    if start[:, 0].any():
        problems.append('ilk gözlemde fark')
    orphan = start & (cause == NONE)
    if orphan.any():
        problems.append(f'{int(orphan.sum())} fark zinciri başarısız bir remove/restore ile başlamıyor')
    origin = np.zeros(host.shape, np.int8)
    label = np.zeros((E, N), np.int8)
    for i in range(T1):
        label = np.where(host[:, i], label | cause[:, i], NONE)
        origin[:, i] = label
    # remove yalnız güvenlik bayrağını yazar; restore aktivite bitlerini de sıfırlar (tarama geçmişi buradan beslenir);
    # hostlar arası bağ yalnız aktiviteyi taşır.
    outside = (d_act | d_scan) & ((origin & (BY_RESTORE | BY_COUPLING)) == 0)
    if outside.any():
        problems.append(f'obs_blue: restore ya da hostlar arası bağ kaynaklı olmayan {int(outside.sum())} '
                        'aktivite/tarama farkı')
    unsafe = d_safe & ((origin & (BY_REMOVE | BY_RESTORE)) == 0)
    if unsafe.any():
        problems.append(f'obs_blue: remove/restore kaynaklı olmayan {int(unsafe.sum())} güvenlik biti farkı')

    own = (cause & (BY_REMOVE | BY_RESTORE)) > 0
    label_hosts = np.arange(N) != USER0
    counts = {
        'failed_remove_actions': int(((a >= REMOVE) & (a < RESTORE) & (s == -1)).sum()),
        'failed_restore_actions': int(((a >= RESTORE) & (s == -1)).sum()),
        'chains_remove': int((start & ((cause & BY_REMOVE) > 0)).sum()),
        'chains_restore': int((start & ((cause & BY_RESTORE) > 0)).sum()),
        'chains_coupling_only': int((start & ~own & (cause > 0)).sum()),
        'host_steps_changed_remove': int((origin == BY_REMOVE).sum()),
        'host_steps_changed_restore': int((origin == BY_RESTORE).sum()),
        'host_steps_changed_other': int(((origin != NONE) & (origin != BY_REMOVE) & (origin != BY_RESTORE)).sum()),
        'label_host_steps_changed': int(host[:, :, label_hosts].sum()),
        'label_host_safety_steps_changed': int(d_safe[:, :, label_hosts].sum()),
        'label_host_activity_scan_steps_changed': int((d_act | d_scan)[:, :, label_hosts].sum()),
        'user0_steps_changed': int(host[:, :, USER0].sum()),
        'episodes_changed': int(host.any(axis=(1, 2)).sum()),
    }
    return problems, counts


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description='Kayıtlı veri sürümünü zorlanan Blue aksiyonlarıyla yeniden simüle eder.')
    p.add_argument('--src', required=True, help='kaynak veri sürümü (meta.json, shard_*.npz, splits.npz)')
    p.add_argument('--out', required=True, help='boş ya da olmayan hedef klasör')
    p.add_argument('--env', default='FixedObsCAGE', choices=sorted(ENVS))
    args = p.parse_args()

    if os.path.isdir(args.out) and os.listdir(args.out):
        sys.exit(f'{args.out} boş değil; var olan veri sürümlerinin üzerine yazılmaz.')
    meta_path = os.path.join(args.src, 'meta.json')
    with open(meta_path) as f:
        meta = json.load(f)
    os.makedirs(args.out, exist_ok=True)

    totals, per_shard, all_problems, t0 = {}, [], [], time.time()
    for s in meta['shards']:
        with np.load(os.path.join(args.src, s['file'])) as z:
            src = {k: z[k] for k in z.files}
        data, _ = run_shard(s['red'], s['blue'], s['episodes'], meta['steps_per_episode'], meta['remove_bugs'],
                            s['seed'], env_cls=ENVS[args.env], blue_actions=src['act_blue'])
        problems, counts = gate(src, data)
        np.savez_compressed(os.path.join(args.out, s['file']), **data)
        per_shard.append({'file': s['file'], 'red': s['red'], 'blue': s['blue'], 'problems': problems, **counts})
        for k, v in counts.items():
            totals[k] = totals.get(k, 0) + v
        all_problems += [f"{s['file']}: {q}" for q in problems]
        status = 'kapı geçti' if not problems else 'KAPI: ' + '; '.join(problems)
        print(f"{s['file']}: {s['red']:8s} x {s['blue']:22s} değişen bölüm {counts.get('episodes_changed', 0):4d}, "
              f"zincir remove/restore {counts.get('chains_remove', 0):5d}/{counts.get('chains_restore', 0):5d}  "
              f"{status}  ({time.time() - t0:.0f} s)", flush=True)

    shutil.copyfile(os.path.join(args.src, 'splits.npz'), os.path.join(args.out, 'splits.npz'))
    src_name = meta.get('name', os.path.basename(os.path.normpath(args.src)))
    out = dict(meta)
    out.update({
        'name': os.path.basename(os.path.normpath(args.out)),
        'source': (f'CybORG++ MiniCAGE ({args.env}); {src_name} kaydından yeniden simülasyon, '
                   'Blue aksiyonları kayıttan zorlandı'),
        'observation_fix': FIX_NOTE if args.env == 'FixedObsCAGE' else None,
        'resimulated_from': {'name': src_name, 'meta_sha256': _sha256(meta_path), 'command': meta.get('command')},
        'gate': {'passed': not all_problems, 'problems': all_problems, 'totals': totals, 'shards': per_shard},
        'versions': {'python': platform.python_version(), 'numpy': np.__version__},
        'generation_seconds': round(time.time() - t0, 1),
        'command': ' '.join(sys.argv),
    })
    with open(os.path.join(args.out, 'meta.json'), 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print('toplam:', json.dumps(totals, ensure_ascii=False))
    if all_problems:
        print(f'KAPI GEÇMEDİ ({len(all_problems)} sorun); veri {args.out} altında incelemeye bırakıldı.')
        sys.exit(1)
    print(f'kapı geçti: {args.out} ({out["generation_seconds"]} s)')


if __name__ == '__main__':
    main()

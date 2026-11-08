"""Görev başına hızlı referans modeller (torch, CPU). Sonuçlar baselines.json'a yazılır.

    venv_cyborg/bin/python -m minicage.dataset.baselines
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

from minicage.dataset import loaders as L
from minicage.dataset.env import FixedCAGE
from minicage.dataset.generate import ROOT
from minicage.dataset.policies import make_red, make_blue

torch.manual_seed(0)
torch.set_num_threads(4)


def subsample(n, max_n, seed=0):
    idx = np.arange(n)
    return idx if n <= max_n else np.sort(np.random.default_rng(seed).choice(n, max_n, replace=False))


def mlp(d_in, d_out, h=256):
    return nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, d_out))


def fit(model, X, y, loss_fn, epochs=5, bs=1024, lr=1e-3):
    X, y = torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss_fn(model(X[b]), y[b]).backward()
            opt.step()
    return model


@torch.no_grad()
def predict(model, X, bs=65536):
    X = torch.as_tensor(X, dtype=torch.float32)
    return torch.cat([model(X[i:i + bs]) for i in range(0, len(X), bs)])


def macro_f1(y, p, classes):
    f1s = []
    for c in classes:
        tp, fp, fn = ((p == c) & (y == c)).sum(), ((p == c) & (y != c)).sum(), ((p != c) & (y == c)).sum()
        if tp + fn > 0:
            f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f1s))


def detection(raw, sp, results):
    for k in [1, 4, 8]:
        detection_k(raw, sp, results, k)


def detection_k(raw, sp, results, k):
    H = len(L.LABEL_HOSTS)
    tr = sp['train'][subsample(len(sp['train']), 4000)]
    X, y, _ = L.detection_view(raw, tr, k)
    model = fit(mlp(X.shape[1], H * 3), X, y.astype(np.int64),
                lambda o, t: nn.functional.cross_entropy(o.view(-1, 3), t.view(-1)))
    for split in ['test', 'ood_test']:
        Xs, ys, ya = L.detection_view(raw, sp[split], k)
        pred = predict(model, Xs).view(-1, H, 3).argmax(-1).numpy()
        f1 = np.mean([macro_f1(ys[:, h], pred[:, h], [0, 1, 2]) for h in range(H) if ys[:, h].any()])
        # saf kural: son gözlemde "istismar edildi" bayrağı olan host = ele geçirilmiş
        last = Xs[:, -78:-26].reshape(-1, 13, 4)[:, L.LABEL_HOSTS, 3] == 1
        any_pred, any_rule = (pred > 0).any(1), last.any(1)
        bin_f1 = lambda p: macro_f1(ya, p.astype(np.int8), [1])
        results += [
            dict(task='tespit (host x 3 sınıf)', split=split, model=f'MLP, k={k}', metric='ort. host makro-F1',
                 value=f1, reference=f"çoğunluk sınıfı: {np.mean([macro_f1(ys[:, h], np.zeros_like(ys[:, h]), [0, 1, 2]) for h in range(H) if ys[:, h].any()]):.3f}"),
            dict(task='tespit (ağda ihlal var mı)', split=split, model=f'MLP, k={k}', metric='F1',
                 value=bin_f1(any_pred), reference=f'uyarı kuralı: {bin_f1(any_rule):.3f}'),
        ]


class GRUClassifier(nn.Module):
    def __init__(self, d_in, d_out, h=64):
        super().__init__()
        self.gru, self.head = nn.GRU(d_in, h, batch_first=True), nn.Linear(h, d_out)

    def forward(self, x):
        return self.head(self.gru(x)[1][-1])


def auc(y, score):
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    pos = y == 1
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


def risk(raw, sp, results, horizon=10, k=4):
    tr = sp['train'][subsample(len(sp['train']), 6000)]
    X, y = L.risk_view(raw, tr, horizon, k)
    model = fit(mlp(X.shape[1], 1), X, y.astype(np.float32),
                lambda o, t: nn.functional.binary_cross_entropy_with_logits(o.view(-1), t))
    for split in ['test', 'ood_test']:
        Xs, ys = L.risk_view(raw, sp[split], horizon, k)
        score = predict(model, Xs).view(-1).numpy()
        # saf kural: son gözlemde ent2 veya opserv için istismar uyarısı varsa → impact gelecek
        last = Xs[:, -78:-26].reshape(-1, 13, 4)[:, [3, 7], 3].any(1)
        results.append(dict(task=f'OpServer erken uyarı ({horizon} adım)', split=split, model=f'MLP, k={k}',
                            metric='F1 / AUC', value=macro_f1(ys, (score > 0).astype(np.int8), [1]),
                            auc=auc(ys, score), pos_rate=float(ys.mean()),
                            reference=f'uyarı kuralı F1: {macro_f1(ys, last.astype(np.int8), [1]):.3f}'))


def attacker(raw, sp, results):
    for n in [3, 5, 10]:
        X, y = L.attacker_id_view(raw, sp['train'], n)
        model = fit(GRUClassifier(X.shape[-1], 2), X, y, nn.functional.cross_entropy, epochs=10, bs=256)
        for split in ['test', 'ood_test']:
            Xs, ys = L.attacker_id_view(raw, sp[split], n)
            acc = float((predict(model, Xs).argmax(-1).numpy() == ys).mean())
            results.append(dict(task='saldırgan tanıma', split=split, model=f'GRU, ilk {n} adım',
                                metric='doğruluk', value=acc, reference='şans: 0.500'))


def next_red(raw, sp, results, k=4):
    tr = sp['train'][subsample(len(sp['train']), 4000)]
    X, y = L.next_red_action_view(raw, tr, k)
    model = fit(mlp(X.shape[1], len(L.RED_ACTION_TYPES)), X, y, nn.functional.cross_entropy)
    for split in ['test', 'ood_test']:
        Xs, ys = L.next_red_action_view(raw, sp[split], k)
        acc = float((predict(model, Xs).argmax(-1).numpy() == ys).mean())
        results.append(dict(task='Red aksiyon tipi tahmini', split=split, model=f'MLP, k={k}', metric='doğruluk',
                            value=acc, reference=f'çoğunluk: {np.bincount(ys).max() / len(ys):.3f}'))


def rollout(policy_fn, red_name, episodes=500, seed=777):
    np.random.seed(seed)
    env = FixedCAGE(num_envs=episodes, remove_bugs=True)
    s, _ = env.reset()
    red, tot = make_red(red_name), np.zeros(episodes)
    for _ in range(100):
        s, r, _, _ = env.step(red.get_action(s['Red']), policy_fn(s['Blue']))
        tot += r['Blue'][:, 0]
    return tot.mean()


def behaviour_cloning(raw, meta, sp, results, teacher='restore_decoys'):
    D = L.d4rl_view(raw, sp['train'], blue_policies=[teacher], meta=meta)
    model = fit(mlp(D['observations'].shape[1], 53), D['observations'], D['actions'],
                nn.functional.cross_entropy, epochs=10)
    policy = lambda obs: predict(model, obs).argmax(-1).numpy().reshape(-1, 1)
    for red_name in meta['red_policies']:
        teacher_agent = make_blue(teacher)
        results.append(dict(task='davranış klonlama (BC)', split='ortamda 500 bölüm', model=f'MLP ← {teacher}',
                            metric=f'bölüm ödülü vs {red_name}', value=rollout(policy, red_name),
                            reference=f'öğretmen: {rollout(teacher_agent.get_action, red_name):.1f}'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=os.path.join(ROOT, 'data', 'minicage_v1'))
    args = p.parse_args()
    raw, meta = L.load_raw(args.data)
    sp = L.load_splits(args.data)
    results = []
    for fn in [detection, risk, attacker, next_red]:
        start = len(results)
        fn(raw, sp, results)
        print(*[f"{r['task']:28s} {r['split']:9s} {r['model']:18s} {r['metric']:22s} {r['value']:.3f}  ({r['reference']})"
                for r in results[start:]], sep='\n', flush=True)
    behaviour_cloning(raw, meta, sp, results)
    for r in results[-2:]:
        print(f"{r['task']} {r['metric']}: {r['value']:.1f}  ({r['reference']})")
    for r in results:
        r['task'] = f"{r['task']} [{r['split']}]"
    with open(os.path.join(args.data, 'baselines.json'), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()

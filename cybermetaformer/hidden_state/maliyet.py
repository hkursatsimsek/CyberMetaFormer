"""Gecikme ve maliyet (revizyon madde 10): CPU'da tek örnekli (bs = 1) çevrimiçi adım gecikmesi ve çalışma belleği.

gecikme.py'nin protokolü, model listesi genişletilerek (CMF B2, M2'nin en iyi iki pencere modeli Transformer k64 ve
TimeMixer k64) ve bellek ölçümü eklenerek uygulanır:
- Çevrimiçi adım: CMF varyantları `CyberMetaFormer.step` (Algoritma 1); GRU tam taşınan gizli durumla tek hücre
  güncellemesi (gecikme.GRUStep); pencere modelleri son k gözlemlik tamponu her adımda kaydırıp ağı yeniden koşar
  (WindowStep; tampon başta sıfır ve maskeli, değerlendirmedeki pencereyle aynı). Ölçümden önce her model 4 test
  bölümünde adım kipini değerlendirme yoluyla karşılaştırır (CMF ve GRU: bölüm kipi; pencere modelleri:
  evaluate.predict_logits; FP32, en büyük |Δ| < 1e-3).
- Gecikme: her iş parçacığı ayarında (1 ve torch'un varsayılanı) ısınma, ardından ROUNDS tur; her turda modeller
  dönüşümlü olarak EPISODES bölüm × 101 adım koşar. Medyan, p95, p99, ortalama ± sd ve tur medyanlarının sd'si.
- Bellek: eğitilebilir parametre sayısı (makale tablolarındaki ve efficiency.py'deki tanım), bütün parametrelerin
  baytı (FP32), çevrimiçi durum baytı (adım kipinin taşıdığı tensörler) ve çalışma belleği. Çalışma
  belleği her model için ayrı bir alt süreçte ölçülür: torch ve veri yüklendikten sonraki tepe RSS ile model yüklenip
  1 iş parçacığında ısınma + bir tur koşulduktan sonraki tepe RSS arasındaki fark (macOS ru_maxrss, bayt). Torch'un
  ayırıcı önbelleğini de içerir; mutlak değer değil, modeller arası karşılaştırma içindir.
- Ortam: CPU modeli, performans/verim çekirdekleri, mantıksal CPU, bellek, macOS, Python/torch/numpy sürümleri, ölçüm
  öncesi ve sonrası yük ortalaması, ölçüm sırasında başka Python sürecinin olup olmadığı. Makine başka iş koşmazken
  ölçülmelidir (betik başka bir ağır Python süreci görürse uyarır ve kaydeder).

Çıktı: results/hidden_state/makale/maliyet/{olcum.json, table.md}

    venv_ts/bin/python -m cybermetaformer.hidden_state.maliyet
    venv_ts/bin/python -m cybermetaformer.hidden_state.maliyet --table     # kayıtlı olcum.json'dan yalnız tablo
"""
import argparse
import json
import os
import resource
import subprocess
import sys
import tempfile

import numpy as np
import torch
import yaml

from cybermetaformer.hidden_state import data as D
from cybermetaformer.hidden_state import evaluate as E
from cybermetaformer.hidden_state import gecikme as G
from cybermetaformer.hidden_state import train as T
from cybermetaformer.hidden_state.models import build_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, 'results', 'hidden_state')
OUT = os.path.join(RESULTS, 'makale', 'maliyet')
# ad → (aşama, koşu klasörü, etiket); seed 0'ın ağırlıkları
MODELS = {
    'cybermetaformer': ('asama3', 'cybermetaformer_k101_s0', G.LABEL['cybermetaformer']),
    'cybermetaformer-b2': ('asama3', 'cybermetaformer-b2_k101_s0', 'CyberMetaFormer B2 (larger width budget)'),
    'cybermetaformer-host_tam': ('asama3', 'cybermetaformer-host_tam_k101_s0', G.LABEL['cybermetaformer-host_tam']),
    'cybermetaformer-konv_gru': ('asama3', 'cybermetaformer-konv_gru_k101_s0', G.LABEL['cybermetaformer-konv_gru']),
    'cybermetaformer-gru_host_topo': ('asama3', 'cybermetaformer-gru_host_topo_k101_s0',
                                      G.LABEL['cybermetaformer-gru_host_topo']),
    'cybermetaformer-gru_host_attn': ('asama3', 'cybermetaformer-gru_host_attn_k101_s0',
                                      G.LABEL['cybermetaformer-gru_host_attn']),
    'cybermetaformer-host_yok': ('asama3', 'cybermetaformer-host_yok_k101_s0', G.LABEL['cybermetaformer-host_yok']),
    'gru_full': ('asama3', 'gru_full_k101_s0', G.LABEL['gru_full']),
    'transformer_k64': ('asama2', 'transformer_k64_s0', 'Transformer, window k = 64'),
    'timemixer_k64': ('asama2', 'timemixer_k64_s0', 'TimeMixer, window k = 64'),
}
WARM, ROUNDS, EPISODES, CHECK_EP = 202, 5, 10, 4


class WindowStep:
    """Pencere modelinin çevrimiçi adımı: son k ölçekli gözlemlik tampon + maske, her adımda tam ileri geçiş."""

    def __init__(self, model, k):
        self.model, self.k = model, k

    def init_state(self, batch=1, device='cpu'):
        return {'x': torch.zeros(batch, self.k, D.OBS_DIM, device=device),
                'm': torch.zeros(batch, self.k, dtype=torch.bool, device=device)}

    @torch.no_grad()
    def step(self, x, state):
        state['x'] = torch.cat([state['x'][:, 1:], x[:, None]], 1)
        state['m'] = torch.cat([state['m'][:, 1:], torch.ones_like(state['m'][:, :1])], 1)
        return self.model(state['x'], state['m'])


def load(name):
    stage, run, _ = MODELS[name]
    d = os.path.join(RESULTS, stage, run)
    cfg = {**T.DEFAULTS, **yaml.safe_load(open(os.path.join(d, 'config.yaml')))}
    assert cfg['scale'] == 'global', f'{name}: ölçekleme {cfg["scale"]}'
    torch.manual_seed(0)
    model = build_model(cfg['model'], cfg['k'], **(cfg.get('model_args') or {}))
    model.load_state_dict(torch.load(os.path.join(d, 'model.pt'), map_location='cpu'))
    model = model.float().eval()
    if getattr(model, 'episode_mode', False):
        return model, (G.GRUStep(model) if name == 'gru_full' else model), cfg
    return model, WindowStep(model, cfg['k']), cfg


@torch.no_grad()
def check(name, st, model, cfg, obs_int8):
    """Adım kipi ile değerlendirme yolunun en büyük mutlak logit farkı (CHECK_EP bölüm)."""
    x = D.scale_obs(torch.as_tensor(obs_int8[:CHECK_EP])).float()
    if isinstance(st, WindowStep):
        n = len(obs_int8[:CHECK_EP])
        wd = D.WindowData(_Arrays(obs_int8[:CHECK_EP]), np.arange(n))
        ref = torch.as_tensor(E.predict_logits(model, wd, cfg['k'], cfg['scale'], 4096))         # (n, T, 12, 3)
        state = st.init_state(n, 'cpu')
        out = torch.stack([st.step(x[:, t], state) for t in range(x.shape[1])], 1)[:, 1:]
    else:
        return G.check(st, x)
    err = float((out - ref).abs().max())
    assert err < 1e-3, f'{name}: adım kipi değerlendirme yolundan {err:.2e} sapıyor'
    return err


class _Arrays:
    def __init__(self, obs):
        n = len(obs)
        self.obs = obs
        self.labels = np.zeros((n, obs.shape[1] - 1, len(D.LABEL_HOSTS)), np.int8)
        self.prev_action = np.full((n, obs.shape[1]), D.NO_ACTION, np.int8)


def other_python_processes():
    """Bu süreç dışında %CPU'su yüksek Python süreçleri (ölçümün boş makinede yapıldığının kaydı)."""
    try:
        ps = subprocess.run(['ps', '-axo', 'pid=,pcpu=,comm='], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    out = []
    for line in ps.splitlines():
        pid, cpu, comm = line.split(None, 2)
        if int(pid) != os.getpid() and 'python' in comm.lower() and float(cpu) > 20:
            out.append({'pid': int(pid), 'pcpu': float(cpu)})
    return out


def worker(name, x_path):
    """Alt süreç: tek modelin çalışma belleği (tepe RSS artışı, bayt) → stdout'a JSON."""
    torch.set_num_threads(1)
    x = torch.load(x_path)
    base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    model, st, _ = load(name)
    G.run_steps(st, x, max(1, WARM // x.shape[1]))
    G.run_steps(st, x, EPISODES)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(json.dumps({'baseline_peak_rss_bytes': base, 'peak_rss_bytes': peak, 'working_memory_bytes': peak - base}))


def main():
    busy = other_python_processes()
    if busy:
        print(f'UYARI: başka yoğun Python süreçleri var: {busy}', flush=True)
    env = G.environment()
    env['busy_python_processes_before'] = busy
    default_threads = env['torch_default_threads']
    store = D.EpisodeStore()
    with store.final_evaluation():
        obs = store.obs[store.split('test')[:EPISODES]]
    x = D.scale_obs(torch.as_tensor(obs)).float()
    S = x.shape[1]
    models = {}
    for name in MODELS:
        model, st, cfg = load(name)
        models[name] = {'step': st, 'params': sum(p.numel() for p in model.parameters() if p.requires_grad),
                        'params_total': sum(p.numel() for p in model.parameters()),
                        'param_bytes': sum(p.numel() * p.element_size() for p in model.parameters()),
                        'state_bytes': G.state_bytes(st.init_state(1, 'cpu')), 'window': cfg.get('k'),
                        'step_check_max_abs': check(name, st, model, cfg, obs)}
        print(f"{name:32s} param {models[name]['params']:>9,}  durum {models[name]['state_bytes']:>9,} B  "
              f"adım–değerlendirme |Δ| {models[name]['step_check_max_abs']:.1e}", flush=True)
    results = {}
    for threads in sorted({1, default_threads}):
        torch.set_num_threads(threads)
        for name in MODELS:                                                  # ısınma
            G.run_steps(models[name]['step'], x, max(1, WARM // S))
        rounds = {name: [] for name in MODELS}
        for _ in range(ROUNDS):
            for name in MODELS:
                rounds[name].append(G.run_steps(models[name]['step'], x, EPISODES))
        for name in MODELS:
            res = G.summarize([v for rr in rounds[name] for v in rr], rounds[name])
            results.setdefault(name, {})[str(threads)] = res
            print(f"{threads} iş parçacığı  {name:32s} medyan {res['median_ms']:.4f} ms  p95 {res['p95_ms']:.4f}  "
                  f"p99 {res['p99_ms']:.4f}  tur sd {res['round_median_sd_ms']:.4f}", flush=True)
    env['load_average_after'] = list(os.getloadavg())
    env['busy_python_processes_after'] = other_python_processes()
    with tempfile.TemporaryDirectory() as tmp:                               # çalışma belleği, model başına süreç
        x_path = os.path.join(tmp, 'x.pt')
        torch.save(x, x_path)
        for name in MODELS:
            r = subprocess.run([sys.executable, '-m', 'cybermetaformer.hidden_state.maliyet', '--worker', name, x_path],
                               capture_output=True, text=True, check=True, cwd=os.path.dirname(ROOT))
            models[name]['memory'] = json.loads(r.stdout.strip().splitlines()[-1])
            print(f"{name:32s} çalışma belleği {models[name]['memory']['working_memory_bytes'] / 2 ** 20:.1f} MiB",
                  flush=True)
    out = {'environment': env,
           'protocol': {'warmup_steps': max(1, WARM // S) * S, 'rounds': ROUNDS, 'steps_per_round': EPISODES * S,
                        'batch_size': 1, 'dtype': 'float32', 'threads': sorted({1, default_threads}),
                        'memory': 'peak RSS increase in a separate process (1 thread, warm-up + one round)'},
           'models': {n: {k: v for k, v in m.items() if k != 'step'} | {'label': MODELS[n][2], 'run': MODELS[n][1],
                                                                       'latency': results[n]}
                      for n, m in models.items()}}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'olcum.json'), 'w') as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(OUT, 'table.md'), 'w') as f:
        f.write(table_md(out))
    print('→', OUT)


def table_md(out):
    env, pr = out['environment'], out['protocol']
    th = [str(t) for t in pr['threads']]
    L = ['| Model | Trainable params | Weights (KiB) | Online state (bytes) | Working memory (MiB) | ' + ' | '.join(
        f'{t} thread{"s" if t != "1" else ""}: median / p95 / p99 (ms)' for t in th) + ' |',
         '|---|---:|---:|---:|---:|' + '---:|' * len(th)]
    for n, m in out['models'].items():
        L.append(f"| {m['label']} | {m['params']:,} | {m['param_bytes'] / 1024:,.0f} | {m['state_bytes']:,} | "
                 f"{m['memory']['working_memory_bytes'] / 2 ** 20:.1f} | " + ' | '.join(
                     f"{m['latency'][t]['median_ms']:.3f} / {m['latency'][t]['p95_ms']:.3f} / "
                     f"{m['latency'][t]['p99_ms']:.3f}" for t in th) + ' |')
    busy = env.get('busy_python_processes_before') or env.get('busy_python_processes_after')
    note = (f"Note. One online step for one network (batch size 1, FP32, PyTorch {env['torch']} eager, NumPy "
            f"{env['numpy']}, Python {env['python']}) on the CPU of an {env['cpu']} ({env['performance_cores']} "
            f"performance + {env['efficiency_cores']} efficiency cores, {int(env['memory_bytes']) / 2 ** 30:.0f} GiB), "
            f"macOS {env['macos']}; torch default = {env['torch_default_threads']} threads. Each model was warmed up "
            f"for {pr['warmup_steps']} steps, then timed for {pr['rounds']} rounds × {pr['steps_per_round']} steps "
            f"(perf_counter_ns per step call), models interleaved within each round; the online state is reset at "
            f"every episode start and the reset is not timed. CyberMetaFormer variants use the step mode "
            f"(Algorithm 1); the full-history GRU carries its hidden state; window models shift a k-observation "
            f"buffer and rerun the network at every step. Online state = tensors carried between steps. Working "
            f"memory = increase of the peak resident set size of a separate process after loading the model and "
            f"running warm-up plus one round with 1 thread (includes allocator caches; for comparison between "
            f"models). Load average before / after: {', '.join(f'{v:.2f}' for v in env['load_average_before'])} / "
            f"{', '.join(f'{v:.2f}' for v in env['load_average_after'])}. Latency excludes observation encoding and "
            f"I/O. Only one network size per architecture was measured, so no scaling claim is made."
            + (f" Other busy Python processes were present during the measurement: {busy}." if busy else ''))
    return '\n'.join(L) + '\n\n' + note + '\n'


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--table', action='store_true')
    p.add_argument('--worker', nargs=2, metavar=('MODEL', 'X_PATH'))
    a = p.parse_args()
    if a.worker:
        worker(*a.worker)
    elif a.table:
        with open(os.path.join(OUT, 'olcum.json')) as f:
            olcum = json.load(f)
        with open(os.path.join(OUT, 'table.md'), 'w') as f:
            f.write(table_md(olcum))
    else:
        main()

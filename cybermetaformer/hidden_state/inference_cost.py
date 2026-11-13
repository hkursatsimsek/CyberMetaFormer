"""Latency and cost (revision item 10): single-sample (bs = 1) online step latency and working memory on CPU.

The protocol of step_latency.py, with the model list extended (CMF B2, the two best M2 window models Transformer k64 and
TimeMixer k64) and a memory measurement added:
- Online step: CMF variants `CyberMetaFormer.step` (Algorithm 1); full GRU a single cell update with the carried
  hidden state (step_latency.GRUStep); window models shift a buffer of the last k observations at every step and rerun
  the network (WindowStep; the buffer starts zero and masked, the same window as in evaluation). Before timing, each
  model compares its step mode with the evaluation path on 4 test episodes (CMF and GRU: episode mode; window models:
  evaluate.predict_logits; FP32, largest |Δ| < 1e-3).
- Latency: in each thread setting (1 and the torch default) a warm-up, then ROUNDS rounds; in each round the models
  run EPISODES episodes × 101 steps in turn. Median, p95, p99, mean ± sd and the sd of the round medians.
- Memory: trainable parameter count (the definition of the paper tables and efficiency.py), bytes of all parameters
  (FP32), online state bytes (tensors carried by the step mode) and working memory. Working
  memory is measured for each model in a separate subprocess: the difference between the peak RSS after loading torch
  and the data and the peak RSS after loading the model and running warm-up + one round on 1 thread (macOS ru_maxrss,
  bytes). It includes the torch allocator cache; it is meant for comparison between models, not as an absolute value.
- Environment: CPU model, performance/efficiency cores, logical CPUs, memory, macOS, Python/torch/numpy versions, load
  average before and after timing, whether another Python process was running during timing. Measure while nothing
  else runs on the machine (the script warns and records it if it sees another heavy Python process).

Output: results/hidden_state/makale/inference_cost/{olcum.json, table.md}

    venv_ts/bin/python -m cybermetaformer.hidden_state.inference_cost
    venv_ts/bin/python -m cybermetaformer.hidden_state.inference_cost --table     # only the table, from the saved olcum.json
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
from cybermetaformer.hidden_state import step_latency as G
from cybermetaformer.hidden_state import train as T
from cybermetaformer.hidden_state.models import build_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, 'results', 'hidden_state')
OUT = os.path.join(RESULTS, 'makale', 'inference_cost')
# name → (stage, run folder, label); weights of seed 0
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
    """Online step of a window model: buffer of the last k scaled observations + mask, a full forward pass per step."""

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
    assert cfg['scale'] == 'global', f'{name}: scaling {cfg["scale"]}'
    torch.manual_seed(0)
    model = build_model(cfg['model'], cfg['k'], **(cfg.get('model_args') or {}))
    model.load_state_dict(torch.load(os.path.join(d, 'model.pt'), map_location='cpu'))
    model = model.float().eval()
    if getattr(model, 'episode_mode', False):
        return model, (G.GRUStep(model) if name == 'gru_full' else model), cfg
    return model, WindowStep(model, cfg['k']), cfg


@torch.no_grad()
def check(name, st, model, cfg, obs_int8):
    """Largest absolute logit difference between the step mode and the evaluation path (CHECK_EP episodes)."""
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
    assert err < 1e-3, f'{name}: step mode deviates from the evaluation path by {err:.2e}'
    return err


class _Arrays:
    def __init__(self, obs):
        n = len(obs)
        self.obs = obs
        self.labels = np.zeros((n, obs.shape[1] - 1, len(D.LABEL_HOSTS)), np.int8)
        self.prev_action = np.full((n, obs.shape[1]), D.NO_ACTION, np.int8)


def other_python_processes():
    """Python processes with high %CPU other than this one (record that the measurement ran on an idle machine)."""
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
    """Subprocess: working memory of a single model (peak RSS increase, bytes) → JSON on stdout."""
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
        print(f'WARNING: other busy Python processes: {busy}', flush=True)
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
        print(f"{name:32s} param {models[name]['params']:>9,}  state {models[name]['state_bytes']:>9,} B  "
              f"step–evaluation |Δ| {models[name]['step_check_max_abs']:.1e}", flush=True)
    results = {}
    for threads in sorted({1, default_threads}):
        torch.set_num_threads(threads)
        for name in MODELS:                                                  # warm-up
            G.run_steps(models[name]['step'], x, max(1, WARM // S))
        rounds = {name: [] for name in MODELS}
        for _ in range(ROUNDS):
            for name in MODELS:
                rounds[name].append(G.run_steps(models[name]['step'], x, EPISODES))
        for name in MODELS:
            res = G.summarize([v for rr in rounds[name] for v in rr], rounds[name])
            results.setdefault(name, {})[str(threads)] = res
            print(f"{threads} threads  {name:32s} median {res['median_ms']:.4f} ms  p95 {res['p95_ms']:.4f}  "
                  f"p99 {res['p99_ms']:.4f}  round sd {res['round_median_sd_ms']:.4f}", flush=True)
    env['load_average_after'] = list(os.getloadavg())
    env['busy_python_processes_after'] = other_python_processes()
    with tempfile.TemporaryDirectory() as tmp:                               # working memory, one process per model
        x_path = os.path.join(tmp, 'x.pt')
        torch.save(x, x_path)
        for name in MODELS:
            r = subprocess.run([sys.executable, '-m', 'cybermetaformer.hidden_state.inference_cost', '--worker', name, x_path],
                               capture_output=True, text=True, check=True, cwd=os.path.dirname(ROOT))
            models[name]['memory'] = json.loads(r.stdout.strip().splitlines()[-1])
            print(f"{name:32s} working memory {models[name]['memory']['working_memory_bytes'] / 2 ** 20:.1f} MiB",
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

"""Paper (M4, item 4; the A4 column of item 10): single-sample (bs = 1) online step latency on CPU.

Models are loaded with the trained weights of seed 0. Timing does not depend on the weights, but the step mode is
checked with real weights.
- CyberMetaFormer and its variants: `CyberMetaFormer.step` (conv buffer, SSD or GRU state; cost independent of t).
- Full GRU: single cell update with the carried hidden state + output head (`rnn(x_t, h)`; the same computation as the
  k = 1 window in efficiency.py, but with the state carried).
Before timing, each model compares its step mode with the episode mode on 4 test episodes (FP32, largest |Δ| < 1e-3).

Protocol: in each thread setting (1 and the torch default) each model warms up for 202 steps; then 5 rounds, each
round running the models in turn (alternating, so drift over time is shared between models) for 10 episodes × 101
steps = 1010 steps. The state is reset at each episode start; the reset is not timed. Measured: wall time of a single
`step` call (perf_counter_ns). Reported: median, p95, p99, mean ± sd of the 5 × 1010 steps and the sd of the round
medians.
Also the parameter count and FP32 bytes, the online state bytes (tensors carried by the step mode) and the
environment: CPU model, performance/efficiency core counts, torch/numpy/Python/macOS versions, load average before and
after timing, peak RSS. Measure while nothing else is running on the machine.

Output: results/hidden_state/makale/step_latency/{olcum.json, table.md}

    venv_ts/bin/python -m cybermetaformer.hidden_state.step_latency
"""
import json
import os
import platform
import resource
import subprocess
import sys
import time

import numpy as np
import torch

from cybermetaformer.hidden_state import data as D
from cybermetaformer.hidden_state.ood_suite import load_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'results', 'hidden_state', 'makale', 'step_latency')
MODELS = ['cybermetaformer', 'cybermetaformer-host_tam', 'cybermetaformer-konv_gru', 'cybermetaformer-gru_host_topo',
          'cybermetaformer-gru_host_attn', 'gru_full', 'cybermetaformer-host_yok']
LABEL = {'cybermetaformer': 'CyberMetaFormer (conv + SSD, topology bias)',      # same names as the paper table
         'cybermetaformer-host_tam': 'CMF, no topology bias',
         'cybermetaformer-konv_gru': 'CMF, GRU instead of SSD',
         'cybermetaformer-gru_host_topo': 'Host-token GRU, topology bias',
         'cybermetaformer-gru_host_attn': 'Host-token GRU, no topology bias',
         'gru_full': 'Full-history GRU (flat 78-d input)',
         'cybermetaformer-host_yok': 'CMF, no host mixer'}
WARM, ROUNDS, EPISODES = 202, 5, 10


class GRUStep:
    """Online step of the full GRU: carried hidden state, single cell update, output head."""

    def __init__(self, model):
        self.model = model

    def init_state(self, batch=1, device='cpu'):
        r = self.model.rnn
        return {'h': torch.zeros(r.num_layers, batch, r.hidden_size, device=device)}

    @torch.no_grad()
    def step(self, x, state):
        out, state['h'] = self.model.rnn(x[:, None], state['h'])
        return self.model.head(out[:, 0]).view(x.shape[0], len(D.LABEL_HOSTS), -1)

    def forward_episode(self, x):
        return self.model.forward_episode(x)


def stepper(name):
    model, _ = load_model(name, 0, 'cpu')
    model = model.float().eval()
    return model, (GRUStep(model) if name == 'gru_full' else model)


def state_bytes(state):
    if isinstance(state, torch.Tensor):
        return state.numel() * state.element_size()
    if isinstance(state, dict):
        return sum(state_bytes(v) for v in state.values())
    if isinstance(state, (list, tuple)):
        return sum(state_bytes(v) for v in state)
    return 0                                                     # step counter (int)


@torch.no_grad()
def check(st, x):
    ref = st.forward_episode(x[:4])
    state = st.init_state(4, 'cpu')
    out = torch.stack([st.step(x[:4, t], state) for t in range(x.shape[1])], 1)
    err = float((out - ref).abs().max())
    assert err < 1e-3, f'step mode deviates from the episode mode by {err:.2e}'
    return err


@torch.no_grad()
def run_steps(st, x, n_ep):
    """n_ep episodes × S steps; the state is reset at each episode start. Returns: time per step (ns)."""
    S, out = x.shape[1], []
    for e in range(n_ep):
        state = st.init_state(1, 'cpu')
        xe = x[e:e + 1]
        for t in range(S):
            t0 = time.perf_counter_ns()
            st.step(xe[:, t], state)
            out.append(time.perf_counter_ns() - t0)
    return out


def summarize(ns, rounds):
    a = np.asarray(ns, np.float64) / 1e6                         # ms
    per_round = [float(np.median(np.asarray(r, np.float64) / 1e6)) for r in rounds]
    return {'n_steps': int(a.size), 'median_ms': float(np.median(a)), 'p95_ms': float(np.percentile(a, 95)),
            'p99_ms': float(np.percentile(a, 99)), 'mean_ms': float(a.mean()), 'sd_ms': float(a.std(ddof=1)),
            'round_medians_ms': per_round, 'round_median_sd_ms': float(np.std(per_round, ddof=1))}


def sysctl(key):
    try:
        return subprocess.run(['sysctl', '-n', key], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment():
    return {'cpu': sysctl('machdep.cpu.brand_string'), 'performance_cores': sysctl('hw.perflevel0.physicalcpu'),
            'efficiency_cores': sysctl('hw.perflevel1.physicalcpu'), 'logical_cpus': sysctl('hw.ncpu'),
            'memory_bytes': sysctl('hw.memsize'), 'macos': platform.mac_ver()[0], 'platform': platform.platform(),
            'python': sys.version.split()[0], 'torch': torch.__version__, 'numpy': np.__version__,
            'torch_default_threads': torch.get_num_threads(), 'load_average_before': list(os.getloadavg())}


def main():
    env = environment()
    default_threads = env['torch_default_threads']
    store = D.EpisodeStore()
    with store.final_evaluation():
        x = D.scale_obs(torch.as_tensor(store.obs[store.split('test')[:EPISODES]])).float()
    S = x.shape[1]
    models = {}
    for name in MODELS:
        model, st = stepper(name)
        models[name] = {'step': st, 'params': sum(p.numel() for p in model.parameters()),
                        'param_bytes': sum(p.numel() * p.element_size() for p in model.parameters()),
                        'state_bytes': state_bytes(st.init_state(1, 'cpu')), 'step_check_max_abs': check(st, x)}
        print(f"{name:32s} param {models[name]['params']:>9,}  state {models[name]['state_bytes']:>9,} B  "
              f"step–episode |Δ| {models[name]['step_check_max_abs']:.1e}", flush=True)
    results = {}
    for threads in sorted({1, default_threads}):
        torch.set_num_threads(threads)
        for name in MODELS:                                                  # warm-up
            run_steps(models[name]['step'], x, max(1, WARM // S))
        rounds = {name: [] for name in MODELS}
        for r in range(ROUNDS):
            for name in MODELS:
                rounds[name].append(run_steps(models[name]['step'], x, EPISODES))
        for name in MODELS:
            res = summarize([v for rr in rounds[name] for v in rr], rounds[name])
            results.setdefault(name, {})[str(threads)] = res
            print(f"{threads} threads  {name:32s} median {res['median_ms']:.4f} ms  p95 {res['p95_ms']:.4f}  "
                  f"p99 {res['p99_ms']:.4f}  round sd {res['round_median_sd_ms']:.4f}", flush=True)
    env['load_average_after'] = list(os.getloadavg())
    env['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss          # macOS: bayt
    out = {'environment': env, 'protocol': {'warmup_steps': max(1, WARM // S) * S, 'rounds': ROUNDS,
                                            'steps_per_round': EPISODES * S, 'batch_size': 1, 'dtype': 'float32',
                                            'threads': sorted({1, default_threads})},
           'models': {n: {k: v for k, v in m.items() if k != 'step'} | {'label': LABEL[n], 'latency': results[n]}
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
    L = ['| Model | Params | Online state (bytes) | ' + ' | '.join(
        f'{t} thread{"s" if t != "1" else ""}: median / p95 (ms)' for t in th) + ' |',
         '|---|---:|---:|' + '---:|' * len(th)]
    for n, m in out['models'].items():
        L.append(f"| {LABEL.get(n, m['label'])} | {m['params']:,} | {m['state_bytes']:,} | " + ' | '.join(
            f"{m['latency'][t]['median_ms']:.3f} / {m['latency'][t]['p95_ms']:.3f}" for t in th) + ' |')
    note = (f"Note. One online step for one network (batch size 1, FP32, PyTorch {env['torch']} eager) on the CPU of "
            f"an {env['cpu']} ({env['performance_cores']} performance + {env['efficiency_cores']} efficiency cores), "
            f"macOS {env['macos']}. Each model was warmed up for {pr['warmup_steps']} steps, then timed for "
            f"{pr['rounds']} rounds × {pr['steps_per_round']} steps, models interleaved within each round; the online "
            'state is reset at every episode start and the reset is not timed. CyberMetaFormer variants use the step '
            'mode (Algorithm 1); the full-history GRU carries its hidden state and performs one cell update per step. '
            'The online state counts the tensors the step mode carries between steps (FP32 buffers and states; for '
            'CyberMetaFormer variants also the 13 × 4 int64 last-event table). Latency excludes '
            'observation encoding and I/O.')
    return '\n'.join(L) + '\n\n' + note + '\n'


if __name__ == '__main__':
    if sys.argv[1:] == ['--table']:                  # without timing, only rewrite the table from the saved olcum.json
        with open(os.path.join(OUT, 'olcum.json')) as f:
            olcum = json.load(f)
        with open(os.path.join(OUT, 'table.md'), 'w') as f:
            f.write(table_md(olcum))
    else:
        main()

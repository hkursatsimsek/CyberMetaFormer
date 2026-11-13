| Model | Trainable params | Weights (KiB) | Online state (bytes) | Working memory (MiB) | 1 thread: median / p95 / p99 (ms) | 6 threads: median / p95 / p99 (ms) |
|---|---:|---:|---:|---:|---:|---:|
| CyberMetaFormer (conv + SSD, topology bias) | 186,665 | 729 | 190,112 | 7.1 | 0.519 / 0.528 / 0.536 | 0.832 / 0.857 / 0.880 |
| CyberMetaFormer B2 (larger width budget) | 2,828,473 | 11,049 | 806,624 | 28.2 | 0.971 / 0.987 / 1.015 | 1.425 / 1.455 / 1.484 |
| CMF, no topology bias | 186,617 | 729 | 190,112 | 7.3 | 0.503 / 0.512 / 0.519 | 0.813 / 0.833 / 0.848 |
| CMF, GRU instead of SSD | 198,329 | 775 | 40,352 | 6.8 | 0.461 / 0.467 / 0.473 | 0.766 / 0.788 / 0.805 |
| Host-token GRU, topology bias | 184,499 | 721 | 10,400 | 6.4 | 0.338 / 0.345 / 0.350 | 0.584 / 0.600 / 0.615 |
| Host-token GRU, no topology bias | 184,451 | 721 | 10,400 | 6.5 | 0.324 / 0.331 / 0.336 | 0.560 / 0.576 / 0.592 |
| CMF, no host mixer | 201,001 | 785 | 237,536 | 6.8 | 0.445 / 0.450 / 0.455 | 0.631 / 0.651 / 0.667 |
| Full-history GRU (flat 78-d input) | 183,588 | 717 | 1,024 | 3.8 | 0.044 / 0.045 / 0.048 | 0.044 / 0.046 / 0.048 |
| Transformer, window k = 64 | 2,399,268 | 9,372 | 20,032 | 22.4 | 0.845 / 0.900 / 0.913 | 0.910 / 0.933 / 0.960 |
| TimeMixer, window k = 64 | 144,516 | 574 | 20,032 | 7.1 | 1.028 / 1.044 / 1.054 | 1.436 / 1.469 / 1.500 |

Note. One online step for one network (batch size 1, FP32, PyTorch 2.14.1 eager, NumPy 2.5.4, Python 3.12.15) on the CPU of an Apple M3 Pro (6 performance + 6 efficiency cores, 36 GiB), macOS 27.2; torch default = 6 threads. Each model was warmed up for 202 steps, then timed for 5 rounds × 1010 steps (perf_counter_ns per step call), models interleaved within each round; the online state is reset at every episode start and the reset is not timed. CyberMetaFormer variants use the step mode (Algorithm 1); the full-history GRU carries its hidden state; window models shift a k-observation buffer and rerun the network at every step. Online state = tensors carried between steps. Working memory = increase of the peak resident set size of a separate process after loading the model and running warm-up plus one round with 1 thread (includes allocator caches; for comparison between models). Load average before / after: 2.30, 1.85, 1.28 / 1.85, 1.78, 1.29. Latency excludes observation encoding and I/O. Only one network size per architecture was measured, so no scaling claim is made.

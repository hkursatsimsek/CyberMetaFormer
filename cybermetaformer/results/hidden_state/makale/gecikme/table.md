| Model | Params | Online state (bytes) | 1 thread: median / p95 (ms) | 6 threads: median / p95 (ms) |
|---|---:|---:|---:|---:|
| CyberMetaFormer (conv + SSD, topology bias) | 186,665 | 190,112 | 0.523 / 0.538 | 0.844 / 0.884 |
| CMF, no topology bias | 186,617 | 190,112 | 0.507 / 0.521 | 0.823 / 0.859 |
| CMF, GRU instead of SSD | 198,329 | 40,352 | 0.466 / 0.478 | 0.780 / 0.813 |
| Host-token GRU, topology bias | 184,499 | 10,400 | 0.340 / 0.353 | 0.592 / 0.620 |
| Host-token GRU, no topology bias | 184,451 | 10,400 | 0.326 / 0.337 | 0.568 / 0.597 |
| Full-history GRU (flat 78-d input) | 183,588 | 1,024 | 0.044 / 0.046 | 0.044 / 0.046 |
| CMF, no host mixer | 201,001 | 237,536 | 0.449 / 0.459 | 0.640 / 0.673 |

Note. One online step for one network (batch size 1, FP32, PyTorch 2.14.1 eager) on the CPU of an Apple M3 Pro (6 performance + 6 efficiency cores), macOS 27.2. Each model was warmed up for 202 steps, then timed for 5 rounds × 1010 steps, models interleaved within each round; the online state is reset at every episode start and the reset is not timed. CyberMetaFormer variants use the step mode (Algorithm 1); the full-history GRU carries its hidden state and performs one cell update per step. The online state counts the tensors the step mode carries between steps (FP32 buffers and states; for CyberMetaFormer variants also the 13 × 4 int64 last-event table). Latency excludes observation encoding and I/O.

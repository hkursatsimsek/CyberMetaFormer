| Model | Params | OOD F1 | OOD silent recall | OOD N_hist recall | CPU step (ms) | Online state (bytes) |
|---|---:|---:|---:|---:|---:|---:|
| CyberMetaFormer (B1) | 186,665 | 0.9417 | 0.772 | 0.826 | 0.519 | 190,112 |
| CyberMetaFormer (B1), run 2 | 186,665 | 0.9414 | 0.785 | 0.836 | 0.519 | 190,112 |
| CMF, GRU instead of SSD† | 198,329 | 0.9428 | 0.777 | 0.838 | 0.461 | 40,352 |
| Host-token GRU, topology bias | 184,499 | 0.9419 | 0.764 | 0.826 | 0.338 | 10,400 |
| Host-token GRU, no topology bias | 184,451 | 0.9413 | 0.763 | 0.814 | 0.324 | 10,400 |
| Full-history GRU (flat input) | 183,588 | 0.9264 | 0.736 | 0.791 | 0.044 | 1,024 |

Note. Accuracy and recall: seed mean over seeds 0–4; † seeds 0–2. Run 2 is an independent five-seed training of the main configuration on the same hardware as the host-token GRU references, which are compared with it (Table @tbl:hostgru_cift); its architecture and cost are those of the main model. The host-token GRU references keep every CyberMetaFormer component except the temporal mixer, which is a per-host GRU over the same host tokens. CPU step: median single-thread latency of one online step (Apple M3 Pro CPU, FP32, batch size 1), excluding observation encoding and I/O; online state: tensors carried between steps. Seed SDs and test results: Table @tbl:hostgru; measurement protocol, six-thread latency and memory: Table @tbl:inference_cost.

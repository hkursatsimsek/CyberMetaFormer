| Model | Params | Test host macro-F1 | OOD host macro-F1 | Test silent recall | OOD silent recall | OOD N_hist recall |
|---|---:|---:|---:|---:|---:|---:|
| CyberMetaFormer | 186,665 | 0.9801 ± 0.0001 | 0.9417 ± 0.0006 | 0.904 ± 0.006 | 0.772 ± 0.020 | 0.826 ± 0.014 |
| CyberMetaFormer, run 2 | 186,665 | 0.9800 ± 0.0002 | 0.9414 ± 0.0003 | 0.906 ± 0.006 | 0.785 ± 0.011 | 0.836 ± 0.014 |
| CMF, no topology bias | 186,617 | 0.9799 ± 0.0002 | 0.9411 ± 0.0009 | 0.905 ± 0.006 | 0.766 ± 0.017 | 0.817 ± 0.028 |
| CMF, GRU instead of SSD† | 198,329 | 0.9806 ± 0.0001 | 0.9428 ± 0.0009 | 0.908 ± 0.003 | 0.777 ± 0.011 | 0.838 ± 0.012 |
| Host-token GRU, topology bias | 184,499 | 0.9803 ± 0.0002 | 0.9419 ± 0.0011 | 0.904 ± 0.002 | 0.764 ± 0.007 | 0.826 ± 0.010 |
| Host-token GRU, no topology bias | 184,451 | 0.9801 ± 0.0004 | 0.9413 ± 0.0014 | 0.904 ± 0.007 | 0.763 ± 0.017 | 0.814 ± 0.020 |
| Full-history GRU | 183,588 | 0.9768 ± 0.0003 | 0.9264 ± 0.0012 | 0.902 ± 0.002 | 0.736 ± 0.009 | 0.791 ± 0.010 |
| CMF, no host mixer† | 201,001 | 0.9598 ± 0.0003 | 0.8993 ± 0.0009 | 0.790 ± 0.009 | 0.591 ± 0.018 | 0.454 ± 0.038 |

Note. Seed mean ± SD over seeds 0–4; † seeds 0–2. Host macro-F1 as defined in Section 3.6; silent recall over compromised host-steps without a current alert; N_hist recall over silent host-steps whose host has had no alert up to t (online-applicable; Section 3.3). CyberMetaFormer is the main B1 model (conv + SSD, topology bias); run 2 is an independent five-seed training of the same configuration on the same hardware as the host-token GRU references, which are compared with it (Table @tbl:hostgru_cift). The host-token GRU variants keep every CyberMetaFormer component except the temporal mixer, which is a per-host GRU over the same host tokens. The full-history GRU takes the flat 78-d observation. Online CPU step latency and carried state of these models are given in Table @tbl:maliyet.

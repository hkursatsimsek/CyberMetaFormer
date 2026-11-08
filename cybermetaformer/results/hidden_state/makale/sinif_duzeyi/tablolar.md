# Class-level performance (M4 item 3)

## Standard test: per-class metrics (pooled over evaluated hosts; seed mean ± sd)

| Model | Seeds | Class | Precision | Recall | F1 | Host-avg F1 |
|---|---:|---|---:|---:|---:|---:|
| CyberMetaFormer | 5 | clean | 0.988 ± 0.001 | 0.992 ± 0.001 | 0.990 ± 0.000 | 0.988 ± 0.000 |
| CyberMetaFormer | 5 | user | 0.992 ± 0.000 | 0.955 ± 0.001 | 0.973 ± 0.000 | 0.978 ± 0.000 |
| CyberMetaFormer | 5 | root | 0.971 ± 0.003 | 0.962 ± 0.003 | 0.966 ± 0.000 | 0.974 ± 0.000 |
| Full-history GRU | 5 | clean | 0.987 ± 0.000 | 0.989 ± 0.000 | 0.988 ± 0.000 | 0.987 ± 0.000 |
| Full-history GRU | 5 | user | 0.991 ± 0.000 | 0.952 ± 0.001 | 0.971 ± 0.000 | 0.973 ± 0.001 |
| Full-history GRU | 5 | root | 0.963 ± 0.001 | 0.960 ± 0.001 | 0.962 ± 0.000 | 0.970 ± 0.000 |
| Host-token GRU + topo attention | 5 | clean | 0.988 ± 0.000 | 0.992 ± 0.000 | 0.990 ± 0.000 | 0.989 ± 0.000 |
| Host-token GRU + topo attention | 5 | user | 0.992 ± 0.000 | 0.955 ± 0.001 | 0.973 ± 0.000 | 0.978 ± 0.000 |
| Host-token GRU + topo attention | 5 | root | 0.972 ± 0.001 | 0.962 ± 0.001 | 0.967 ± 0.000 | 0.974 ± 0.000 |
| Host-token GRU + attention | 5 | clean | 0.988 ± 0.001 | 0.992 ± 0.001 | 0.990 ± 0.000 | 0.988 ± 0.000 |
| Host-token GRU + attention | 5 | user | 0.992 ± 0.001 | 0.955 ± 0.000 | 0.973 ± 0.000 | 0.978 ± 0.000 |
| Host-token GRU + attention | 5 | root | 0.971 ± 0.004 | 0.962 ± 0.003 | 0.966 ± 0.001 | 0.974 ± 0.001 |

## Standard test: security-relevant error rates

| Model | Root→user | Root→clean | User→clean | User→root | Clean false alarm | Worst host (macro-F1) | Best host (macro-F1) |
|---|---:|---:|---:|---:|---:|---|---|
| CyberMetaFormer | 0.0000 ± 0.0000 | 0.0383 ± 0.0026 | 0.0415 ± 0.0006 | 0.0031 ± 0.0003 | 0.0084 ± 0.0008 | user1 (0.963) | ophost1 (1.000) |
| Full-history GRU | 0.0001 ± 0.0000 | 0.0399 ± 0.0010 | 0.0428 ± 0.0005 | 0.0054 ± 0.0003 | 0.0106 ± 0.0003 | user1 (0.960) | opserv (1.000) |
| Host-token GRU + topo attention | 0.0000 ± 0.0000 | 0.0384 ± 0.0010 | 0.0414 ± 0.0004 | 0.0032 ± 0.0005 | 0.0081 ± 0.0004 | user1 (0.963) | opserv (1.000) |
| Host-token GRU + attention | 0.0000 ± 0.0000 | 0.0384 ± 0.0033 | 0.0414 ± 0.0005 | 0.0031 ± 0.0005 | 0.0084 ± 0.0013 | user1 (0.964) | ophost1 (1.000) |

## Standard test: detection latency (steps; seeds with stored predictions)

| Model | Seeds | Onset | Spells | Detected (breach) | Median | p90 | Detected (exact class) | Median | p90 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| CyberMetaFormer | 5 | all | 56,976 | 0.9962 | 0.0 | 0.0 | 0.9961 | 0.0 | 0.0 |
| CyberMetaFormer | 5 | alert_onset | 52,533 | 1.0000 | 0.0 | 0.0 | 0.9999 | 0.0 | 0.0 |
| CyberMetaFormer | 5 | silent_onset | 4,443 | 0.9516 | 0.6 | 11.0 | 0.9512 | 1.0 | 11.0 |
| Full-history GRU | 5 | all | 56,976 | 0.9961 | 0.0 | 0.0 | 0.9957 | 0.0 | 0.0 |
| Full-history GRU | 5 | alert_onset | 52,533 | 0.9999 | 0.0 | 0.0 | 0.9997 | 0.0 | 0.0 |
| Full-history GRU | 5 | silent_onset | 4,443 | 0.9503 | 1.0 | 9.4 | 0.9492 | 1.0 | 9.4 |
| Host-token GRU + topo attention | 5 | all | 56,976 | 0.9963 | 0.0 | 0.0 | 0.9963 | 0.0 | 0.0 |
| Host-token GRU + topo attention | 5 | alert_onset | 52,533 | 1.0000 | 0.0 | 0.0 | 0.9999 | 0.0 | 0.0 |
| Host-token GRU + topo attention | 5 | silent_onset | 4,443 | 0.9533 | 0.6 | 10.8 | 0.9530 | 1.0 | 10.8 |
| Host-token GRU + attention | 5 | all | 56,976 | 0.9963 | 0.0 | 0.0 | 0.9962 | 0.0 | 0.0 |
| Host-token GRU + attention | 5 | alert_onset | 52,533 | 1.0000 | 0.0 | 0.0 | 0.9999 | 0.0 | 0.0 |
| Host-token GRU + attention | 5 | silent_onset | 4,443 | 0.9526 | 0.4 | 10.8 | 0.9523 | 1.0 | 10.8 |

## Standard test: pooled confusion (row-normalized, seed mean)

**CyberMetaFormer**

| True \ Pred | clean | user | root |
|---|---:|---:|---:|
| clean | 0.9916 | 0.0003 | 0.0082 |
| user | 0.0415 | 0.9553 | 0.0031 |
| root | 0.0383 | 0.0000 | 0.9617 |

**Full-history GRU**

| True \ Pred | clean | user | root |
|---|---:|---:|---:|
| clean | 0.9894 | 0.0003 | 0.0104 |
| user | 0.0428 | 0.9518 | 0.0054 |
| root | 0.0399 | 0.0001 | 0.9600 |

**Host-token GRU + topo attention**

| True \ Pred | clean | user | root |
|---|---:|---:|---:|
| clean | 0.9919 | 0.0003 | 0.0079 |
| user | 0.0414 | 0.9554 | 0.0032 |
| root | 0.0384 | 0.0000 | 0.9616 |

**Host-token GRU + attention**

| True \ Pred | clean | user | root |
|---|---:|---:|---:|
| clean | 0.9916 | 0.0003 | 0.0081 |
| user | 0.0414 | 0.9555 | 0.0031 |
| root | 0.0384 | 0.0000 | 0.9616 |

## OOD test: per-class metrics (pooled over evaluated hosts; seed mean ± sd)

| Model | Seeds | Class | Precision | Recall | F1 | Host-avg F1 |
|---|---:|---|---:|---:|---:|---:|
| CyberMetaFormer | 5 | clean | 0.981 ± 0.002 | 0.983 ± 0.002 | 0.982 ± 0.000 | 0.981 ± 0.000 |
| CyberMetaFormer | 5 | user | 0.994 ± 0.000 | 0.961 ± 0.000 | 0.977 ± 0.000 | 0.972 ± 0.000 |
| CyberMetaFormer | 5 | root | 0.827 ± 0.016 | 0.825 ± 0.016 | 0.826 ± 0.002 | 0.872 ± 0.002 |
| Full-history GRU | 5 | clean | 0.978 ± 0.001 | 0.976 ± 0.001 | 0.977 ± 0.000 | 0.976 ± 0.000 |
| Full-history GRU | 5 | user | 0.990 ± 0.001 | 0.956 ± 0.001 | 0.972 ± 0.000 | 0.965 ± 0.000 |
| Full-history GRU | 5 | root | 0.773 ± 0.005 | 0.795 ± 0.007 | 0.783 ± 0.002 | 0.837 ± 0.004 |
| Host-token GRU + topo attention | 5 | clean | 0.980 ± 0.001 | 0.984 ± 0.001 | 0.982 ± 0.000 | 0.981 ± 0.000 |
| Host-token GRU + topo attention | 5 | user | 0.994 ± 0.000 | 0.962 ± 0.001 | 0.977 ± 0.000 | 0.972 ± 0.000 |
| Host-token GRU + topo attention | 5 | root | 0.834 ± 0.009 | 0.818 ± 0.006 | 0.826 ± 0.003 | 0.873 ± 0.002 |
| Host-token GRU + attention | 5 | clean | 0.980 ± 0.001 | 0.983 ± 0.002 | 0.982 ± 0.001 | 0.981 ± 0.001 |
| Host-token GRU + attention | 5 | user | 0.994 ± 0.001 | 0.962 ± 0.001 | 0.977 ± 0.000 | 0.972 ± 0.000 |
| Host-token GRU + attention | 5 | root | 0.832 ± 0.015 | 0.818 ± 0.014 | 0.824 ± 0.005 | 0.871 ± 0.004 |

## OOD test: security-relevant error rates

| Model | Root→user | Root→clean | User→clean | User→root | Clean false alarm | Worst host (macro-F1) | Best host (macro-F1) |
|---|---:|---:|---:|---:|---:|---|---|
| CyberMetaFormer | 0.0001 ± 0.0000 | 0.1752 ± 0.0161 | 0.0358 ± 0.0006 | 0.0028 ± 0.0004 | 0.0175 ± 0.0024 | user2 (0.899) | ophost0 (1.000) |
| Full-history GRU | 0.0008 ± 0.0001 | 0.2047 ± 0.0074 | 0.0377 ± 0.0008 | 0.0064 ± 0.0002 | 0.0236 ± 0.0009 | user2 (0.875) | opserv (0.999) |
| Host-token GRU + topo attention | 0.0001 ± 0.0001 | 0.1814 ± 0.0057 | 0.0357 ± 0.0003 | 0.0027 ± 0.0005 | 0.0164 ± 0.0011 | user4 (0.901) | ophost0 (1.000) |
| Host-token GRU + attention | 0.0001 ± 0.0001 | 0.1823 ± 0.0144 | 0.0360 ± 0.0006 | 0.0025 ± 0.0003 | 0.0168 ± 0.0020 | user4 (0.898) | opserv (1.000) |

## OOD test: detection latency (steps; seeds with stored predictions)

| Model | Seeds | Onset | Spells | Detected (breach) | Median | p90 | Detected (exact class) | Median | p90 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| CyberMetaFormer | 5 | all | 152,839 | 0.9921 | 0.0 | 0.0 | 0.9918 | 0.0 | 0.0 |
| CyberMetaFormer | 5 | alert_onset | 145,135 | 1.0000 | 0.0 | 0.0 | 0.9998 | 0.0 | 0.0 |
| CyberMetaFormer | 5 | silent_onset | 7,704 | 0.8438 | 2.4 | 22.4 | 0.8410 | 2.4 | 22.4 |
| Full-history GRU | 5 | all | 152,839 | 0.9917 | 0.0 | 0.0 | 0.9904 | 0.0 | 0.0 |
| Full-history GRU | 5 | alert_onset | 145,135 | 0.9999 | 0.0 | 0.0 | 0.9990 | 0.0 | 0.0 |
| Full-history GRU | 5 | silent_onset | 7,704 | 0.8374 | 2.0 | 19.2 | 0.8280 | 2.0 | 19.2 |
| Host-token GRU + topo attention | 5 | all | 152,839 | 0.9920 | 0.0 | 0.0 | 0.9917 | 0.0 | 0.0 |
| Host-token GRU + topo attention | 5 | alert_onset | 145,135 | 1.0000 | 0.0 | 0.0 | 0.9998 | 0.0 | 0.0 |
| Host-token GRU + topo attention | 5 | silent_onset | 7,704 | 0.8417 | 2.8 | 22.0 | 0.8384 | 2.8 | 22.0 |
| Host-token GRU + attention | 5 | all | 152,839 | 0.9918 | 0.0 | 0.0 | 0.9915 | 0.0 | 0.0 |
| Host-token GRU + attention | 5 | alert_onset | 145,135 | 1.0000 | 0.0 | 0.0 | 0.9998 | 0.0 | 0.0 |
| Host-token GRU + attention | 5 | silent_onset | 7,704 | 0.8375 | 2.8 | 21.4 | 0.8346 | 2.8 | 21.4 |

## OOD test: pooled confusion (row-normalized, seed mean)

**CyberMetaFormer**

| True \ Pred | clean | user | root |
|---|---:|---:|---:|
| clean | 0.9825 | 0.0002 | 0.0173 |
| user | 0.0358 | 0.9614 | 0.0028 |
| root | 0.1752 | 0.0001 | 0.8247 |

**Full-history GRU**

| True \ Pred | clean | user | root |
|---|---:|---:|---:|
| clean | 0.9764 | 0.0003 | 0.0232 |
| user | 0.0377 | 0.9560 | 0.0064 |
| root | 0.2047 | 0.0008 | 0.7945 |

**Host-token GRU + topo attention**

| True \ Pred | clean | user | root |
|---|---:|---:|---:|
| clean | 0.9836 | 0.0002 | 0.0162 |
| user | 0.0357 | 0.9616 | 0.0027 |
| root | 0.1814 | 0.0001 | 0.8184 |

**Host-token GRU + attention**

| True \ Pred | clean | user | root |
|---|---:|---:|---:|
| clean | 0.9832 | 0.0002 | 0.0166 |
| user | 0.0360 | 0.9615 | 0.0025 |
| root | 0.1823 | 0.0001 | 0.8176 |


| Measure | Train | Test | OOD |
|---|---:|---:|---:|
| Compromised / all host-steps | 0.224 | 0.223 | 0.113 |
| Silent / compromised host-steps | 0.408 | 0.404 | 0.593 |
| Root share among silent host-steps | 0.980 | 0.979 | 0.975 |
| No alert on the host up to t (N_hist) / silent | 0.260 | 0.253 | 0.301 |
| No alert on the host in the whole episode (N_ep) / silent | 0.242 | 0.235 | 0.215 |
| N_ep / N_hist | 0.931 | 0.930 | 0.716 |
| Previous alert within the last 32 steps† | 0.558 | 0.559 | 0.513 |
| Previous alert within the last 64 steps† | 0.891 | 0.894 | 0.885 |
| Silent host-steps (n) | 1,759,365 | 216,495 | 321,008 |
| N_hist host-steps (n) | 457,746 | 54,731 | 96,486 |
| N_ep host-steps (n) | 426,051 | 50,884 | 69,088 |
| N_hist (episode, host) pairs (n) | 7,229 | 861 | 1,891 |
| N_ep (episode, host) pairs (n) | 5,584 | 676 | 1,022 |
| Episodes | 16,000 | 2,000 | 4,000 |

Note. A host-step is silent when the host is compromised (user or root) and neither the exploit-activity bit nor the compromise flag is set in the current observation. N_hist(h, t): no alert on host h in obs[1..t] (online-applicable); N_ep(h): no alert on host h in obs[1..T] (retrospective; every N_ep host-step is also an N_hist host-step). Both subsets are restricted to silent host-steps (Section 3.3). obs[0] carries no alert in any split. † Among silent host-steps whose host had an earlier alert.

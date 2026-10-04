## Table 1 - Data

| | Train | Dev | Test |
|---|---|---|---|
| Pairs | 56,355 | 8,421 | 15,878 |
| Mean / max source length | 42.5 / 222 | 42.5 / 167 | 42.7 / 260 |
| Mean / max target length | 14.8 / 65 | 14.8 / 44 | 14.9 / 46 |
| Pairs dropped as too long | 19 | 0 | 0 |

## Table 2 - Model and training

| | |
|---|---|
| Trainable parameters | 7,577,600 |
| Epochs trained / best epoch | 20 / 19 |
| Best dev loss | 1.4541 |
| Training time and GPU | 22 min, Tesla T4 |

## Table 3 - Official metrics

| Split | Decoding | Logical form (%) | Execution (%) | Parse failures (%) |
|---|---|---|---|---|
| Dev | greedy | 64.45 | 70.73 | 0.00 |
| Dev | beam (4) | 64.94 | 71.11 | 0.00 |
| Test | beam | 64.99 | 70.97 | 0.01 |

## Table 4 - Component accuracy (dev, beam)

| sel column (%) | agg (%) | WHERE clause (%) |
|---|---|---|
| 92.72 | 89.88 | 75.23 |

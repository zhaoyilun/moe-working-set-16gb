# Results

## Decode progression

| Data label | Configuration | tok/s |
|---|---|---:|
| Measured | Coarse layer offload, `-ngl 10` | 6.538 |
| Measured | Native expert-only CPU, same Stage-1 path | 15.135 |
| Measured | Native + CUDA Graph, historical best | 17.063 |
| Measured | Hybrid fixed cache, 4 GiB, 3×500 | 16.096 |
| Measured | Hybrid fixed cache, 6 GiB, 3×500 | 17.275 |
| Measured | Hybrid fixed cache, 8 GiB, 3×500 | **18.753** |

The 8 GiB result is 23.91% above the same-path Native baseline and 9.91% above the historical Native+Graph best.

## Selective H2D negative result

| Threshold | hit rate | H2D weight/token | promotion time/token | tok/s |
|---:|---:|---:|---:|---:|
| 0 | 31.11% | 0 MB | 0 ms | **12.767** |
| 1 | 32.73% | 3.53 MB | 0.87 ms | 12.269 |
| 2 | 39.86% | 20.67 MB | 4.83 ms | 11.551 |
| 3 | 53.69% | 64.38 MB | 13.60 ms | 12.010 |
| 4 | 61.78% | 114.45 MB | 22.89 ms | 11.152 |

This experiment used an instrumented 6 GiB path and one 100-token run per threshold. It is a mechanism gate, separate from the frozen 8 GiB 3×500 headline.

## Context and pool matrix

| Context | 4 GiB | 6 GiB | 8 GiB | Safe default |
|---:|---:|---:|---:|---:|
| 2,027 | 16.405 | 17.162 | **17.511** | 8 GiB |
| 8,171 | 15.462 | 15.986 | **17.626** | 8 GiB |
| 16,363 | 15.050 | 15.032 | **16.652** | 6 GiB |
| 32,747 | 13.620 | 14.104 | **15.118** | 6 GiB |

These are one-run 100-token context comparisons collected in a later daily-state window. They show relative coupling and leave the earlier 18.753 headline unchanged.

## Prefill v2

Strict same-prompt 16K A/B:

| physical ubatch | tok/s | seconds |
|---:|---:|---:|
| 2,048 | 287.387 | 56.937 |
| 4,096 | 411.174 | 39.796 |
| 8,192, run 1 | 523.859 | 31.235 |
| 8,192, run 2 | 526.970 | 31.051 |
| 8,192, run 3 | 521.179 | 31.396 |

The 8K mean is **524.003 tok/s**, 82.33% above the same-prompt 2K control. A 32K single context check measured 617.342 tok/s.

Numerical boundary: 4K and 8K ubatch produced the same first token and same 32-token greedy continuation. The 2K path matched the first six continuation tokens, then diverged; logits cosine was 0.990457 for 2K versus 8K. Multi-prompt quality evaluation remains a release decision item.

## Raw data

Every table above is backed by a CSV under `data/results/`. See `LIMITATIONS.md` before comparing across measurement windows.

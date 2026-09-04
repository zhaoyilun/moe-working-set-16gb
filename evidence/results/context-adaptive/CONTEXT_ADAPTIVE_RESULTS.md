# Context-Adaptive VRAM Gate

## 结论

**Context–Expert Cache Coupling 已经实测出现。** 在同一批次、同一 forced-token continuation、CUDA Graph ON 的固定缓存矩阵中，8 GiB expert pool 始终给出最高短跑吞吐；但它在 16K 和 32K 的显存余量已经低于 1 GiB。最终策略分成两档：

- **Benchmark-max**：2K / 8K / 16K / 32K 均为 8 GiB；
- **Safe-default**：2K / 8K 用 8 GiB，16K / 32K 用 6 GiB。

这两个答案服务于不同目标：前者回答本机短跑峰值，后者回答日常桌面环境下的默认部署。

## 真实 Decode 矩阵

统一条件：100 token warmup + 100 token measured；动态 promotion 关闭；fixed plan hit 走 GPU、miss 走 Native CPU；性能路径不启用 route readback。

| 数据标签 | Context | actual tokens | Pool | tok/s | p50 | p95 | CUDA Graph |
|---|---:|---:|---:|---:|---:|---:|---:|
| Measured | 2K | 2,027 | 4 GiB | 16.405 | 59.93 ms | 72.64 ms | ON |
| Measured | 2K | 2,027 | 6 GiB | 17.162 | 57.49 ms | 72.40 ms | ON |
| Measured | 2K | 2,027 | 8 GiB | **17.511** | **54.71 ms** | 80.57 ms | ON |
| Measured | 8K | 8,171 | 4 GiB | 15.462 | 62.97 ms | 83.30 ms | ON |
| Measured | 8K | 8,171 | 6 GiB | 15.986 | 60.39 ms | 79.86 ms | ON |
| Measured | 8K | 8,171 | 8 GiB | **17.626** | **54.71 ms** | **71.64 ms** | ON |
| Measured | 16K | 16,363 | 4 GiB | 15.050 | 65.42 ms | 77.48 ms | ON |
| Measured | 16K | 16,363 | 6 GiB | 15.032 | 64.07 ms | 85.03 ms | ON |
| Measured | 16K | 16,363 | 8 GiB | **16.652** | **57.94 ms** | **75.82 ms** | ON |
| Measured | 32K | 32,747 | 4 GiB | 13.620 | 72.15 ms | 92.05 ms | ON |
| Measured | 32K | 32,747 | 6 GiB | 14.104 | 69.40 ms | 89.26 ms | ON |
| Measured | 32K | 32,747 | 8 GiB | **15.118** | **64.18 ms** | **80.01 ms** | ON |

8 GiB 路径从 2K 的 17.511 降到 16K 的 16.652，降幅 4.91%；到 32K 为 15.118，较 2K 下降 13.66%。32K 相对 16K 再下降 9.21%。

### 与冻结 Stage 1 的关系

本矩阵是后续同批次的 100-token context sweep；其中 16K / 8 GiB 为 16.652 tok/s。正式 headline 仍采用 Stage 1 的三轮 500-token 均值 **18.753 tok/s**。当前矩阵只用于同批次横向选择和 context 趋势，未替换已冻结的稳定性基线。

## Cache hit 与 CPU fallback

为避免 route readback 扰动性能数字，cache 计数来自独立的 stats-only companion run；输入 token、state、pool plan 与性能运行一致。该路径每 token 做一次 packed route D2H 与同步，所以这里只采用命中/回退计数，吞吐来自上一节。

| 数据标签 | Context | 4 GiB hit / fallback experts | 6 GiB hit / fallback experts | 8 GiB hit / fallback experts |
|---|---:|---:|---:|---:|
| Measured | 2K | 28.66% / 342.45 | 41.05% / 282.95 | **50.29% / 238.59** |
| Measured | 8K | 28.49% / 343.27 | 40.91% / 283.65 | **50.17% / 239.20** |
| Measured | 16K | 28.55% / 342.98 | 41.06% / 282.93 | **50.13% / 239.36** |
| Measured | 32K | 28.15% / 344.88 | 40.58% / 285.21 | **49.86% / 240.69** |

CPU fallback layers/token 仍约为 46–48，因为一个 layer 只要存在至少一个 miss 就会记为 fallback layer。8 GiB 的主要收益表现为每 token CPU expert 数从约 343 降到约 239，而不是把大量 layer 变成全命中。

固定缓存路径的 H2D weight/token 为 **0 MB**；Stage 2 的 selective H2D 已单独判为 NO-GO。

## 16 GiB 显存账本

| 数据标签 | Context | Pool | CUDA model buffer | KV total | recurrent | workspace | allocator-accounted | allocator headroom |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Measured allocator logs | 2K | 8 GiB | 13,008.91 MiB | 99 MiB | 112.57 MiB | 1,030.19 MiB | 14,251.62 MiB | 2,124.38 MiB |
| Measured allocator logs | 8K | 8 GiB | 13,008.91 MiB | 297 MiB | 112.57 MiB | 1,039.26 MiB | 14,458.69 MiB | 1,917.31 MiB |
| Measured allocator logs | 16K | 8 GiB | 13,008.91 MiB | 561 MiB | 112.57 MiB | 1,842.57 MiB | 15,526.00 MiB | 850.00 MiB |
| Measured + device sample | 16K | 8 GiB | 13,008.91 MiB | 561 MiB | 112.57 MiB | 1,842.57 MiB | 15,526.00 MiB | **386 MiB observed** |
| Measured allocator logs | 32K | 8 GiB | 13,008.91 MiB | 1,089 MiB | 112.57 MiB | 1,858.57 MiB | 16,070.00 MiB | **306.00 MiB** |
| Measured allocator logs | 16K | 6 GiB | 10,866.63 MiB | 561 MiB | 112.57 MiB | 1,842.57 MiB | 13,383.72 MiB | 2,992.28 MiB |
| Measured allocator logs | 32K | 6 GiB | 10,866.63 MiB | 1,089 MiB | 112.57 MiB | 1,858.57 MiB | 13,927.72 MiB | 2,448.28 MiB |

`CUDA model buffer` 已包含固定 GPU core、expert cache 映射与分配填充；其中固定 core 的 Native 实测值为 4,459.62 MiB，packed expert weights 的精确值分别为 4,094.56 / 6,142.94 / 8,191.33 MiB。表格总量使用 CUDA model buffer，避免重复相加 expert pool。

除 16K / 8 GiB 外，本轮没有进行设备峰值轮询；原因是此前已经确认 NVML 高频查询会扰动 WDDM / CUDA Graph 吞吐。其余行使用 llama.cpp allocator 日志，字段明确标为 allocator-accounted，而非设备峰值。

## Auto Policy

规则：

`16,376 MiB total → subtract measured model/KV/recurrent/workspace/output → require 1,024 MiB allocator headroom → choose largest qualified measured pool`

| 数据标签 | Context | Benchmark-max | Safe-default | safe allocator headroom |
|---|---:|---:|---:|---:|
| Measured + policy decision | 2K | 8 GiB / 17.511 tok/s | **8 GiB** | 2,124 MiB |
| Measured + policy decision | 8K | 8 GiB / 17.626 tok/s | **8 GiB** | 1,917 MiB |
| Measured + policy decision | 16K | 8 GiB / 16.652 tok/s | **6 GiB** | 2,992 MiB |
| Measured + policy decision | 32K | 8 GiB / 15.118 tok/s | **6 GiB** | 2,448 MiB |

16K 的单次短矩阵中 4 GiB 与 6 GiB 相差 0.12%，属于短窗口噪声量级；冻结 Stage 1 的三轮 500-token 结果为 16.096 vs 17.275 tok/s，且 6 GiB 仍有约 3 GiB allocator 余量，因此 Safe-default 选择 6 GiB。

## Context 状态与 Prefill

| 数据标签 | Context | actual prompt tokens | state build prefill | prefill tok/s | state size | restore-to-first-token |
|---|---:|---:|---:|---:|---:|---:|
| Measured | 2K | 2,027 | 16.161 s | 125.43 | 177.96 MiB | 0.231 s |
| Measured | 8K | 8,171 | 34.404 s | 237.50 | 376.27 MiB | 0.436 s |
| Measured | 16K | 16,363 | **46.429 s** | **352.43** | 640.67 MiB | 0.807 s |
| Measured | 32K | 32,747 | **105.851 s** | **309.37** | 1,169.49 MiB | 1.345 s |

新建的 2K、8K、32K state 均通过 token count、首 token、验证序列与 logits roundtrip 校验。32K Cold Prefill 仍需约 106 秒，说明 Prefill v2 仍是长上下文体验的独立主线。

## Gate 判断

1. Context 增长同时推高 KV 和部分 workspace；32K 的 8 GiB allocator 余量只剩 306 MiB。
2. 固定 plan 的命中率在同一 forced-token continuation 下仅小幅变化，吞吐下降的主要新增项是 attention/KV/context compute，而非 hit rate突然崩塌。
3. 8 GiB 在 32K 完成一次短跑，证明容量路径可执行；它不进入 Safe-default。
4. 64K 留给后续独立长上下文 Gate。32K 已经回答本轮的耦合问题，继续创建 64K state 会新增数分钟 Cold Prefill 和多 GiB session 文件，却不改变当前 16 GiB 策略判断。

## 证据文件

- `context_pool_matrix.csv`
- `optimal_pool_policy.csv`
- `auto_policy.json`
- `context_state_builds.csv`
- `runs/*/result.json`
- `runs/*/stats-result.json`
- `states/*-state-roundtrip.json`

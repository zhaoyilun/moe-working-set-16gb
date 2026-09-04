# End-to-End Critical Path Gate：最终报告

## 结论

本轮结论收敛为三条彼此独立的基线：

1. **原始粗粒度 Layer Offload（`-ngl 10`）**：完整 decode 实测 **6.538 tok/s**。`llama_decode` 平均 **148.235 ms/token**，其中 CPU-resident 0–38 层直接测得 **132.875 ms/token**，占 `llama_decode` 的 **89.64%**。主要问题是整个层一起放置，使大量 attention、hyperconnection、shared expert、residual 和 next-layer preparation 跟随 routed experts 留在 CPU。
2. **Native Expert-only CPU Offload**：非 expert tensor 放在 GPU，48 层 routed expert tensor 留在 CPU/RAM。3×500 token 实测 **15.094 tok/s**，属于当前新的真实 Native Baseline；同一执行路径中 CPU routed-expert 子图为 **35.223 ms/token**。
3. **Phase-Adaptive Hybrid**：GPU Expert Cache、selective H2D、CPU fallback 目前由单层/短 replay 实测、真实 trace 仿真和 48 层投影组成。本轮没有把它报告成完整模型实测，也没有把旧的 26.921 ms/token 从任一真实 wall time 中直接扣除。

核心判断是 **C：placement / global VRAM**。粗粒度按层 offload 是 6.5 tok/s 的主因；结构感知的 expert-only CPU offload 把同机 decode 提升到约 15–16 tok/s。接下来要验证的 decode 门槛是：集成后的 GPU Expert Cache 是否真实超过 Native，先看 **18–20+ tok/s**。

## 数据来源标签

| 标签 | 本报告中的含义 |
|---|---|
| **Measured** | 完整 llama.cpp 路径或 profiler 直接测量 |
| **Prototype measured** | 独立 Hybrid layer、copy、CPU/GPU expert 或短 replay 的真实测量 |
| **Trace simulation** | 在五类真实 routing trace 上重放 cache 状态和 miss 分布 |
| **Projection** | 把不同但已标定的组件用于规划估算；不视作完整模型 wall-time 实测 |

完整逐项登记见 `evidence_ledger.csv`。

---

## 1. 三条基线

### 1.1 原始粗粒度 `-ngl 10` — Measured

固定条件：同一 GGUF、`n_ctx=2048`、24 threads、greedy sampling、100 warmup、3×500 measured tokens。

| 指标 | 结果 |
|---|---:|
| 完整 token wall mean | **152.957 ms** |
| wall p50 / p95 | **151.818 / 162.436 ms** |
| `llama_decode` mean | **148.235 ms** |
| sampling mean | **4.723 ms** |
| Decode TG | **6.538 tok/s** |
| CPU model buffers | **77,059.72 MiB** |
| GPU model buffer | **12,272.45 MiB** |
| placement | layer 0–38 CPU；layer 39–47 + output GPU |

### 1.2 Native Expert-only CPU Offload — Measured

实际 placement：所有 non-expert tensors 进入 GPU；正则匹配 routed-expert tensor 留在 CPU buffer；48/48 层均采用该结构。

| 指标 | 结果 |
|---|---:|
| 完整 token wall mean | **66.251 ms** |
| wall p50 / p95 | **65.277 / 72.953 ms** |
| `llama_decode` mean | **64.540 ms** |
| sampling mean | **1.711 ms** |
| Decode TG | **15.094 tok/s** |
| 相对 `-ngl 10` | **+130.88% TG；wall -56.69%** |
| CPU mapped model buffers | **88,803.04 MiB** |
| GPU model buffer | **4,459.62 MiB** |
| GPU KV at `n_ctx=2048` | **66.00 MiB** |
| GPU recurrent state | **112.57 MiB** |
| GPU compute buffer | **1,030.50 MiB** |
| observed total GPU memory peak | **6,973 MiB** |
| process working set mean / max | **35.24 / 40.80 GB** |
| CPU utilization mean / p95 / max | **60.43 / 75.40 / 77.01%** |
| GPU utilization mean / p95 / max | **32.85 / 41 / 57%** |

Native 同路径 CPU instrumentation 得到：

| 组件 | ms/token |
|---|---:|
| 48 层 CPU routed-expert 子图 | **35.223** |
| 其中最终 down projection worker | 9.466 |
| 最终 barrier | 3.471 |
| 其余 expert work + 内部 barrier | 22.287 |
| Native `llama_decode` 其余同路径部分 | **29.317** |

这里的 29.317 ms 是 `64.540 - 35.223`，两项来自同一 Native execution path；它仅表示该路径里除 CPU expert 子图以外的剩余时间。

### 1.3 Phase-Adaptive Hybrid — 分级证据

| 内容 | 数值 | 标签 |
|---|---:|---|
| Hybrid branch host span p50 / p95 | 1.0380 / 1.3952 ms/layer | Prototype measured |
| 加独立 router 后 p50 / p95 | 1.1226 / 1.6247 ms/layer | Prototype measured composite |
| 8 GiB LRU held-out hit rate | 70.535% | Trace simulation |
| 48-layer routed-MoE p50 / p95 | 26.921 / 57.391 ms/token | Projection |
| 完整模型 Hybrid TG | 本轮未形成真实 wall-time 样本 | — |

因此本轮 Native 15.094 tok/s 是实测基线；Hybrid 的 18–20 tok/s 区间仍是下一阶段要用完整模型验证的目标。

---

## 2. `-ngl 10` CPU Critical Path

### 2.1 直接回答首要问题

`-ngl 10` 的 `llama_decode = 148.235 ms/token` 中：

- CPU layer 0–38：**132.875 ms/token，89.64%**；
- input embedding / initial prep CPU graph：**1.503 ms/token**；
- 两者合计：**134.377 ms/token，90.65%**；
- 剩余 decode 路径：约 **13.857 ms/token**；
- sampling：**4.723 ms/token**。

这说明 Nsight 中看到的大量 GPU 空闲不是根因本身，而是 CPU 0–38 层串行推进时 GPU 等待的结果。

### 2.2 CPU operator-family breakdown — Measured

| CPU 区间 | ms/token | wall 占比 |
|---|---:|---:|
| FFN residual + next-layer preparation | **52.599** | 34.39% |
| routed MoE combined | **37.514** | 24.53% |
| hyperconnection + attention | **25.564** | 16.71% |
| attention residual + FFN hyperconnection | **10.689** | 6.99% |
| shared expert | **5.890** | 3.85% |
| MoE/shared merge | **0.619** | 0.40% |
| input embedding / initial prep | **1.503** | 0.98% |

补充分解：使用细粒度 marker 的 router/expert 比例校准 37.514 ms combined span，router 约 **5.363 ms/token**，routed expert + merge 约 **32.151 ms/token**。这是“实测比例 × 实测 combined span”，已在 `per_layer_attribution.csv` 单独标记。

CPU worker/barrier 记录采用轻量 endpoint markers：已观测 marker worker 合计 **11.780 ms/token**，marker 后 barrier 下界 **3.018 ms/token**。它们是部分节点观测，不代表全图所有 worker/barrier；全图耗时以上述 graph elapsed 为准。

### 2.3 GPU timeline — Measured

| 项目 | ms/token |
|---|---:|
| GPU kernel active | 4.151 |
| GPU copy-engine active | 0.347 |
| GPU active union | **4.498** |
| token 内 first-to-last GPU span | 159.143 |
| token 内 GPU idle | **157.310** |
| layer 39–47 elapsed spans | 3.728 |
| output / lm_head active | **0.980** |
| layer 39–47 + output final burst | **4.709** |

GPU span 很长是因为 39 个 CPU layer 之间夹有少量 GPU/copy/synchronization 活动；真正 GPU active 只有约 4.5 ms/token。完整 API/critical-partition 数据见 `gpu_idle_analysis.csv` 和 `nsys/nsys_summary.json`。

### 2.4 Per-layer timing

所有 layer 0–47 已写入 `per_layer_attribution.csv`：

- layer 0–38：CPU graph total、attention/hyperconnection、router、routed expert、shared expert、merge、residual/next prep、marker worker/barrier；
- layer 39–47：Nsight GPU final-burst 分层 elapsed 与 placement；
- 同一张表还附有 Native 路径每层 CPU expert graph 时间。

粗粒度路径最慢的 layer 是 layer 0：**4.715 ms**；其余多数 CPU layer 约 **3.1–3.7 ms**。GPU 端 layer 39–47 合计仅 **3.728 ms**。

---

## 3. Context / Prefill / Decode 控制结果 — Native Measured

每个点使用实际 evaluated prompt token 数计算 PP，未用目标 context 代替分母。Prefill 时间是进程内首次 prompt evaluation，不含 model load；本轮未清空 OS page cache。

| Context target | actual prompt tokens | prefill seconds | Prompt Processing | Decode TG | KV VRAM |
|---:|---:|---:|---:|---:|---:|
| 512 | 491 | 11.081 s | **44.31 tok/s** | **16.51 tok/s** | 24.75 MiB |
| 1K | 1,003 | 15.739 s | **63.73 tok/s** | **16.48 tok/s** | 41.25 MiB |
| 2K | 2,027 | 23.893 s | **84.84 tok/s** | **16.60 tok/s** | 74.25 MiB |
| 4K | 4,075 | 40.047 s | **101.76 tok/s** | **16.29 tok/s** | 140.25 MiB |
| 8K | 8,171 | 74.029 s | **110.38 tok/s** | **15.55 tok/s** | 272.25 MiB |
| 16K | 16,363 | 141.507 s | **115.63 tok/s** | **15.05 tok/s** | 536.25 MiB |

从 512 到 16K，Decode TG 从 16.51 降到 15.05 tok/s，下降 **8.81%**。可支持的结论是：

> 在当前 Native expert-only CPU placement 下，到 16K 为止，Attention/KV 本身没有造成 decode 性能断崖。

该 sweep 不代表 Hybrid 长上下文表现。真实 Hybrid 还存在 **Context–Expert Cache Coupling**：

`Context ↑ → KV/compute VRAM ↑ → Expert Pool headroom ↓ → hit rate 可能下降 → H2D / CPU fallback 可能上升 → TG 可能下降`

本轮已有日志可给出 accounted CUDA buffers；仅 2K 的 3×500 control 同时采到了 NVML total peak：

| Context | KV | accounted CUDA buffers | observed total VRAM |
|---:|---:|---:|---:|
| 512 | 24.75 MiB | 5,595.19 MiB | — |
| 1K | 41.25 MiB | 5,624.19 MiB | — |
| 2K | 74.25 MiB | 5,677.39 MiB | 6,973 MiB |
| 4K | 140.25 MiB | 5,746.39 MiB | — |
| 8K | 272.25 MiB | 5,884.39 MiB | — |
| 16K | 536.25 MiB | 6,950.26 MiB | — |

16K 的 compute buffer 从约 1,031 MiB 增至 1,842 MiB，因此显存耦合不只来自 KV。其余 context 未额外采 NVML total，表中保持空值。

### 长上下文用户体验瓶颈

16,363 个实际 prompt tokens 的首次 Prefill 为 **141.507 s / 115.63 tok/s**。这已经成为长上下文交互的主要等待来源。下一阶段 Prefill Streaming Gate 的目标应分级为：

- 工程验证：200–300+ tok/s；
- 最低实用改善：500+ tok/s；
- 主要目标：1000+ tok/s；
- 优秀：2000+ tok/s。

本轮只登记这条下一阶段主线，没有启动新的 Prefill 优化实验。

---

## 4. Top-1 / Top-2 控制实验

### Top-1：placement — Measured

| Control | Candidate | Control TG | Candidate TG | 变化 |
|---|---|---:|---:|---:|
| `-ngl 10`，0–38 整层 CPU | non-expert GPU + experts CPU/RAM | 6.538 | 15.094 | **+130.88%** |

这是本轮决定性的优化。它同时把 attention、shared/dense、residual 和 hyperconnection 从 CPU 粗粒度层放置中解耦出来。

### Top-2：CUDA Graph — Measured A/B

相同 prompt、相同 context position、相同 100 warmup、各 500 measured tokens：

| Control | Candidate | wall | TG | 变化 |
|---|---|---:|---:|---:|
| CUDA Graph disabled | CUDA Graph enabled | 80.381 → 58.607 ms | 12.441 → 17.063 tok/s | **+37.15%** |

日志中 enabled 路径记录 150 次 graph warmup/reuse marker，disabled 路径为 0。该 A/B 说明 Native/后续 Hybrid 都应保留 CUDA Graph。17.063 是此单次相同提示词控制实验的结果；正式 Native Baseline 仍采用更保守的 3×500 pooled **15.094 tok/s**。

---

## 5. 16 GiB Global Budget

2K Native control 的 observed GPU peak 是 **6,973 MiB**。以实际设备 16,376 MiB 为总量，把已有 LRU trace 数据叠加到该 base footprint，仅用于规划：

| Expert cache | projected total peak | headroom | 8GB LRU类 hit/对应 hit | 状态 |
|---:|---:|---:|---:|---|
| 0 GiB | 6,973 MiB | 9,403 MiB | 0% | Native measured |
| 2 GiB | 9,021 MiB | 7,355 MiB | 35.67% | fits |
| 4 GiB | 11,069 MiB | 5,307 MiB | 54.03% | fits |
| 6 GiB | 13,117 MiB | 3,259 MiB | 63.60% | fits |
| 8 GiB | **15,165 MiB** | **1,211 MiB** | **70.53%** | fits, tight |
| 10 GiB | 17,213 MiB | -837 MiB | 76.23% | over budget |
| 12 GiB | 19,261 MiB | -2,885 MiB | 80.46% | over budget |

因此 8 GiB 在 2K Native footprint 上只是紧贴上限的候选，不是可自由叠加的固定池。到 16K 时，仅已记账 CUDA buffers 就比 2K 多约 **1,273 MiB**；真正 Hybrid 要按 context 动态缩减 Expert Pool，并重新测 hit rate 和 TG。

`vram_budget_sweep.csv` 还包含一个明确标为 **Projection** 的同路径规划值：8 GiB 约 19.08 tok/s。该数值只用于决定集成优先级，不列入三条基线的完整模型实测。

---

## 6. 最终边界与下一阶段

### 已确认事实

- `-ngl 10` 的 6.538 tok/s 主要由 CPU 0–38 整层 placement 导致；这 39 层占 `llama_decode` 的 89.64%。
- Native expert-only CPU placement 实测 15.094 tok/s，达到既定 15+ 区间。
- Native 路径到 16K 的 decode 下降约 8.81%，没有 attention/KV 断崖。
- 16K Prefill 使用 16,363 个实际 prompt tokens，速度 115.63 tok/s，耗时 141.507 s。
- 2K base footprint 上 8 GiB expert pool 只剩约 1.18 GiB 设备余量。

### 后续验证项

1. 把 Phase-Adaptive Hybrid 真正接入 Native non-expert GPU path，取得完整模型 wall-time；首个门槛是稳定超过 Native，并验证 18–20+ tok/s。
2. 在集成路径上测 Context–Expert Cache Coupling，形成 `context → KV/workspace → expert capacity → hit rate → TG` 曲线。
3. 独立启动 Prefill Streaming Gate，把当前约百 tok/s 提升到数百，再冲击 1000+ tok/s。

## 一句话总结

**粗粒度按层 offload 是 6.5 tok/s 的主要根因；结构感知的 expert-only CPU offload 已将同机 decode 提升至约 15–16 tok/s，且到 16K context 仍较稳定。下一阶段真正的问题已经收敛为两个：① GPU Expert Cache 是否能在 Native baseline 上继续提高 decode；② Cold Prefill 能否从当前约百 tok/s 提升到数百乃至 1000+ tok/s。**

---

## 主要产物

- `critical_path_summary.json`
- `token_timeline.csv`
- `per_layer_attribution.csv`
- `device_residency_map.csv`
- `gpu_idle_analysis.csv`
- `native_cpu_expert_layer_timing.csv`
- `context_prefill_decode.csv`
- `bottleneck_pareto.csv`
- `optimization_ab_results.csv`
- `vram_budget_sweep.csv`
- `evidence_ledger.csv`
- `nsys/baseline_ngl10.nsys-rep`
- `nsys/baseline_ngl10.sqlite`

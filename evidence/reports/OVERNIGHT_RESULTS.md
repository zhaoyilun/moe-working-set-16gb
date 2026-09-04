# PC Slotstream Overnight Continuous Execution — Final Report

日期：2026-09-04（Asia/Shanghai）

## 先看十项结论

| # | 问题 | 收敛结论 | 数据标签 |
|---:|---|---|---|
| 1 | 今夜完成了什么 | 完成 CPU critical-path 归因、Native/Hybrid Decode 门禁、selective H2D 负门禁、2K–32K Context–Cache 矩阵、Prefill v2、OSS 本地暂存与干净构建 | Measured + 工程产物 |
| 2 | 最佳 Decode | **18.753 tok/s**，16K、固定 8 GiB Expert Pool、CUDA Graph ON、3×500 token | Measured |
| 3 | 最佳稳定 Prefill | **524.003 tok/s**，16,363 tokens、physical ubatch 8K、三轮均值；32K 单轮为 **617.342 tok/s** | Measured |
| 4 | 16K / 32K 表现 | 同批次 8 GiB 矩阵：16K **16.652 tok/s**，32K **15.118 tok/s**；16K 冻结稳定 headline 仍为 18.753 tok/s | Measured |
| 5 | 最优 Expert Pool | Benchmark-max：2K–32K 都选 8 GiB；Safe-default：2K/8K 选 8 GiB，16K/32K 选 6 GiB | Measured + policy decision |
| 6 | CUDA Graph | 必须保持开启；8 GiB Stage 1 ON 比 OFF 高 **38.23%** | Measured |
| 7 | 当前主瓶颈 | Decode：CPU miss expert 与图复用边界；动态缓存：route D2H 同步与碎片 H2D；Prefill：专家权重重复搬运与执行粒度 | Measured attribution |
| 8 | 哪些路线已止损 | 相邻 token selective H2D 为 **NO-GO**；12K physical ubatch 超出 16 GiB 预算预测；64K 留至独立长上下文 Gate | Measured + Projection |
| 9 | OSS 条件 | 已达到：Decode 有真实增益，Prefill 越过 500+；本地 staging、图表、文档、patch、runner、复现脚本、静态检查和干净构建均完成 | 工程验证 |
| 10 | 明早动作 | 先审阅本报告；再选“Prefill 1000+ grouped runtime”或“固定 Decode cache 产品化”；公开发布仍由用户单独决定 | 决策项 |

## 核心结论

粗粒度按层 offload 是 6.5 tok/s 的主要根因；结构感知的 expert-only CPU offload 已把同机 Decode 提升到约 15–16 tok/s。固定 GPU Expert Cache 进一步把完整 Decode 推到 18.753 tok/s。Native 在 16K 前保持稳定，Hybrid 在 32K 仍可运行，但显存余量促使日常策略从 8 GiB 收缩到 6 GiB。Prefill 通过增大 physical ubatch 已稳定跨过 500 tok/s，下一量级目标需要 grouped MoE runtime。

## 数据口径

| 标签 | 定义 |
|---|---|
| **Measured** | 完整 llama.cpp 路径、内部 instrumentation 或系统 profiler 直接测量 |
| **Prototype measured** | 单层、单 expert、copy 或短 replay 原型的真实测量 |
| **Trace simulation** | 在已记录 routing IDs 上重放 cache 策略 |
| **Projection** | 由已标定组件或容量模型推导的规划数据 |

早期 routed-MoE prototype 的 26.921 ms/token 与粗粒度 `-ngl 10` 属于不同端到端执行路径。两组数据保持分列，最终结论未执行跨路径减法。

## 一、三条独立基线

| 基线 | Placement | Decode | 关键解释 | 标签 |
|---|---|---:|---|---|
| 原始粗粒度 Layer Offload | `-ngl 10`；layer 0–38 CPU，39–47 + output GPU | **6.538 tok/s** | CPU 端留住了大量 attention、shared/dense、residual、norm 等非 routed-MoE 算子 | Measured |
| Native expert-only CPU offload | non-expert tensors 全 GPU；48 层 routed experts 位于 CPU/RAM | **15.094 tok/s**（critical-path 3×500）；同路径 Stage 1 control **15.135 tok/s** | 新的真实 Native Baseline | Measured |
| Phase-Adaptive Hybrid Stage 1 | 固定 8 GiB GPU Expert Pool；miss 走 Native CPU expert | **18.753 tok/s**（3×500） | 比同路径 Native 15.135 高 **23.91%** | Measured |

其中 `-ngl 10` 的完整 token wall mean 为 152.957 ms，`llama_decode` 为 148.235 ms；Native 的完整 wall mean 为 66.251 ms，`llama_decode` 为 64.540 ms。

## 二、CPU Critical Path：6.5 tok/s 的根因

### 直接归因

- CPU-resident layer 0–38：**132.875 ms/token**，占粗粒度 `llama_decode` 的 **89.64%**；
- input embedding / initial prep：**1.503 ms/token**；
- CPU graph 合计：**134.377 ms/token**，占 `llama_decode` 的 **90.65%**；
- GPU kernel active：约 **4.151 ms/token**；
- GPU active union：约 **4.498 ms/token**；
- layer 39–47 GPU final burst：约 **3.728 ms/token**；
- output / lm_head：约 **0.980 ms/token**。

所以 Nsight 中的 GPU 空闲是结果：CPU 0–38 层串行推进时，GPU 长时间等待。

### CPU operator-family 拆分

| CPU operator family | ms/token | 完整 wall 占比 |
|---|---:|---:|
| FFN residual + next-layer preparation | **52.599** | 34.39% |
| routed MoE combined | **37.514** | 24.53% |
| hyperconnection + attention | **25.564** | 16.71% |
| attention residual + FFN hyperconnection | **10.689** | 6.99% |
| shared expert | **5.890** | 3.85% |
| MoE/shared merge | **0.619** | 0.40% |
| input embedding / initial prep | **1.503** | 0.98% |

细粒度 marker 比例映射到 routed MoE combined span 后，router 约 5.363 ms/token，routed expert + merge 约 32.151 ms/token。CPU worker marker 合计 11.780 ms/token，barrier marker 下界为 3.018 ms/token；二者属于部分节点观测。

Native 同路径 48 层 CPU expert 子图为 **35.223 ms/token**，其余 Native `llama_decode` 为 **29.317 ms/token**。这次相减发生在同一执行路径内，口径成立。

### Per-layer 证据

- `results/end-to-end-critical-path/per_layer_attribution.csv`：layer 0–47 统一表；
- `results/end-to-end-critical-path/cpu_layer_timing.csv`：粗粒度路径 layer 0–38 CPU 算子族；
- `results/end-to-end-critical-path/native_cpu_expert_layer_timing.csv`：Native 路径 layer 0–47 CPU routed-expert 子图；
- `results/end-to-end-critical-path/gpu_layer_timing.csv`：layer 39–47 GPU final burst。

粗粒度路径最慢的是 layer 0，均值 4.715 ms；其余多数 CPU 层约 3.1–3.7 ms。Native CPU expert 每层均已记录 20 个样本。

## 三、Hybrid Decode Runtime

### Stage 1：固定缓存通过

统一条件：canonical 16K state，CUDA Graph ON，100 token warmup，500 measured tokens，三轮独立运行。

| 路径 | tok/s 均值 | p50 | p95 | 结论 |
|---|---:|---:|---:|---|
| Native expert-only CPU | 15.135 | 63.46 ms | 79.84 ms | control |
| Hybrid 4 GiB | 16.096 | 61.06 ms | 74.32 ms | 正收益 |
| Hybrid 6 GiB | 17.275 | 56.97 ms | 72.11 ms | 正收益 |
| Hybrid 8 GiB | **18.753** | **52.34 ms** | **67.77 ms** | 冻结 headline |
| Hybrid 8 GiB，Graph OFF | 13.567 | 73.22 ms | 89.29 ms | 启动开销显著 |

8 GiB 资源轮观测峰值为 **15,990 / 16,376 MiB**，设备余量 **386 MiB**。这证明它适合作为 benchmark-max，而日常默认需保留更宽的显存余量。

### Stage 2：selective H2D 止损

统一条件：canonical 16K、6 GiB、Graph ON、100 warmup + 100 measured、同一 forced continuation。

| 阈值 | tok/s | hit rate | CPU fallback layers/token | H2D weight/token | promotion/token |
|---:|---:|---:|---:|---:|---:|
| T0 stats only | **12.767** | 31.11% | 47.72 | 0 MB | 0 ms |
| T1 miss-set ≤1 | 12.269 | 32.73% | 47.08 | 3.53 MB | 0.87 ms |
| T2 miss-set ≤2 | 11.551 | 39.86% | 45.08 | 20.67 MB | 4.83 ms |
| T3 miss-set ≤3 | 12.010 | 53.69% | 42.08 | 64.38 MB | 13.60 ms |
| T4 miss-set ≤4 | 11.152 | **61.78%** | 40.11 | **114.45 MB** | **22.89 ms** |

命中率上升却未转化为吞吐。主税项是：每 token packed route D2H + scheduler synchronize、gate/up/down 碎片 slice H2D、滞后一拍的预测和替换抖动。Stage 2 T0/T2 logits cosine 为 0.999814702，argmax 同为 2128，因此负结论指向性能机制。运行时默认保持 fixed plan，`promote_threshold=-1`。

Nsight Systems 2023.3.3 的两次短采集均在目标进程退出后留下 collector 驻留，本轮没有引用其数字；Stage 2 归因使用内部计数。

## 四、Context–Expert Cache Coupling

### 同批次 Decode 矩阵

| Context | actual prompt tokens | 4 GiB | 6 GiB | 8 GiB | 8 GiB hit | 8 GiB CPU fallback experts/token |
|---:|---:|---:|---:|---:|---:|---:|
| 2K | 2,027 | 16.405 | 17.162 | **17.511** | 50.29% | 238.59 |
| 8K | 8,171 | 15.462 | 15.986 | **17.626** | 50.17% | 239.20 |
| 16K | 16,363 | 15.050 | 15.032 | **16.652** | 50.13% | 239.36 |
| 32K | 32,747 | 13.620 | 14.104 | **15.118** | 49.86% | 240.69 |

每格为 100 warmup + 100 measured 的单次短矩阵。它用于同批次 context 趋势与 pool 选择；16K 正式稳定 headline 继续采用 Stage 1 的 18.753 tok/s。

8 GiB 从 2K 到 16K 下降 4.91%，到 32K 下降 13.66%。fixed plan 的 hit rate 变化很小，主要新增成本来自 attention/KV/context compute 与显存余量收缩，而非 cache hit 突然坍塌。

### 16 GiB VRAM 策略

| Context | 8 GiB allocator total | 8 GiB headroom | 6 GiB headroom | Benchmark-max | Safe-default |
|---:|---:|---:|---:|---:|---:|
| 2K | 14,251.62 MiB | 2,124.38 MiB | — | 8 GiB | **8 GiB** |
| 8K | 14,458.69 MiB | 1,917.31 MiB | — | 8 GiB | **8 GiB** |
| 16K | 15,526.00 MiB | 850.00 MiB；设备采样余量 386 MiB | 2,992.28 MiB | 8 GiB | **6 GiB** |
| 32K | 16,070.00 MiB | **306.00 MiB** | 2,448.28 MiB | 8 GiB | **6 GiB** |

Safe-default 规则：基于 16,376 MiB 总量，扣除实测 model/KV/recurrent/workspace/output 后，要求至少 1,024 MiB allocator headroom，再选择容量最大的已测 pool。

## 五、Native Context 校准

以下表格使用实际 evaluated prompt token 数计算 Prompt Processing 速度。

| Context target | actual prompt tokens | Cold Prefill | Prefill tok/s | Decode tok/s | KV VRAM | accounted CUDA buffers |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 491 | 11.081 s | 44.31 | 16.51 | 24.75 MiB | 5,595.19 MiB |
| 1K | 1,003 | 15.739 s | 63.73 | 16.48 | 41.25 MiB | 5,624.19 MiB |
| 2K | 2,027 | 23.893 s | 84.84 | 16.60 | 74.25 MiB | 5,677.39 MiB |
| 4K | 4,075 | 40.047 s | 101.76 | 16.29 | 140.25 MiB | 5,746.39 MiB |
| 8K | 8,171 | 74.029 s | 110.38 | 15.55 | 272.25 MiB | 5,884.39 MiB |
| 16K | 16,363 | 141.507 s | **115.63** | **15.05** | 536.25 MiB | 6,950.26 MiB |

512 → 16K 的 Native Decode 仅下降 **8.81%**。可支持的结论是：在 Native expert-only CPU placement 下，到 16K 为止 Attention/KV 没有造成 Decode 断崖。Hybrid 还占用数 GiB Expert Pool，因此其长上下文表现由上节的独立矩阵回答。

## 六、Prefill Acceleration Gate v2

### 16K 严格同 prompt A/B

| physical ubatch | 秒 | tok/s | 对 2K control |
|---:|---:|---:|---:|
| 2,048 | 56.937 | 287.387 | control |
| 4,096 | 39.796 | 411.174 | +43.07% |
| 8,192 run 1 | 31.235 | 523.859 | +82.28% |
| 8,192 run 2 | 31.051 | 526.970 | +83.37% |
| 8,192 run 3 | 31.396 | 521.179 | +81.35% |

8K ubatch 三轮均值 **524.003 tok/s**，CV 0.45%，相对同 prompt 2K control 高 **82.33%**。32,747-token context 的 8K ubatch 单轮为 **617.342 tok/s / 53.045 s**。

### 数值边界

- 2K / 4K / 8K 的首 token argmax 均为 321；
- 4K 与 8K 的 32-token greedy continuation 完全一致；
- 2K 与大 ubatch 的 continuation 在第 7 token 分叉；
- 2K vs 8K logits cosine 为 0.990457。

这符合量化路径受 batch partition 影响的归约差异。性能门禁通过；产品默认前还需多 prompt 质量评估。

### 显存与下一结构

16K / 8K ubatch allocator total 为 13,294.14 MiB，余量 3,081.86 MiB；32K 为 13,822.14 MiB，余量 2,553.86 MiB。12K ubatch 按当前近线性 workspace 模型会越过 16,376 MiB 预算，因此本轮容量 sweep 在 8K 收口。

下一量级结构：

`prompt chunk → router → expert buckets → unique experts → contiguous/bulk staging → grouped GPU GEMM → scatter/weighted merge`

当前门槛状态：200–300+ 已超过；500+ 已通过；1000+ 是下一主目标；2000+ 为后续优秀目标。

## 七、OSS 本地暂存

已生成本地发布暂存：

- llama.cpp patch 与两个 runner；
- Windows / Linux 复现脚本；
- 4/6/8 GiB 固定 plan、runtime 示例；
- 聚合 Decode、Context、VRAM、Prefill、selective-H2D 结果；
- README、Architecture、Results、Limitations、build/benchmark/hardware 文档；
- 5 张结果图；
- Reddit、llama.cpp Discussion、中文长文草稿；
- license/third-party/upstream proposal 清单。

发布暂存验证：

| 检查 | 结果 |
|---|---|
| clean out-of-tree configure/build | **PASS，360/360** |
| `slotstream-prefix-cache.exe` | **PASS，生成成功** |
| `slotstream-hybrid-decode.exe` | **PASS，生成成功** |
| patch reverse-check | **PASS** |
| JSON / CSV / Python / PowerShell / Bash parse | **PASS** |
| private path scan | **0 hits** |
| common secret pattern scan | **0 hits** |
| staging git status | **clean** |
| public remotes | **0** |
| public publication actions | **0** |

本地 staging commit：`622b3114a7fb2c3c38e64922a98130361e761aad`。

## 八、代码与检查点

- llama.cpp experimental checkpoint：`2d4f3154a2d93c3a4d6d4a415c404f1b397d8dcb`，工作树 clean；
- base revision：`4e97ac86ebe2c4cb8212d98d2641ad6768810896`；
- OSS staging checkpoint：`622b3114a7fb2c3c38e64922a98130361e761aad`，工作树 clean、remote count 0；
- Stage 1：`STAGE1_FINAL.md`；
- Stage 2：`STAGE2_FINAL.md`；
- Decode 汇总：`DECODE_RUNTIME_FINAL.md`；
- Context：`results/context-adaptive/CONTEXT_ADAPTIVE_RESULTS.md`；
- Prefill v2：`results/prefill-gate-v2/PREFILL_V2_RESULTS.md`；
- 机器可读总表：`results/overnight/final_summary.json`。

## 九、明早最短决策路径

1. **审阅数据口径**：冻结 18.753 tok/s Decode 与 524.003 tok/s Prefill 为当前两个 headline；32K Prefill 617.342 标注 single run。
2. **选择下一主线**：若目标是长上下文等待时间，进入 Prefill 1000+ grouped runtime；若目标是可日用 Decode，先把 safe-default 8G/6G policy 和 fixed plan 做成产品入口。
3. **确定 Prefill 质量标准**：准备多 prompt 集，比较 4K 与 8K ubatch 的 logits、首 token、greedy prefix 与任务输出。
4. **决定公开节奏**：确定仓库名、版权主体、MIT 文件、README 主推 safe-default 或 benchmark-max，并逐篇审阅发布草稿。
5. **维持已冻结方向**：Graph ON、fixed cache、Native CPU miss；selective H2D 保持实验关闭。

## 范围边界

- 本轮没有更改硬件、BIOS、驱动或系统服务；
- 本轮没有执行公开 push、发帖或远端创建；
- 64K context 与 Prefill grouped kernel 留给后续独立 Gate；
- 现有报告保留 Measured、Prototype measured、Trace simulation、Projection 四种来源，headline 只取 Measured。


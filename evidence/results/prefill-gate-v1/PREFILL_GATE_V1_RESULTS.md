# Prefill Acceleration Gate v1 Results

## 结论

**Prefill v1 达到停止条件 B，下一步返回 Hybrid Decode Runtime。**

- 固定 16K prefix/KV 状态已可靠保存和复用；独立读档后，模型常驻时恢复到首 token 为 **0.807 s**。
- `16K + 500 decode` 的 Warm Development 实测为 **37.623 s**，低于约 40 s 的研发门槛；新进程含模型加载为 **44.337 s**。
- Cold Prefill 通过 `n_ubatch: 512 → 2048` 达到 **352.43 tok/s / 46.429 s**。相对同一测试程序的 512 control（158.36 tok/s）提升 **122.55%**；相对继承的 115.63 tok/s headline 为 **204.78%**，跨程序差异不作为因果收益。
- 500 tok/s / 35 s Success Gate 尚未到达；当前已经满足“研发循环约 40 s + Cold Prefill 明显提升”的停止条件。

## 数据来源

| 标签 | 含义 |
|---|---|
| Measured | 当前完整 llama.cpp 路径、NVML 或 Nsight 直接测量 |
| Prototype measured | 真实 Qwen 单层/真实 expert 的独立原型测量 |
| Trace simulation | 用真实 routing trace 做窗口聚合或容量重放 |
| Projection | 由不同已标定组件组合出的规划值 |

## 与 Decode 三条基线的关系

| 路径 | Decode | 口径 |
|---|---:|---|
| 原始粗粒度 layer offload，`-ngl 10` | 约 6.538 tok/s | Measured；layer 0–38 的大量非 routed-MoE 算子留在 CPU |
| Native expert-only CPU offload | 约 15.094 tok/s；CUDA Graph ON 为 17.063 tok/s | Measured；非 expert tensor 在 GPU，routed expert 权重在 CPU/RAM |
| Phase-Adaptive Hybrid，GPU Expert Cache | 待完整 runtime 测量 | selective H2D / CPU fallback 当前分别标为 Prototype measured、Trace simulation 或 Projection |

本轮 Prefill v1 继承第二条 Native placement。Hybrid 数据继续保持独立；本文未用原型耗时与 Native wall time相减。

## 继承的 Context Sweep

| context target | actual prompt tokens | Prefill 秒 | Prefill tok/s | Decode tok/s | 标签 |
|---:|---:|---:|---:|---:|---|
| 512 | 491 | 11.081 | 44.31 | 16.508 | Measured |
| 1024 | 1,003 | 15.739 | 63.73 | 16.483 | Measured |
| 2048 | 2,027 | 23.893 | 84.84 | 16.600 | Measured |
| 4096 | 4,075 | 40.047 | 101.76 | 16.291 | Measured |
| 8192 | 8,171 | 74.029 | 110.38 | 15.552 | Measured |
| 16384 | 16,363 | 141.507 | 115.63 | 15.053 | Measured |

512 → 16K 时，Decode 从 16.508 降至 15.053 tok/s，降幅 8.81%。这表明当前 Native placement 到 16K 尚未出现 Attention/KV 导致的性能断崖。Hybrid Runtime 还会加入数 GB Expert Cache，Context 增长会压缩 Expert Pool 并影响命中率、H2D 和 CPU fallback，因此 Hybrid 长上下文表现留待接入后实测。

## 1. Prefix/KV 复用

当前 revision `4e97ac86ebe2c4cb8212d98d2641ad6768810896` 已有 `llama_state_save_file` / `llama_state_load_file`。Qwen 的 recurrent state 也进入序列化。采用当前官方 completion/session 的协议：**在 prompt 最后一个 token 之前保存，恢复后在原 position 重放最后一个 token 以重建 logits**。

| 项目 | 16K 结果 | 标签 |
|---|---:|---|
| actual prompt tokens | 16,363 | Measured |
| cache save | 0.466 s | Measured |
| cache file | 0.626 GiB | Measured |
| cache load | 0.587 s | Measured |
| last-token replay | 0.218 s | Measured |
| restore → first token | **0.807 s** | Measured |
| fresh process → first token | 7.521 s | Measured |
| restored decode, 500 token | 13.581 tok/s | Measured |

一致性采用同一 session 协议做 cold/warm 对照：prompt tokens、首 token、后续 32 个 greedy token 全部相同；logits RMSE=0, max-abs=0, cosine=1。

边界：这里证明的是同一官方 session 协议下 cold/warm 状态等价。直接 full-batch Prefill 与“最后一个 token 前保存、恢复后重放”的分段协议会改变数值归约分区，因此本文未宣称任意 batch partition 都是 bit-exact。后续 Hybrid Decode A/B 统一从这份 canonical session state 起跑。

## 2. Cold Prefill

继承的产品基线仍为：16,363 token，141.507 s，115.63 tok/s。新测试程序给出同程序 control 158.36 tok/s；它的用途是建立 ubatch A/B，不覆盖继承的 headline。

| 16K 路径 | n_ubatch | 秒 | tok/s | 标签 |
|---|---:|---:|---:|---|
| 继承 baseline | 512 | 141.507 | 115.63 | Measured |
| 同程序 control | 512 | 103.328 | 158.36 | Measured |
| Prefill v1 | **2048** | **46.429** | **352.43** | Measured |

2K chunk sweep 单调上升，测量范围内最佳点是 2048：64/128/256/512/1024/2048 分别为 32.41/50.46/76.01/105.63/131.59/**150.46 tok/s**。

数值控制：2K 下 512 与 2048 的 logits cosine=0.993633、RMSE=0.264350、max-abs=1.500296；argmax 和后续 32-token greedy 序列相同。ubatch 改变了量化归约次序，因此不是 bit-exact。

## 3. 真实 Prefill Critical Path

2,027-token Nsight capture（n_ubatch=512）：

| 项目 | 结果 | 标签 |
|---|---:|---|
| Prefill wall | 19.232 s | Measured |
| H2D | 9.335 s / 101,173.4 MB / 68,159 copies | Measured |
| H2D / token | **49.91 MB** / 33.63 copies | Measured |
| H2D effective bandwidth | 10.84 GB/s | Measured |
| 所有 GPU kernels | 1.064 s | Measured |
| GPU kernel interval / wall | 5.53% | Measured |
| NVML GPU util（16K control 近似 Prefill 窗口） | mean 50.52%, peak 61% | Measured |
| Host CPU（32 logical processors） | mean 3.48% | Measured |

GPU kernel 内部合计：matmul 675.53 ms、linear attention 117.24 ms、router/dispatch 86.88 ms、full attention 11.01 ms、norm 23.09 ms、其余 149.98 ms。matmul kernel 名称没有 tensor identity，因此 routed/shared/dense 在该项中合并呈现。

**主瓶颈是 CPU-resident expert 权重的碎片化 H2D 与同步。** `cudaMemcpyAsync` host 调用累计 17.846 s；设备 H2D 9.335 s，而全部 GPU compute 仅 1.064 s。当前 Native 名称中的“CPU/RAM”描述的是 expert 权重驻留位置；Prefill 执行实际是选中权重搬到 GPU 后运行 CUDA kernel，而不是由 CPU AVX2 算完全部 expert。

## 4. 真实 Expert Occupancy

真实 1024-token Prefill trace、layer 0–46 的窗口聚合：

| chunk | mean unique experts/layer | mean p50 tokens/active expert |
|---:|---:|---:|
| 16 | 83.57 | 1.14 |
| 32 | 119.84 | 1.81 |
| 64 | 141.62 | 2.73 |
| 128 | 167.23 | 4.33 |
| 256 | 197.13 | 6.90 |
| 512 | 234.99 | 11.02 |
| 1024 | 291.62 | 15.32 |


从 512 到 1024，active experts 只由约 235 增至 292，而每个 active expert 的中位复用由 11.0 增至 15.3；这正是大 ubatch 减少重复传权重的来源。layer 47 只为最终输出 token 计算，因此单独留在原始 CSV 中，不进入上述层均值。

## 5. qlen crossover 与 Grouped MoE

- CPU AVX2 top-10 与 GPU resident top-10 均为真实 Qwen expert 的 Prototype measured。
- 把 24.576 MB packed INT4 bulk H2D 和实测 dequant 组件串行加入后，crossover 落在 **qlen 4–8**；通过跨 expert pipeline overlap，2–4 仍是可争取区间，属于 Projection。
- 真实路由中，active-expert bucket 的 p50 在 chunk 128 达到 4.33，chunk 256 达到 6.90，因此 128–256 已进入 crossover 区间。
- resident GPU 相对 CPU AVX2：q4 **4.18×**、q8 **8.02×**、q16 **15.30×**。这是组件级数据。
- 已有 Hybrid 单层 prototype 的 mixed output：relative RMSE=0.000557003、cosine=0.999999642、max-abs=3.69214e-05。
- Prefill v1 的完整 llama.cpp 提升来自现有 selected-expert CUDA 路径扩大 ubatch：2K 为 **1.424×**，16K 同程序为 **2.226×**。自定义 grouped streaming runtime 留到 Prefill v2。

## 6. VRAM

- 16K / ubatch 512：模型 4459.62 MiB，KV+recurrent 561 MiB，compute 1842.57 MiB；Prefill 窗口 NVML peak **8188 MiB**。
- 16K / ubatch 2048：compute 增至 2270.28 MiB，日志记账总量 **7290.9 MiB**。
- 按实测 runtime-overhead 外推，额外 4 GiB staging 后约 12712 MiB，仍留约 3663 MiB；额外 8 GiB 后约 16808 MiB，越过 16375 MiB 设备总量。Prefill v2 应从 4 GiB 左右 staging 起步。

## 7. 研发循环

| 模式 | 16K prefix/prefill | 500 decode | total | 标签 |
|---|---:|---:|---:|---|
| 继承 cold baseline | 141.507 s | 29.303 s | 170.810 s | Projection |
| 优化后 cold | 46.429 s | 29.303 s | 75.732 s | Projection |
| Warm Development，模型常驻 | 0.807 s | 36.816 s | **37.623 s** | Measured |
| Warm Development，新进程 | 7.521 s 到首 token | 36.816 s | **44.337 s** | Measured |

## 8. 十二个收口答案

1. **16K Prefix/KV 可保存并复用。**
2. **Warm restore：0.807 s；重复 cache load 本体 0.587 s。**
3. **Cold Prefill：继承基线 115.63 tok/s；Prefill v1 最终 352.43 tok/s。** 同程序因果增益为 +122.55%。
4. **主要瓶颈：selected expert 权重的碎片化 H2D、host copy API 与同步；CPU arithmetic 和 attention 都不是首要项。**
5. **qlen crossover：串行 staging 口径在 4–8；理想 overlap 口径 2–4 属于 Projection。**
6. **最佳 chunk/physical ubatch：已测范围内 2048。**
7. **GPU utilization：NVML mean 50.52%、peak 61%；Nsight 中 compute-kernel interval 仅占 wall 5.53%，大量时间在 copy。**
8. **H2D：49.91 MB/token，33.63 次 copy/token（2K / ubatch 512 实测）。**
9. **Grouped MoE：resident 组件在 q16 为 15.30×；完整自定义 grouped runtime 本轮未接入。完整 llama.cpp 通过扩大 ubatch 得到 2K 1.424×、16K 同程序 2.226×。**
10. **16K + 500 decode：约 171 s → 37.623 s（模型常驻实测），缩短约 77.97%。**
11. **Prefill v1 值得停止：是。** Gate B 与 Development Throughput Gate 已通过；500 tok/s 留给 v2。
12. **下一步：返回 Hybrid Decode Runtime。** Decode 默认使用 Warm Development Mode；最终产品报告再执行 Cold Start。

## 产物

- `cold_prefill_baseline.csv`
- `prefix_cache_results.json`
- `prefill_critical_path.json`
- `prefill_routing_occupancy.csv`
- `qlen_crossover.csv`
- `chunk_sweep.csv`
- `grouped_moe_benchmark.json`
- `vram_prefill_budget.csv`
- `development_iteration_time.csv`
- `optimization_ab.csv`
- 原始 Prefix、routing、CPU instrumentation 与 Nsight 文件位于各子目录。

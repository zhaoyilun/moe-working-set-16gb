# PC / CUDA Slotstream 结果总览

## 第三关：Integration Gate

第三关已完成。最终选择 **C：Phase-Adaptive Hybrid**。

- 五类真实 workload：5000 decode tokens、240000 个 routed-layer 决策、2400000 次 expert request。
- 8 GB LRU held-out hit rate：**70.535%**；0–4 miss 占 **76.949%**。
- 双池折中：**2 GB adaptive static + 6 GB dynamic**；若允许纯动态池，0+8 GB 的 hit/p50 更高。
- 真实 4 resident + 2 promotion + 4 CPU mixed layer：**1.1226 / 1.6247 ms routed-layer composite p50/p95**；mixed cosine **0.99999964**。
- 当前 prototype 校准的 48-layer routed-MoE projection：**26.921 / 57.391 ms/token**。
- 完整 decode projection：**6.072 tok/s p50、5.124 tok/s p95**；p50 相对带 callback 的 trace runtime 5.553 tok/s 有约 9.35% 空间，p95 仍需完整 decoder 验证。

完整证据与八个直接答案见 [`INTEGRATION_GATE_RESULTS.md`](INTEGRATION_GATE_RESULTS.md)。

---

## 第二关：CPU–GPU Hybrid microbenchmark

## 结论

选择 **C：混合路线**，但 CPU 不是“所有 miss 的默认去处”，而是 decode 阶段的大 miss 批次执行池：

1. **GPU 常驻池**：按真实频率固定最热专家。
2. **GPU 动态池**：CLOCK 作为基础淘汰，叠加频率窗口的升降级。
3. **CPU 冷专家池**：decode 且同层冷专家较多时直接算；本机稳健阈值从 `cold_count >= 8` 起，若只优化 p50 可放宽到 `>= 4`。
4. **RAM→VRAM 流式路径**：保留；冷专家只有 1–4 个时它更合算，也是 prefill 的主路径。

核心门槛已通过：真实形状、真实权重记录、DRAM 冷态下，top-10 CPU 专家整层计算为 **0.7809 ms p50 / 0.9149 ms p95**。加上 5,120 B 激活的 GPU→CPU→GPU 同步往返后为 **0.7970 / 0.9443 ms**，低于上一轮 top-10 的 **1.0402 ms 暴露 PCIe 时间**。

## 研究结论

### Fiddler

Fiddler 的基本闭环正是本次路径：dense/热专家留在 GPU；GPU miss 时传 activation 到 CPU，CPU 算专家，再传回合并后的输出。它针对小 batch 的关键判断是“传小 activation，而不是传大权重”。仓库实现仍以未量化 Mixtral 和 PyTorch CPU 为主，并明确提示缺少 AVX-512 时 CPU 路径偏慢。本实验用 KTransformers 的 AVX2 内核补上这一处。

- https://github.com/efeslab/fiddler
- https://arxiv.org/abs/2402.07033

### KTransformers / kt-kernel

当前代码已经具备本原型需要的三块成熟结构：

- `gpu_experts_mask`：同一 MoE 层里按专家选择 CPU 或 GPU；
- `frequency` 与 `dynamic-expert-update`：按历史分布初始化和动态更新 GPU 专家；
- AVX2 后端：`RAWINT4`、`GPTQ_INT4`、`MXFP4`，并在本机 CPUID 上识别到 AVX2、FMA、AVX-VNNI。

本机缺少 AVX-512/AMX，所以选 AVX2 RAWINT4 作为最接近 Slotstream 记录的成熟起点。

- https://github.com/kvcache-ai/ktransformers
- https://github.com/kvcache-ai/ktransformers/commit/9481291

### SlotWise / OSDI 2026

SlotWise 的结构分工与本次测量一致：decode 使用 CPU–GPU 协作，prefill 使用流式加载专家；公开结果来自双路 EPYC、AVX-512/FP8 和双 RTX 5090 等平台，其绝对吞吐不外推到 14900KF。这里采用它的阶段分工，不搬用它的数字。

- https://www.usenix.org/conference/osdi26/presentation/wang-wenxin
- https://arxiv.org/abs/2606.10493

### 最新 Slotstream measurements

当前检出的 Slotstream 为 `5bf0e67ac050f2d7407eb72e2cc52e818c9509dd`。它的真实 8,073-token 运行在 60 experts/layer 时报告 **0.593–0.600** 聚合命中率；60×48×2,764,800 B = **7.962624 GB**。源码淘汰策略是 CLOCK。

仓库的 `PLAN.md` 仍写着原始 per-token 路由 trace 待提交。因此当前有真实聚合 hit counter，却没有足以重放 W50/W90、reuse distance、转移概率和多策略 A/B 的完整 ID 序列。本文不从聚合计数反推这些量。

- https://github.com/carloslfu/slotstream/blob/main/MEASUREMENTS.md

## Slotstream 九件套与 KTransformers 格式

Qwen3.8-Flash-Next routed expert 的精确形状是 `hidden=2560`、`intermediate=640`、`group=64`、`topk=10`。每个专家按 gate/up/down 各 weight/scales/biases，共九件：

| 投影 | weight | scales | biases | 每专家字节 |
|---|---|---|---|---:|
| gate | U32 `[640,320]` | BF16 `[640,40]` | BF16 `[640,40]` | 921,600 |
| up | U32 `[640,320]` | BF16 `[640,40]` | BF16 `[640,40]` | 921,600 |
| down | U32 `[2560,80]` | BF16 `[2560,10]` | BF16 `[2560,10]` | 921,600 |
| 合计 | | | | **2,764,800** |

格式对照：

| 格式 | nibble/轴 | group 元数据 | 解码公式 | 与 MLX 九件套关系 |
|---|---|---|---|---|
| MLX affine INT4 | U32 只是 8 个 nibble 的容器；字节内 low nibble 对应偶数 K | group64，BF16 scale + BF16 bias | `q*s+b` | 原始格式 |
| KT RAWINT4 | `[E,N,K/2]` U8；字节内 low/hi 依次对应 `2j/2j+1` | group64，BF16 scale，加载后 FP32 | `(q-8)*s` | 字节排列相合，数学零点不同 |
| KT GPTQ_INT4 | `[K/8,N]` I32，沿 K 每字 8 个 nibble | `[K/group,N]` FP32 scale | 对称固定零点 8 | 需要转置/重排，也缺少独立 bias |
| KT MXFP4 | 每字节两个 E2M1 nibble | group32 FP32 scale | E2M1 查表后乘 scale | 数值编码和 group 均不同 |
| KT MOE_INT4 | 在线从 BF16/FP16 量化到通用内核布局 | 内核专属 scale/packing | 通用量化路径 | 当前 Intel AVX2 构建未选此后端 |

真实记录里的隐含零点 `-bias/scale` 范围约 **6.98–14.02**，并非固定 8。离线重数量化到对称 RAWINT4 的三投影相对 RMSE 为 **10.85%–11.44%**，所以运行主线没有采用该近似。

实际做法是最小修改 RAWINT4 AVX2 内核：保留原始 nibble，加载 BF16 scale/bias 为 FP32，在融合点积里按

```text
q*s+b = (q-8)*s + (8*s+b)
```

计算。CPU 内存中每专家为 2,457,600 B packed weight + 307,200 B FP32 scales + 307,200 B FP32 biases = **3,072,000 B**。

## 本机实验

### 输入与验证

- CPU：Intel Core i9-14900KF，24 physical / 32 logical；AVX2、FMA、AVX-VNNI；无 AVX-512/AMX。
- GPU：RTX 4080 SUPER 16 GB，CUDA 12.3。
- 权重：从固定 revision `e9d552f83de4665d243d5c9cf73201a1ca6c16d7` 的 safetensors 分片用 HTTP Range 取 layer 0 专家 0–9 的真实九件套，共 27,648,000 B；router 也是同一分片的真实 BF16 `[512,2560]`。
- 内核一致性：相对 L1 **0.002989**，相对 RMSE **0.003767**；参考为直接 FP32 `q*s+b` 重建和 FP32 GEMM，内核中间缓冲为 BF16。

### Decode：DRAM 冷态线程扫描

每次计时前读取 64 MiB cache-scrub buffer，使跨层访问更接近 48 层权重流，而不是让 10 个专家一直留在 LLC。

| workers | top-10 p50 ms | p95 ms | 说明 |
|---:|---:|---:|---|
| 1 | 4.9811 | 5.4065 | 单线程 |
| 2 | 2.6391 | 2.9924 | |
| 4 | 1.4898 | 1.6839 | |
| 8 | 1.0484 | 1.2457 | P-core 数量代理组 |
| 16 | 0.9169 | 1.0703 | |
| **24** | **0.7809** | **0.9149** | physical-count 代理组，最佳 |
| 32 | 32.1700 | 42.2744 | WSL 把拓扑呈现成 16 cores×2；kt-kernel 对 16 号之后的 core 绑定失败，形成严重过度调度 |

WSL 没有向 hwloc 暴露真实 P/E efficiency class，因此 8/24/32 分别只是 P-count、physical-count、HT-count 的代理组，不宣称为纯净的 P-only/P+E 隔离实验。当前运行配置应取 **24 workers**，关闭 32-worker 组。

24-worker 下，各冷专家数的最佳结果：

| CPU 冷专家数 | 最佳 workers | p50 ms | p95 ms | 名义流带宽 GB/s |
|---:|---:|---:|---:|---:|
| 1 | 8 | 0.2531 | 0.3490 | 12.14 |
| 2 | 16 | 0.3339 | 0.4663 | 18.40 |
| 4 | 24 | 0.4225 | 0.6006 | 29.08 |
| 8 | 24 | 0.7005 | 0.8260 | 35.08 |
| 10 | 24 | **0.7809** | **0.9149** | **39.34** |

真实 8 GB 聚合命中率约 0.60，对 top-10 意味着平均 4 个 miss：CPU+activation 的 p50 约 0.4386 ms，略低于 4 专家 H2D 的 0.4751 ms；但 p95 约 0.6300 ms，高于 H2D 的约 0.4927 ms。因此 `>=4` 只适合 p50 优先，默认用 `>=8` 才同时改善尾延迟。

top-10 组成：

- 调度/提交/同步空路径：0.0365 ms，约 **4.7%**；
- LLC warm 的融合专家内核：0.6172 ms；
- DRAM cold：0.7809 ms；cold-warm 差 0.1637 ms，约 **21.0%**；
- 剩余约 **74.4%** 是融合的 nibble 解码、FP32 FMA、SwiGLU、中间 BF16 和 weighted merge。

解量化与 GEMV 在同一个 AVX2 inner loop 中交错；单独报两个百分比会变成估算，所以保留为“融合内核”一项。

### 激活往返与真实 router

- 5,120 B BF16 activation D2H，同步，CPU 边界，5,120 B BF16 output H2D，再同步：**16.1 µs p50 / 29.4 µs p95**，5,000 次，pinned host memory。
- 真实 layer-0 BF16 router：cuBLAS GEMV → CUDA top-10 softmax → 仅回传 10 个 id/weight：**84.6 µs p50 / 229.5 µs p95**，3,000 次。
- 串行求和的完整 routed layer：**0.8816 ms p50 / 1.1738 ms p95**。router 对“CPU 还是 H2D”两条路径是共同成本；做选择时比较的是 CPU+activation 对 PCIe 暴露时间。

### Prefill crossover

CPU 为 exact affine AVX2；GPU 为同一真实 affine 权重一次解码到 FP16 后常驻，用 cuBLAS 跑完整 top-10 gate/up→SiLU→down→merge。GPU 流式冷权重的稳态还要与上一轮 1.1718 ms top-10 H2D pipeline 取较大者。

| qlen | CPU top-10 p50 ms | GPU resident p50 ms |
|---:|---:|---:|
| 256 | 118.056 | 0.609 |
| 512 | 237.080 | 0.793 |
| 1024 | 460.789 | 1.363 |
| 2048 | 1,016.765 | 2.551 |
| 4096 | 2,328.438 | 5.826 |

小 qlen 补点给出 crossover：qlen 1 的 CPU cold top-10 为 0.781 ms，qlen 2 为 1.092 ms，qlen 4 为 2.009 ms；GPU 流式稳态约 1.172 ms。因此本机交点在 **qlen 2–4**：decode 单 token/双 token 适合 CPU，大于等于 4 的批开始流式 GPU，256–4k prefill 全部走 GPU。

## 调度器

已实现 `scripts/trace_scheduler.py`，输入 JSONL：

```json
{"layer": 0, "experts": [134, 322, 439, 103, 339, 481, 500, 1, 72, 174], "weights": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], "qlen": 1}
```

它支持 CLOCK、LRU、LFU、static-hot、hybrid，输出 hit/miss、CPU 专家比例、流式比例、eviction、promotion、W50/W90、entropy、复用距离和相邻 top-k 重合率。`router_sample.jsonl` 用本机真实 layer-0 router 结果完成了运行冒烟；一个 token 只证明数据通路，不拿它做 locality 结论。

建议状态机：

```text
prefill or qlen >= 4
    -> GPU resident hit / RAM->VRAM stream

decode qlen <= 2
    -> static-hot GPU hit
    -> dynamic CLOCK hit
    -> miss_count 1..4: H2D
    -> miss_count 8..10: CPU exact affine AVX2
    -> miss_count 5..7: 用实时 p95 估计器二选一

window frequency rises repeatedly
    -> CPU/dynamic expert promoted to GPU dynamic pool
frequency decays + CLOCK second chance expires
    -> demote to CPU cold pool
```

## 四个直接答案

1. **真实 top-10 CPU 专家能否低于 1.0402 ms 暴露 PCIe？** 能。0.7809 ms；加激活往返后 0.7970 ms，余量 0.2432 ms。
2. **最大瓶颈在哪里？** top-10 时不是 activation，而是融合专家内核；其中 DRAM 冷态罚时约 21%，融合解量化/GEMV/SwiGLU/merge 约 74%，调度约 5%。
3. **热/动态/冷专家怎样分？** 静态频率头部固定 GPU，具有短期复用的中段进入 GPU CLOCK 动态池，decode 中大 miss 批次留 CPU，小 miss 批次仍 H2D；prefill 流式 GPU。
4. **8 GB VRAM 池在真实路由上的命中率？** 当前公开真实运行给出 **59.3%–60.0%** 聚合命中率；这是 8,073-token prefill 加 16-token decode 的运行计数，非 decode-only trace。

## 产物

- `results/summary.json`：最终机器可读结论。
- `results/cpu_cold_t*.json`：1/2/4/8/16/24/32 workers × top-k 1/2/4/8/10。
- `results/activation_roundtrip.json`：激活同步往返。
- `results/router_topk.json`：真实 router。
- `results/gpu_prefill.json`、`results/prefill_q*.json`：crossover。
- `results/affine_kernel_validation.json`：内核一致性。
- `results/rawint4_layer0_experts0_9/conversion_report.json`：对称重数量化误差。
- `scripts/trace_scheduler.py`：trace-driven 调度器。

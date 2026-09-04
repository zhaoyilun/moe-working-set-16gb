# PC Slotstream Integration Gate：真实 Routing Trace + 最小 Hybrid MoE Layer

## 结论

最终选择 **C：Phase-Adaptive Hybrid**。

这次真实数据把结论收紧成一句话：**低 miss 层保留 GPU cache / H2D，高 miss 层优先整层 CPU；三路径混合能力保留，但当前 4 resident + 2 promotion + 4 CPU 组合不应成为默认路径。**

原因不是概念判断，而是四组直接证据：

1. 五类真实 decode trace 中，8 GB LRU 的命中率为 **70.535%**，有 **76.949%** 的 layer-token 落在 0–4 miss 区间，GPU cache + 小批 H2D 有足够覆盖面。
2. 三路径混合层数值正确，mixed output 对 FP32 reference 的 cosine 为 **0.99999964**，relative RMSE 为 **0.0005570**。
3. 三路径混合层的完整分支时间为 **1.0380 / 1.3952 ms p50/p95**；加上既有 router 实测后为 **1.1226 / 1.6247 ms**。它比纯 RAM→VRAM→GPU 的 p50 更快，但比纯 CPU routed layer 更慢。
4. 真实 trace 逐层回放后，当前 prototype 校准的 48-layer routed-MoE 为 **26.921 / 57.391 ms/token p50/p95**；p50 显著优于全 CPU 的 42.317 ms，但 p95 略高于全 CPU 的约 56.34 ms。因此应按阶段、miss 数量和尾延迟目标动态选择，而不是固定全 GPU、全 CPU或每层强制三路混合。

## 数据分级

- **Measured**：真实模型 router trace、真实 Qwen affine INT4 权重、真实 hidden、真实 KTransformers AVX2 CPU kernel、真实 CUDA H2D/GPU expert/merge、80 次单层 timeline、2/4/8 层 replay。
- **Simulated from real trace**：真实 48 层逐 token 路由上的 cache 状态、miss、promotion、eviction 和 CPU/H2D 决策；每个 workload 前 20% token 用于 hot-set calibration，后 80% 用于统计。
- **Projected**：48-layer routed-MoE latency，以及把它与当前 instrumented decoder trunk proxy 合并后的完整 decode tok/s。

## 真实 trace 范围

模型：`unsloth/Qwen3.8-Flash-Next-GGUF`，revision `38bb39ee97821de2c9009abb7e93950eec396e66`。运行时：llama.cpp `b10666` / commit `4e97ac86ebe2c4cb8212d98d2641ad6768810896`。

| workload | prompt tokens | decode tokens | 相邻 token top-10 平均交集 | W50 | W90 | W95 | reuse token distance p50/p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 中文技术 | 155 | 1000 | 3.433 | 1975 | 7974 | 10073 | 3 / 92 |
| 英文技术 | 138 | 1000 | 3.593 | 1933 | 7519 | 9333 | 3 / 96 |
| 代码 | 173 | 1000 | 3.377 | 1691 | 7262 | 9255 | 3 / 106 |
| 数学逻辑 | 209 | 1000 | 3.185 | 1737 | 7750 | 9920 | 4 / 98 |
| 长上下文 | 450 | 1000 | 3.195 | 2023 | 7416 | 9295 | 4 / 97 |

合计：**5000 decode tokens、240000 个 routed-layer 决策、2400000 次 expert request**。每个 decode token 都有 48 层完整 top-10；最大 routing weight sum error 为 `2.384e-7`。

跨 workload 的 frequency cosine 均值为 **0.741**，top-10% hot-page Jaccard 均值只有 **0.317**。长期热点有公共部分，但任务相关漂移很明显，因此 adaptive hot set 有意义。

## 八个核心答案

### 1. 8 GB expert pool 的真实 hit rate

后 4000 个 held-out decode token 上，**LRU = 70.535%**，对应 2893 个 `(layer, expert)` slot。分 workload 为：

- 中文技术：72.738%
- 英文技术：72.167%
- 代码：71.609%
- 数学逻辑：69.279%
- 长上下文：66.881%

这比上一轮约 59.3%–60.0% 的聚合值高，但两者统计口径不同：本轮是 decode-only、逐 token、20% calibration 后的 replay。

### 2. 每层 top-10 的 0–10 miss 分布

统计口径：8 GB LRU，五类 workload 合并，192000 个 held-out layer-token。

| miss | count | percent |
|---:|---:|---:|
| 0 | 25572 | 13.319% |
| 1 | 36515 | 19.018% |
| 2 | 35471 | 18.474% |
| 3 | 28740 | 14.969% |
| 4 | 21444 | 11.169% |
| 5 | 15355 | 7.997% |
| 6 | 10796 | 5.623% |
| 7 | 7568 | 3.942% |
| 8 | 5125 | 2.669% |
| 9 | 3388 | 1.765% |
| 10 | 2026 | 1.055% |

调度区间：**0–4 miss = 76.949%**，**5–7 = 17.562%**，**8–10 = 5.489%**。

### 3. Static Hot + Dynamic Pool 最佳分区

- 若允许纯动态池，p50 与 hit rate 的最佳点都是 **0 GB static + 8 GB dynamic LRU/Hybrid**：hit **70.535%**，cost-model p50 **25.145 ms/token**。
- 若要求 static 和 dynamic 两池都存在，综合吞吐选 **2 GB static + 6 GB dynamic，adaptive Hybrid**：hit **68.453%**，p50/p95 **25.486 / 46.729 ms**。
- 只优化 p95 时，**4 GB + 4 GB** 为 **46.629 ms**，仅比 2+6 低 0.100 ms，同时 p50 升到 25.891 ms、hit 降到 67.065%。

因此本轮实现参数选 **2+6**；它是双池约束下的折中点。真实数据也说明“8 GB 必须划出很大的永久 static 区”这个前提并不成立。

### 4. 单个真实 Hybrid routed layer 的 p50/p95

实际组合：4 个 GPU resident、2 个 RAM→VRAM promotion 后 GPU 执行、4 个 KTransformers AVX2 CPU direct。

- 三路径执行 + weighted merge 的 host span：**1.0380 / 1.3952 ms**。
- 加既有真实 router 0.0846 / 0.2295 ms：**1.1226 / 1.6247 ms routed-layer composite**。

router 是前一轮的独立实测，本轮没有把它伪装成同一个计时区间。

正确性：

| path | max abs | relative RMSE | cosine |
|---|---:|---:|---:|
| GPU partial | 2.889e-5 | 0.0002544 | 1.0000000 |
| CPU partial | 4.894e-5 | 0.0035252 | 0.9999942 |
| mixed final | 3.692e-5 | 0.0005570 | 0.9999996 |

### 5. GPU hot、CPU cold、H2D promotion 重叠量

80 次 timeline 的 p50：

- GPU hot 与 H2D：**0.3701 ms**，H2D 在 resident compute 内完全隐藏，resident 结束后的 DMA 暴露为 **0.0000 ms**。
- CPU 与 GPU critical span：**0.1895 ms**。
- GPU hot 与 CPU：**0.0763 ms**。
- CPU 与 H2D：**0.0000 ms**；三路同时 overlap 也是 **0.0000 ms**。
- GPU 完成后等待 CPU output 进入 merge：**0.2396 ms**。
- merge 与 host finalize：**0.1886 ms**。

p50-nearest timeline：

```text
activation D2H          0.0 ->   24.5 us
GPU resident            0.0 ->  461.8 us
H2D promotion           0.0 ->  365.5 us
GPU dequant           365.5 ->  438.3 us
GPU promoted          461.8 ->  562.2 us
CPU cold              437.2 ->  869.3 us
CPU return + merge    890.9 ->  939.2 us
host total              0.0 -> 1042.0 us
```

关键路径是：**host 提交 GPU 工作延迟了 CPU 启动 → CPU cold branch 晚于 GPU 完成 → CPU return/merge 收尾**。一次额外的 split-dispatch 尝试把 CPU/GPU overlap 提高到 0.376 ms，却把 host p50 增加到 1.134 ms；所以“重叠更多”本身不是目标，critical path 才是目标。

### 6. 与纯 RAM→VRAM→GPU 相比

按共同的 expert-path 口径：

- 纯 streaming：1.1718 ms p50。
- 当前 Hybrid：1.0380 ms p50。
- latency 减少 **0.1338 ms / 11.42%**，等价 throughput speedup **12.89%**。

加共同 router 后，1.2564 ms 对 1.1226 ms，latency 减少约 **10.65%**。p95 方向相反：当前三路径 composite 1.6247 ms，高于 streaming 参考约 1.4084 ms，尾延迟增加约 15.4%。

### 7. 与纯 CPU cold 相比

纯 CPU routed layer 为 **0.8816 / 1.1738 ms**；当前三路径 composite 为 **1.1226 / 1.6247 ms**。

当前固定三路径组合的 p50 **慢 27.34%**，p95 **慢 38.42%**。所以 5–7 miss 区间不应机械拆成三路；cost model 应同时比较“保持 GPU hit 并混合”和“整层 top-10 CPU”两个候选时间。

### 8. 48-layer routed-MoE 与完整 decode projection

选择 2 GB adaptive static + 6 GB dynamic Hybrid，真实 trace 逐层回放：

| 口径 | routed-MoE p50 | routed-MoE p95 |
|---|---:|---:|
| 理想 overlap cost model | 25.459 ms/token | 46.731 ms/token |
| 当前 prototype timeline 校准 | **26.921 ms/token** | **57.391 ms/token** |
| 48×纯 CPU routed-layer 参考 | 42.317 ms/token | 56.340 ms/token |

这里逐 token 累加每层真实 miss 与调度决策，然后在 token 维度取 percentile；不是 `48 × 平均 layer latency`。

完整 decoder 使用一个明确标注的 trunk proxy：五份带同步 callback 的 llama.cpp trace runtime 中位数 180.09 ms/token，减去 `48 × 0.8816115 ms` 的纯 CPU routed-MoE，得到 **137.773 ms/token**。callback 开销仍留在 proxy 中，因此这是偏保守的 projection。

| 完整 decode projection | p50 | p95 |
|---|---:|---:|
| 理想 overlap | 163.232 ms / **6.126 tok/s** | 184.504 ms / **5.420 tok/s** |
| 当前 prototype 校准 | 164.694 ms / **6.072 tok/s** | 195.164 ms / **5.124 tok/s** |
| 当前 instrumented trace runtime | 180.09 ms / **5.553 tok/s** | — |

含义：p50 有约 9.35% 的 projected throughput 空间，p95 仍是短板。这个结果支持进入完整 decoder 验证，但它仍属于 projection，不是完整 Hybrid decoder 的实测吞吐。

## 多层 replay

| depth | p50 total | p95 total | p50 per layer |
|---:|---:|---:|---:|
| 2 | 2.121 ms | 2.798 ms | 1.061 ms |
| 4 | 4.334 ms | 4.775 ms | 1.084 ms |
| 8 | 8.799 ms | 9.434 ms | 1.100 ms |

从 2 到 8 层，p50 per-layer 增加约 3.7%。短 replay 没有出现 allocator 抖动或连续恶化；48 层真实 decoder 仍应作为下一关的实测对象。

## 最终调度规则

```text
prefill / qlen >= 4
    -> GPU resident + RAM→VRAM streaming

decode
    -> 8 GB cache lookup
    -> miss <= 6: GPU hit + selective H2D
    -> miss >= 7: 比较 mixed critical path 与 whole-layer CPU
    -> 当前三路径实测成本下，5–7 miss 默认候选为 whole-layer CPU
    -> promotion 只依据历史频率、reuse distance 与 eviction cost
```

不存在 exact next-layer prefetch；所有 promotion 都只使用当前及历史信息。

## 最终决策

**C：Phase-Adaptive Hybrid。**

- A 的纯分层 Slotstream 会放弃高 miss 时 CPU 的确定性优势。
- B 的固定 CPU–GPU Hybrid 会把当前三路径尾延迟问题扩散到每层。
- C 能利用 76.949% 的低 miss 层，同时在三路径混合比整层 CPU 更慢时切回 CPU，并继续让 prefill 走 GPU streaming。
- D 与本轮 p50 trace projection 不一致：真实 replay 仍给出约 9.35% 的完整 decode p50 吞吐空间；是否兑现由下一关完整 decoder 实测决定。

## 可追溯产物

- `results/integration-gate/traces/*.bin`：五份原始 binary routing trace。
- `results/integration-gate/routing_trace_summary.json`：frequency、entropy、adjacent overlap、2–128 window locality、reuse distance、W50–W99、per-layer 与跨 workload 比较。
- `results/integration-gate/cache_policy_comparison.csv` / `.json`：510 组 cache policy 结果。
- `results/integration-gate/hybrid_reference_cpu.json`：真实 router 与 FP32/CPU reference。
- `results/integration-gate/hybrid_layer_benchmark.json`：正确性、80 次单层计时、2/4/8 层 replay、raw timeline。
- `results/integration-gate/hybrid_layer_benchmark_split_dispatch.json`：高 overlap 但更慢的 dispatch 对照。
- `results/integration-gate/timeline/`：timeline CSV、摘要和 p50-nearest 文本图。
- `results/integration-gate/end_to_end_projection.json`：Measured / Simulated / Projected 分级与完整 projection。
- `results/integration-gate/integration_gate_summary.json`：八个答案的机器可读汇总。

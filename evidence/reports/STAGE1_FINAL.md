# Hybrid Decode Runtime — Stage 1 Final

## 结论

**已确认事实（Measured）**：固定 8 GiB GPU Expert Working Set 在 canonical 16K 会话、100 token warmup、500 token 测量、三轮独立运行下达到 **18.753 tok/s**，比同路径 Native expert-only CPU baseline 的 **15.135 tok/s** 提高 **23.91%**，比历史 Golden Native **17.063 tok/s** 提高 **9.91%**。

这证明了完整 decoder 内的主机制：

> GPU cache hit → GPU；cache miss → Native CPU expert

不是单独 kernel 的推演，而是端到端 decode 实测。

## 500-token 稳定性

| 数据标签 | 配置 | Graph | 轮数 | tok/s 均值 | 范围 | p50 | p95 |
|---|---|---:|---:|---:|---:|---:|---:|
| Measured | Native expert-only CPU | ON | 3 | 15.135 | 15.081–15.182 | 63.46 ms | 79.84 ms |
| Measured | Hybrid 4 GiB | ON | 3 | 16.096 | 16.039–16.181 | 61.06 ms | 74.32 ms |
| Measured | Hybrid 6 GiB | ON | 3 | 17.275 | 17.235–17.326 | 56.97 ms | 72.11 ms |
| Measured | Hybrid 8 GiB | ON | 3 | **18.753** | **18.748–18.761** | **52.34 ms** | **67.77 ms** |
| Measured | Hybrid 8 GiB | OFF | 1 | 13.567 | single run | 73.22 ms | 89.29 ms |

Graph ON 相对 OFF 提高 **38.23%**。动态调度接入必须保留图复用；这不是次要优化，而是运行时成立的前提之一。

## 显存预算

16K、8 GiB pool 的资源观测峰值为 **15,990 MiB / 16,376 MiB**，最小观测余量 **386 MiB**。采样进程显著扰动该轮速度，因此该轮只用于资源峰值，性能结论取上表的无采样三轮。

| 数据标签 | 项目 | MiB |
|---|---|---:|
| Measured | GPU model buffer（含固定模型、专家池映射及分配填充） | 13,008.91 |
| Exact allocation | packed expert weights | 8,191.33 |
| Measured | attention KV | 408.00 |
| Measured | indexer KV | 153.00 |
| Measured | recurrent state | 112.57 |
| Measured | CUDA compute buffer | 1,842.57 |
| Measured | CUDA output buffer | 0.95 |
| Measured | observed process peak | **15,990** |

各项来自不同分配器口径，不能直接用逐项和替代设备峰值；决策采用设备采样峰值。8 GiB 已位于 16 GiB 卡的紧边界，第二阶段只在现有池内置换，不额外叠加数 GiB staging pool。

## Cache / routing 证据边界

### Measured

- 4 GiB 短窗口运行时计数：hit rate **26.06%**，CPU fallback **48 layers/token**。
- 正确性窗口：首 token 相同，前 32 token 序列相同；首 logits cosine **0.999239**。
- 精确 router ID 与 routing weight 未逐项相等；量化专家在 CPU 与 CUDA kernel 上的数值路径差异会沿 hidden state 累积。该事实保留在正确性结论里。

### Trace simulation from Measured routing IDs

现有 callback 会导致 CUDA Graph 反复 warmup，因此 8 GiB 的 500-token 计数没有作为运行时实测值写入。用 canonical 会话的已测 routing IDs 回放固定 plan，34-token 窗口得到：

- 8 GiB hit rate：**44.32%**
- CPU fallback：**47.09 layers/token**
- CPU fallback：**267.26 experts/token**

这组数字用于理解该 canonical 短窗口，不与五 workload 的 5,000-token held-out trace 混写。

### Trace simulation

五 workload、5,000 decode tokens 的既有 8 GiB held-out trace hit rate 为 **70.535%**；miss 0–4 / 5–7 / 8–10 分别为 **76.949% / 17.562% / 5.489%**。

## Stage 1 冻结判断

Stage 1 的端到端性能门禁通过：固定 GPU expert working set 有稳定、可复现的真实收益。下一步只验证一个新增机制：**在 8 GiB 总池不变的前提下，使用相邻 token 同层路由作为预测，对小 miss-set 做 selective H2D + dynamic promotion；大 miss-set 保持 Native CPU。**

## 证据文件

- `results/hybrid-decode-runtime/native_vs_hybrid.csv`
- `results/hybrid-decode-runtime/vram_budget.csv`
- `results/hybrid-decode-runtime/cache_runtime_stats.json`
- `results/hybrid-decode-runtime/correctness.json`


# Prefill Acceleration Gate v2

## 结论

**500+ 基本门禁通过。** 现有 llama.cpp selected-expert CUDA 路径通过增大 physical ubatch，在 16K 同 prompt 三轮达到：

- **523.859 / 526.970 / 521.179 tok/s**
- 均值 **524.003 tok/s**
- 范围 521.179–526.970 tok/s
- CV 0.45%
- 平均 Cold Prefill 约 **31.23 秒**

本轮没有引入自定义 grouped kernel。8K ubatch 已经把现有成熟路径推过 500；1000+ 的下一步才需要正式的 expert bucketing、bulk staging 与 grouped GPU execution。

## 严格同 prompt A/B

历史 Prefill v1 的 352.433 tok/s 使用另一条 prompt seed，首 token 为 248046；本轮 long-context seed 的首 token为 321。两条都是 Measured，但因果增益采用下面的同 prompt series。

| 数据标签 | Context | physical ubatch | 秒 | tok/s | vs 2K control |
|---|---:|---:|---:|---:|---:|
| Measured | 16K / 16,363 tokens | 2,048 | 56.937 | 287.387 | control |
| Measured | 16K / 16,363 tokens | 4,096 | 39.796 | 411.174 | +43.07% |
| Measured | 16K / 16,363 tokens | 8,192 | 31.235 | 523.859 | +82.28% |
| Measured | 16K / 16,363 tokens | 8,192 | 31.051 | 526.970 | +83.37% |
| Measured | 16K / 16,363 tokens | 8,192 | 31.396 | 521.179 | +81.35% |

三轮均值相对同 prompt 2K control 提高 **82.33%**。与历史 v1 headline 相比高 48.68%，该百分比只描述两组实测数值差，不作为单变量 A/B。

## 32K Context Check

| 数据标签 | actual prompt tokens | physical ubatch | 秒 | tok/s |
|---|---:|---:|---:|---:|
| Measured, single run | 32,747 | 8,192 | **53.045** | **617.342** |

32K 高于 16K tok/s，主要说明大 prompt 对固定调度成本的摊销更充分；这是单轮 context check，正式稳定 headline 仍采用 16K 三轮均值 524.003 tok/s。

## 数值边界

同一 16K prompt 的 Prefill 末端 logits 与 32-token greedy continuation：

| 数据标签 | A vs B | logits cosine | RMSE | max abs | argmax | common greedy prefix | 32-token sequence |
|---|---|---:|---:|---:|---|---:|---|
| Measured | ubatch 2K vs 4K | 0.996569 | 0.233968 | 1.63104 | 321 = 321 | 6 | different |
| Measured | ubatch 2K vs 8K | 0.990457 | 0.401766 | 3.32212 | 321 = 321 | 6 | different |
| Measured | ubatch 4K vs 8K | 0.992680 | 0.336871 | 3.54281 | 321 = 321 | **32** | **equal** |

当前量化路径对 batch partition 存在数值归约差异。2K、4K、8K 的首 token argmax 相同；4K 与 8K 的后续 32 个 greedy token 全部一致；2K 与大 ubatch 在第 7 个 token 分叉。由此得到两个独立判断：

1. **性能门禁**：8K ubatch 稳定通过 500+；
2. **数值验收**：已有单 prompt 证据支持首 token 与 4K/8K continuation 稳定，面向默认产品配置仍需多 prompt eval，而不是宣称跨 ubatch bit-exact。

## 显存

| 数据标签 | Context | ubatch | model | KV | recurrent | CUDA compute | allocator-accounted | allocator headroom |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Measured allocator logs | 16K | 2K | 4,459.62 MiB | 561 MiB | 112.57 MiB | 2,270.28 MiB | 7,404.42 MiB | 8,971.58 MiB |
| Measured allocator logs | 16K | 4K | 4,459.62 MiB | 561 MiB | 112.57 MiB | 4,080 MiB | 9,214.14 MiB | 7,161.86 MiB |
| Measured allocator logs | 16K | 8K | 4,459.62 MiB | 561 MiB | 112.57 MiB | 8,160 MiB | 13,294.14 MiB | 3,081.86 MiB |
| Measured allocator logs | 32K | 8K | 4,459.62 MiB | 1,089 MiB | 112.57 MiB | 8,160 MiB | 13,822.14 MiB | 2,553.86 MiB |

本轮未进行新一轮 NVML 峰值轮询，表中余量来自 llama.cpp allocator 日志。12K physical ubatch 按当前近线性 workspace 增长会把 CUDA compute 推到约 12 GiB，叠加 model/KV 后越过 16,376 MiB 设备预算，因此当前容量 sweep 在 8K 收口。

## 为什么大 ubatch 有效

现有真实 Prefill trace 已经表明：chunk 从 512 增至 1024 时，每层 active experts 只从约 235 增到 292，而每个 active expert 的中位 token 复用从 11.02 增到 15.32。更大的 physical ubatch 把更多 token 聚进同一 selected-expert 执行窗口，摊薄重复传权重、host copy API 与同步成本。

已有 v1 Nsight 证据仍适用来解释方向：2K / ubatch 512 的 H2D 为 49.91 MB/token、33.63 copies/token，GPU kernels 仅占 wall interval 的 5.53%。v2 没有新增 timeline，因此不把该旧窗口的精确比例当作 8K ubatch 当前值。

## Gate 判断

- **工程验证 200–300+**：已超过；
- **最低实用改善 500+**：已通过，16K 三轮均值 524.003；
- **主要目标 1000+**：留给 grouped MoE runtime；
- **优秀 2000+**：仍是远期目标。

下一轮 1000+ 的最小结构保持为：

`prompt chunk → router → expert buckets → unique experts → contiguous/bulk staging → grouped GPU GEMM → scatter/weighted merge`

当前量化格式、scale/bias semantics 与 offline repack 需要先对齐成熟 CUDA/CUTLASS实现。运行时逐次格式转换不进入方案。

## 证据文件

- `ubatch_sweep.csv`
- `vram_prefill_v2.csv`
- `correctness-comparison.json`
- `correctness-comparison-2048-vs-4096.json`
- `correctness-comparison-4096-vs-8192.json`
- `ubatch-*/result.json`
- `correctness-ubatch-*/result.json`
- `32k-ubatch-8192/result.json`

# PC / CUDA Slotstream 第一轮实测报告

日期：2026-09-03  
目标：先验证 `RAM -> PCIe -> VRAM -> expert GEMM`，再判断是否进入完整 runtime。

## 结论

这轮得到的是一个清晰的“部分通过”结果：

1. **容量与基本链路通过。** 1/2/4/6/8 GB VRAM pool 均成功分配和写入；
   pageable、pinned、单段和九段 copy 路径均实际运行，最终源码加入了回读逐字节校验。
2. **Pinned H2D 达到可用但不是无限快的水平。** 三轮无探针干扰的中位结果是
   **23.46 GB/s**；单个 2,764,800 B expert 的 p50 是 **0.1261 ms**。
3. **低命中率路径仍明显受 PCIe 限制。** top-10 全 miss 时，每层搬运
   27.648 MB，耗时 **1.1697 ms**；等 FLOPs GEMM 只有 **0.1339 ms**。
4. **双流主要隐藏了计算，而没有隐藏大部分传输。** overlap 后总时间
   **1.1718 ms/layer**，其中仍有约 **1.0402 ms/layer** 是超出计算时间的传输，
   仅隐藏 **11.1%** 的 copy 时间。
5. **值得继续做下一道验证门，但还不应开始完整模型移植。** 下一步应先得到
   4/6/8 GB pool 下的真实 routing trace 命中率，并接入真实 INT4 expert kernel。
   如果命中率长期明显低于约 90%，这台机器的 decode 很可能仍以 PCIe 为主；
   如果能接近或超过 90%，PCIe 才有机会退出 routed-MoE 主瓶颈。

## 三个最关键的真实数字

| 问题 | 本机结果 | 测量边界 |
|---|---:|---|
| 单个真实大小 expert，pinned RAM -> VRAM | p50 **0.1261 ms**；p95 **0.1302 ms** | 2,764,800 B synthetic payload，字节量与真实记录一致 |
| 连续多 expert H2D | **23.46 GB/s** | batches 8/10/20、pool 1–8 GB、三轮结果中位数 |
| top-10 H2D 与等 FLOPs GEMM 重叠后 | 总计 **1.1718 ms/layer**；暴露 copy **1.0402 ms/layer** | 已知下一批 ID 的理想双流上界；GEMM 为 FP16 surrogate |

## 环境

- Windows 11 专业工作站版，10.0.26200
- Intel Core i9-14900KF，96 GB DDR5
- NVIDIA GeForce RTX 4080 SUPER，16,376 MiB
- NVIDIA driver 610.88
- CUDA driver API 13.3；本地 CUDA Toolkit/runtime 12.3
- MSVC 19.44.35222；CUDA architecture `sm_89`
- 负载时 PCIe 实际升到 **Gen4 x16**
- 上游源码：`carloslfu/slotstream` commit
  `5bf0e67ac050f2d7407eb72e2cc52e818c9509dd`

## “语义分页”在这个原型中的准确含义

普通虚拟内存按固定字节页分页；Slotstream 按模型结构分页：

```text
页号       = (layer_id, expert_id)
页内容     = 该 routed expert 的 9 个量化 tensor piece
页大小     = 2,764,800 B
页表       = expert key -> VRAM slot
页命中     = 直接执行 expert kernel
缺页       = RAM/SSD 读取 -> slot -> 执行
淘汰       = 当前上游实现采用 CLOCK
提前取页   = router 已经给出的 expert IDs
```

它之所以叫“语义”分页，是因为分页单位不是任意 4 KiB，而是对模型计算有完整含义的
expert。PC 版再多一层：SSD 是冷仓库，96 GB RAM 是大容量 expert store，16 GB VRAM
是热 slot pool。

## 源码核验到的模型事实

- 48 层；每层 512 routed experts；每 token 每层激活 top-10 routed experts。
- hidden size 2560；expert intermediate size 640。
- expert 为 4-bit、group size 64。
- 每个 expert 包含 gate/up/down 三组 packed weight、scale、bias，共九段。
- 每段字节数为：`819200, 51200, 51200`，三组重复，总计 2,764,800 B。
- 全部 routed experts 约 67.95 GB。
- slot pool 是跨 48 层共享的全局 pool，key 是 `(layer, expert)`。
- decode miss 会先选 victim，pin 住本层工作集，再批量读入并写入 pool。
- 精确的下一层 expert ID 依赖当前层输出；因此“layer N 计算时精确预取
  layer N+1”不是免费成立的条件。

## H2D 结果

三轮无持续 `nvidia-smi` 轮询的聚合结果：

| Host/layout | 大 batch 中位带宽 | 含义 |
|---|---:|---|
| pinned + contiguous record | **23.46 GB/s** | PC runtime 在 RAM 中按 GPU 消费布局重排后的主路径 |
| pinned + nine pieces per expert | **20.60 GB/s** | 直接保留九段形状并逐段提交，约低 12% |
| pageable + contiguous record | **14.28 GB/s** | 性能和尾延迟都更差 |

VRAM pool 从 1 GB 增到 8 GB 没有改变链路带宽。这符合机制：pool 容量改变的是
命中率，不是 PCIe 的单次传输速度。

`nine_pieces_per_expert` 测的是每个 expert 拆成九次 CUDA copy 的提交开销；它尚未
包含从九个大型 host tensor 按 expert 做 CPU gather，也未实现真实九个 VRAM tensor
之间的 scatter。因此 20.60 GB/s 仍是“分段 DMA”结果，不是完整换入路径的总成本。

NVIDIA 自带 `bandwidthTest` 对 10/20 expert 大小的独立核验得到：

- CPU timing：21.31–21.57 GB/s；
- CUDA event timing：约 23.04–23.07 GB/s。

它与三轮主基准的 23.46 GB/s 同一量级。另有一轮在每 200 ms 调用
`nvidia-smi` 的情况下只得到 16.84 GB/s；该轮保留在原始结果中，但没有参与主聚合。
这说明频繁驱动探针会干扰这类短周期传输测试。

## Copy / compute overlap

4 GB pool、top-10、每轮 80 个 pipeline cycles、三轮中位数：

| 模式 | ms/layer |
|---|---:|
| copy only | **1.1697** |
| 三个 FP16 等 FLOPs GEMM | **0.1339** |
| copy -> compute 串行 | **1.3871** |
| 双缓冲但仍串行 | **1.3867** |
| 双 CUDA stream + event | **1.1718** |
| overlap p95 | **1.1742** |
| 仍暴露的 copy | **1.0402** |

核心解释：`copy >> compute`，所以 overlap 以后几乎所有 surrogate compute 都藏进了
copy，但 copy 本身只被遮住约 11%。这与“85% 传输可隐藏”的乐观假设方向相反。

## Cache hit rate 对 PCIe 的影响

使用实测 23.457 GB/s：

```text
H2D_bytes/token = 2,764,800 * 10 * 48 * (1 - h)
                = 1.327104 GB * (1 - h)
```

| 命中率 h | H2D GB/token | 原始 H2D ms/token | H2D-only 上限 tok/s | 理想连续 overlap 后仍暴露的 copy ms |
|---:|---:|---:|---:|---:|
| 0% | 1.327 | 56.58 | 17.68 | 50.15 |
| 25% | 0.995 | 42.43 | 23.57 | 36.01 |
| 50% | 0.664 | 28.29 | 35.35 | 21.86 |
| 75% | 0.332 | 14.14 | 70.70 | 7.72 |
| 90% | 0.133 | 5.66 | 176.75 | 0.00* |
| 95% | 0.066 | 2.83 | 353.51 | 0.00* |

`*` 这里是把 48 层总传输当成连续流，并允许与 6.43 ms/token 的 surrogate MoE
计算任意重叠的数学上界。真实 decoder 受逐层 router 依赖、离散 miss 数和其他 kernel
排序限制，不应直接把它当成最终速度。

H2D-only 上限也不是模型 token/s；attention、router、shared expert、KV cache、
量化/反量化和同步仍会继续增加总时间。

## 已确认事实 / 推断 / 尚待测量

### 已确认事实

- 当前机器能建立并写入 8 GB VRAM pool。
- pinned、连续布局可稳定达到约 23.46 GB/s。
- top-10 全 miss 的 27.648 MB 搬运约为 1.17 ms/layer。
- 在当前 surrogate 下，双流只隐藏约 11% copy。

### 合理推断

- PC runtime 应在模型加载时把 routed experts 整理成连续 RAM record，或者至少按
  piece 批处理；直接提交九段小 copy 会损失约 12% 带宽。
- 4–8 GB pool 真正能否成立，主要由真实路由局部性决定，而不是由 pool 写入速度决定。
- 90% 左右是值得重点观察的命中率区域：在理想连续 overlap 模型里，PCIe 与当前
  surrogate MoE compute 在这里发生主导权切换。

### 尚待测量

- 4/6/8 GB global pool 对真实 decode traces 的 hit rate。
- CUDA 上真实 INT4 group-64 expert kernel 的执行时间和数值正确性。
- streamed INT4 bytes 被真实 kernel 消费时的 copy/compute overlap。
- dense trunk、shared expert、attention、KV cache 同时占用显存后，实际能留给 pool
  的容量。
- Windows WDDM 与 Linux CUDA 的差异。

## 下一步主线

1. 从真实模型采集 router trace，只做 4/6/8 GB global CLOCK/LRU replay，得到真实
   `hit_rate -> H2D bytes/token`。
2. 接入一个真实 INT4 group-64 expert GEMV/GEMM kernel，让刚搬入的 2.7648 MB
   record 被实际计算消费。
3. 重测 top-10 miss 的串行与双流路径。
4. 只有当真实 trace 与真实 kernel 的组合仍有合理 token/s，再进入单层真实模型；
   仍不直接做完整 decode。

## 原始结果索引

- `results/full_round2/`、`full_round3/`、`full_round4/`：三轮主结果。
- `results/aggregate/aggregate_copy_results.csv`：按测试单元取三轮中位数。
- `results/aggregate/aggregate_overlap_results.csv`：重叠结果三轮中位数。
- `results/aggregate/summary.json`：关键数字与 cache-hit 推导。
- `results/full/`：带高频 `nvidia-smi` 探针的扰动轮，作为测量方法警示保留。
- `results/source_validation/`：最终源码重新编译后的 copy 回读校验与烟雾测试。

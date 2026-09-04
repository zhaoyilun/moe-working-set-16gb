# Prefill Grouped Runtime — 代码理解笔记

日期：2026-09-04。checkpoint `2d4f3154a2d93c3a4d6d4a415c404f1b397d8dcb`（base `4e97ac86`，补丁 14 文件 +702 行）。

## 一、现有补丁的结构（Decode 侧）

### 1. 模型加载（`llama-model.cpp`）
- `llama_model_params.hybrid_expert_cache_plan`（CSV：`layer,expert` 行）→ `load_hybrid_expert_cache_plan()` 每层得到常驻 expert 列表。
- 每层经 `ml.create_extra_tensor()` 在 GPU buffer 上建 6 个张量：
  - `ffn_{gate,up,down}_exps_cache`：形状 `[ne0, ne1, slots]` 的 packed 权重（只含 plan 内 expert，按 slot 排列）；
  - `ffn_exps_gpu_map / cpu_map / hit_mask`：512 项的 I32/I32/F32 映射表。
- 加载后逐 slot `tensor_get(source) → tensor_set(cache)` 填充（staging buffer 中转）。
- `qwen35moe.cpp / qwen4exp.cpp` 的 `build_layer_ffn` 把这 6 个张量传入 `build_moe_ffn`。

### 2. 图构建（`llama-graph.cpp::build_moe_ffn`，hybrid 分支）
```
selected_experts [10, n_tokens]                     ← router topk（既有逻辑）
  gpu_ids  = gpu_map[selected]   （命中→slot id；miss→0，由 hit_mask 掩掉）
  cpu_ids  = cpu_map[selected]   （miss→原 expert id；命中→-1）
  hit_mask = hit_mask[selected]  [1, 10, n_tokens]
  cpu_out  = mul_mat_id(cpu_{up,gate}_exps, cur, cpu_ids) → swiglu → mul_mat_id(cpu_down, ·, cpu_ids)
  gpu_out  = mul_mat_id(cache_{up,gate}, cur, gpu_ids) → swiglu → mul_mat_id(cache_down, ·, gpu_ids)
  gpu_out *= hit_mask
  experts  = cpu_out + gpu_out
```
- 负 id 语义：CPU 端 `ggml_compute_forward_mul_mat_id`（`ggml-cpu.c`）与 repack 模板（`repack.cpp`）在发现负 id 时先把输出 lane 清零并跳过这些行 → CPU 只算 miss，GPU 只算命中，相加即全量。**该语义对任意 n_tokens 成立，prefill 可直接复用。**
- `ffn_moe_routes_pack_cont/concat`（threshold ≥ 0 时）把每层 selected_experts 拼成一个 I32 输出张量 `t_moe_selected_experts_packed`，供运行期 D2H 读取（Stage 2 动态晋升用，默认关闭）。

### 3. 运行时（`llama-context.cpp`）
- 固定 plan：只在加载期填充，运行期零改动（Stage 1 主线）。
- `hybrid_cache_process_routes()`（threshold ≥ 0 且 n_tokens==1）：每 token D2H 48×10 route ids + 同步，LRU 选 victim，逐 expert 逐 tensor 拷权重，更新三张 map。→ 这是 Stage 2 实测 NO-GO 的税项来源，保持关闭。
- `hybrid_cache_stats_get/reset`：精确计数器（hits/misses/h2d bytes/直方图）。

## 二、当前 Prefill 为什么慢（base 路径，与 plan 无关）

Prefill（无 plan 时 `build_moe_ffn` 走原生分支）在 8K physical ubatch 下，`ggml_mul_mat_id` 由 CUDA 后端执行，落入 `ggml-cuda.cu::ggml_cuda_mul_mat_id` 的**通用 fallback**（因为 expert 权重在 host buffer，分组 MMQ 路径要求权重驻留设备）：

1. `cudaMemcpyAsync` 把 ids D2H + **`cudaStreamSynchronize`** —— 每层每 ubatch 一次硬同步；
2. **host 端三重循环分桶** `ne02 × ne12 × n_expert_used`（8K ubatch 时 512×8192×10 ≈ 42M 次迭代/层）；
3. 设备上按 expert 排序 activations（get_rows scatter/gather）；
4. **逐 expert 调 `ggml_cuda_mul_mat`，`src0_slice.data = host_ptr + i02*nb02`** —— 每个 unique expert 每 tensor 一次独立 H2D（pageable、无打包、gate/up/down 三次），实测有效带宽 10.84 GB/s（pinned 可达 23.46）；
5. 输出 gather 回原 token 顺序。

v1 实测（2K ctx / ubatch 512）：49.91 MB/token、33.63 copies/token、host copy API 17.8 s vs 设备 H2D 9.3 s、GPU compute 只占 wall 5.5%。

**关键认识：这条路径其实已经是"分组"的**——同一 expert 在一个 ubatch 内的所有 token 复用一次权重拷贝（ubatch 越大收益越大，这就是 2K→8K ubatch 287→524 tok/s 的来源）。它的浪费在：pageable 逐片拷贝（带宽利用率 46%）、每层硬同步、host 分桶循环、无流水。

## 三、Grouped Runtime 的设计含义

改造目标路径 = 保留分组思想，消灭四个浪费点：

1. **Pinned 打包 staging**：每层每 ubatch 的 unique experts（8K chunk 实测待测，1024 chunk 时 ~292/层）把 gate+up+down 连续打包进 pinned ring buffer，一次/少数几次大块 async H2D @ ~23 GB/s → 带宽利用率 2.2×。
2. **复用 hybrid cache 图结构**：把"每 ubatch 的 unique expert 集"当作瞬时 plan 写进现有 `ffn_*_exps_cache` + 三张 map（或独立 staging slot pool），`build_moe_ffn` 的 hybrid 分支原样工作，负 id 语义已兼容任意 batch。这样 prefill 与 decode 共用一套机制。
3. **设备端分桶**：用分组 MMQ 路径已有的 `ggml_cuda_launch_mm_ids_helper`（on-device bucketing + expert_bounds）替代 host 循环与同步；权重已在 GPU 后，`ggml_cuda_mul_mat_q` 的批量 MMQ 直接可用。
4. **双缓冲流水**：slot pool 两块交替——当前层/批的 GEMM 与下一批的 H2D 重叠（Gate1 已测双流可隐藏 11% copy，配合更大块应更高）。

### 上限推算（16K prompt 实测 trace，2026-09-04）

真实占用（`routing/occupancy-16k.csv`，47 层均值，prompt 16,363 tokens）：

| chunk | unique/层 mean | unique/层 p95 | 每 8K 窗口权重字节（47 层） |
|---:|---:|---:|---:|
| 1024 | 232 | 419 | — |
| 2048 | 288 | 449 | — |
| 4096 | 346 | 474 | — |
| **8192** | **396** | **491** | **396 × 2,764,800 B × 47 ≈ 51.5 GB** |

流式分页上限（8K chunk，expert 部分，不含 attention/其他层）：

| 通道 | 带宽 | 8K chunk 耗时 | tok/s 上限 |
|---|---:|---:|---:|
| pinned 打包批量 H2D（Gate1 实测） | 23.46 GB/s | 2.20 s | **~3,730** |
| 当前 pageable 碎片拷贝有效带宽（v1 实测） | 10.84 GB/s | 4.75 s | ~1,730 |
| 当前实际（8K ubatch 完整路径） | — | — | **524（实测）** |

结论：当前 524 tok/s 距 pageable 带宽上限还有 3.3×（同步 + host 分桶 + 无流水的放大），距 pinned 上限 7.1×。1000+ 目标只需把 pageable→pinned + 消除每层同步（1.7×）即可达成；2000+ 需要接近 pinned 上限（要求 H2D 与 GEMM 流水重叠）。

## 五、实现期发现（2026-09-04 下午，全部 Measured）

### 发现 1：当前 prefill 的真实搬运机制 —— 调度器整张量 per-op staging

仪表化实证（mmq.cu 内 per-op 诊断，`ab/staging-u2048/stderr.log`）：
- 模型放置：49/49 层全部标记 GPU offload；非 expert 权重 4,459.62 MiB 在 CUDA0；**expert 权重 88.85 GiB 在两个 CPU_Mapped（mmap）buffer**；
- `-ngl -2` ⇒ `full_offload=true` ⇒ patched graph cb 把全部 MoE op 强制放到层设备（GPU）⇒ **ggml_backend_sched 为每个 MUL_MAT_ID 把整层 512-expert 张量（3 个）整块 H2D 到 VRAM 池，再跑分组 MMQ**（op 入口处 src0 的 buft 已是 CUDA0）；
- 每个ubatch 的 H2D 总量 = 全部 expert 字节 ≈ 88.85 GiB（与 ubatch 大小无关！）；2K/2048 单 ubatch 实测 12.6 s ≈ 88.85 GiB ÷ 10.8 GB/s（pageable）+ ~4 s 计算 ✓ 定量吻合；
- 推论：16K/8192 = 2 ubatch = 2×88.85 GiB ≈ 16.4 s 拷贝 + 计算 → 与 524 tok/s（31.2 s）吻合。
- v1 时代的 Nsight（49.91 MB/token、1.5 MB avg × 68k 次 = 逐 expert 拷贝）与今日行为不同——v1 时走的是 mul_mat_id 同步 fallback 的逐 expert 路径；当前 dispatch 到分组 MMQ + 调度器整张量拷贝。两代机制都实测成立，分属不同 revision/条件。

### 发现 2（负实验 A）：mmq.cu 内 staging 补丁无效

补丁位置正确但永远不触发——op 执行时 src0 已被调度器拷进 VRAM（buft=CUDA0）。保留（对"我们自己控制放置"的后续设计有用），env `GGML_CUDA_MOE_DEVICE_STAGING`。

### 发现 3（负实验 B）：cudaHostRegister 整个 mmap 反而 8× 变慢

上游自带 `ggml_backend_cuda_register_host_buffer`（env `GGML_CUDA_REGISTER_HOST`），已在 llama-model.cpp 加载收尾处接线。实测：88.85 GiB 中 46.42 GiB 注册成功、40.83 GiB 失败；**prefill 40.96 tok/s（8× 慢）**——部分锁页把未注册区域逐出页缓存，后续拷贝退化为 NVMe 回读；跑完后下一个未注册 run 也慢（50 s），页缓存回暖后恢复 13.26 s。结论：96 GB RAM 无法同时锁 88.85 GiB + 保持系统运转；整文件 pinned 不可行。

### 修正后的正确设计（下一步实现）

**自定义 barrier op + 共享 VRAM 窗口 + 每层 unique-expert staging**：
1. 共享窗口：3 个 [ne0, ne1, 512] VRAM 张量（gate/up/down，~4.05 GiB 总），47 层的 cache 视图都指向它——同 stream 顺序执行保证层 L 的 GEMM 在 barrier L 写完窗口后、barrier L+1 覆写前执行；prefill 不开 CUDA Graph，无重排风险；
2. barrier op（自定义 ggml op，CUDA 实现里做 host 工作）：D2H 该层 ids（10×n_tokens I32，2048-ubatch 仅 80 KB）→ host 求唯一集（~288-396 个）→ 从 **pinned ring（2×~1.1 GiB，CPU 线程并从 mmap 拷入）** 发起 ring→VRAM 批量 DMA → 更新该层 gpu_map/cpu_map/hit_mask → 返回；47 次/ubatch，每次同步 ~0.3-1 ms，占 <5%；
3. expert 计算走现有 hybrid 图分支（命中=窗口，miss=0 因为全量 staged；负 id 语义已兼容任意 batch）；
4. 预算：窗口 4.05 GiB + 4K-ubatch compute ~4 GiB + model 4.46 + KV ⇒ 8K ubatch 需下调窗口至 384 slot（3.0 GiB）或用 4K ubatch；
5. 预期（Measured 口径推算，标 Projection）：每 ubatch 拷贝从 88.85 GiB 降到 unique×47×2.64 MiB（2048→37 GiB，8192→49 GiB），ring 流水 ~20 GB/s ⇒ 2048-ubatch ~1,100 tok/s，8192-ubatch ~2,300 tok/s（含计算）。

## 六、实现与实测（2026-09-04 晚，全部 Measured）

实现落地（llama.cpp-routing-trace 内，env `LLAMA_PREFILL_STAGING=1` 开启）：
- `llama-model.cpp`：按（角色×类型）共享 VRAM 窗口（本模型 6 窗口张量、512 slot、2,837.5 MiB），各层 cache 张量别名到窗口；三张 map 全 miss 初始化；
- `llama-graph.cpp`：staging 模式在 router top-k 后插入锚点节点 `ffn_moe_stage-<il>`（cont）；n_tokens ≥ 32 时图只含 GPU 路径（无 CPU fallback、无 hit mask）；
- `llama-context.cpp`：eval-callback 包装器（透传用户回调）；锚点触发时 D2H ids → 排序去重 → 相邻 expert 合并成 run → 12 线程并行打包进 2.84 GiB **pinned arena** → 每 tensor 分块（256 MiB）DMA 进窗口 → 改写该层三张 map。调度器对 callback 节点段间同步，保证图内顺序。

迭代过程：
| 版本 | 16K 配置 | tok/s | 备注 |
|---|---|---:|---|
| control（同 build 无 staging） | 8192 ubatch（最优） | 566.7 | 调度器整张量路径 |
| v1 逐 run 小拷贝（每拷贝同步） | 2K/2048 | 115.0 | 比 control 慢：3.5 GB/s 有效 |
| v2 pinned + 并行打包 + 分块 DMA | 2K/2048 | 187.2 | |
| **v3 修 phantom** | 2K/2048 | **315.8（+104%）** | reshape 视图继承锚点名被误当 ids——用 op==CONT && ne[0]==10 过滤 |
| v3 | 16K/4096 | 631.9 | 已超 control 最优 |
| v3/v4 | 16K/6144 | 682.3 | 8192 触发 WDDM（窗口 2.84G + compute 7G 超预算），NO-GO |
| **v4 三轮均值** | **16K/6144** | **720.8（CV 1.1%）** | **+27.3% vs control 最优** |

数值门禁（G1，2K/2048 同 build A/B）：logits cosine **0.999992847**、RMSE 0.010075、max-abs 0.050556、argmax 一致（248046）；全部验证 token 序列一致。

剩余差距到 1000+：每 6144-ubatch barrier 合计 ~3.5 s（pack ~25ms + DMA ~55ms/层 串行）vs 计算 ~4 s——需要 (a) 层内 pack∥DMA 流水（barrier 80→~50 ms），(b) 跨层预取/双缓冲把 staging 藏进计算。decode（<32 token）在 staging 模式下每 token 全量 staging（慢但正确）——staging 模式当前定位为 prefill 专用配置。

### 测量卫生注记
- 大 mmap 页缓存状态主导 prefill 重复测量：任何把 88 GiB 挤出缓存的实验（锁页、大内存压力）都会让后续 run 变 4×；prefill A/B 前必须确认页缓存暖（首跑弃置或对比第二跑）。
- staging 模式 8192 ubatch 触发 WDDM（compute ~7 GiB + 窗口 2.84 GiB + model 4.46 GiB）；6144 为实测甜点。

## 七、第一段（层内流水）结论与直接映射增量实验（2026-09-04 深夜，Measured）

1. pack∥DMA 流水实现后持平（~700 tok/s）：分相计时证明瓶颈是 **mmap 源读取 ~10.4-11 GB/s**（12→24 线程无变化），不是 DMA 也不是同步次数。流水化无法突破源读取墙。
2. 依 trace 交叠分析（chunk 2+ 需求 96.6% 被之前并集覆盖、并集 max 505/512）实现了**直接映射（slot=expert）+ 每层增量 staging**——理论上 950-1,250 tok/s。实测翻车两次后回退：
   - v7：按块跨度 DMA 覆盖了"已驻留专家"的窗口槽（span 含 gap）→ 首 token 795 错；
   - v9/v10：改逐 run 写入（含 raw cudaMemcpyAsync 与 tensor_set 两版）仍错；VERIFY 诊断出现**深层矛盾**：2K 单 ubatch 输出正确但"写入后立即回读比对"在 1060 处 mismatch（连刚写的 expert 都不一致）——同一 offset 的 set/get 不一致，怀疑与 6 个共享窗口张量的 buffer 别名/寻址有关，未解（LLAMA_PREFILL_STAGING_VERIFY 可复现）。
   - raw cudaMemcpyAsync 直接调用还有独立问题：不带 cudaSetDevice 会 AV；带上下文后仍写不进 GEMM 读到的位置。已弃用。
3. 回退到 v4 形态（位置布局 + pinned 并行打包 + 连续跨度 DMA + 每 ubatch 重写映射）：**恢复正确**（16K/6144 = 684.6 单轮；v4 三轮均值 720.8 仍为该形态的正式数字）。
4. 直接映射增量（通往 950+）的下一步 = 先解共享窗口 set/get 不一致之谜（VERIFY 已就位，最小复现：2K + LLAMA_PREFILL_STAGING_VERIFY=1）。

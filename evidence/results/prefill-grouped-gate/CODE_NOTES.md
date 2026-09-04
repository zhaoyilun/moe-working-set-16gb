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

### 上限推算（待 8K chunk trace 修正）
- 每 8K chunk 每层 unique experts ≈ 410（外推值）× 2.637 MiB × 48 层 ≈ 51.9 GiB / 23.46 GB/s ≈ 2.21 s → **~3,700 tok/s 流式上限**（16K ctx，不含 attention）。
- 同样字节数走当前 pageable 路径 ≈ 4.79 s → 1,710 tok/s；实测 524，差额是同步 + host 分桶 + 无流水 → 优化空间被三重放大。

## 四、待办

1. 修 routing 工具的 16K prompt 分批 bug（`llama_routing_trace.cpp` 单次 `llama_batch_get_one` 全量提交，超 n_batch=2048 断言失败；改分块提交）。
2. 采 16K prompt trace → chunk 1024/2048/4096/8192 的真实 unique/层 与复用直方图。
3. 决定 staging slot pool 形态（复用 decode plan cache + 瞬时扩展 vs 独立 prefill pool）与 VRAM 预算（v1 推算 16K/2048 基础上 +4 GiB staging ≈ 12.7 GiB 可行）。
4. 数值门禁设计：分组路径 vs 当前路径的 logits / 首 token / greedy continuation（预期同量化同归约序，接近 bit-exact，需实测）。

# Prefill Grouped Runtime — 设计文档 v1

日期：2026-09-04。目标：16K prefill 从 524 tok/s 提到 **1000+（一级）**，为 2000+（二级，流水化）留出结构空间。

## 决策记录

### 决策 1：改在哪里 —— ggml-cuda 的 `mul_mat_id` 通用 fallback，不动 llama.cpp 图结构

理由：
- 当前 524 tok/s 路径的税全部落在 `ggml-cuda.cu::ggml_cuda_mul_mat_id` 的 fallback：ids D2H + 硬同步 + host 三重循环分桶 + 逐 expert pageable 逐 tensor 拷贝 + 逐 expert `ggml_cuda_mul_mat` + gather。
- 三张备选方案的取舍：
  - A. per-layer 拆图（router 先跑、读回 ids、再 staging、再 GEMM）：需要把 48 层图拆成 96 段，llama-graph/调度器手术大，且引入 47 个跨段同步点；
  - B. 复用 decode 的 per-layer 固定 cache 张量做 prefill：每层需要本 ubatch 自己的 unique 集（396/512），47 层同时驻留 = 47×396×2.64 MiB ≈ 48 GiB，不可行；共享窗口又要求层间打断图执行，回到 A；
  - **C.（选定）原地升级 fallback**：路由 ids 已在图内算好、作为 `ids` 张量传给该 op——不需要拆图就能拿到去重依据；staging/GEMM/流水全部封装在这一个函数里；对上层完全透明，hybrid decode 路径（负 id 走 ggml-cpu）零影响。
- Stage 2 的"每 token route D2H 同步税"不会重现：这里是**每层每 ubatch**一次（8K tokens 摊薄，47 次/8192 tokens vs decode 的 47 次/token）。

### 决策 2：两级实现

**一级（本次）：pinned 打包 + 去掉 host 分桶 + 基本异步**
1. ids D2H 改为一次性 320 KB（8K ubatch）pinned 异步拷 + 仅同步一次（或后续用 `ggml_cuda_launch_mm_ids_helper` 设备端分桶 + 2 KB expert_bounds 回读，先做简单版）。
2. host 分桶循环保留但只跑一遍（现状是每 tensor 每层都跑）→ 改为 op 内一次分桶，按 expert 排序 token（沿用现有 src1_sorted 机制）。
3. 逐 expert 逐 tensor 的 pageable `ggml_cuda_mul_mat(host_slice)` 改为：**pinned ring buffer（2×8 MiB）+ 设备 slot 镜像**，`cudaMemcpyAsync` H2D，然后对设备驻留 slice 调 `ggml_cuda_mul_mat`（设备 src0 无二次拷贝）。
4. 预期：带宽 10.84 → ~20-23 GB/s，加上拷贝次数减少，**目标 1,000–1,400 tok/s**。

**二级（一级达标后）：双缓冲流水 + 设备端分桶**
- copy stream 与 compute stream 分离，expert i 的 GEMM 与 expert i+1/i+2 的 H2D 重叠（事件同步），GEMM 消耗率约 100+ TFLOPS 折算权重消耗 >60 GB/s，远超 23.46 GB/s 供给 → 稳态完全 copy-bound，理论上限 ~3,730 tok/s；考虑 attention 与其他开销，**现实目标 2,000–2,800 tok/s**。

### 决策 3：数值口径
- 同量化（IQ4_XS INT4）、同 MMQ 内核族、同 per-expert 归约序——与现路径的预期差异接近 bit-exact；
- 门禁仍按项目标准执行：2K/16K 下 logits RMSE/max-abs/cosine、首 token argmax、32-token greedy continuation 对照；首 token 与 continuation 需完全一致，logits cosine ≥ 0.9999 为过。

### 决策 4：VRAM
- 设备 slot 镜像 2×8 MiB（ring）+ 现有 compute/pool 结构不变；16K/8K ubatch 现实测 allocator 13,294 MiB、余量 3,082 MiB——本改动不新增大块分配。

## 验证计划（门禁）

| 阶段 | 内容 | 通过标准 |
|---|---|---|
| G0 编译 | ggml-cuda.cu 修改后全量 build + 现有 decode 回归（16K hybrid 3×500） | decode 18.753 ±5% 内、首 token 一致 |
| G1 功能 | 2K/16K prefill A/B：现路径 vs 新路径 logits/首 token/32-token continuation | cosine ≥0.9999、序列一致 |
| G2 性能 | 16K 3×500-token 独立轮 prefill tok/s | ≥1,000 tok/s（一级） |
| G3 长上下文 | 262K 单轮 prefill | 无回归即记录，目标 ≥800（一级推算 332×~2.4） |
| G4 汇总 | 报告 + 聚合表更新 | 四类证据标签齐全 |

## 风险
1. fallback 入口条件（`ggml_cuda_mul_mat_id_needs_sync`）与 CUDA Graph 记录冲突——prefill 不开 graph，先在 prefill 路径验证，decode 不经过此路径（负 id 走 CPU、命中走 cache 张量），风险隔离；
2. 8K ubatch 下 host 分桶即便只跑一遍仍是 42M 迭代/层——若成瓶颈，二级的设备端分桶直接消掉；
3. ring buffer 与 MMQ 的对齐要求（block 边界）需按 128 B 对齐。

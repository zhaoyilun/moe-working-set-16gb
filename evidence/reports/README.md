# pc-slotstream-hybrid-gate

本目录是 PC Slotstream 第二关的可运行原型：真实 MLX affine INT4 专家记录、KTransformers AVX2 CPU 专家、CUDA router/activation/prefill 基准和 trace-driven 调度器。

结论与数字见 [RESULTS.md](RESULTS.md)，机器可读汇总见 `results/summary.json`。

## 目录

- `benchmarks/bench_cpu_moe.py`：CPU routed MLP。
- `benchmarks/activation_roundtrip.cu`：5,120 B 双向 activation。
- `benchmarks/router_topk.cu`：真实 BF16 router 和 top-10。
- `benchmarks/gpu_prefill.cu`：真实权重、完整 GPU top-10 prefill。
- `scripts/fetch_mlx_expert.py`：按 safetensors byte range 取真实专家。
- `scripts/trace_scheduler.py`：CLOCK/LRU/LFU/static/hybrid 重放。
- `patches/ktransformers-mlx-affine-avx2.patch`：对 KTransformers 的最小 affine 与 WSL 拓扑补丁。

## 重跑顺序

1. 把 `patches/ktransformers-mlx-affine-avx2.patch` 应用到当前 `work/ktransformers`。
2. 在 WSL Ubuntu 22.04 中按 AVX2、AMX off、AVX-512 off、CUDA off 构建 `kt-kernel`。
3. 用 `scripts/fetch_mlx_expert.py --expert 0 --count 10` 获取 layer 0 的十个专家。
4. 运行 `scripts/run_cpu_matrix.sh`。
5. Windows 侧用 CMake/Ninja 构建三个 CUDA target，再运行并写入 `results/`。
6. 运行 `scripts/summarize_results.py`。

本轮未触发上一轮 PCIe benchmark；对照数字直接读取 `work/pc-slotstream/results/aggregate/summary.json`。

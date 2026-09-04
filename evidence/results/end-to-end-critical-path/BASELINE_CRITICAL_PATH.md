# Decode Critical Path Baseline

## Frozen comparison point

- Model: `Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf`
- llama.cpp revision: `4e97ac86ebe2c4cb8212d98d2641ad6768810896`
- Execution: Windows CUDA build, 24 CPU threads, Flash Attention on, CUDA Graphs on
- Placement: `-ngl 10`; layers 0–38 on CPU, layers 39–47 plus output on CUDA0
- Model buffers: CPU mapped 77,059.72 MiB; CUDA0 12,272.45 MiB
- Context buffers at `n_ctx=2048`: CPU KV 49.50 MiB; CUDA KV 16.50 MiB; CUDA compute 1,038.02 MiB; pinned host compute 13.43 MiB
- Prompt: `routing/prompts/02_en_technical.txt`, 138 prompt tokens
- Decode measurement: 100 warm-up tokens, 500 measured tokens, three runs, greedy sampling

## Measured end-to-end decode

| Metric | Result |
|---|---:|
| Mean token wall time | 152.957 ms |
| Median token wall time | 151.818 ms |
| p95 token wall time | 162.436 ms |
| Mean `llama_decode` | 148.235 ms |
| Mean sampling | 4.723 ms |
| Token generation throughput | 6.538 tok/s |

The three run means were 155.708, 152.187, and 150.976 ms/token. The first run was colder; pooled values above retain all three runs.

## CUDA-side observation

A separate 20-token Nsight Systems capture observed:

| Metric | Mean per token |
|---|---:|
| Token wall time under capture | 161.808 ms |
| GPU kernel active time | 4.151 ms |
| GPU copy-engine active time | 0.347 ms |
| GPU total active union | 4.498 ms |
| First-to-last GPU operation span | 159.143 ms |
| GPU idle inside token | 157.310 ms |
| CUDA synchronization activity | 8.415 ms |
| H2D traffic | 90 copies / 1.127 MB |
| D2H traffic | 40 copies / 2.055 MB |
| D2D traffic | 58 copies / 2.119 MB |

The 159 ms GPU projection is a span, not 159 ms of GPU computation. GPU operations are interleaved with long CPU-resident layer intervals; actual GPU active union is about 4.5 ms/token in this placement.

## Scope boundary

The earlier 26.921 ms/token Hybrid routed-MoE prototype belongs to a different execution path. It is retained as an independent component result and is not subtracted from the 152.957 ms/token llama.cpp baseline.

## Profiling-only instrumentation

The CPU attribution build adds bounded marker timing around layer-stage endpoints. It is used only in short profiling runs. Normal baseline and optimization A/B runs use the profiler disabled. Absolute performance claims come from the uninstrumented path; the marker run supplies the distribution across CPU layers and operator families.

## Gate state before optimization

The baseline is now frozen. The next decisions are based on end-to-end comparisons under the same prompt, context, sampling, and thread count. The 16 GB VRAM evaluation treats model residency, KV, expert cache, and workspace as one global budget.

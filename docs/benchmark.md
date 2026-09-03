# Benchmark Protocol

## Decode headline

- restore a saved 16,363-token state
- replay the final prompt token
- 100 warmup tokens
- 500 measured tokens
- three independent process runs
- greedy or fixed-token continuation held constant within an A/B series
- CUDA Graph ON
- report mean tok/s plus p50/p95 token latency

## Context matrix

- actual states: 2,027 / 8,171 / 16,363 / 32,747 tokens
- pools: 4 / 6 / 8 GiB fixed plans
- 100 warmup + 100 measured tokens
- one performance run per cell
- separate 100-token stats-only companion run per cell

The companion run provides hit rate and CPU fallback counts. Its throughput is excluded because packed route readback adds a synchronization point.

## Cold Prefill headline

- 16,363 prompt tokens
- `n_batch = n_ubatch = 8192`
- three independent process runs
- report tokens divided by Prefill wall time
- model load and context initialization excluded from Prefill wall time

## Evidence discipline

Keep these four labels in every exported row:

1. Measured
2. Prototype measured
3. Trace simulation
4. Projection

Results from different prompt seeds, executables, sampling windows, or graph states stay in separate series. Component measurements are not subtracted from an unrelated end-to-end wall time.

## VRAM policy

Use both:

- allocator-accounted buffers from llama.cpp logs
- device-level peak when a low-perturbation measurement exists

The reference safe policy requires 1,024 MiB allocator-accounted headroom. A benchmark-max result below that reserve stays marked as benchmark-max.

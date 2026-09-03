# Architecture

## Unified abstraction: model working-set virtualization

The model is larger than VRAM, but each token touches only a sparse subset of routed experts. The runtime therefore separates **capacity** from **working set**, similar to virtual memory:

| Runtime resource | Role |
|---|---|
| System RAM | Complete routed-expert capacity |
| VRAM | Non-routed GPU core plus hot expert working set |
| GPU | Primary tensor execution |
| CPU | Cold expert and miss-burst execution |

The analogy stops at page granularity: an expert is a structured tensor group, routing is model-driven, and moving an expert has compute and graph-scheduling consequences beyond ordinary page faults.

## Placement

All suitable attention, norm, shared/dense FFN, router, and output tensors reside on GPU. Routed expert tensors reside in RAM, with selected packed experts copied into a fixed VRAM pool at model load.

## Decode state machine

```text
route IDs
  -> cache lookup
  -> split selected experts into resident hits and CPU misses
  -> evaluate both subsets
  -> weighted merge
  -> next GPU-core segment
```

The default is a fixed plan because it preserves CUDA Graph reuse and has no per-token policy synchronization. The experimental promotion mode adds one packed 48×10 route readback per token, then classifies the prior token's same-layer miss set. Thresholds 1–4 all regressed end-to-end throughput.

## CUDA Graph boundary

CUDA Graph is part of the performance architecture rather than an optional launch optimization. On the 8 GiB fixed-cache path, Graph ON measured 18.753 tok/s and Graph OFF measured 13.567 tok/s, a 38.23% difference.

## Context-adaptive budget

```text
device total
  - CUDA model buffer
  - attention/indexer KV
  - recurrent state
  - compute workspace
  - output/runtime reserve
  = candidate expert-pool headroom
```

The measured policy requires at least 1,024 MiB allocator-accounted headroom. It selects 8 GiB at 2K/8K and 6 GiB at 16K/32K.

## Prefill phase

Decode benefits from a long-lived hot-expert cache. Prefill benefits from broad token grouping and weight reuse inside a large chunk. The current v2 result uses an 8K physical ubatch with the existing selected-expert CUDA path. The next 1000+ design is:

```text
prompt chunk
  -> router
  -> expert buckets
  -> unique expert list
  -> contiguous staging
  -> grouped GPU GEMM
  -> weighted scatter/merge
```

## Deliberately excluded complexity

- per-expert futures and task graphs
- one stream per expert
- additional multi-GiB Decode staging pool
- runtime weight-format conversion
- dynamic promotion as the default path

These features add scheduling state before the measured minimum mechanism has earned the complexity.

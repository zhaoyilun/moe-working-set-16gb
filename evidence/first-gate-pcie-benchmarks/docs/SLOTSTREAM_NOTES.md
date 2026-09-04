# Slotstream source analysis for a PC/CUDA prototype

Upstream examined: `carloslfu/slotstream` commit
`5bf0e67ac050f2d7407eb72e2cc52e818c9509dd` (2026-09-02).

## Core model geometry

- 48 MoE layers.
- 512 routed experts per layer.
- 10 routed experts plus one shared expert are active per token/layer.
- Hidden size 2560; routed-expert intermediate size 640.
- Expert quantization is 4-bit, group size 64.
- One routed expert occupies exactly 2,764,800 bytes across nine pieces:
  gate/up/down, each with packed weight, BF16 scales, and BF16 biases.
- All routed experts occupy about 67.95 GB in the examined MLX checkpoint.

## What the slot pool actually does

The runtime keeps one global pool shared by all layers. Its key is
`(layer, expert)`, not just `expert`, because equal expert IDs in different layers
refer to different weights. The current implementation uses CLOCK eviction. A
decode step performs:

1. deduplicate the routed expert IDs;
2. map cache hits to existing slots;
3. select victims for misses and pin the whole current working set;
4. read the nine exact tensor slices for every missing expert;
5. batch-scatter the quantized bytes into nine resident pool tensors;
6. execute `gatherQuantizedMM` using slot IDs;
7. unpin after the layer step.

The pool is global so hotter layers can borrow capacity from colder layers.

## Why this is more than ordinary offload

Under the measured MLX path, gathering top-10 experts from a memory-mapped
512-expert tensor materialized the whole layer tensor. N-gram row lookup similarly
materialized a whole shard. Explicit record reads into a bounded resident pool
separate total model capacity from the active compute working set.

On a discrete NVIDIA GPU, that same mechanism becomes a two-level active store:

```text
NVMe checkpoint -> RAM expert store -> PCIe -> VRAM slot pool -> CUDA kernel
```

The RAM-to-VRAM hop is new and is the first quantity this prototype measures.

## Important limit on cross-layer prefetch

The next layer's router consumes the current layer's output. Therefore exact
layer-N+1 expert IDs are generally unavailable while layer N is still computing.
The upstream project built and removed a background cross-layer read-ahead path
after paired runs were slower. Its useful overlap happens within a known routed
group/prefill sweep.

The CUDA benchmark's two-stream result is consequently an *ideal known-next-batch
upper bound*. Reaching it in a real decoder requires a correct source of early
expert IDs, such as same-layer batching, speculative routing that is validated,
or overlap between transfer and other already-independent work.

## Benchmark interpretation

Transfer measurements use synthetic byte buffers whose size exactly matches the
real serialized expert record. Two layouts are measured:

- `contiguous_record`: one H2D operation per expert batch, representing a RAM
  store repacked for the PC runtime;
- `nine_pieces_per_expert`: nine H2D operations per expert, representing the
  CUDA submission fragmentation of the original tensor-piece geometry. It does
  not include CPU gathering from nine large strided host tensors or a real
  nine-pool destination scatter.

The overlap compute is three FP16 strided-batched GEMMs with the real
`2560 <-> 640` geometry. It matches the routed expert's GEMM FLOP count, but it
does not execute the real 4-bit dequantization, SiLU/multiply, router, shared
expert, attention, or full model. It is an overlap microbenchmark, not a token/s
claim.

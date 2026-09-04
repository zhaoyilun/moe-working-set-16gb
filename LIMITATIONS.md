# Limitations

1. Measurements cover one Windows workstation, one GPU generation, one model family, and one quantization family.
2. The 18.753 tok/s Decode headline is a 16K canonical state with three 500-token runs. Context-matrix cells are later one-run 100-token comparisons.
3. Fixed-cache hit metrics come from a separate stats-only companion path because route readback perturbs CUDA Graph timing.
4. The 8 GiB pool has only 386 MiB observed device headroom at 16K. At 32K its allocator-accounted headroom is 306 MiB. These are benchmark-max configurations rather than desktop defaults.
5. Selective H2D is a negative implementation result, not a general statement about all dynamic expert caches.
6. Prefill throughput depends strongly on prompt routing. The 524.003 tok/s headline is a strict same-prompt three-run series; the earlier 352.433 tok/s v1 number used another prompt seed.
7. Quantized Prefill is batch-partition dependent. Larger ubatches preserve the tested first-token argmax, while 2K versus 8K greedy sequences diverge after six tokens on the tested prompt.
8. Nsight Systems 2023.3.3 did not produce a usable Stage-2 report after target exit; Stage-2 attribution uses internal counters.
9. Long-context cells left unmeasured: q4-q4 KV with the 4 GiB pool and f16-f16 KV at 262K context. Other GPUs, DDR5 frequency changes, and multi-prompt quality evaluation are outside this snapshot.
10. Model weights, session files, raw prompts, token continuations, full routing traces, hostnames, and personal filesystem paths are excluded from this staging area.

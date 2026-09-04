# llama.cpp Discussion draft

## Title

Experiment: fixed GPU expert working set with Native CPU miss execution for Qwen sparse MoE — 16K through 262K context on one 16 GiB GPU

## Summary

This is an engineering proposal and result package for tensor-structured sparse-MoE placement in llama.cpp. The experiment starts from a pinned revision (`4e97ac8`) and keeps all suitable non-routed tensors on CUDA while routed expert tensors remain in host memory. A fixed subset of packed experts is materialized in a GPU pool at model load; graph construction splits selected expert IDs into resident GPU hits and Native CPU misses, then merges weighted outputs.

Reference system: RTX 4080 Super 16 GiB, i9-14900KF, 96 GiB DDR5-4000, Windows, Qwen3.8-Flash-Next 176.94B IQ4_XS-family GGUF (QWEN4EXP architecture).

## Measured results — short/medium context

- Native expert-only CPU, same Stage-1 path: 15.135 tok/s
- fixed cache 4 GiB: 16.096 tok/s
- fixed cache 6 GiB: 17.275 tok/s
- fixed cache 8 GiB: 18.753 tok/s

Each result above is a three-run 500-token mean at a restored 16K state. The 8 GiB path improved 23.91% over the same-path Native control. Short-context repeats: 19.531 tok/s at 2K and 18.321 tok/s at 8K (three-run means, coefficient of variation under 1%).

CUDA Graph remained enabled. The 8 GiB Graph-OFF control measured 13.567 tok/s, so preserving graph reuse is central to the design.

## Measured results — long context (128K–262K gate)

KV quantization profiles were swept (f16-f16, q8-q8, q4-q4, q8-q4 K/V mix) with cache round-trip cosine = 1 at every step, then Decode, retrieval, and VRAM were measured per context step:

- 262K context, q8-q4 KV + 4 GiB pool: **5.776 tok/s** (p50 170.7 ms / p95 197.2 ms), peak 13,611 MiB, 2,765 MiB headroom
- 262K context, q4-q4 KV + 6 GiB pool: 5.649 tok/s — the q8-q4+4G combination is slightly faster while using less VRAM
- Long-context retrieval at 262K: 5/5 cases correct with q8-q4 KV (4/5 with q4-q4; the miss was a 384-token output-budget truncation, not a retrieval failure)
- Session restore into a 262K state: about 14 s
- Honest framing: the hybrid cache gain shrinks from +23.91% at 16K to about +4.6% at 262K, because attention/KV time dominates per-token cost as context grows

One negative result is kept deliberately: q8-q4 KV with a 6 GiB pool at 262K fits on paper but collapsed to 0.71 tok/s mean (p95 7,477.7 ms) under Windows WDDM shared-memory paging, and the collapse is residency-dependent — a healthy 5+5-token run of the same config stayed at 5.508 tok/s with only 637 MiB headroom. That is the field evidence behind our ≥1,024 MiB allocator-headroom rule and the context ladder: 8 GiB pool through 128K, 6 GiB at 192K, 4 GiB at 262K.

## Dynamic-promotion result

I added an experimental packed 48×10 route observation tensor and adjacent-token promotion threshold. Thresholds 1–4 increased hit rate up to 61.78%, but all regressed throughput. Promotion moved up to 114.45 MB/token and added 22.89 ms/token inside the promotion routine. The default remains fixed-cache mode (`threshold=-1`).

## Prior art referenced

- llama.cpp / ggml (MIT) — the base this patches.
- Slotstream by Carlos Galarza (carloslfu/slotstream, MIT) — expert-slot streaming for the same model on Apple Silicon; inspired treating experts as a streamed working set. No source copied.
- Fiddler (efeslab/fiddler, arXiv 2402.07033) — the CPU-executes-missed-experts trade (copy small activations, not large weights) that the Native CPU miss path relies on. No source copied.
- KTransformers (kvcache-ai, Apache-2.0) — read as a heterogeneous MoE comparison point. No source copied.

## Potential upstream pieces

1. expert tensor placement override
2. fixed packed expert working set
3. public hit/miss/fallback counters
4. packed route observation for diagnostics

The repository package includes the patch, runners, raw per-run evidence (JSON + stderr logs for every number above), aggregate CSVs, exact base revision, and limitations. Before a PR, I would like feedback on API shape, model generality, and where a fixed expert working-set policy belongs in llama.cpp.

## Work in progress

At 262K context the cold prefill costs 786.9 s (331.8 tok/s) — the system's largest single wait. An expert-bucketing plus grouped-GEMM prefill path targeting 1000+ tok/s is actively being developed; the first implementation round is not landed yet, and measured results will be added to the repository as they arrive.

Repository: https://github.com/zhaoyilun/moe-working-set-16gb

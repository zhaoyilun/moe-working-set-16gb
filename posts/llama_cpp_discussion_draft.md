# llama.cpp Discussion draft

## Title

Experiment: fixed GPU expert working set with Native CPU miss execution for Qwen sparse MoE (16 GiB GPU, 16K–262K context)

## Body

I've been experimenting with tensor-structured sparse-MoE placement in llama.cpp and put together a result package I'd like feedback on. The starting point was a simple observation: for a sparse MoE that's way bigger than VRAM, the question isn't "which layers go to CPU" but "which tensors". I keep everything that suits CUDA (attention, norms, router, shared/dense FFN) on the GPU and leave routed expert weights in host RAM. At model load, a fixed subset of packed experts gets materialized in a GPU pool; graph construction splits the selected expert IDs into resident hits (GPU) and misses (Native CPU expert), then merges the weighted outputs.

My test box: RTX 4080 Super 16 GiB, i9-14900KF, 96 GiB DDR5-4000, Windows, Qwen3.8-Flash-Next 176.94B IQ4_XS-family GGUF (QWEN4EXP arch), pinned llama.cpp revision `4e97ac8`.

What I measured at 16K context (three 500-token runs each):

- same-path Native expert-only CPU control: 15.135 tok/s
- fixed cache 4 GiB: 16.096 tok/s
- fixed cache 6 GiB: 17.275 tok/s
- fixed cache 8 GiB: 18.753 tok/s

So +23.91% over the control at 8 GiB. Short-context repeats: 19.531 tok/s at 2K, 18.321 at 8K. One thing that surprised me: CUDA Graph is doing a lot of heavy lifting here — the same 8 GiB config with graphs off drops to 13.567 tok/s.

I also ran the context ladder out to 262K with mixed KV quantization (q8_0 K / q4_0 V):

- 262K, q8-q4 KV + 4 GiB pool: 5.776 tok/s (p50 170.7 ms / p95 197.2 ms), peak 13,611 MiB
- q4-q4 KV + 6 GiB pool: 5.649 tok/s — the q8-q4 + 4 GiB combo is slightly faster while using less VRAM
- retrieval at 262K: 5/5 with q8-q4 (4/5 with q4-q4, and that miss was an output-budget truncation, not a retrieval failure)
- restoring a 262K session takes about 14 s

Being honest about this part: the cache gain shrinks from +23.9% at 16K to about +4.6% at 262K. Attention/KV dominates per-token time out there, so the expert working set stops being the bottleneck.

The most painful lesson was a WDDM one. q8-q4 KV with a 6 GiB pool at 262K fits on paper — a short 5+5-token run of it looked fine at 5.508 tok/s with 637 MiB headroom — but a 250-token run collapsed to 0.71 tok/s mean with p95 at 7.5 s, because total demand crossed physical VRAM and Windows started paging through shared memory. And it's nondeterministic: whether you collapse depends on residency. That's why I landed on keeping at least ~1 GiB of allocator headroom as a hard rule, with the pool stepping down 8 GiB → 6 GiB → 4 GiB as context goes 128K → 192K → 262K.

I also tried adjacent-token selective H2D promotion (packed 48×10 route observation tensor). Hit rate went from ~31% to 61.78%, and every threshold setting made throughput worse — up to 114.45 MB/token moved and 22.89 ms/token inside the promotion routine alone. Cache optimization is about end-to-end service time, not hit rate. That path stays in the tree but disabled by default (`threshold=-1`).

On prefill: physical ubatch 2K → 8K took the same 16K prompt from 287.387 to 524.003 tok/s (three-run mean; 617.342 single-run at 32K). At 262K cold prefill is still 786.9 s (331.8 tok/s), which is the biggest single wait in the system. I'm actively working on expert bucketing + grouped GEMM to push past 1000 tok/s — the first implementation round hasn't landed yet, and measured results will go into the repo as they arrive.

Pieces I think could be upstreamed, and where I'd genuinely like feedback:

1. expert tensor placement override
2. fixed packed expert working set
3. public hit/miss/fallback counters
4. packed route observation for diagnostics

Before turning any of this into a PR, I'd rather hear where people think a fixed expert working-set policy belongs in llama.cpp, whether the API shape makes sense, and how to generalize beyond this one model.

To be clear about lineage: this builds on llama.cpp itself as the base; the expert-slot streaming idea came from reading Slotstream (carloslfu — same model on a Mac, streamed from SSD); the CPU-executes-missed-experts trade is straight out of Fiddler (efeslab, arXiv 2402.07033); KTransformers was my heterogeneous-execution comparison. No source copied from any of them.

Everything above ran on the full llama.cpp path, and every number has per-run JSON + stderr logs behind it: https://github.com/zhaoyilun/moe-working-set-16gb (patch, runners, ~1150 evidence files, build/repro docs). Limitations are listed there too — one machine, one model family, and a few matrix cells still unmeasured.

# Reddit draft

## Title

Running a 176B sparse MoE on a 16 GB RTX 4080 Super: 6.5 → 18.8 tok/s decode, and it still does 5.8 tok/s at 262K context (llama.cpp patch, full evidence in repo)

## Body

I wanted a local setup for Qwen3.8-Flash-Next (176.94B total params, IQ4_XS GGUF) that I could actually live with day to day, on a regular Windows PC — RTX 4080 Super 16 GB, i9-14900KF, 96 GB DDR5. This is my own project; the patch, runners, and the log behind every single number are here: https://github.com/zhaoyilun/moe-working-set-16gb

The mental model that ended up working:

- RAM holds the full expert capacity
- VRAM holds the non-expert core plus a hot expert working set
- GPU runs the cache hits, CPU runs the misses natively
- prefill and decode get different policies

The progression was pretty stark. Coarse `-ngl 10` layer offload: 6.538 tok/s. Keeping routed experts in RAM but everything else on GPU: about 15–16 tok/s. Fixed 8 GiB expert cache: 18.753 tok/s at 16K (three 500-token runs), 19.5 / 18.3 at 2K / 8K. Fun detail: CUDA Graph on vs off on the exact same config is 18.753 vs 13.567 tok/s — keeping graph reuse matters about as much as getting cache hits.

At 262K context with q8_0-K / q4_0-V KV and a 4 GiB pool it still holds 5.776 tok/s (p50 170.7 ms, peak 13.6 GiB), retrieval checks pass 5/5, and restoring a 262K session takes about 14 s. Not claiming magic, though — the cache's advantage over no-cache shrinks from +23.9% at 16K to about +4.6% at 262K because attention time takes over. Long context is a KV problem more than an expert problem.

Two gotchas worth flagging for Windows folks:

1. The WDDM cliff is real and sneaky. q8-q4 KV + 6 GiB pool at 262K fits on paper — I watched a healthy run of that exact config do 5.5 tok/s with just 637 MiB to spare — and then a longer run collapsed to 0.71 tok/s mean with p95 latencies around 7.5 s, because total demand crossed physical VRAM and Windows started paging through shared memory. Same config, nondeterministic outcome depending on residency. I now treat ≥1 GiB of allocator headroom as a hard rule, with the pool stepping 8 → 6 → 4 GiB as context goes 128K → 192K → 262K.

2. Chasing hit rate made things slower. Adjacent-token expert promotion pushed hit rate from 31% to 62%, and throughput dropped to 11–12.8 tok/s — the per-token route readback, the sync boundary, and fragmented weight-slice transfers ate everything the extra hits gained. It's still in the code, disabled by default.

Prefill: physical ubatch 2K → 8K takes a 16K prompt from 287 to 524 tok/s (three-run mean; 617 at 32K in a single run). At 262K cold prefill is still 786.9 s, and that's the thing I'm actively working on — expert bucketing + grouped GEMM to get past 1000 tok/s. First attempt hasn't landed yet; I'll put numbers in the repo when they're real.

Since people (rightly) ask: built on llama.cpp as the base. The expert-slot streaming idea came from reading Slotstream (carloslfu — same model on a Mac, streamed from SSD). The CPU-runs-missed-experts trade is straight out of Fiddler (efeslab, arXiv 2402.07033). KTransformers was my comparison reference. No source copied from any of them.

Caveats: one machine, one model/quant family, Windows-specific VRAM behavior, a couple of matrix cells unmeasured (q4-q4 + 4 GiB and f16-f16 at 262K). Every number in the repo is labeled measured / prototype measured / trace simulation / projection, and the per-run JSON + stderr logs are all in there (~1150 files).

Happy to dig into the WDDM behavior or the cache policy in the comments — those were the two things I got wrong first.

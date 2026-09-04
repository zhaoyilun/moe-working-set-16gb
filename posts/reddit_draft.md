# Reddit draft

## Title

Running a 176B sparse MoE on a 16 GB RTX 4080 Super: 6.5 to 18.8 tok/s decode, still 5.8 tok/s at 262K context (llama.cpp patch + full evidence)

## Body

I have been prototyping a phase-adaptive heterogeneous MoE runtime on a normal Windows PC: RTX 4080 Super 16 GB, i9-14900KF, and 96 GB DDR5-4000. The model is Qwen3.8-Flash-Next, 176.94B total parameters, IQ4_XS-family GGUF.

The useful mental model:

- RAM stores complete expert capacity.
- VRAM stores the non-expert GPU core and a hot expert working set.
- GPU runs resident expert hits.
- CPU handles cold expert misses.
- Prefill and Decode use different policies.

Measured Decode progression:

- coarse `-ngl 10` layer offload: 6.538 tok/s
- expert-only CPU placement: about 15–16 tok/s
- fixed 8 GiB GPU expert cache: 18.753 tok/s, three 500-token runs at 16K context
- short context: 19.531 tok/s at 2K, 18.321 tok/s at 8K (three-run means)

CUDA Graph mattered a lot: 18.753 tok/s ON versus 13.567 OFF on the 8 GiB path.

Long context is where it got interesting. With q8-q4 KV quantization (q8_0 K / q4_0 V) and a 4 GiB expert pool, the same runtime holds **5.776 tok/s at 262K context** (p50 170.7 ms, p95 197.2 ms), with peak VRAM at 13.6 GiB. Long-context retrieval scored 5/5 at 262K, and restoring a 262K session takes about 14 s.

Two honest caveats:

1. The hybrid cache gain shrinks as context grows: +23.9% over the Native control at 16K, but only about +4.6% at 262K, because attention/KV time starts dominating per-token cost. The cache is not magic at long context.
2. I hit a WDDM cliff worth knowing about on Windows: q8-q4 KV with a 6 GiB pool at 262K fits on paper, but collapsed to 0.71 tok/s mean (worst-case p95 7.5 s) once total demand crossed physical VRAM and shared-memory paging kicked in. Worse, it is nondeterministic — a short run of the same config looked healthy with only 637 MiB headroom. So the config ladder I settled on keeps at least ~1 GiB of allocator headroom: 8 GiB pool through 128K, 6 GiB at 192K, 4 GiB at 262K.

I also tested adjacent-token selective H2D. Hit rate rose from 31% to 62%, yet throughput fell to 11–12.8 tok/s. The main issue was the per-token route readback/synchronization boundary plus fragmented expert-slice transfers. That path stays experimental and disabled.

For Cold Prefill, increasing physical ubatch to 8192 produced a 524.003 tok/s three-run mean for 16,363 tokens, with a 32K single check at 617.342 tok/s. At 262K the cold prefill cost is 786.9 s (331.8 tok/s) — still the system's biggest single wait. I'm actively working on an expert-bucketing + grouped-GEMM prefill path to push past 1000 tok/s; the first implementation round hasn't landed yet, and I'll add measured results to the repo as they arrive.

Prior art this builds on (read and referenced, no source copied): llama.cpp as the base, Slotstream (carloslfu) for the expert-slot streaming idea on Apple Silicon, Fiddler (efeslab, arXiv 2402.07033) for the CPU-executes-missed-experts trade, and KTransformers as a heterogeneous MoE comparison.

Limitations: one machine, one model/quant family, Windows-only VRAM behavior observed, some cells of the matrix unmeasured (q4-q4 with the 4 GiB pool and f16-f16 at 262K). Everything above is measured on the full llama.cpp path; the repo labels every number as measured / prototype measured / trace simulation / projection.

I published the patch, both benchmark runners, the reproduction scripts, and the complete raw evidence trail (per-run JSON + stderr logs behind every number, ~1150 files) here:

https://github.com/zhaoyilun/moe-working-set-16gb

Happy to answer questions about the WDDM behavior or the cache policy — those were the two things I got wrong first.

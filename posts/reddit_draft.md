# Reddit draft

## Title

Running a 176B sparse MoE on a 16 GB RTX 4080 Super: from 6.5 to 18.8 tok/s with expert working-set virtualization

## Body

I have been prototyping a phase-adaptive heterogeneous MoE runtime on a normal Windows PC: RTX 4080 Super 16 GB, i9-14900KF, and 96 GB DDR5-4000.

The model is a 176.94B sparse MoE GGUF. The useful mental model is:

- RAM stores complete expert capacity.
- VRAM stores the non-expert GPU core and a hot expert working set.
- GPU runs resident expert hits.
- CPU handles cold expert misses.
- Prefill and Decode use different policies.

Measured Decode progression:

- coarse `-ngl 10` layer offload: 6.538 tok/s
- expert-only CPU placement: about 15–16 tok/s
- fixed 8 GiB GPU expert cache: 18.753 tok/s, three 500-token runs at 16K context

CUDA Graph mattered a lot: 18.753 tok/s ON versus 13.567 OFF on the 8 GiB path.

I also tested adjacent-token selective H2D. Hit rate rose from 31% to 62%, yet throughput fell to 11–12.8 tok/s. The main issue was the per-token route readback/synchronization boundary plus fragmented expert-slice transfers. That path stays experimental and disabled.

For Cold Prefill, increasing physical ubatch to 8192 produced a 524.003 tok/s three-run mean for 16,363 tokens, with a 32K single check at 617.342 tok/s. The next 1000+ direction is expert bucketing plus grouped GEMM rather than a still-larger workspace.

Important limitations: one machine, one model/quant family, tight 8 GiB Decode headroom, and batch-partition-dependent Prefill numerics. I am preparing the patch, aggregate CSVs, and reproduction notes for review before deciding how to publish it.

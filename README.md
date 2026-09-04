# MoE Working Set on 16 GB VRAM

> A phase-adaptive heterogeneous MoE runtime experiment for running models larger than VRAM on consumer discrete GPUs — llama.cpp patch, benchmark runners, and the full measurement evidence trail.

## What is this?

This repository contains a llama.cpp patch, two benchmark runners, reproducible configurations, and a complete evidence ledger for a 176.94B-parameter sparse MoE model (Qwen3.8-Flash-Next, IQ4_XS-family GGUF) on a 16 GiB GPU plus 96 GiB system RAM.

The core model is simple:

- **RAM = model capacity layer**
- **VRAM = high-speed working set**
- **GPU = primary compute**
- **CPU = cold expert fallback**
- **Prefill and Decode use different policies**

## Why?

Coarse layer offload placed complete early layers on CPU and produced 6.538 tok/s. Structure-aware placement moved non-expert operators to GPU while keeping routed experts in RAM, raising Decode to about 15–16 tok/s. A fixed GPU expert working set then reached 18.753 tok/s at 16K context, and the same runtime holds 5.776 tok/s at 262K context with q8-q4 KV quantization and a 4 GiB pool.

## Reference hardware

- NVIDIA GeForce RTX 4080 SUPER, 16 GiB
- Intel Core i9-14900KF
- 96 GiB DDR5-4000
- PCIe 4.0 x16
- Windows, CUDA, llama.cpp base revision `4e97ac86ebe2c4cb8212d98d2641ad6768810896`
- Qwen3.8-Flash-Next-GGUF, IQ4_XS family

## Headline results

| Data label | Path | Result |
|---|---|---:|
| Measured | Coarse layer offload, `-ngl 10` | 6.538 tok/s Decode |
| Measured | Native expert-only CPU placement | about 15–16 tok/s Decode |
| Measured | Fixed 8 GiB GPU expert cache, 16K, 3×500 | **18.753 tok/s Decode** |
| Measured | Short context, 2K / 8K, 3-run means | 19.531 / 18.321 tok/s |
| Measured | Hybrid Decode at 262K context, q8-q4 KV + 4 GiB pool | **5.776 tok/s**, p50 170.7 ms, peak 13,611 MiB |
| Measured | Cold Prefill, 16,363 tokens, 8K ubatch, 3 runs | **524.003 tok/s mean** |
| Measured, single run | Cold Prefill, 32,747 tokens, 8K ubatch | **617.342 tok/s** |
| Measured | Session restore to a 262K restored state | about 14 s |

Context scaling is honest about diminishing returns: the hybrid cache gains
+23.91% over the Native control at 16K, but only about +4.6% at 262K, because
attention/KV time dominates per-token cost as context grows.

### The WDDM boundary (negative result, kept on purpose)

q8-q4 KV with a 6 GiB pool at 262K context fits on paper (~637 MiB peak
headroom observed in a healthy short run) but collapsed to 0.71 tok/s mean
(p95 7,477.7 ms) in a 250-token run: once total demand crosses physical VRAM
under Windows WDDM, shared-memory paging destroys latency, and whether it
happens is residency-dependent, i.e. nondeterministic. This is the direct
field evidence behind the ≥1,024 MiB allocator-headroom safe-default rule and
the context ladder:

| Context step | q8-q4 KV safe-default pool |
|---|---|
| ≤128K | 8 GiB |
| 192K | 6 GiB |
| 262K | 4 GiB |

## How it works

### Decode

```text
router
  -> fixed cache lookup
  -> GPU expert for hits
  -> Native CPU expert for misses
  -> shared GPU core continues under CUDA Graph
```

An adjacent-token selective-H2D experiment raised hit rate but reduced end-to-end throughput because per-token synchronization and fragmented expert-slice transfers dominated. It remains disabled by default.

### Prefill

```text
large prompt ubatch
  -> more tokens share each active expert
  -> fewer repeated weight transfers per token
  -> existing selected-expert CUDA path
```

Increasing physical ubatch from 2K to 8K raised the same-prompt 16K Prefill path from 287.387 to a 524.003 tok/s three-run mean. At 262K the cold prefill cost is 786.9 s (331.8 tok/s) — the system's largest single wait.

The first round of the expert-staging prefill runtime (eval-callback barrier + shared VRAM windows + pinned pack/DMA) measured **720.8 tok/s** on the 16K prompt at physical ubatch 6144 — +27.3% over its same-condition control, with numerics agreement at cosine 0.9999928. Pushing past 1000 tok/s continues; the runtime patch lives in the routing-trace repository and the run data in `evidence/results/prefill-grouped-gate/`.

## Evidence labels

- **Measured**: full llama.cpp path or direct system measurement.
- **Prototype measured**: real Qwen layer/expert component in a standalone harness.
- **Trace simulation**: cache/policy replay over recorded routing IDs.
- **Projection**: component or capacity model used for planning.

Headline numbers above are Measured.

## Repository map

- `patches/`: patch against the pinned llama.cpp revision
- `tools/`: prefix-state and Hybrid Decode runners
- `scripts/`: parameterized reproduction scripts
- `configs/`: public-format examples (fixed plans per pool size, runtime example)
- `data/results/`: sanitized aggregate CSVs
- `data/synthetic/`: synthetic fixtures
- `evidence/`: **raw evidence trail** — per-run JSON, stderr logs, stage reports (in Chinese), the first PCIe micro-benchmark gate, and the routing-trace tool; see [evidence/EVIDENCE_INDEX.md](evidence/EVIDENCE_INDEX.md)
- `docs/`: build, hardware, benchmark, and design notes
- `posts/`: unpublished draft announcements
- `charts/`: generated figures sourced from CSV

Start with [docs/build.md](docs/build.md), then [docs/benchmark.md](docs/benchmark.md).

## References and prior art

This project stands on prior work it read and referenced, without copying source:

- **llama.cpp / ggml** (MIT) — the base framework this patches.
- **Slotstream** by Carlos Galarza ([carloslfu/slotstream](https://github.com/carloslfu/slotstream), MIT) — expert-slot streaming for the same model family on Apple Silicon; the inspiration for treating experts as a streamed working set. This project takes the complementary Windows/CUDA route with RAM as the capacity layer.
- **Fiddler** ([efeslab/fiddler](https://github.com/efeslab/fiddler), [arXiv 2402.07033](https://arxiv.org/abs/2402.07033)) — the CPU-executes-missed-experts trade (copy small activations instead of large weights) that the Native CPU miss path relies on.
- **KTransformers** ([kvcache-ai/ktransformers](https://github.com/kvcache-ai/ktransformers), Apache-2.0) — comparison point for heterogeneous MoE execution.

See [THIRD_PARTY.md](THIRD_PARTY.md) for the component-by-component inventory and [LIMITATIONS.md](LIMITATIONS.md) for what is not claimed.

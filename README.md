# PC Slotstream Prototype

> A phase-adaptive heterogeneous MoE runtime experiment for running models larger than VRAM on consumer discrete GPUs.

## What is this?

This staging area contains a llama.cpp patch, two benchmark runners, reproducible configurations, and an evidence ledger for a 176.94B-parameter sparse MoE model on a 16 GiB GPU plus 96 GiB system RAM.

The core model is simple:

- **RAM = model capacity layer**
- **VRAM = high-speed working set**
- **GPU = primary compute**
- **CPU = cold expert fallback**
- **Prefill and Decode use different policies**

## Why?

Coarse layer offload placed complete early layers on CPU and produced 6.538 tok/s. Structure-aware placement moved non-expert operators to GPU while keeping routed experts in RAM, raising Decode to about 15–16 tok/s. A fixed GPU expert working set then reached 18.753 tok/s.

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
| Measured | Cold Prefill, 16,363 tokens, 8K ubatch, 3 runs | **524.003 tok/s mean** |
| Measured, single run | Cold Prefill, 32,747 tokens, 8K ubatch | **617.342 tok/s** |

The 8 GiB Decode cache is the throughput winner, while the safe context-adaptive default uses 8 GiB through 8K context and 6 GiB at 16K/32K.

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

Increasing physical ubatch from 2K to 8K raised the same-prompt 16K Prefill path from 287.387 to a 524.003 tok/s three-run mean.

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
- `configs/`: public-format examples
- `data/results/`: sanitized aggregate CSVs
- `data/synthetic/`: synthetic fixtures
- `docs/`: build, hardware, benchmark, and design notes
- `posts/`: unpublished draft announcements
- `charts/`: generated figures sourced from CSV

Start with [docs/build.md](docs/build.md), then [docs/benchmark.md](docs/benchmark.md).

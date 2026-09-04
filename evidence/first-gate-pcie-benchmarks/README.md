# PC Slotstream: first CUDA feasibility gate

This repository measures the first question that decides whether a hierarchical
PC Slotstream is worth building:

```text
RAM expert store -> PCIe -> VRAM slot pool
                         ||
                 equivalent expert GEMM
```

It does not load or run the full model. The transfer payload exactly matches the
2,764,800-byte routed-expert record measured from Qwen3.8-Flash-Next-MLX-4bit.

## Result from this workstation

- one pinned expert: 0.1261 ms p50;
- sustained pinned H2D: 23.46 GB/s;
- top-10 known-next-batch overlap: 1.1718 ms/layer total, with 1.0402 ms of
  transfer still exposed;
- gate decision: continue with real routing traces and a real INT4 kernel, not
  with a full decoder yet.

See `RESULTS.md` for the evidence boundaries and cache-hit table.

## Build and run on this machine

```powershell
PowerShell -ExecutionPolicy Bypass -File .\scripts\build_and_run.ps1
```

The default matrix covers:

- VRAM pools: 1, 2, 4, 6, and 8 decimal GB;
- batches: 1, 2, 4, 8, 10, and 20 experts;
- pageable versus pinned host memory;
- one contiguous copy versus nine copies per expert;
- serialized copy/compute, double-buffered serialized execution, and two-stream
  overlap with CUDA events.

Raw results are written to:

```text
results/latest/metadata.txt
results/latest/copy_results.csv
results/latest/overlap_results.csv
```

## Result boundaries

The H2D numbers are real measurements on the current GPU. The overlap compute is
an FP16, equal-GEMM-FLOP surrogate. It does not include the production INT4
kernel or full decoder. See `docs/SLOTSTREAM_NOTES.md` before interpreting the
overlap percentage.

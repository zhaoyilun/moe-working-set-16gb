# Evidence Index

This directory holds the raw measurement evidence behind the aggregate CSVs in
`data/results/` and the reports in the repository root. Everything here was
produced on the single reference machine described in
[docs/hardware.md](../docs/hardware.md) unless a file says otherwise in its own
metadata.

## Layout

- `reports/` — the eight stage reports written as the research progressed
  (`README`, `RESULTS`, `STAGE1_FINAL`, `STAGE2_FINAL`, `DECODE_RUNTIME_FINAL`,
  `INTEGRATION_GATE_RESULTS`, `LONG_CONTEXT_128K_262K_FINAL`, `OVERNIGHT_RESULTS`).
  These are historical engineering notes, written in Chinese; numbers in the
  English `../RESULTS.md` and `../README.md` are taken from the same runs.
- `results/` — raw run outputs, organized by experiment gate:
  - `context-128k-gate/` — 32K–262K long-context gate: KV round-trip checks,
    long-context retrieval, decode at each context step, VRAM sampling runs,
    and the WDDM shared-memory paging collapse negative sample (q8-q4 KV with
    a 6 GiB pool at 262K context).
  - `context-short-gate/` — 2K/8K short-context decode repeats and
    three-way isolation diagnostics (KV profile / ubatch / protocol length).
  - `context-adaptive/` — the 2K–32K context × 4/6/8 GiB pool matrix behind
    the context-adaptive policy CSV.
  - `hybrid-decode-runtime/` — fixed-cache hybrid decode development: smoke
    runs, correctness checks against the Native CPU path, LRU plan exports,
    calibration and CUDA Graph isolation runs.
  - `prefill-gate-v1/` (and `v2`, `v3`) — Prefill ubatch scaling, critical-path
    analysis, prefix-cache runs, routing traces.
  - `prefill-grouped-gate/` — first round of the expert-staging prefill
    runtime (v4: eval-callback barrier, shared VRAM windows, pinned pack/DMA):
    A/B runs, staging runs, design notes (`GROUPED_RUNTIME_DESIGN.md`,
    `CODE_NOTES.md`), and routing captures. Headline: 16K prompt at physical
    ubatch 6144 = 720.8 tok/s (+27.3% over same-condition control, cosine
    0.9999928). The 52.8 MB `routing/prefill-16k.sstrace` is excluded
    (regenerable via `profiling-scripts/capture_routing_trace.ps1`).
  - `economic-adaptation/` — ubatch knee, workspace sizing, and stability
    confirmation runs.
  - `integration-gate/` — router-trace fixtures and smoke checks used to
    validate the trace format end to end.
  - `end-to-end-critical-path/` — nsys/CPU-attribution artifacts for the
    coarse-offload baseline.
  - `overnight/` — overnight soak results.
- `first-gate-pcie-benchmarks/` — the very first gate: PCIe bandwidth,
  host-to-device copy, and copy/compute overlap micro-benchmarks that sized
  the expert-transfer budget before any runtime work started.
- `profiling-scripts/` — the driver scripts behind the gates: 262K context
  gate, prefill staging runtime, prefill A/B comparisons, short-context
  repeats/diagnostics, and routing-trace capture.
- `routing-trace-tool/` — the routing-trace patch source (updated with the
  grouped-runtime fixes), trace-format definitions, and the analysis script
  used to inspect expert selection.

## Sanitization (what changed, and what did not)

- Local username paths (`C:\Users\<name>`, plus `AppData` paths) were replaced
  with the placeholder `user0` in text files. 297 files, 1193 occurrences.
  No measurement values, timestamps, or identifiers were altered.
- Excluded as regenerable or non-redistributable (about 27.8 GB):
  - `session.bin` / `*-session.bin` — serialized KV/session state, reproducible
    with the runners in `../tools/` from any local copy of the model.
  - `*.tokens` — tokenized prompt materializations produced by
    `../profiling` tools (the `context_prompt_tool` path).
  - `*.gguf`, quantized expert-weight slices (`rawint4*`, `*_qweight.bin`),
    activation dumps (`*.hidden.f32`) — model-derived binary data.
  - `*.exe`, `*.so`, `*.dll`, `*.nsys-rep`, build trees, `__pycache__`.
- Kept: all JSON results, stderr/stdout logs, CSVs, nsys-exported SQLite
  analysis databases, routing traces (`.sstrace`), stage reports, and metadata.

If a number in the aggregate CSVs looks surprising, the per-run JSON and its
`.stderr.log` sibling in this directory are the source of truth.

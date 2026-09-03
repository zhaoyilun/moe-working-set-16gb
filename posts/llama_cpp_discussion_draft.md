# llama.cpp Discussion draft

## Title

Experiment: fixed GPU expert working set with Native CPU miss execution for Qwen sparse MoE

## Summary

This is an engineering proposal and result package for tensor-structured sparse-MoE placement in llama.cpp. The experiment starts from a pinned revision and keeps all suitable non-routed tensors on CUDA while routed expert tensors remain in host memory. A fixed subset of packed experts is materialized in a GPU pool at model load; graph construction splits selected expert IDs into resident GPU hits and Native CPU misses, then merges weighted outputs.

Reference system: RTX 4080 Super 16 GiB, i9-14900KF, 96 GiB DDR5-4000, Windows, Qwen4exp 176.94B IQ4_XS-family GGUF.

## Measured results

- Native expert-only CPU, same Stage-1 path: 15.135 tok/s
- fixed cache 4 GiB: 16.096 tok/s
- fixed cache 6 GiB: 17.275 tok/s
- fixed cache 8 GiB: 18.753 tok/s

Each result above is a three-run 500-token mean at a restored 16K state. The 8 GiB path improved 23.91% over the same-path Native control.

CUDA Graph remained enabled. The 8 GiB Graph-OFF control measured 13.567 tok/s, so preserving graph reuse is central to the design.

## Dynamic-promotion result

I added an experimental packed 48×10 route observation tensor and adjacent-token promotion threshold. Thresholds 1–4 increased hit rate up to 61.78%, but all regressed throughput. Promotion moved up to 114.45 MB/token and added 22.89 ms/token inside the promotion routine. The default remains fixed-cache mode (`threshold=-1`).

## Potential upstream pieces

1. expert tensor placement override
2. fixed packed expert working set
3. public hit/miss/fallback counters
4. packed route observation for diagnostics

The repository package includes the patch, runners, raw aggregate CSVs, exact base revision, and limitations. Before a PR, I would like feedback on API shape, model generality, and where a fixed expert working-set policy belongs in llama.cpp.

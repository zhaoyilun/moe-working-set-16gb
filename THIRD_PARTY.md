# Third-Party Components

## llama.cpp / ggml

- Role: model loader, graph builder, CPU/CUDA backends, CUDA Graph, state serialization.
- Base revision used by the experiment: `4e97ac86ebe2c4cb8212d98d2641ad6768810896`.
- License: MIT.
- Distribution form here: patch only.

## Slotstream

- Role: prior-art inspiration for expert working-set streaming.
- Upstream project examined locally; no source copied into the current patch.
- License observed in the checked-out reference: MIT, copyright Carlos Galarza, 2026.

## KTransformers

- Role: comparison point for heterogeneous MoE execution and future grouped-expert work.
- No source copied into the current patch.
- License observed in the checked-out reference: Apache License 2.0.

## NVIDIA tooling

- CUDA toolkit builds and runs the CUDA backend.
- Nsight Systems was used in earlier profiling attempts.
- SDK binaries and profiler output are excluded from this staging area.

# Third-Party Components

## llama.cpp / ggml

- Role: model loader, graph builder, CPU/CUDA backends, CUDA Graph, state serialization.
- Base revision used by the experiment: `4e97ac86ebe2c4cb8212d98d2641ad6768810896`.
- License: MIT.
- Distribution form here: patch only.

## Slotstream

- Role: prior-art inspiration for expert-slot working-set streaming; the
  "slot" vocabulary in this project's internal working names comes from there.
- Upstream: https://github.com/carloslfu/slotstream (Carlos Galarza, 2026) —
  a Swift/Metal engine that runs the same Qwen3.8-Flash-Next model on a 48 GB
  Mac by streaming expert weights from SSD. The present work takes the
  complementary route: system RAM as the capacity layer, VRAM as the working
  set, on Windows/CUDA via a llama.cpp patch.
- Upstream project examined locally; no source copied into the current patch.
- License observed in the checked-out reference: MIT, copyright Carlos Galarza, 2026.

## Fiddler

- Role: prior art for executing MoE experts on the CPU while keeping the rest
  of the layer on the GPU — its core observation (copy small activations to
  CPU instead of copying large expert weights to GPU) is the same trade this
  project's "Native CPU miss path" relies on.
- Upstream: https://github.com/efeslab/fiddler, paper
  https://arxiv.org/abs/2402.07033.
- Upstream repository cloned and read during research; no source copied into
  the current patch.
- License: upstream repository carries its own license terms (Apache-2.0
  family); none of its source is distributed here.

## KTransformers

- Role: comparison point for heterogeneous MoE execution and future grouped-expert work.
- Upstream: https://github.com/kvcache-ai/ktransformers.
- No source copied into the current patch.
- License observed in the checked-out reference: Apache License 2.0.

## NVIDIA tooling

- CUDA toolkit builds and runs the CUDA backend.
- Nsight Systems was used in earlier profiling attempts.
- SDK binaries and profiler output are excluded from this staging area.

## Model

- Qwen3.8-Flash-Next GGUF (IQ4_XS family), 176.94B total parameters.
- Model weights are not distributed here; users supply a locally obtained copy.
  The GGUF quantization carries the upstream model's own terms.

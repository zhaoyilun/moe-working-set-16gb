# License Compatibility

## Current composition

| Component | License | Included here? | Treatment |
|---|---|---:|---|
| llama.cpp / ggml | MIT | Patch against source | Preserve upstream copyright and MIT terms |
| Slotstream reference project | MIT | No copied source | Conceptual/reference attribution only |
| KTransformers | Apache-2.0 | No copied source | Comparative implementation reference only |
| NVIDIA CUDA / Nsight | Proprietary SDK/tool terms | Build/runtime dependency | No redistribution in this staging area |
| Qwen model and GGUF quantization | Separate model terms | No weights | User supplies a locally obtained model |

## Recommended release form

For a llama.cpp fork, retain the upstream MIT license and notices. For standalone benchmark runners and original glue code, MIT is the simplest compatible choice, subject to the maintainer's final copyright and license decision.

No Apache-2.0 source is present in the current patch, so an Apache NOTICE file is not required by the present file set. If KTransformers or CUTLASS-derived source is added later, repeat this audit at file level and include all required notices.

This document is an engineering inventory rather than a legal opinion.

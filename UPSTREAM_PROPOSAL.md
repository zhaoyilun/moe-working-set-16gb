# Upstream Proposal Draft

## Candidate upstreamable pieces

1. Tensor-level expert placement override for Qwen sparse MoE models.
2. Fixed packed GPU expert working set with Native CPU miss execution.
3. Public runtime counters for cache hit, miss, CPU fallback, and transfer volume.
4. A packed route-ID observation tensor that avoids hundreds of separate readbacks.

## Pieces that should remain experimental

- adjacent-token dynamic promotion
- per-token scheduler synchronization
- fixed policies tuned to one 16 GiB card
- model-specific benchmark harnesses

## Evidence requested before an upstream PR

- multi-model correctness
- Linux and Windows builds
- at least two GPU generations
- multi-prompt Decode stability
- API naming review
- proof that the default-disabled path has negligible overhead

The present staging area is a discussion package, not a submitted PR.

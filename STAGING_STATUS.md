# OSS Staging Status

## Scope complete

- patched llama.cpp runtime snapshot
- fixed-cache Decode and selective-H2D experiment
- Prefix-state and Decode runners
- aggregate performance/context/VRAM CSVs
- context-adaptive policy
- Prefill 500+ result package
- build and benchmark instructions
- license and third-party inventory
- publication drafts
- generated charts

## Publication state

This directory is local preparation only. It has no configured public remote and no publication action has been performed.

## Clean build verification

- Environment: Windows x64 Native Tools, Ninja, CUDA 12.3
- CUDA compiler override: `--allow-unsupported-compiler` plus the matching MSVC STL version define
- Result: **PASS**, 360/360 build actions completed
- Built targets: `slotstream-prefix-cache.exe`, `slotstream-hybrid-decode.exe`
- Build outputs stay outside this staging repository and are excluded from the package.

## Remaining maintainer decisions

1. choose repository name and ownership
2. confirm the license for original standalone files
3. choose safe-default versus benchmark-max README emphasis
4. run a second-machine reproduction
5. accept a multi-prompt Prefill numerical criterion
6. select and edit publication drafts

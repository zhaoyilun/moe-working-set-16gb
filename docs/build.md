# Build

## 1. Prepare the pinned llama.cpp checkout

```powershell
git clone https://github.com/ggml-org/llama.cpp LLAMA_DIR
git -C LLAMA_DIR checkout 4e97ac86ebe2c4cb8212d98d2641ad6768810896
git -C LLAMA_DIR apply PATCH_DIR/llama.cpp-hybrid-expert-cache.patch
```

`LLAMA_DIR` and `PATCH_DIR` are placeholders. Use local paths on the target machine.

## 2. Configure the benchmark tools

Windows with Visual Studio 2022, CUDA, and Ninja: open an **x64 Native Tools** shell first, then run:

```powershell
cmake -G Ninja -S TOOLS_DIR -B BUILD_DIR `
  -DLLAMA_CPP_DIR=LLAMA_DIR `
  -DGGML_CUDA=ON `
  '-DCMAKE_CUDA_FLAGS=--allow-unsupported-compiler -Xcompiler=/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH' `
  -DCMAKE_BUILD_TYPE=Release
cmake --build BUILD_DIR -j 10
```

The CUDA flag above is part of the measured Windows toolchain: CUDA 12.3 plus the current Visual Studio 17.14 host compiler. Use a supported host compiler instead when preparing a clean public build.

Linux with a CUDA-capable compiler toolchain:

```bash
cmake -G Ninja -S TOOLS_DIR -B BUILD_DIR \
  -DLLAMA_CPP_DIR=LLAMA_DIR \
  -DGGML_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build BUILD_DIR -j
```

## 3. Expected binaries

- `slotstream-prefix-cache`: cold Prefill, state save/load, and roundtrip checks
- `slotstream-hybrid-decode`: fixed-cache and experimental promotion Decode runs

## 4. Patch defaults

- `hybrid_expert_cache_promote_threshold = -1`: fixed-cache default
- `0`: stats-only packed route readback
- `1..10`: experimental adjacent-token promotion threshold

The benchmark result uses `-1` for normal Decode performance. Stats-only and promotion modes have an extra per-token synchronization boundary.

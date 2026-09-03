param(
    [Parameter(Mandatory = $true)][string] $LlamaDir,
    [Parameter(Mandatory = $true)][string] $Model,
    [Parameter(Mandatory = $true)][string] $PromptFile,
    [Parameter(Mandatory = $true)][string] $CachePlan,
    [string] $BuildDir = (Join-Path $PSScriptRoot '..\build'),
    [string] $OutputDir = (Join-Path $PSScriptRoot '..\reproduction-output'),
    [int] $PromptTokens = 16363,
    [int] $ContextSize = 17408,
    [int] $Threads = 24,
    [int] $Ubatch = 8192
)

$ErrorActionPreference = 'Stop'
$toolsDir = (Resolve-Path (Join-Path $PSScriptRoot '..\tools')).Path
New-Item -ItemType Directory -Force -Path $BuildDir, $OutputDir | Out-Null

if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    throw 'Run this script from a Visual Studio x64 Native Tools shell'
}

cmake -G Ninja -S $toolsDir -B $BuildDir `
    -DLLAMA_CPP_DIR=$LlamaDir `
    -DGGML_CUDA=ON `
    '-DCMAKE_CUDA_FLAGS=--allow-unsupported-compiler -Xcompiler=/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH' `
    -DCMAKE_BUILD_TYPE=Release
cmake --build $BuildDir -j 10

$prefixRunner = Get-ChildItem -LiteralPath $BuildDir -Recurse -File -Filter 'slotstream-prefix-cache.exe' |
    Select-Object -First 1 -ExpandProperty FullName
$decodeRunner = Get-ChildItem -LiteralPath $BuildDir -Recurse -File -Filter 'slotstream-hybrid-decode.exe' |
    Select-Object -First 1 -ExpandProperty FullName

if (-not $prefixRunner -or -not $decodeRunner) {
    throw 'Expected benchmark binaries are missing from BUILD_DIR'
}

$state = Join-Path $OutputDir 'session.bin'
& $prefixRunner `
    -m $Model `
    --prompt-file $PromptFile `
    --state-file $state `
    --output-json (Join-Path $OutputDir 'prefill-roundtrip.json') `
    --prompt-tokens $PromptTokens `
    --measured-tokens 1 `
    --verify-tokens 8 `
    -c $ContextSize `
    -t $Threads `
    --n-batch $Ubatch `
    --n-ubatch $Ubatch

for ($run = 1; $run -le 3; $run++) {
    & $decodeRunner `
        -m $Model `
        --state-file $state `
        --output-json (Join-Path $OutputDir ("decode-run-{0}.json" -f $run)) `
        --hybrid-cache-plan $CachePlan `
        --promote-threshold -1 `
        --verify-tokens 1 `
        --warmup-tokens 100 `
        --measured-tokens 500 `
        -c $ContextSize `
        -t $Threads `
        --n-batch 2048 `
        --n-ubatch 512
}

Get-ChildItem -LiteralPath $OutputDir -File -Filter '*.json' | Select-Object Name, Length, LastWriteTime

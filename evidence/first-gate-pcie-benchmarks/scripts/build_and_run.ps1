param(
    [string]$BuildDirectory = "build",
    [string]$ResultDirectory = "results/latest",
    [string]$PoolGB = "1,2,4,6,8",
    [string]$Batches = "1,2,4,8,10,20"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VsDevCmd = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"

if (-not (Test-Path -LiteralPath $VsDevCmd)) {
    throw "Visual Studio developer environment was not found at $VsDevCmd"
}

Push-Location $ProjectRoot
try {
    $configure = 'cmake -S . -B "{0}" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler -Xcompiler=/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"' -f $BuildDirectory
    $build = 'cmake --build "{0}" --parallel' -f $BuildDirectory
    $command = '"{0}" -arch=x64 && {1} && {2}' -f $VsDevCmd, $configure, $build
    cmd.exe /d /s /c $command
    if ($LASTEXITCODE -ne 0) { throw "Build failed with exit code $LASTEXITCODE" }

    $exe = Join-Path $BuildDirectory "slotstream_bench.exe"
    & $exe --pool-gb $PoolGB --batches $Batches --output-dir $ResultDirectory
    if ($LASTEXITCODE -ne 0) { throw "Benchmark failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
}

param(
    [Parameter(Mandatory = $true)] [ValidateSet("retrieval", "decode", "vram")] [string] $Phase,
    [Parameter(Mandatory = $true)] [ValidateSet("q4-q4", "q8-q4")] [string] $KvProfile,
    [Parameter(Mandatory = $true)] [string] $Exe,
    [ValidateSet("4", "6", "8")] [string] $PoolGib = "6",
    [string] $Model = "E:\AI\Models\Qwen3.8-Flash-Next-UD-IQ4_XS\Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$gate = Join-Path $root "results\context-128k-gate"
$promptFile = Join-Path $gate "context-prompts\context_262k.txt"
$plans = Join-Path $root "results\hybrid-decode-runtime\plans"
$cacheK = if ($KvProfile -eq "q4-q4") { "q4_0" } else { "q8_0" }
$cacheV = "q4_0"
$stateFile = Join-Path $gate "262k\kv-roundtrip\$KvProfile\session.bin"

function Invoke-WithSampling {
    param([string] $ExePath, [string[]] $Arguments, [string] $Stdout, [string] $Stderr, [string] $SamplesCsv)
    $process = Start-Process -FilePath $ExePath -ArgumentList $Arguments -PassThru `
        -WindowStyle Hidden -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr
    $logicalProcessors = [Environment]::ProcessorCount
    $rows = [System.Collections.Generic.List[object]]::new()
    $stopwatch = [Diagnostics.Stopwatch]::StartNew()
    $lastWall = $stopwatch.Elapsed.TotalSeconds
    $lastCpu = 0.0
    try {
        $process.Refresh()
        $lastCpu = $process.TotalProcessorTime.TotalSeconds
    } catch {}
    while (-not $process.HasExited) {
        Start-Sleep -Milliseconds 1000
        try {
            $process.Refresh()
            $now = $stopwatch.Elapsed.TotalSeconds
            $cpuNow = $process.TotalProcessorTime.TotalSeconds
            $dt = [Math]::Max(0.001, $now - $lastWall)
            $cpuPercent = 100.0 * ($cpuNow - $lastCpu) / $dt / $logicalProcessors
            $lastWall = $now
            $lastCpu = $cpuNow
            $gpuRaw = & nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>$null
            $gpu = @(0.0, 0.0, 0.0)
            if ($gpuRaw) {
                $gpu = $gpuRaw.Trim().Split(",") | ForEach-Object { [double]($_.Trim()) }
            }
            $rows.Add([pscustomobject]@{
                elapsed_s           = [Math]::Round($now, 3)
                cpu_percent         = [Math]::Round($cpuPercent, 2)
                working_set_gib     = [Math]::Round($process.WorkingSet64 / 1GB, 3)
                gpu_util_percent    = $gpu[0]
                gpu_mem_used_mib    = $gpu[1]
                gpu_mem_total_mib   = $gpu[2]
            })
        } catch {}
    }
    $process.WaitForExit()
    $process.Refresh()
    $exitCode = $process.ExitCode
    $rows | Export-Csv -LiteralPath $SamplesCsv -NoTypeInformation -Encoding utf8
    return $exitCode
}

if ($Phase -eq "retrieval") {
    $outDir = Join-Path $gate "262k\retrieval\$KvProfile"
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    $json = Join-Path $outDir "load-only-384.json"
    $stdout = Join-Path $outDir "stdout.log"
    $stderr = Join-Path $outDir "stderr.log"
    $arguments = @(
        "-m", $Model,
        "--prompt-file", $promptFile,
        "--state-file", $stateFile,
        "--output-json", $json,
        "--prompt-tokens", "261099",
        "--measured-tokens", "384",
        "--n-batch", "2048",
        "--n-ubatch", "16",
        "--cache-type-k", $cacheK,
        "--cache-type-v", $cacheV,
        "-c", "262144",
        "-t", "24",
        "-ngl", "-2",
        "--load-only"
    )
    $p = Start-Process -FilePath $Exe -ArgumentList $arguments -PassThru `
        -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    $p.WaitForExit()
    $p.Refresh()
    if ($p.ExitCode -ne 0) {
        Get-Content -LiteralPath $stderr -Tail 60
        throw "retrieval $KvProfile failed with exit code $($p.ExitCode)"
    }
    Get-Content -LiteralPath $json -Raw | ConvertFrom-Json | Select-Object mode, prompt_tokens, cache_load_ms, restored_decode_tok_s, restored_decode_p50_ms, warm_development_iteration_ms, fresh_process_iteration_ms | ConvertTo-Json
}
elseif ($Phase -eq "decode") {
    $outDir = Join-Path $gate "262k\decode-adaptive-u16"
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    $name = "$KvProfile-pool$($PoolGib)g-run-1"
    $json = Join-Path $outDir "$name.json"
    $logits = Join-Path $outDir "$name.f32"
    $stdout = Join-Path $outDir "$name.stdout.log"
    $stderr = Join-Path $outDir "$name.stderr.log"
    $resources = Join-Path $outDir "$name.resources.csv"
    if (-not (Test-Path $resources)) { New-Item -ItemType File -Path $resources | Out-Null }
    $arguments = @(
        "-m", $Model,
        "--state-file", $stateFile,
        "--output-json", $json,
        "--logits-file", $logits,
        "--hybrid-cache-plan", (Join-Path $plans "lru-$($PoolGib)-gib.csv"),
        "--input-tokens-file", (Join-Path $plans "canonical-16k-native-continuation-601.tokens"),
        "--verify-tokens", "1",
        "--warmup-tokens", "50",
        "--measured-tokens", "200",
        "-c", "262144",
        "--n-batch", "2048",
        "--n-ubatch", "16",
        "--cache-type-k", $cacheK,
        "--cache-type-v", $cacheV
    )
    $p = Start-Process -FilePath $Exe -ArgumentList $arguments -PassThru `
        -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    $p.WaitForExit()
    $p.Refresh()
    if ($p.ExitCode -ne 0) {
        Get-Content -LiteralPath $stderr -Tail 60
        throw "decode $KvProfile failed with exit code $($p.ExitCode)"
    }
    Get-Content -LiteralPath $json -Raw | ConvertFrom-Json | Select-Object mode, state_tokens, warmup_tokens, measured_tokens, decode_tok_s, decode_p50_ms, decode_p95_ms | ConvertTo-Json
}
else {
    $outDir = Join-Path $gate "262k\vram-adaptive-u16"
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    $name = "$KvProfile-pool$($PoolGib)g-run-1"
    $json = Join-Path $outDir "$name.json"
    $logits = Join-Path $outDir "$name.f32"
    $stdout = Join-Path $outDir "$name.stdout.log"
    $stderr = Join-Path $outDir "$name.stderr.log"
    $resources = Join-Path $outDir "$name.resources.csv"
    $arguments = @(
        "-m", $Model,
        "--state-file", $stateFile,
        "--output-json", $json,
        "--logits-file", $logits,
        "--hybrid-cache-plan", (Join-Path $plans "lru-$($PoolGib)-gib.csv"),
        "--input-tokens-file", (Join-Path $plans "canonical-16k-native-continuation-601.tokens"),
        "--verify-tokens", "1",
        "--warmup-tokens", "5",
        "--measured-tokens", "5",
        "-c", "262144",
        "--n-batch", "2048",
        "--n-ubatch", "16",
        "--cache-type-k", $cacheK,
        "--cache-type-v", $cacheV
    )
    $exitCode = Invoke-WithSampling -ExePath $Exe -Arguments $arguments -Stdout $stdout -Stderr $stderr -SamplesCsv $resources
    if ($exitCode -ne 0) {
        Get-Content -LiteralPath $stderr -Tail 60
        throw "vram $KvProfile failed with exit code $exitCode"
    }
    $samples = Import-Csv -LiteralPath $resources
    [pscustomobject]@{
        samples            = $samples.Count
        peak_gpu_mem_mib   = ($samples | Measure-Object gpu_mem_used_mib -Maximum).Maximum
        peak_working_gib   = ($samples | Measure-Object working_set_gib -Maximum).Maximum
        gpu_mem_total_mib  = ($samples | Select-Object -Last 1).gpu_mem_total_mib
    } | ConvertTo-Json
}

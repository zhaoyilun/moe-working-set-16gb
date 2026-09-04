param(
    [Parameter(Mandatory = $true)] [string] $Exe,
    [Parameter(Mandatory = $true)] [string] $Model,
    [Parameter(Mandatory = $true)] [string] $PromptFile,
    [Parameter(Mandatory = $true)] [string] $OutputDirectory,
    [int] $PromptTokens = 16363,
    [int] $ContextSize = 17408,
    [int] $NBatch = 2048,
    [int] $NUBatch = 512,
    [ValidateSet("f16", "q8_0", "q4_0")] [string] $CacheTypeK = "f16",
    [ValidateSet("f16", "q8_0", "q4_0")] [string] $CacheTypeV = "f16",
    [int] $Threads = 24,
    [int] $MeasuredTokens = 100,
    [int] $VerifyTokens = 32,
    [string] $Name = "16k"
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null

$stdout = Join-Path $OutputDirectory "$Name-console.stdout.log"
$stderr = Join-Path $OutputDirectory "$Name-console.stderr.log"
$result = Join-Path $OutputDirectory "$Name-result.json"
$state = Join-Path $OutputDirectory "$Name-session.bin"
$samples = Join-Path $OutputDirectory "$Name-resource-samples.csv"

foreach ($path in @($stdout, $stderr, $result, $state, $samples)) {
    if (Test-Path -LiteralPath $path) {
        Remove-Item -LiteralPath $path -Force
    }
}

$arguments = @(
    "-m", $Model,
    "--prompt-file", $PromptFile,
    "--state-file", $state,
    "--output-json", $result,
    "--prompt-tokens", "$PromptTokens",
    "--measured-tokens", "$MeasuredTokens",
    "--verify-tokens", "$VerifyTokens",
    "--n-batch", "$NBatch",
    "--n-ubatch", "$NUBatch",
    "--cache-type-k", $CacheTypeK,
    "--cache-type-v", $CacheTypeV,
    "-c", "$ContextSize",
    "-t", "$Threads",
    "-ngl", "-2"
)

$process = Start-Process -FilePath $Exe -ArgumentList $arguments -PassThru `
    -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr

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
            private_gib         = [Math]::Round($process.PrivateMemorySize64 / 1GB, 3)
            gpu_util_percent    = $gpu[0]
            gpu_mem_used_mib    = $gpu[1]
            gpu_mem_total_mib   = $gpu[2]
        })
    } catch {}
}

$process.WaitForExit()
$rows | Export-Csv -LiteralPath $samples -NoTypeInformation -Encoding utf8

if ($process.ExitCode -ne 0) {
    Get-Content -LiteralPath $stderr -Tail 100
    throw "Prefix-cache run failed with exit code $($process.ExitCode)"
}

$measurement = Get-Content -LiteralPath $result -Raw | ConvertFrom-Json
$summary = [pscustomobject]@{
    exit_code                    = $process.ExitCode
    sample_count                 = $rows.Count
    peak_working_set_gib         = ($rows | Measure-Object working_set_gib -Maximum).Maximum
    peak_private_gib             = ($rows | Measure-Object private_gib -Maximum).Maximum
    peak_gpu_mem_used_mib        = ($rows | Measure-Object gpu_mem_used_mib -Maximum).Maximum
    mean_gpu_util_percent        = [Math]::Round(($rows | Measure-Object gpu_util_percent -Average).Average, 2)
    prompt_tokens                = $measurement.prompt_tokens
    n_batch                      = $measurement.n_batch
    n_ubatch                     = $measurement.n_ubatch
    cache_type_k                 = $measurement.cache_type_k
    cache_type_v                 = $measurement.cache_type_v
    cold_prefill_s               = [Math]::Round($measurement.cold_prefill_ms_excluding_save / 1000, 3)
    cold_prefill_tok_s           = [Math]::Round($measurement.cold_prefill_tok_s, 3)
    cache_save_s                 = [Math]::Round($measurement.cache_save_ms / 1000, 3)
    cache_size_gib               = [Math]::Round($measurement.cache_file_bytes / 1GB, 3)
    cache_load_s                 = [Math]::Round($measurement.cache_load_ms / 1000, 3)
    replay_ms                    = [Math]::Round($measurement.last_token_replay_ms, 3)
    warm_ready_s                 = [Math]::Round($measurement.warm_restore_to_first_token_ms / 1000, 3)
    prompt_tokens_equal          = $measurement.prompt_tokens_equal
    first_token_equal            = $measurement.first_token_equal
    generated_tokens_equal       = $measurement.generated_tokens_equal
    logits_rmse                  = $measurement.logits_rmse
    logits_max_abs               = $measurement.logits_max_abs
    logits_cosine                = $measurement.logits_cosine
    restored_decode_tok_s        = [Math]::Round($measurement.restored_decode_tok_s, 3)
}

$summary | ConvertTo-Json -Depth 4

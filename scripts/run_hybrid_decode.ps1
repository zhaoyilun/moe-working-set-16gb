param(
    [Parameter(Mandatory = $true)] [string] $Exe,
    [Parameter(Mandatory = $true)] [string] $Model,
    [Parameter(Mandatory = $true)] [string] $StateFile,
    [Parameter(Mandatory = $true)] [string] $OutputDirectory,
    [Parameter(Mandatory = $true)] [string] $Name,
    [string] $CachePlan = "",
    [string] $InputTokensFile = "",
    [int] $Runs = 3,
    [int] $VerifyTokens = 32,
    [int] $WarmupTokens = 100,
    [int] $MeasuredTokens = 500,
    [int] $ContextSize = 17408,
    [int] $Threads = 24,
    [int] $NBatch = 2048,
    [int] $NUbatch = 16,
    [ValidateSet("f16", "q8_0", "q4_0")] [string] $CacheTypeK = "f16",
    [ValidateSet("f16", "q8_0", "q4_0")] [string] $CacheTypeV = "f16",
    [int] $SampleIntervalMs = 2000,
    [switch] $DisableCudaGraphs,
    [switch] $NoResourceSampling
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null

$oldGraphSetting = $env:GGML_CUDA_DISABLE_GRAPHS
if ($DisableCudaGraphs) {
    $env:GGML_CUDA_DISABLE_GRAPHS = "1"
} else {
    Remove-Item Env:GGML_CUDA_DISABLE_GRAPHS -ErrorAction SilentlyContinue
}

try {
    for ($run = 1; $run -le $Runs; $run++) {
        $prefix = Join-Path $OutputDirectory "$Name-run-$run"
        $json = "$prefix.json"
        $logits = "$prefix.f32"
        $stdout = "$prefix.stdout.log"
        $stderr = "$prefix.stderr.log"
        $samples = "$prefix.resources.csv"

        $arguments = @(
            "-m", $Model,
            "--state-file", $StateFile,
            "--output-json", $json,
            "--logits-file", $logits,
            "--verify-tokens", "$VerifyTokens",
            "--warmup-tokens", "$WarmupTokens",
            "--measured-tokens", "$MeasuredTokens",
            "-c", "$ContextSize",
            "-t", "$Threads",
            "--n-batch", "$NBatch",
            "--n-ubatch", "$NUbatch",
            "--cache-type-k", $CacheTypeK,
            "--cache-type-v", $CacheTypeV,
            "-ngl", "-2"
        )
        if ($CachePlan) {
            $arguments += @("--hybrid-cache-plan", $CachePlan)
        }
        if ($InputTokensFile) {
            $arguments += @("--input-tokens-file", $InputTokensFile)
        }

        $process = Start-Process -FilePath $Exe -ArgumentList $arguments -PassThru -WindowStyle Hidden `
            -RedirectStandardOutput $stdout -RedirectStandardError $stderr
        $logicalProcessors = [Environment]::ProcessorCount
        $rows = [System.Collections.Generic.List[object]]::new()
        $watch = [Diagnostics.Stopwatch]::StartNew()
        $lastWall = 0.0
        $lastCpu = 0.0
        if ($NoResourceSampling) {
            $process.WaitForExit()
        }
        while (-not $NoResourceSampling -and -not $process.HasExited) {
            Start-Sleep -Milliseconds $SampleIntervalMs
            try {
                $process.Refresh()
                $now = $watch.Elapsed.TotalSeconds
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
                    elapsed_s = [Math]::Round($now, 3)
                    cpu_percent = [Math]::Round($cpuPercent, 2)
                    working_set_gib = [Math]::Round($process.WorkingSet64 / 1GB, 3)
                    gpu_util_percent = $gpu[0]
                    gpu_mem_used_mib = $gpu[1]
                    gpu_mem_total_mib = $gpu[2]
                })
            } catch {}
        }
        $process.WaitForExit()
        $rows | Export-Csv -LiteralPath $samples -NoTypeInformation -Encoding utf8
        if ($process.ExitCode -ne 0) {
            Get-Content -LiteralPath $stderr -Tail 100
            throw "$Name run $run exited with $($process.ExitCode)"
        }
        $measurement = Get-Content -LiteralPath $json -Raw | ConvertFrom-Json
        [pscustomobject]@{
            run = $run
            decode_tok_s = $measurement.decode_tok_s
            decode_p50_ms = $measurement.decode_p50_ms
            decode_p95_ms = $measurement.decode_p95_ms
            peak_gpu_mem_used_mib = ($rows | Measure-Object gpu_mem_used_mib -Maximum).Maximum
            mean_gpu_util_percent = [Math]::Round(($rows | Measure-Object gpu_util_percent -Average).Average, 3)
            mean_cpu_percent = [Math]::Round(($rows | Measure-Object cpu_percent -Average).Average, 3)
            peak_working_set_gib = ($rows | Measure-Object working_set_gib -Maximum).Maximum
        } | ConvertTo-Json -Compress
    }
} finally {
    if ($null -eq $oldGraphSetting) {
        Remove-Item Env:GGML_CUDA_DISABLE_GRAPHS -ErrorAction SilentlyContinue
    } else {
        $env:GGML_CUDA_DISABLE_GRAPHS = $oldGraphSetting
    }
}

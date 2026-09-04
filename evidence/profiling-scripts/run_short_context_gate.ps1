param(
    [Parameter(Mandatory = $true)] [ValidateSet("2k", "8k")] [string] $Context,
    [Parameter(Mandatory = $true)] [string] $PrefixExe,
    [Parameter(Mandatory = $true)] [string] $DecodeExe,
    [string] $Model = "E:\AI\Models\Qwen3.8-Flash-Next-UD-IQ4_XS\Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf"
)

$root = Split-Path -Parent $PSScriptRoot
$gate = Join-Path $root "results\context-short-gate"
$promptFile = Join-Path $root "results\context-128k-gate\context-prompts\context_32k.txt"
$plans = Join-Path $root "results\hybrid-decode-runtime\plans"
$tokens = Join-Path $plans "canonical-16k-native-continuation-601.tokens"

function Invoke-Runner {
    param([string] $ExePath, [string[]] $Arguments, [string] $Directory, [string] $Name)
    $stdout = Join-Path $Directory "$Name.stdout.log"
    $stderr = Join-Path $Directory "$Name.stderr.log"
    $p = Start-Process -FilePath $ExePath -ArgumentList $Arguments -PassThru `
        -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    $p.WaitForExit()
    $p.Refresh()
    return $p.ExitCode
}

$cfg = @{ "2k" = @{ T = 2027; C = 3072 }; "8k" = @{ T = 8171; C = 9216 } }[$Context]
$outDir = Join-Path $gate $Context
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

# Phase 1: build q8-q4 state via roundtrip (also verifies save/load fidelity)
$stateDir = Join-Path $outDir "kv-roundtrip-q8-q4"
New-Item -ItemType Directory -Force -Path $stateDir | Out-Null
$state = Join-Path $stateDir "session.bin"
$rtJson = Join-Path $stateDir "roundtrip.json"
$rc = Invoke-Runner $PrefixExe @(
    "-m", $Model,
    "--prompt-file", $promptFile,
    "--state-file", $state,
    "--output-json", $rtJson,
    "--prompt-tokens", "$($cfg.T)",
    "--measured-tokens", "1",
    "--verify-tokens", "8",
    "--n-batch", "2048", "--n-ubatch", "2048",
    "--cache-type-k", "q8_0", "--cache-type-v", "q4_0",
    "-c", "$($cfg.C)", "-t", "24", "-ngl", "-2"
) $stateDir "roundtrip"
if ($rc -ne 0 -or -not (Test-Path $rtJson)) { throw "state build failed rc=$rc" }
$rt = Get-Content -LiteralPath $rtJson -Raw | ConvertFrom-Json
"STATE $Context tokens=$($rt.prompt_tokens) prefill_tok_s=$([Math]::Round($rt.cold_prefill_tok_s,1)) cosine=$($rt.logits_cosine) size_mib=$([Math]::Round($rt.cache_file_bytes/1MB,1))"

# Phase 2: hybrid decode, 8 GiB pool, 50 warmup + 200 measured, ubatch 16, no sampling
$decDir = Join-Path $outDir "decode-8g-u16"
New-Item -ItemType Directory -Force -Path $decDir | Out-Null
$decJson = Join-Path $decDir "result.json"
$rc = Invoke-Runner $DecodeExe @(
    "-m", $Model,
    "--state-file", $state,
    "--output-json", $decJson,
    "--input-tokens-file", $tokens,
    "--hybrid-cache-plan", (Join-Path $plans "lru-8-gib.csv"),
    "--promote-threshold", "-1",
    "--verify-tokens", "1",
    "--warmup-tokens", "50",
    "--measured-tokens", "200",
    "-c", "$($cfg.C)", "-t", "24", "-ngl", "-2",
    "--n-batch", "2048", "--n-ubatch", "16",
    "--cache-type-k", "q8_0", "--cache-type-v", "q4_0"
) $decDir "run"
if ($rc -ne 0 -or -not (Test-Path $decJson)) { throw "decode failed rc=$rc" }
$d = Get-Content -LiteralPath $decJson -Raw | ConvertFrom-Json
"DECODE $Context tok_s=$([Math]::Round($d.decode_tok_s,3)) p50_ms=$([Math]::Round($d.decode_p50_ms,1)) p95_ms=$([Math]::Round($d.decode_p95_ms,1)) first=$($d.first_token) graph=$($d.cuda_graphs)"

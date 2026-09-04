param(
    [Parameter(Mandatory = $true)] [ValidateSet("2k", "8k")] [string] $Context,
    [int] $Runs = 3
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$exe = 'C:\Users\user0\Documents\Codex\2026-09-03\chatgpt-conversation-6a98f719-c4cc-83ec-8dd8\work\build\slotstream-tools-262k\slotstream-hybrid-decode.exe'
$model = 'E:\AI\Models\Qwen3.8-Flash-Next-UD-IQ4_XS\Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf'
$plans = Join-Path $root 'results\hybrid-decode-runtime\plans'
$cfg = @{ "2k" = @{ C = 3072 }; "8k" = @{ C = 9216 } }[$Context]
$outDir = Join-Path $root "results\context-short-gate\$Context\decode-8g-u16-repeat"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$tokRates = @()
for ($i = 1; $i -le $Runs; $i++) {
    $json = Join-Path $outDir "run-$i.json"
    & $exe -m $model `
        --state-file (Join-Path $root "results\context-short-gate\$Context\kv-roundtrip-q8-q4\session.bin") `
        --output-json $json `
        --input-tokens-file (Join-Path $plans 'canonical-16k-native-continuation-601.tokens') `
        --hybrid-cache-plan (Join-Path $plans 'lru-8-gib.csv') `
        --promote-threshold -1 `
        --verify-tokens 1 --warmup-tokens 50 --measured-tokens 200 `
        -c $cfg.C -t 24 -ngl -2 --n-batch 2048 --n-ubatch 16 `
        --cache-type-k q8_0 --cache-type-v q4_0 2> (Join-Path $outDir "run-$i.stderr.log") | Out-Null
    $d = Get-Content -LiteralPath $json -Raw | ConvertFrom-Json
    $tokRates += $d.decode_tok_s
    'run {0}: tok_s={1:N3} p50={2:N1}ms p95={3:N1}ms' -f $i, $d.decode_tok_s, $d.decode_p50_ms, $d.decode_p95_ms
}
$mean = ($tokRates | Measure-Object -Average).Average
$std = [Math]::Sqrt((($tokRates | ForEach-Object { [Math]::Pow($_ - $mean, 2) } | Measure-Object -Sum).Sum) / ($Runs - 1))
'SUMMARY {0}: mean={1:N3} tok/s cv={2:N2}% runs={3}' -f $Context, $mean, (100 * $std / $mean), $Runs

param(
    [string] $Context = "2k",
    [int] $CtxSize = 3072
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$exe = 'C:\Users\user0\Documents\Codex\2026-09-03\chatgpt-conversation-6a98f719-c4cc-83ec-8dd8\work\build\slotstream-tools-262k\slotstream-hybrid-decode.exe'
$model = 'E:\AI\Models\Qwen3.8-Flash-Next-UD-IQ4_XS\Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf'
$plans = Join-Path $root 'results\hybrid-decode-runtime\plans'
$diag = Join-Path $root "results\context-short-gate\diagnostics"
New-Item -ItemType Directory -Force -Path $diag | Out-Null

function Run-Dec {
    param([string] $Name, [string] $State, [int] $Ubatch, [int] $Warm, [int] $Meas, [string] $Ck, [string] $Cv)
    $json = Join-Path $diag "$Name.json"
    & $exe -m $model --state-file $State --output-json $json `
        --input-tokens-file (Join-Path $plans 'canonical-16k-native-continuation-601.tokens') `
        --hybrid-cache-plan (Join-Path $plans 'lru-8-gib.csv') --promote-threshold -1 `
        --verify-tokens 1 --warmup-tokens $Warm --measured-tokens $Meas `
        -c $CtxSize -t 24 -ngl -2 --n-batch 2048 --n-ubatch $Ubatch `
        --cache-type-k $Ck --cache-type-v $Cv 2> (Join-Path $diag "$Name.stderr.log") | Out-Null
    $d = Get-Content -LiteralPath $json -Raw | ConvertFrom-Json
    '{0}: tok_s={1:N3} p50={2:N1}ms p95={3:N1}ms' -f $Name, $d.decode_tok_s, $d.decode_p50_ms, $d.decode_p95_ms
}

# Test 1: exact reproduction of yesterday's run - old f16 state, ubatch 512, 100+100
Run-Dec 't1-f16-u512-100x100' (Join-Path $root 'results\context-adaptive\states\2k-session.bin') 512 100 100 'f16' 'f16'
# Test 2: new q8-q4 state, ubatch 512 - isolates ubatch vs KV profile
Run-Dec 't2-q8q4-u512-100x100' (Join-Path $root "results\context-short-gate\$Context\kv-roundtrip-q8-q4\session.bin") 512 100 100 'q8_0' 'q4_0'
# Test 3: repeat of today's config - variance check
Run-Dec 't3-q8q4-u16-50x200-r2' (Join-Path $root "results\context-short-gate\$Context\kv-roundtrip-q8-q4\session.bin") 16 50 200 'q8_0' 'q4_0'

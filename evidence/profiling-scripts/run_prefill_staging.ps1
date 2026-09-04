param(
    [Parameter(Mandatory = $true)] [string] $Name,
    [int] $PromptTokens = 2048,
    [int] $CtxSize = 3072,
    [int] $Ubatch = 2048,
    [int] $Slots = 512,
    [switch] $Roundtrip
)
$root = Split-Path -Parent $PSScriptRoot
$exe = 'C:\Users\user0\Documents\Codex\2026-09-03\chatgpt-conversation-6a98f719-c4cc-83ec-8dd8\work\build\slotstream-tools-262k\slotstream-prefix-cache.exe'
$model = 'E:\AI\Models\Qwen3.8-Flash-Next-UD-IQ4_XS\Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf'
$promptFile = Join-Path $root 'results\context-128k-gate\context-prompts\context_32k.txt'
$outDir = Join-Path $root "results\prefill-grouped-gate\staging\$Name"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$env:LLAMA_PREFILL_STAGING = '1'
$env:LLAMA_PREFILL_STAGING_SLOTS = "$Slots"

$modeArgs = @('--cold-only')
if ($Roundtrip) { $modeArgs = @() }

$p = Start-Process -FilePath $exe -ArgumentList (@(
    "-m", $model,
    "--prompt-file", $promptFile,
    "--state-file", (Join-Path $outDir "session.bin"),
    "--output-json", (Join-Path $outDir "result.json"),
    "--logits-file", (Join-Path $outDir "logits.f32"),
    "--prompt-tokens", "$PromptTokens",
    "--measured-tokens", "1",
    "--verify-tokens", "8",
    "--n-batch", "$Ubatch", "--n-ubatch", "$Ubatch",
    "--cache-type-k", "f16", "--cache-type-v", "f16",
    "-c", "$CtxSize", "-t", "24", "-ngl", "-2"
) + $modeArgs) -PassThru -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $outDir "stdout.log") `
    -RedirectStandardError (Join-Path $outDir "stderr.log")
$p.WaitForExit(); $p.Refresh()
$d = Get-Content -LiteralPath (Join-Path $outDir "result.json") -Raw | ConvertFrom-Json
'{0}: exit={1} prefill_s={2:N3} tok_s={3:N3} first={4}' -f $Name, $p.ExitCode, ($d.cold_prefill_ms/1000), $d.cold_prefill_tok_s, $d.first_token

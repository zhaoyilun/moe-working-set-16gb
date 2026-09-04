param(
    [Parameter(Mandatory = $true)] [ValidateSet("control", "staging")] [string] $Arm,
    [int] $PromptTokens = 16363,
    [int] $CtxSize = 17408,
    [int] $Ubatch = 8192
)
$root = Split-Path -Parent $PSScriptRoot
$exe = 'C:\Users\user0\Documents\Codex\2026-09-03\chatgpt-conversation-6a98f719-c4cc-83ec-8dd8\work\build\slotstream-tools-262k\slotstream-prefix-cache.exe'
$model = 'E:\AI\Models\Qwen3.8-Flash-Next-UD-IQ4_XS\Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf'
$promptFile = Join-Path $root 'results\context-128k-gate\context-prompts\context_32k.txt'
$outDir = Join-Path $root "results\prefill-grouped-gate\ab\$Arm-u$Ubatch"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

if ($Arm -eq 'staging') { $env:GGML_CUDA_MOE_DEVICE_STAGING = '1' }
else { Remove-Item Env:GGML_CUDA_MOE_DEVICE_STAGING -ErrorAction SilentlyContinue }

$p = Start-Process -FilePath $exe -ArgumentList @(
    "-m", $model,
    "--prompt-file", $promptFile,
    "--state-file", (Join-Path $outDir "session.bin"),
    "--output-json", (Join-Path $outDir "probe.json"),
    "--logits-file", (Join-Path $outDir "logits.f32"),
    "--prompt-tokens", "$PromptTokens",
    "--measured-tokens", "1",
    "--verify-tokens", "8",
    "--n-batch", "$Ubatch", "--n-ubatch", "$Ubatch",
    "--cache-type-k", "f16", "--cache-type-v", "f16",
    "-c", "$CtxSize", "-t", "24", "-ngl", "-2",
    "--cold-only"
) -PassThru -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $outDir "stdout.log") `
    -RedirectStandardError (Join-Path $outDir "stderr.log")
$p.WaitForExit(); $p.Refresh()
$d = Get-Content -LiteralPath (Join-Path $outDir "probe.json") -Raw | ConvertFrom-Json
'{0}: exit={1} prefill_s={2:N3} tok_s={3:N3} first_token={4}' -f $Arm, $p.ExitCode, ($d.cold_prefill_ms/1000), $d.cold_prefill_tok_s, $d.first_token

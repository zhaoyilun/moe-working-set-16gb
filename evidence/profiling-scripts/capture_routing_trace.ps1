param(
    [Parameter(Mandatory = $true)] [string] $Name,
    [Parameter(Mandatory = $true)] [int] $PromptTokens,
    [Parameter(Mandatory = $true)] [int] $CtxSize,
    [string] $PromptFile = "",
    [string] $Exe = 'C:\Users\user0\Documents\Codex\2026-09-03\chatgpt-conversation-6a98f719-c4cc-83ec-8dd8\work\build\slotstream-tools-262k\slotstream-routing-trace.exe',
    [string] $Model = 'E:\AI\Models\Qwen3.8-Flash-Next-UD-IQ4_XS\Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf'
)
$root = Split-Path -Parent $PSScriptRoot
$outDir = Join-Path $root 'results\prefill-grouped-gate\routing'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
if (-not $PromptFile) { $PromptFile = Join-Path $root 'results\context-128k-gate\context-prompts\context_32k.txt' }
& $Exe -m $Model `
    --trace (Join-Path $outDir "$Name.sstrace") `
    --meta (Join-Path $outDir "$Name.meta.json") `
    --prompt-file $PromptFile `
    --prompt-tokens $PromptTokens `
    --cpu-moe -ngl -2 -c $CtxSize -t 24 2> (Join-Path $outDir "$Name.stderr.log")
"exit=$LASTEXITCODE"
Get-Item (Join-Path $outDir "$Name.sstrace") | Select-Object Name, Length

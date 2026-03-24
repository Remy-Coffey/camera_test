$ErrorActionPreference = "Stop"

$baseUrl = if ($env:CAMERA_TEST_LLM_BASE_URL) { $env:CAMERA_TEST_LLM_BASE_URL } else { "http://127.0.0.1:11434" }
$textEnabled = if ($env:CAMERA_TEST_LLM_ENABLED) { $env:CAMERA_TEST_LLM_ENABLED } else { "1" }
$textModel = if ($env:CAMERA_TEST_LLM_MODEL) { $env:CAMERA_TEST_LLM_MODEL } else { "qwen2.5:7b" }
$videoEnabled = if ($env:CAMERA_TEST_VIDEO_LLM_ENABLED) { $env:CAMERA_TEST_VIDEO_LLM_ENABLED } else { "1" }
$videoModel = if ($env:CAMERA_TEST_VIDEO_LLM_MODEL) { $env:CAMERA_TEST_VIDEO_LLM_MODEL } else { "minicpm-v:8b" }

Write-Host "Checking Ollama at $baseUrl"

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Error "找不到 ollama 命令，请先安装 Ollama。"
}

try {
    $null = Invoke-RestMethod -Uri "$baseUrl/api/tags" -Method Get -TimeoutSec 5
} catch {
    Write-Error "无法连接到 Ollama 服务，请先启动 Ollama 桌面应用。"
}

$installed = @()
foreach ($line in (& ollama list | Select-Object -Skip 1)) {
    $text = [string]$line
    if ([string]::IsNullOrWhiteSpace($text)) { continue }
    $name = ($text -split "\s+")[0]
    if (-not [string]::IsNullOrWhiteSpace($name)) {
        $installed += $name.Trim()
    }
}

$textInstalled = $installed -contains $textModel
$videoInstalled = $installed -contains $videoModel

Write-Host "Reachable: true"
Write-Host ("Text enabled: {0}" -f $textEnabled)
Write-Host ("Text model: {0}" -f $textModel)
Write-Host ("Text model installed: {0}" -f $textInstalled)
Write-Host ("Video enabled: {0}" -f $videoEnabled)
Write-Host ("Video model: {0}" -f $videoModel)
Write-Host ("Video model installed: {0}" -f $videoInstalled)
Write-Host "Installed models:"
$installed | ForEach-Object { Write-Host " - $_" }

$missing = @()
if ($textEnabled -eq "1" -and -not $textInstalled) {
    $missing += "text:$textModel"
}
if ($videoEnabled -eq "1" -and -not $videoInstalled) {
    $missing += "video:$videoModel"
}

if ($missing.Count -gt 0) {
    Write-Host "Missing configured models:"
    $missing | ForEach-Object { Write-Host " - $_" }
    exit 2
}

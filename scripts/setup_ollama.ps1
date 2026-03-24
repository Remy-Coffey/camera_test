$ErrorActionPreference = "Stop"

$env:CAMERA_TEST_LLM_BASE_URL = if ($env:CAMERA_TEST_LLM_BASE_URL) { $env:CAMERA_TEST_LLM_BASE_URL } else { "http://127.0.0.1:11434" }
$env:CAMERA_TEST_LLM_ENABLED = if ($env:CAMERA_TEST_LLM_ENABLED) { $env:CAMERA_TEST_LLM_ENABLED } else { "1" }
$env:CAMERA_TEST_LLM_MODEL = if ($env:CAMERA_TEST_LLM_MODEL) { $env:CAMERA_TEST_LLM_MODEL } else { "qwen2.5:7b" }
$env:CAMERA_TEST_VIDEO_LLM_ENABLED = if ($env:CAMERA_TEST_VIDEO_LLM_ENABLED) { $env:CAMERA_TEST_VIDEO_LLM_ENABLED } else { "1" }
$env:CAMERA_TEST_VIDEO_LLM_MODEL = if ($env:CAMERA_TEST_VIDEO_LLM_MODEL) { $env:CAMERA_TEST_VIDEO_LLM_MODEL } else { "minicpm-v:8b" }

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Error "找不到 ollama 命令，请先安装 Ollama。"
}

Write-Host "Checking Ollama service..."
try {
    $null = Invoke-RestMethod -Uri "$env:CAMERA_TEST_LLM_BASE_URL/api/tags" -Method Get -TimeoutSec 5
} catch {
    Write-Error "无法连接到 Ollama 服务，请先启动 Ollama 桌面应用。"
}

$targets = @()
if ($env:CAMERA_TEST_LLM_ENABLED -eq "1") {
    $targets += @{ Kind = "text"; Model = $env:CAMERA_TEST_LLM_MODEL }
}
if ($env:CAMERA_TEST_VIDEO_LLM_ENABLED -eq "1") {
    $targets += @{ Kind = "video"; Model = $env:CAMERA_TEST_VIDEO_LLM_MODEL }
}

if ($targets.Count -eq 0) {
    Write-Host "当前没有启用任何模型，无需下载。"
    exit 0
}

$list = & ollama list
foreach ($target in $targets) {
    Write-Host ("Checking {0} model {1} ..." -f $target.Kind, $target.Model)
    if ($list -notmatch [regex]::Escape($target.Model)) {
        Write-Host ("Model not found, pulling {0} ..." -f $target.Model)
        & ollama pull $target.Model
        $list = & ollama list
    } else {
        Write-Host "Model already installed."
    }
}

Write-Host "Ollama is ready."
Write-Host "说明：只有当前启用的模型会被检查和下载；moondream:1.8b / llava:7b 只有在你切换到它们时才需要 pull。"

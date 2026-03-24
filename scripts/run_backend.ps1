[CmdletBinding()]
param(
    [switch]$Reload
)

$ErrorActionPreference = "Stop"

$env:CAMERA_TEST_LLM_ENABLED = if ($env:CAMERA_TEST_LLM_ENABLED) { $env:CAMERA_TEST_LLM_ENABLED } else { "1" }
$env:CAMERA_TEST_LLM_MODEL = if ($env:CAMERA_TEST_LLM_MODEL) { $env:CAMERA_TEST_LLM_MODEL } else { "qwen2.5:7b" }
$env:CAMERA_TEST_LLM_BASE_URL = if ($env:CAMERA_TEST_LLM_BASE_URL) { $env:CAMERA_TEST_LLM_BASE_URL } else { "http://127.0.0.1:11434" }
$env:CAMERA_TEST_VIDEO_LLM_ENABLED = if ($env:CAMERA_TEST_VIDEO_LLM_ENABLED) { $env:CAMERA_TEST_VIDEO_LLM_ENABLED } else { "1" }
$env:CAMERA_TEST_VIDEO_LLM_MODEL = if ($env:CAMERA_TEST_VIDEO_LLM_MODEL) { $env:CAMERA_TEST_VIDEO_LLM_MODEL } else { "minicpm-v:8b" }

Write-Host ("Text LLM enabled: {0}" -f $env:CAMERA_TEST_LLM_ENABLED)
Write-Host ("Text model: {0}" -f $env:CAMERA_TEST_LLM_MODEL)
Write-Host ("Video LLM enabled: {0}" -f $env:CAMERA_TEST_VIDEO_LLM_ENABLED)
Write-Host ("Video model: {0}" -f $env:CAMERA_TEST_VIDEO_LLM_MODEL)

& "$PSScriptRoot\check_ollama.ps1"

Set-Location (Join-Path $PSScriptRoot "..\backend")
$uvicornArgs = @("-3.11", "-m", "uvicorn", "main:app", "--port", "8000")

if ($Reload) {
    Write-Host "Backend mode: development reload enabled"
    $uvicornArgs += "--reload"
} else {
    Write-Host "Backend mode: standard"
}

py @uvicornArgs

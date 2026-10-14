$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Error "ARGUS is not set up yet. Run .\setup.ps1 first."
    exit 1
}

Set-Location -LiteralPath $projectRoot
& $venvPython (Join-Path $projectRoot "main.py")
exit $LASTEXITCODE

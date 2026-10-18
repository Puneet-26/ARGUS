$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonCommand = Get-Command python -ErrorAction SilentlyContinue

if (-not $pythonCommand) {
    Write-Error "Python was not found. Install Python 3.10, 3.11, or 3.12, then reopen PowerShell."
    exit 1
}

$pythonExe = $pythonCommand.Source
$pythonVersionText = & $pythonExe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Could not run Python at $pythonExe."
    exit 1
}
$pythonVersion = [version]$pythonVersionText
if ($pythonVersion -lt [version]"3.10" -or $pythonVersion -ge [version]"3.13") {
    Write-Error "ARGUS supports Python 3.10 through 3.12. Found Python $pythonVersionText at $pythonExe."
    exit 1
}

$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Host "Creating project virtual environment with Python $pythonVersionText..."
    & $pythonExe -m venv (Join-Path $projectRoot ".venv")
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Could not create .venv. Check that Python includes the venv module."
        exit 1
    }
}

Write-Host "Installing ARGUS dependencies into .venv..."
& $venvPython -m pip install -r (Join-Path $projectRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) {
    Write-Error "Dependency installation failed. Check the pip output and network access, then rerun setup.ps1."
    exit 1
}

Write-Host "Setup complete. Launch ARGUS with: .\run.ps1"

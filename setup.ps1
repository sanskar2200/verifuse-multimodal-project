$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
python -m venv .venv
if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.10 or newer and try again.' }
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& '.\.venv\Scripts\python.exe' scripts/run_experiments.py
if ($LASTEXITCODE -ne 0) { throw 'Experiment setup failed.' }
Write-Output 'Setup complete. Run .\start.ps1 to open the local server.'

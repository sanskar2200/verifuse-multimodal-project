$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    $taskBundledPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
    if (Test-Path -LiteralPath $taskBundledPython) {
        $taskPython = $taskBundledPython
    } else {
        $taskPython = (Get-Command python -ErrorAction Stop).Source
    }
}
& $taskPython -c 'import numpy, PIL'
if ($LASTEXITCODE -ne 0) { throw 'Dependencies are missing. Run setup.ps1 first.' }
if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'results\experiments.json')) -or -not (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'models\bundle.json'))) {
    & $taskPython scripts/run_experiments.py
    if ($LASTEXITCODE -ne 0) { throw 'Experiment setup failed.' }
}
& $taskPython app.py @args

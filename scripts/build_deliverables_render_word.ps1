$ErrorActionPreference = 'Stop'
$projectDir = Split-Path -Parent $PSScriptRoot
$sourcePath = Join-Path $projectDir 'output\VeriFuse_Paper.docx'
$exportDir = Join-Path $projectDir 'output\.build\word'
New-Item -ItemType Directory -Force -Path $exportDir | Out-Null
$pdfPath = Join-Path $exportDir 'VeriFuse_Paper_Word.pdf'
$wordApp = $null
$paperDocument = $null
try {
    $wordApp = New-Object -ComObject Word.Application
    $wordApp.Visible = $false
    $wordApp.DisplayAlerts = 0
    $paperDocument = $wordApp.Documents.Open($sourcePath, $false, $true)
    $paperDocument.ExportAsFixedFormat($pdfPath, 17)
    Write-Output $pdfPath
} finally {
    if ($null -ne $paperDocument) { $paperDocument.Close(0) }
    if ($null -ne $wordApp) { $wordApp.Quit() }
}

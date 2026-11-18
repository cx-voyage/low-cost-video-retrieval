# Run once per new terminal:   . .\env.ps1
#
# 1) Sets the working directory to the project root.
# 2) Points `py` at the project venv interpreter (never use the system Python).
# 3) Switches the console to UTF-8 so Chinese output from Python shows correctly.
#
# NOTE: keep this file PURE ASCII.
#   Windows PowerShell 5.1 decodes a BOM-less .ps1 file as ANSI (GBK on zh-CN),
#   so any Chinese text here would show up as mojibake.

$VenvPy = "D:\video-retrieval\.venv\Scripts\python.exe"

if (-not (Test-Path $VenvPy)) {
    Write-Host "[ERROR] venv python not found: $VenvPy" -ForegroundColor Red
} else {
    function global:py { & $VenvPy @args }
    Set-Location "D:\video-retrieval"

    # console -> UTF-8 (otherwise Python's UTF-8 stdout is decoded as GBK)
    try {
        chcp 65001 > $null
        [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
        $OutputEncoding = New-Object System.Text.UTF8Encoding $false
        $env:PYTHONIOENCODING = "utf-8"
    } catch {
        Write-Host "[warn] failed to switch console to UTF-8" -ForegroundColor Yellow
    }

    Write-Host "project : D:\video-retrieval" -ForegroundColor Cyan
    Write-Host "python  : $VenvPy" -ForegroundColor Cyan
    Write-Host "console : UTF-8 (chcp 65001)" -ForegroundColor Cyan
    Write-Host "usage   : py code\make_figure.py" -ForegroundColor DarkGray
}

<#
.SYNOPSIS
    Developer task runner for GeoVisionGPT (Windows / PowerShell).

.DESCRIPTION
    Wraps the commands used day to day so they are consistent between local development
    and CI. Mirrors .github/workflows/backend-ci.yml.

.EXAMPLE
    .\scripts\dev.ps1 setup
    .\scripts\dev.ps1 test
    .\scripts\dev.ps1 check
    .\scripts\dev.ps1 run
#>

param(
    [Parameter(Position = 0)]
    [ValidateSet('setup', 'run', 'test', 'lint', 'format', 'types', 'check', 'migrate', 'revision')]
    [string]$Task = 'check',

    [Parameter(Position = 1)]
    [string]$Message = 'auto'
)

$ErrorActionPreference = 'Stop'
$Backend = Join-Path $PSScriptRoot '..\backend' | Resolve-Path
$Python = Join-Path $Backend '.venv\Scripts\python.exe'

# A predictable dev secret: the app refuses to start with a key under 32 characters, and
# this is never used outside local development.
if (-not $env:GVG_SECRET_KEY) {
    $env:GVG_SECRET_KEY = 'local-development-secret-key-at-least-32-characters'
}

function Invoke-Py {
    param([string[]]$Arguments)
    Push-Location $Backend
    try {
        & $Python @Arguments
        if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code $LASTEXITCODE" }
    }
    finally { Pop-Location }
}

switch ($Task) {
    'setup' {
        Push-Location $Backend
        try {
            uv venv --python 3.11
            uv pip install -e ".[dev]"
        }
        finally { Pop-Location }
        Write-Host "Environment ready. Run: .\scripts\dev.ps1 test" -ForegroundColor Green
    }
    'run'      { Invoke-Py @('-m', 'uvicorn', 'app.main:app', '--reload', '--port', '8000') }
    'test'     { Invoke-Py @('-m', 'pytest', '-q') }
    'lint'     { Invoke-Py @('-m', 'ruff', 'check', '.') }
    'format'   { Invoke-Py @('-m', 'ruff', 'format', '.') }
    'types'    { Invoke-Py @('-m', 'mypy', 'app') }
    'migrate'  { Invoke-Py @('-m', 'alembic', 'upgrade', 'head') }
    'revision' { Invoke-Py @('-m', 'alembic', 'revision', '--autogenerate', '-m', $Message) }
    'check' {
        Invoke-Py @('-m', 'ruff', 'check', '.')
        Invoke-Py @('-m', 'ruff', 'format', '--check', '.')
        Invoke-Py @('-m', 'mypy', 'app')
        Invoke-Py @('-m', 'pytest', '-q')
        Write-Host "All checks passed." -ForegroundColor Green
    }
}

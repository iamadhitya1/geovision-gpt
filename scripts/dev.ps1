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
    .\scripts\dev.ps1 demo     # everything for a live demo, one command
#>

param(
    [Parameter(Position = 0)]
    [ValidateSet('setup', 'run', 'test', 'lint', 'format', 'types', 'check', 'migrate', 'revision', 'demo')]
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
    'demo' {
        # Everything a live demo needs: backend (creates its own SQLite tables), a production-mode
        # frontend (no first-visit compile pauses, unlike `npm run dev`), and the browser.
        $Frontend = Join-Path $PSScriptRoot '..\frontend' | Resolve-Path
        $Checkpoints = Join-Path $Backend 'checkpoints'
        if (-not (Test-Path (Join-Path $Checkpoints 'answer_vocab_generalist.json'))) {
            Write-Warning "No answer_vocab_generalist.json in $Checkpoints - questions will fail until the trained checkpoints are copied there. Run: gh release download v1.0-trained-models --dir backend/checkpoints --pattern '*.pt' --pattern '*.json' (see README.md)."
        }
        # No `alembic upgrade` here on purpose: the default SQLite profile creates any missing
        # tables itself at startup, and alembic refuses a database it didn't create ("table
        # users already exists"). Migrations are for the PostgreSQL profile.
        # Child windows inherit GVG_SECRET_KEY from this session. Warm the segmentation model up
        # in the background so the first "highlight" click doesn't pause for several seconds.
        $env:GVG_SEGMENTATION_WARM_UP = 'true'
        Start-Process powershell -ArgumentList '-NoExit', '-Command', "Set-Location '$Backend'; & '$Python' -m uvicorn app.main:app --port 8000"
        Start-Process powershell -ArgumentList '-NoExit', '-Command', "Set-Location '$Frontend'; npm run build; npm run start"
        Write-Host "Backend on :8000, frontend building then serving on :3000 (about 30s)." -ForegroundColor Green
        Start-Sleep -Seconds 40
        Start-Process 'http://localhost:3000'
    }
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

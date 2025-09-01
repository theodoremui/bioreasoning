# PowerShell script to run the BioMCP server (streamable-http)

# Set PYTHONPATH to the project root
$env:PYTHONPATH = Split-Path -Parent $PSScriptRoot
Write-Host "Set PYTHONPATH to: $env:PYTHONPATH" -ForegroundColor Cyan

# Check if we're in a virtual environment
if (-not $env:VIRTUAL_ENV) {
    Write-Host "Not in virtual environment. Activating .venv..." -ForegroundColor Yellow
    if (Test-Path ".venv\Scripts\Activate.ps1") {
        & ".venv\Scripts\Activate.ps1"
        Write-Host "Virtual environment activated." -ForegroundColor Green
    } else {
        Write-Host "Error: .venv\\Scripts\\Activate.ps1 not found. Please create a virtual environment first." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "Already in virtual environment: $env:VIRTUAL_ENV" -ForegroundColor Green
}

# Change to project root directory
Set-Location (Split-Path -Parent $PSScriptRoot)

# Set default port if not provided
if (-not $env:BIOMCP_PORT) {
    $env:BIOMCP_PORT = "8132"
}
Write-Host "Starting BioMCP server on port $env:BIOMCP_PORT ..." -ForegroundColor Cyan

# Run the BioMCP server
uv run bioagents/mcp/biomcp_server.py

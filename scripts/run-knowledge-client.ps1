# PowerShell script to run the knowledge client streamlit app

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
        Write-Host "Error: .venv\Scripts\Activate.ps1 not found. Please create a virtual environment first." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "Already in virtual environment: $env:VIRTUAL_ENV" -ForegroundColor Green
}

# Check if streamlit is installed
try {
    python -c "import streamlit" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Streamlit is already installed." -ForegroundColor Green
    } else {
        throw "Streamlit not found"
    }
} catch {
    Write-Host "Streamlit not found. Installing streamlit..." -ForegroundColor Yellow
    pip install streamlit
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Streamlit installed successfully." -ForegroundColor Green
    } else {
        Write-Host "Error: Failed to install streamlit." -ForegroundColor Red
        exit 1
    }
}

# Change to project root directory
Set-Location (Split-Path -Parent $PSScriptRoot)

# Run the streamlit app
$PORT = 8501
Write-Host "Starting streamlit app at port $PORT..." -ForegroundColor Cyan
streamlit run frontend/app.py --server.port $PORT 
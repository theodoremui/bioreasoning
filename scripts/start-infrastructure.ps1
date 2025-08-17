# BioReasoning Infrastructure Startup Script (PowerShell)

param(
    [Parameter(Position=0)]
    [string]$Command = "start"
)

function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Function to detect Docker Compose command
function Get-DockerComposeCommand {
    # Try the newer docker compose first
    try {
        docker compose version | Out-Null
        return "docker compose"
    }
    catch {
        # Fall back to older docker-compose
        try {
            docker-compose --version | Out-Null
            return "docker-compose"
        }
        catch {
            return $null
        }
    }
}

# Function to check if Docker is running
function Test-Docker {
    try {
        docker info | Out-Null
        Write-Success "Docker is running"
        return $true
    }
    catch {
        Write-Error "Docker is not running. Please start Docker Desktop and try again."
        return $false
    }
}

# Function to check if Docker Compose is available
function Test-DockerCompose {
    $dockerComposeCmd = Get-DockerComposeCommand
    if ($dockerComposeCmd) {
        Write-Success "Docker Compose is available ($dockerComposeCmd)"
        return $true
    }
    else {
        Write-Error "Docker Compose is not installed. Please install Docker Compose and try again."
        return $false
    }
}

# Function to check for port conflicts
function Test-PortConflicts {
    $ports = @("5432", "16686", "4317", "4318", "8080")
    $conflicts = @()
    
    foreach ($port in $ports) {
        try {
            $connections = netstat -ano | Select-String ":$port "
            if ($connections) {
                $conflicts += $port
            }
        }
        catch {
            Write-Warning "Could not check port $port for conflicts"
        }
    }
    
    if ($conflicts.Count -gt 0) {
        Write-Warning "Port conflicts detected on: $($conflicts -join ', ')"
        Write-Warning "Some services may fail to start. Consider stopping conflicting services."
        Write-Host ""
        Write-Host "Common solutions:" -ForegroundColor Yellow
        Write-Host "  - Stop local PostgreSQL: net stop postgresql" -ForegroundColor Gray
        Write-Host "  - Stop local MySQL: net stop mysql" -ForegroundColor Gray
        Write-Host "  - Check for other services using these ports" -ForegroundColor Gray
    }
    else {
        Write-Success "No port conflicts detected"
    }
}

# Function to start services
function Start-Services {
    Write-Status "Starting infrastructure services..."
    
    $dockerComposeCmd = Get-DockerComposeCommand
    
    try {
        # Check if compose.yaml exists
        if (-not (Test-Path "compose.yaml")) {
            Write-Error "compose.yaml file not found in current directory"
            return $false
        }
        
        # Start services
        $result = Invoke-Expression "$dockerComposeCmd up -d"
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Services started successfully"
            return $true
        }
        else {
            Write-Error "Failed to start services (exit code: $LASTEXITCODE)"
            return $false
        }
    }
    catch {
        Write-Error "Failed to start services: $_"
        return $false
    }
}

# Function to check if services are running
function Test-ServicesRunning {
    $dockerComposeCmd = Get-DockerComposeCommand
    
    try {
        $result = Invoke-Expression "$dockerComposeCmd ps --format json" | ConvertFrom-Json
        $runningServices = $result | Where-Object { $_.State -eq "running" }
        
        if ($runningServices.Count -eq 0) {
            return $false
        }
        
        Write-Status "Found $($runningServices.Count) running services"
        return $true
    }
    catch {
        Write-Warning "Could not check service status: $_"
        return $false
    }
}

# Function to wait for services to be ready
function Wait-ForServices {
    Write-Status "Waiting for services to be ready..."
    
    $dockerComposeCmd = Get-DockerComposeCommand
    $maxAttempts = 30
    $attempt = 1
    
    # First, wait for services to be running
    Write-Status "Waiting for services to start..."
    while ($attempt -le $maxAttempts) {
        if (Test-ServicesRunning) {
            Write-Success "All services are running"
            break
        }
        
        if ($attempt -eq $maxAttempts) {
            Write-Error "Services failed to start within timeout"
            return $false
        }
        
        Write-Host "." -NoNewline
        Start-Sleep -Seconds 2
        $attempt++
    }
    
    # Wait for PostgreSQL
    Write-Status "Waiting for PostgreSQL to be ready..."
    $attempt = 1
    
    while ($attempt -le $maxAttempts) {
        try {
            # Use docker compose exec with better error handling
            $result = Invoke-Expression "$dockerComposeCmd exec -T postgres pg_isready -U llama -d notebookllama 2>&1"
            if ($LASTEXITCODE -eq 0) {
                Write-Success "PostgreSQL is ready"
                break
            }
        }
        catch {
            # Ignore errors during startup
        }
        
        if ($attempt -eq $maxAttempts) {
            Write-Error "PostgreSQL failed to start within timeout"
            Write-Host "   Check logs with: $dockerComposeCmd logs postgres" -ForegroundColor Yellow
            return $false
        }
        
        Write-Host "." -NoNewline
        Start-Sleep -Seconds 2
        $attempt++
    }
    
    # Wait for Jaeger
    Write-Status "Waiting for Jaeger to be ready..."
    $attempt = 1
    
    while ($attempt -le $maxAttempts) {
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:16686/api/services" -UseBasicParsing -TimeoutSec 5 -ErrorAction Stop
            if ($response.StatusCode -eq 200) {
                Write-Success "Jaeger is ready"
                break
            }
        }
        catch {
            # Ignore errors during startup
        }
        
        if ($attempt -eq $maxAttempts) {
            Write-Error "Jaeger failed to start within timeout"
            Write-Host "   Check logs with: $dockerComposeCmd logs jaeger" -ForegroundColor Yellow
            return $false
        }
        
        Write-Host "." -NoNewline
        Start-Sleep -Seconds 2
        $attempt++
    }
    
    # Wait for Adminer
    Write-Status "Waiting for Adminer to be ready..."
    $attempt = 1
    
    while ($attempt -le $maxAttempts) {
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:8080" -UseBasicParsing -TimeoutSec 5 -ErrorAction Stop
            if ($response.StatusCode -eq 200) {
                Write-Success "Adminer is ready"
                break
            }
        }
        catch {
            # Ignore errors during startup
        }
        
        if ($attempt -eq $maxAttempts) {
            Write-Error "Adminer failed to start within timeout"
            Write-Host "   Check logs with: $dockerComposeCmd logs adminer" -ForegroundColor Yellow
            return $false
        }
        
        Write-Host "." -NoNewline
        Start-Sleep -Seconds 2
        $attempt++
    }
    
    return $true
}

# Function to display service information
function Show-ServiceInfo {
    Write-Host ""
    Write-Success "Infrastructure services are ready!"
    Write-Host ""
    Write-Host "Service Information:" -ForegroundColor Cyan
    Write-Host "===================" -ForegroundColor Cyan
    Write-Host "📊 Jaeger Tracing UI:     http://localhost:16686"
    Write-Host "🗄️  Adminer (Database):   http://localhost:8080"
    Write-Host "🔌 PostgreSQL:            localhost:5432"
    Write-Host "📡 OpenTelemetry HTTP:    localhost:4318"
    Write-Host "📡 OpenTelemetry gRPC:    localhost:4317"
    Write-Host ""
    Write-Host "Database Credentials:" -ForegroundColor Cyan
    Write-Host "====================" -ForegroundColor Cyan
    Write-Host "Username: llama"
    Write-Host "Password: S*********1"
    Write-Host "Database: notebookllama"
    Write-Host ""
    Write-Host "Next Steps:" -ForegroundColor Cyan
    Write-Host "===========" -ForegroundColor Cyan
    Write-Host "1. Set up your .env file with API keys: .\scripts\setup-env.ps1"
    Write-Host "2. Start the MCP server: .\scripts\run-docs-server.ps1"
    Write-Host "3. Start the Streamlit client: .\scripts\run-docs-client.ps1"
    Write-Host ""
}

# Function to check service status
function Get-ServiceStatus {
    Write-Status "Checking service status..."
    
    $dockerComposeCmd = Get-DockerComposeCommand
    
    Write-Host ""
    Write-Host "Service Status:" -ForegroundColor Cyan
    Write-Host "===============" -ForegroundColor Cyan
    try {
        Invoke-Expression "$dockerComposeCmd ps"
    }
    catch {
        Write-Error "Failed to get service status: $_"
    }
    
    Write-Host ""
    Write-Host "Recent Logs:" -ForegroundColor Cyan
    Write-Host "============" -ForegroundColor Cyan
    try {
        Invoke-Expression "$dockerComposeCmd logs --tail=10"
    }
    catch {
        Write-Error "Failed to get logs: $_"
    }
}

# Function to show logs
function Show-Logs {
    $dockerComposeCmd = Get-DockerComposeCommand
    try {
        Invoke-Expression "$dockerComposeCmd logs -f"
    }
    catch {
        Write-Error "Failed to show logs: $_"
    }
}

# Function to stop services
function Stop-Services {
    Write-Status "Stopping infrastructure services..."
    $dockerComposeCmd = Get-DockerComposeCommand
    
    try {
        Invoke-Expression "$dockerComposeCmd down"
        Write-Success "Services stopped"
    }
    catch {
        Write-Error "Failed to stop services: $_"
    }
}

# Function to restart services
function Restart-Services {
    Write-Status "Restarting infrastructure services..."
    $dockerComposeCmd = Get-DockerComposeCommand
    
    try {
        Invoke-Expression "$dockerComposeCmd down"
        Invoke-Expression "$dockerComposeCmd up -d"
        Write-Success "Services restarted"
    }
    catch {
        Write-Error "Failed to restart services: $_"
    }
}

function Show-Help {
    Write-Host "Usage: .\start-infrastructure.ps1 [command]"
    Write-Host ""
    Write-Host "Commands:"
    Write-Host "  start     Start infrastructure services (default)"
    Write-Host "  status    Check service status"
    Write-Host "  logs      Show service logs"
    Write-Host "  stop      Stop all services"
    Write-Host "  restart   Restart all services"
    Write-Host "  help      Show this help message"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\start-infrastructure.ps1"
    Write-Host "  .\start-infrastructure.ps1 status"
    Write-Host "  .\start-infrastructure.ps1 logs"
}

function Start-Infrastructure {
    Write-Host "🚀 BioReasoning Infrastructure Startup" -ForegroundColor Cyan
    Write-Host "=====================================" -ForegroundColor Cyan
    Write-Host ""
    
    # Check prerequisites
    if (-not (Test-Docker)) { 
        Write-Host ""
        Write-Host "💡 Troubleshooting:" -ForegroundColor Yellow
        Write-Host "1. Make sure Docker Desktop is installed and running" -ForegroundColor Gray
        Write-Host "2. Check Docker Desktop status in system tray" -ForegroundColor Gray
        Write-Host "3. Restart Docker Desktop if needed" -ForegroundColor Gray
        return 
    }
    
    if (-not (Test-DockerCompose)) { 
        Write-Host ""
        Write-Host "💡 Troubleshooting:" -ForegroundColor Yellow
        Write-Host "1. Docker Compose should be included with Docker Desktop" -ForegroundColor Gray
        Write-Host "2. Try updating Docker Desktop" -ForegroundColor Gray
        Write-Host "3. Check Docker installation: docker --version" -ForegroundColor Gray
        return 
    }
    
    Test-PortConflicts
    
    # Start services
    if (-not (Start-Services)) { 
        Write-Host ""
        Write-Host "💡 Troubleshooting:" -ForegroundColor Yellow
        Write-Host "1. Check if compose.yaml exists in current directory" -ForegroundColor Gray
        Write-Host "2. Check Docker Desktop has enough resources" -ForegroundColor Gray
        Write-Host "3. Try: docker compose logs" -ForegroundColor Gray
        return 
    }
    
    # Wait for services to be ready
    if (-not (Wait-ForServices)) { 
        Write-Host ""
        Write-Host "💡 Troubleshooting:" -ForegroundColor Yellow
        Write-Host "1. Check service logs: docker compose logs [service-name]" -ForegroundColor Gray
        Write-Host "2. Ensure ports are not in use by other applications" -ForegroundColor Gray
        Write-Host "3. Try restarting Docker Desktop" -ForegroundColor Gray
        return 
    }
    
    # Show service information
    Show-ServiceInfo
    
    # Check final status
    Get-ServiceStatus
}

# Main script logic
switch ($Command.ToLower()) {
    "start" {
        Start-Infrastructure
    }
    "status" {
        Get-ServiceStatus
    }
    "logs" {
        Show-Logs
    }
    "stop" {
        Stop-Services
    }
    "restart" {
        Restart-Services
    }
    "help" {
        Show-Help
    }
    default {
        Write-Error "Unknown command: $Command"
        Write-Host "Use .\start-infrastructure.ps1 help for usage information"
        exit 1
    }
}

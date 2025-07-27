# BioReasoning Environment Checker (PowerShell)
# This script helps diagnose issues with .env file configuration

param(
    [switch]$Verbose,
    [switch]$Help
)

# Function to write colored output
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

function Show-Help {
    Write-Host "Usage: .\check-env.ps1 [options]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Verbose    Show detailed information about each variable"
    Write-Host "  -Help       Show this help message"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\check-env.ps1                    # Basic environment check"
    Write-Host "  .\check-env.ps1 -Verbose           # Detailed environment check"
}

# Function to load and validate .env file
function Load-EnvFile {
    param([string]$FilePath)
    
    $envVars = @{}
    $errors = @()
    $warnings = @()
    
    if (-not (Test-Path $FilePath)) {
        throw "File not found: $FilePath"
    }
    
    try {
        $content = Get-Content $FilePath -ErrorAction Stop
        for ($lineNum = 1; $lineNum -le $content.Length; $lineNum++) {
            $line = $content[$lineNum - 1].Trim()
            
            # Skip comments and empty lines
            if ($line.StartsWith('#') -or [string]::IsNullOrWhiteSpace($line)) {
                continue
            }
            
            # Parse variable assignment
            if ($line.Contains('=')) {
                $parts = $line.Split('=', 2)
                $varName = $parts[0].Trim()
                $varValue = if ($parts.Length -gt 1) { $parts[1].Trim() } else { "" }
                
                # Validate variable name
                if ($varName -match '^[a-zA-Z_][a-zA-Z0-9_]*$') {
                    # Remove quotes if present
                    if ($varValue.StartsWith('"') -and $varValue.EndsWith('"')) {
                        $varValue = $varValue.Substring(1, $varValue.Length - 2)
                    } elseif ($varValue.StartsWith("'") -and $varValue.EndsWith("'")) {
                        $varValue = $varValue.Substring(1, $varValue.Length - 2)
                    }
                    
                    $envVars[$varName] = $varValue
                } else {
                    $errors += "Line $lineNum`: Invalid variable name '$varName'"
                }
            } else {
                $warnings += "Line $lineNum`: No '=' found in line"
            }
        }
    }
    catch {
        throw "Failed to read .env file: $_"
    }
    
    return @{
        Variables = $envVars
        Errors = $errors
        Warnings = $warnings
    }
}

# Function to validate API key format
function Test-ApiKey {
    param([string]$ApiKey, [string]$Provider)
    
    if ([string]::IsNullOrWhiteSpace($ApiKey)) {
        return $false
    }
    
    switch ($Provider) {
        "OpenAI" {
            # OpenAI keys typically start with 'sk-' and are 51 characters long
            return $ApiKey -match '^sk-[a-zA-Z0-9]{48}$'
        }
        "ElevenLabs" {
            # ElevenLabs keys are typically 28 characters long
            return $ApiKey.Length -eq 28 -and $ApiKey -match '^[a-zA-Z0-9]+$'
        }
        default {
            return $ApiKey.Length -gt 10
        }
    }
}

# Function to check environment status
function Test-EnvironmentStatus {
    param([hashtable]$EnvVars, [switch]$Verbose)
    
    Write-Host "🔍 BioReasoning Environment Checker" -ForegroundColor Cyan
    Write-Host "===================================" -ForegroundColor Cyan
    Write-Host ""
    
    # Load and validate .env file
    try {
        $envData = Load-EnvFile -FilePath ".env"
        $envVars = $envData.Variables
        $errors = $envData.Errors
        $warnings = $envData.Warnings
        
        Write-Success "✅ .env file found and parsed"
        Write-Host ""
        
        # Show summary
        Write-Host "📊 Environment Summary:" -ForegroundColor Cyan
        Write-Host "   Total variables: $($envVars.Count)"
        Write-Host "   Errors: $($errors.Count)"
        Write-Host "   Warnings: $($warnings.Count)"
        Write-Host ""
        
        # Show errors if any
        if ($errors.Count -gt 0) {
            Write-Error "❌ Errors found:"
            foreach ($error in $errors) {
                Write-Host "   $error" -ForegroundColor Red
            }
            Write-Host ""
        }
        
        # Show warnings if any
        if ($warnings.Count -gt 0) {
            Write-Warning "⚠️  Warnings found:"
            foreach ($warning in $warnings) {
                Write-Host "   $warning" -ForegroundColor Yellow
            }
            Write-Host ""
        }
        
    } catch {
        Write-Error "❌ .env file not found or invalid!"
        Write-Host "   Please create a .env file or run .\scripts\setup-env.ps1"
        return $false
    }
    
    # Define variable categories
    $requiredVars = @('OPENAI_API_KEY', 'ELEVENLABS_API_KEY')
    $optionalVars = @('PHOENIX_API_KEY', 'PHOENIX_ENDPOINT', 'OTLP_ENDPOINT')
    $databaseVars = @('pgql_user', 'pgql_psw', 'pgql_db')
    $observabilityVars = @('ENABLE_OBSERVABILITY')
    
    # Check required variables
    Write-Host "🔑 Required Variables:" -ForegroundColor Cyan
    $requiredStatus = @{}
    foreach ($varName in $requiredVars) {
        if ($envVars.ContainsKey($varName)) {
            $value = $envVars[$varName]
            if (-not [string]::IsNullOrWhiteSpace($value)) {
                $maskedValue = if ($value.Length -gt 20) { "$($value.Substring(0, 20))..." } else { $value }
                Write-Success "   ✅ $varName`: $maskedValue"
                
                # Validate API key format
                $provider = if ($varName -eq 'OPENAI_API_KEY') { 'OpenAI' } else { 'ElevenLabs' }
                if (Test-ApiKey -ApiKey $value -Provider $provider) {
                    Write-Success "      ✅ Format appears valid"
                } else {
                    Write-Warning "      ⚠️  Format may be invalid"
                }
                
                $requiredStatus[$varName] = $true
            } else {
                Write-Warning "   ⚠️  $varName`: Empty value"
                $requiredStatus[$varName] = $false
            }
        } else {
            Write-Error "   ❌ $varName`: Not found"
            $requiredStatus[$varName] = $false
        }
    }
    
    Write-Host ""
    Write-Host "🔧 Optional Variables:" -ForegroundColor Cyan
    foreach ($varName in $optionalVars) {
        if ($envVars.ContainsKey($varName)) {
            $value = $envVars[$varName]
            $maskedValue = if ($value.Length -gt 30) { "$($value.Substring(0, 30))..." } else { $value }
            Write-Success "   ✅ $varName`: $maskedValue"
        } else {
            Write-Host "   ⚪ $varName`: Not set" -ForegroundColor Gray
        }
    }
    
    Write-Host ""
    Write-Host "🗄️  Database Configuration:" -ForegroundColor Cyan
    foreach ($varName in $databaseVars) {
        if ($envVars.ContainsKey($varName)) {
            $value = $envVars[$varName]
            if (-not [string]::IsNullOrWhiteSpace($value)) {
                Write-Success "   ✅ $varName`: $value"
            } else {
                Write-Warning "   ⚠️  $varName`: Empty value (will use default)"
            }
        } else {
            Write-Warning "   ⚠️  $varName`: Not set (will use default)"
        }
    }
    
    Write-Host ""
    Write-Host "📊 Observability Configuration:" -ForegroundColor Cyan
    foreach ($varName in $observabilityVars) {
        if ($envVars.ContainsKey($varName)) {
            $value = $envVars[$varName]
            Write-Success "   ✅ $varName`: $value"
        } else {
            Write-Warning "   ⚠️  $varName`: Not set (will use default: true)"
        }
    }
    
    # Check OTLP endpoint specifically
    if ($envVars.ContainsKey('OTLP_ENDPOINT')) {
        $otlpValue = $envVars['OTLP_ENDPOINT']
        Write-Success "   ✅ OTLP_ENDPOINT`: $otlpValue"
    } else {
        Write-Warning "   ⚠️  OTLP_ENDPOINT`: Not set (will use default: http://localhost:4318/v1/traces)"
    }
    
    Write-Host ""
    Write-Host "🎙️  Podcast Generation Status:" -ForegroundColor Cyan
    Write-Host "-----------------------------" -ForegroundColor Cyan
    
    $openaiOK = $requiredStatus.ContainsKey('OPENAI_API_KEY') -and $requiredStatus['OPENAI_API_KEY']
    $elevenlabsOK = $requiredStatus.ContainsKey('ELEVENLABS_API_KEY') -and $requiredStatus['ELEVENLABS_API_KEY']
    
    if ($openaiOK -and $elevenlabsOK) {
        Write-Success "   ✅ Ready - Both API keys are configured"
    } elseif ($openaiOK) {
        Write-Warning "   ⚠️  Partially ready - Missing ELEVENLABS_API_KEY"
    } elseif ($elevenlabsOK) {
        Write-Warning "   ⚠️  Partially ready - Missing OPENAI_API_KEY"
    } else {
        Write-Error "   ❌ Not ready - Missing both API keys"
    }
    
    Write-Host ""
    Write-Host "💡 Next Steps:" -ForegroundColor Cyan
    if (-not $openaiOK -or -not $elevenlabsOK) {
        Write-Host "   1. Run .\scripts\setup-env.ps1 to configure missing API keys"
        Write-Host "   2. Get your API keys from:"
        Write-Host "      - OpenAI: https://platform.openai.com/api-keys"
        Write-Host "      - ElevenLabs: https://elevenlabs.io/speech-synthesis"
    } else {
        Write-Host "   1. Start the infrastructure: .\scripts\start-infrastructure.ps1"
        Write-Host "   2. Start the MCP server: .\scripts\run-knowledge-server.ps1"
        Write-Host "   3. Start the Streamlit client: .\scripts\run-knowledge-client.ps1"
        Write-Host "   4. Upload a document and try podcast generation!"
    }
    
    Write-Host ""
    Write-Success "✅ Environment check completed successfully!"
    
    return $true
}

# Main script logic
function Main {
    if ($Help) {
        Show-Help
        return
    }
    
    try {
        Test-EnvironmentStatus -Verbose:$Verbose
    }
    catch {
        Write-Error "Environment check failed: $_"
        exit 1
    }
}

# Handle command line arguments
if ($Help) {
    Show-Help
} else {
    Main
} 
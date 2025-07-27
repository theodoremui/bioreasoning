# Error Fixes and Troubleshooting Guide

This document outlines the errors found in the BioReasoning system and their systematic fixes.

## Issues Identified and Fixed

### 1. Port Mismatch Error ❌ → ✅ FIXED

**Problem**: The workflow was trying to connect to `localhost:8000` but the MCP server runs on port 8131.

**Error**: `httpx.ReadError` when trying to connect to MCP server

**Fix Applied**: 
- Modified `bioagents/knowledge/workflow.py` line 8
- Changed `"http://localhost:8000/mcp"` to `"http://localhost:8131/mcp"`

**Files Modified**:
- `bioagents/knowledge/workflow.py`

### 2. OpenTelemetry Connection Error ❌ → ✅ FIXED

**Problem**: System trying to connect to `localhost:4318` for observability but failing with connection errors.

**Error**: `httpx.ReadError` when connecting to OpenTelemetry endpoint

**Fix Applied**:
- Added health check before initializing OpenTelemetry
- Made observability gracefully degrade when endpoint is unavailable
- Added proper error handling and fallback behavior

**Files Modified**:
- `frontend/pages/2_Documents.py`

### 3. Pydub Regex Warning ⚠️ → ✅ FIXED

**Problem**: Invalid escape sequence `\(` in regex patterns causing SyntaxWarning.

**Warning**: 
```
SyntaxWarning: invalid escape sequence '\('
  elif re.match('(dbl)p?( \(default\))?$', token):
```

**Fix Applied**:
- Created `fixes/pydub_regex_fix.py` script to patch the virtual environment
- Converts regex patterns to raw strings (r'') to properly handle backslashes
- Creates backup before modifying

**Files Created**:
- `fixes/pydub_regex_fix.py`

### 4. Missing FFmpeg Warning ⚠️ → ⚠️ MANUAL INSTALLATION REQUIRED

**Problem**: Runtime warning about missing ffmpeg/avconv for audio processing.

**Warning**:
```
RuntimeWarning: Couldn't find ffmpeg or avconv - defaulting to ffmpeg, but may not work
```

**Solution**: Manual installation required (see instructions below)

## How to Apply Fixes

### Automatic Fixes

1. **Port and OpenTelemetry fixes** are already applied to the codebase
2. **Pydub regex fix** can be applied by running:

```bash
python fixes/pydub_regex_fix.py
```

### Manual Fixes

#### Install FFmpeg

**Windows**:
```bash
# Option 1: Using winget (recommended)
winget install ffmpeg

# Option 2: Using Chocolatey
choco install ffmpeg

# Option 3: Manual installation
# 1. Download from https://ffmpeg.org/download.html
# 2. Extract to C:\ffmpeg
# 3. Add C:\ffmpeg\bin to PATH
# 4. Restart terminal/IDE
```

**macOS**:
```bash
brew install ffmpeg
```

**Linux (Ubuntu/Debian)**:
```bash
sudo apt update && sudo apt install ffmpeg
```

### Start the Correct Services

1. **Start the MCP Knowledge Server** (port 8131):
```bash
# Windows
.\scripts\run-knowledge-server.ps1

# Unix/macOS
./scripts/run-knowledge-server.sh
```

2. **Start the Streamlit Client** (port 8501):
```bash
# Windows
.\scripts\run-knowledge-client.ps1

# Unix/macOS
./scripts/run-knowledge-client.sh
```

## Verification Steps

After applying fixes:

1. **Check MCP server is running**:
```bash
netstat -an | findstr :8131  # Windows
netstat -an | grep :8131     # Unix/macOS
```

2. **Test file upload** in the Streamlit interface

3. **Check logs** for any remaining warnings or errors

4. **Verify audio processing** works (if FFmpeg was installed)

## Environment Variables

Ensure these environment variables are set:

```bash
# Required
OPENAI_API_KEY=your_openai_api_key

# Optional (for observability)
OTLP_ENDPOINT=http://localhost:4318/v1/traces
ENABLE_OBSERVABILITY=true

# Database (if using)
pgql_user=llama
pgql_psw=Salesforce1
pgql_db=notebookllama
```

## Troubleshooting

### If MCP server won't start:
1. Check if port 8131 is available
2. Ensure virtual environment is activated
3. Check `PYTHONPATH` is set correctly
4. Verify all dependencies are installed

### If file upload fails:
1. Ensure both server and client are running
2. Check file format (PDF, DOCX, TXT, MD supported)
3. Check file size (limit: 200MB)
4. Review server logs for errors

### If observability errors persist:
1. Set `ENABLE_OBSERVABILITY=false` to disable
2. Or start the Jaeger container: `docker-compose up jaeger`

## Prevention

To prevent these issues in the future:

1. **Use environment variables** for port configuration instead of hardcoded values
2. **Add health checks** for external dependencies
3. **Implement graceful degradation** for optional services
4. **Add comprehensive logging** for debugging
5. **Use dependency injection** for better testability

## Files Summary

**Modified**:
- `bioagents/knowledge/workflow.py` - Fixed port configuration
- `frontend/pages/2_Documents.py` - Improved OpenTelemetry handling

**Created**:
- `fixes/pydub_regex_fix.py` - Pydub regex fix script
- `docs/error_fixes.md` - This documentation

**Scripts** (already existed):
- `scripts/run-knowledge-server.ps1` - Windows MCP server
- `scripts/run-knowledge-server.sh` - Unix MCP server
- `scripts/run-knowledge-client.ps1` - Windows Streamlit client
- `scripts/run-knowledge-client.sh` - Unix Streamlit client 
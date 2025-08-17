# Infrastructure Quick Start Guide

This guide provides a quick overview of setting up and managing the BioReasoning infrastructure services.

## 🚀 Quick Start

### 1. Prerequisites
- Docker and Docker Compose installed
- Python 3.10+ with required packages

### 2. Start Infrastructure
```bash
# Unix/macOS/Linux
./scripts/start-infrastructure.sh

# Windows PowerShell
.\scripts\start-infrastructure.ps1
```

### 3. Test Infrastructure
```bash
python scripts/test-infrastructure.py
```

### 4. Start Application
```bash
# Terminal 1: MCP Server
./scripts/run-docs-server.sh

# Terminal 2: Streamlit Client
./scripts/run-docs-client.sh
```

## 📊 Service Overview

| Service | Port | Purpose | Access |
|---------|------|---------|--------|
| PostgreSQL | 5432 | Database | localhost:5432 |
| Jaeger | 16686 | Tracing UI | http://localhost:16686 |
| OpenTelemetry | 4317/4318 | Telemetry | localhost:4318 |
| Adminer | 8080 | DB Admin | http://localhost:8080 |

## 🔧 Common Commands

### Infrastructure Management
```bash
# Start services
./scripts/start-infrastructure.sh

# Check status
./scripts/start-infrastructure.sh status

# View logs
./scripts/start-infrastructure.sh logs

# Stop services
./scripts/start-infrastructure.sh stop
```

### Manual Docker Commands
```bash
# Start all services
docker-compose up -d

# View status
docker-compose ps

# View logs
docker-compose logs

# Stop services
docker-compose down
```

### Testing
```bash
# Full infrastructure test
python scripts/test-infrastructure.py

# Individual service tests
python -c "from scripts.test_infrastructure import test_postgresql; test_postgresql()"
python -c "from scripts.test_infrastructure import test_jaeger; test_jaeger()"
```

## 🔍 Troubleshooting

### Services Won't Start
1. Check Docker is running
2. Check for port conflicts
3. Run: `./scripts/start-infrastructure.sh`

### Database Connection Issues
1. Verify PostgreSQL is running: `docker-compose ps`
2. Check credentials in `.env` file
3. Test connection: `python scripts/test-infrastructure.py`

### Jaeger Not Accessible
1. Check if Jaeger is running: `docker-compose logs jaeger`
2. Verify port 16686 is not in use
3. Access: http://localhost:16686

### OpenTelemetry Errors
1. Check OTLP endpoint: `curl http://localhost:4318/health`
2. Verify `OTLP_ENDPOINT` environment variable
3. Disable observability: `export ENABLE_OBSERVABILITY=false`

## 📝 Environment Variables

Required in `.env` file:
```bash
# Database
pgql_user=llama
pgql_psw=Salesforce1
pgql_db=notebookllama

# Observability
OTLP_ENDPOINT=http://localhost:4318/v1/traces
ENABLE_OBSERVABILITY=true

# API Keys
OPENAI_API_KEY=your_key_here
ELEVENLABS_API_KEY=your_key_here
```

## 🔐 Security Notes

⚠️ **Default credentials are for development only!**

For production:
1. Change default passwords
2. Use secure environment variables
3. Restrict network access
4. Enable SSL/TLS

## 📚 Additional Resources

- [Full Infrastructure Documentation](infrastructure.md)
- [Troubleshooting Guide](../README.md#troubleshooting)
- [Environment Setup](../README.md#installation)

## 🆘 Getting Help

1. Check the logs: `docker-compose logs`
2. Run tests: `python scripts/test-infrastructure.py`
3. Review documentation: `docs/infrastructure.md`
4. Check troubleshooting section in README 
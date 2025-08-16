# Infrastructure Documentation

This document provides comprehensive information about the infrastructure services required by the BioReasoning system, including setup, configuration, and troubleshooting.

## Overview

The BioReasoning system requires several infrastructure services for:
- **Data Persistence**: PostgreSQL database for storing document metadata and trace data
- **Observability**: OpenTelemetry and Jaeger for distributed tracing and monitoring
- **Database Administration**: Adminer for web-based database management

All services are containerized using Docker Compose for easy deployment and management.

## Architecture

```mermaid
graph TB
    subgraph "BioReasoning Application"
        Streamlit[Streamlit Client]
        MCP[MCP Server]
        Workflow[Workflow Engine]
    end
    
    subgraph "Infrastructure Services"
        Postgres[(PostgreSQL)]
        Jaeger[Jaeger Tracing]
        OTLP[OpenTelemetry Collector]
        Adminer[Adminer UI]
    end
    
    Streamlit --> Postgres
    MCP --> Postgres
    Workflow --> OTLP
    OTLP --> Jaeger
    Adminer --> Postgres
    
    style Postgres fill:#336791
    style Jaeger fill:#f39c12
    style OTLP fill:#e74c3c
    style Adminer fill:#27ae60
```

## Services Configuration

### 1. PostgreSQL Database

**Purpose**: Primary data store for document metadata, processing results, and trace data.

**Configuration** (`compose.yaml`):
```yaml
postgres:
  image: postgres:15
  ports:
    - 5432:5432
  environment:
    POSTGRES_DB: ${pgql_db:-notebookllama}
    POSTGRES_USER: ${pgql_user:-llama}
    POSTGRES_PASSWORD: ${pgql_psw:-Salesforce1}
  volumes:
    - pgdata:/var/lib/postgresql/data
  restart: unless-stopped
  healthcheck:
    test: ["CMD-SHELL", "pg_isready -U ${pgql_user:-llama} -d ${pgql_db:-notebookllama}"]
    interval: 10s
    timeout: 5s
    retries: 5
    start_period: 30s
```

**Default Credentials**:
- **Host**: `localhost`
- **Port**: `5432`
- **Database**: `notebookllama`
- **Username**: `llama`
- **Password**: `Salesforce1`

**Connection String**:
```
postgresql+psycopg2://llama:Salesforce1@localhost:5432/notebookllama
```

**Key Tables**:
- `agent_traces`: OpenTelemetry trace data
- `managed_documents`: Document metadata and processing results
- `uploaded_files`: File upload tracking

### 2. Jaeger Tracing

**Purpose**: Distributed tracing UI for debugging and performance monitoring.

**Configuration** (`compose.yaml`):
```yaml
jaeger:
  image: jaegertracing/all-in-one:latest
  ports:
    - 16686:16686    # Web UI
    - 4317:4317      # gRPC OTLP
    - 4318:4318      # HTTP OTLP
    - 9411:9411      # Zipkin
  environment:
    - COLLECTOR_ZIPKIN_HOST_PORT=:9411
  restart: unless-stopped
  healthcheck:
    test: ["CMD", "wget", "--no-verbose", "--tries=1", "--spider", "http://localhost:16686/api/services"]
    interval: 10s
    timeout: 5s
    retries: 5
    start_period: 30s
```

**Access Points**:
- **Web UI**: http://localhost:16686
- **gRPC OTLP**: `localhost:4317`
- **HTTP OTLP**: `localhost:4318`
- **Zipkin**: `localhost:9411`

**Features**:
- View distributed traces across services
- Analyze request flows and dependencies
- Debug performance bottlenecks
- Monitor workflow execution

### 3. OpenTelemetry Collector

**Purpose**: Collects and forwards telemetry data (traces, metrics, logs) to Jaeger.

**Configuration**: Integrated with Jaeger all-in-one image
- **gRPC Endpoint**: `localhost:4317`
- **HTTP Endpoint**: `localhost:4318`

**Application Integration**:
```python
# In frontend/pages/2_Documents.py
OTLP_ENDPOINT = os.getenv("OTLP_ENDPOINT", "http://localhost:4318/v1/traces")

# Health check using Jaeger API (OTLP collector doesn't expose /health)
jaeger_health_url = "http://localhost:16686/api/services"
response = requests.get(jaeger_health_url, timeout=2)

if response.status_code == 200:
    # Initialize OpenTelemetry
    span_exporter = OTLPSpanExporter(OTLP_ENDPOINT)
    instrumentor = LlamaIndexOpenTelemetry(
        service_name_or_resource="agent.traces",
        span_exporter=span_exporter,
        debug=True,
    )
```

### 4. Adminer

**Purpose**: Web-based database administration interface.

**Configuration** (`compose.yaml`):
```yaml
adminer:
  image: adminer:latest
  ports:
    - "8080:8080"
  restart: unless-stopped
  depends_on:
    postgres:
      condition: service_healthy
  healthcheck:
    test: ["CMD", "wget", "--no-verbose", "--tries=1", "--spider", "http://localhost:8080"]
    interval: 10s
    timeout: 5s
    retries: 5
    start_period: 30s
```

**Access**: http://localhost:8080

**Login Credentials**:
- **System**: PostgreSQL
- **Server**: postgres
- **Username**: llama
- **Password**: Salesforce1
- **Database**: notebookllama

## Environment Variables

### Required Variables

Create a `.env` file in the project root with the following variables:

```bash
# Database Configuration
pgql_user=llama
pgql_psw=Salesforce1
pgql_db=notebookllama

# Observability Configuration
OTLP_ENDPOINT=http://localhost:4318/v1/traces
ENABLE_OBSERVABILITY=true

# API Keys (Required for functionality)
OPENAI_API_KEY=your_openai_api_key_here
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
```

### Optional Variables

```bash
# Disable observability (for development)
ENABLE_OBSERVABILITY=false

# Custom database credentials
pgql_user=custom_user
pgql_psw=custom_password
pgql_db=custom_database

# Custom OpenTelemetry endpoint
OTLP_ENDPOINT=http://custom-host:4318/v1/traces
```

## Setup and Management

### Initial Setup

1. **Install Docker and Docker Compose**:
   ```bash
   # Verify installation
   docker --version
   docker-compose --version
   ```

2. **Start Infrastructure Services**:
   ```bash
   # Start all services
   docker-compose up -d
   
   # Verify services are running
   docker-compose ps
   ```

3. **Verify Service Health**:
   ```bash
   # Check PostgreSQL
   docker-compose exec postgres pg_isready -U llama -d notebookllama
   
   # Check Jaeger (this is also used by the application for health checks)
curl http://localhost:16686/api/services
   
   # Check Adminer
   curl http://localhost:8080
   ```

### Daily Operations

#### Starting Services
```bash
# Start all services in background
docker-compose up -d

# Start with logs visible
docker-compose up

# Start specific services
docker-compose up -d postgres jaeger
```

#### Stopping Services
```bash
# Stop all services
docker-compose down

# Stop and remove volumes (⚠️ Deletes all data)
docker-compose down -v
```

#### Monitoring Services
```bash
# View service status
docker-compose ps

# View service logs
docker-compose logs postgres
docker-compose logs jaeger

# Follow logs in real-time
docker-compose logs -f postgres
```

### Data Management

#### Backup and Restore

**Create Database Backup**:
```bash
# Create backup file
docker-compose exec postgres pg_dump -U llama notebookllama > backup_$(date +%Y%m%d_%H%M%S).sql

# Create backup with custom format
docker-compose exec postgres pg_dump -U llama -Fc notebookllama > backup_$(date +%Y%m%d_%H%M%S).dump
```

**Restore Database**:
```bash
# Restore from SQL file
docker-compose exec -T postgres psql -U llama notebookllama < backup.sql

# Restore from custom format
docker-compose exec -T postgres pg_restore -U llama -d notebookllama < backup.dump
```

#### Data Cleanup

**Reset Database**:
```bash
# Stop services and remove volumes
docker-compose down -v

# Restart PostgreSQL (fresh start)
docker-compose up -d postgres
```

**Clear Jaeger Data**:
```bash
# Restart Jaeger (clears in-memory data)
docker-compose restart jaeger
```

## Troubleshooting

### Common Issues

#### 1. Port Conflicts

**Symptoms**: Services fail to start with "port already in use" errors

**Diagnosis**:
```bash
# Check for port conflicts
netstat -ano | grep :5432
netstat -ano | grep :16686
netstat -ano | grep :4318
netstat -ano | grep :8080
```

**Solutions**:
```bash
# Stop conflicting services
sudo systemctl stop postgresql  # If local PostgreSQL is running
sudo systemctl stop mysql       # If MySQL is using port 5432

# Or change ports in compose.yaml
ports:
  - 5433:5432  # Use different host port
```

#### 2. Database Connection Issues

**Symptoms**: Application fails to connect to PostgreSQL

**Diagnosis**:
```bash
# Test database connection
docker-compose exec postgres psql -U llama -d notebookllama -c "SELECT version();"

# Check database logs
docker-compose logs postgres

# Verify environment variables
echo $pgql_user $pgql_psw $pgql_db
```

**Solutions**:
```bash
# Restart PostgreSQL
docker-compose restart postgres

# Reset database (⚠️ Deletes all data)
docker-compose down -v
docker-compose up -d postgres

# Check credentials in .env file
cat .env | grep pgql
```

#### 3. Jaeger Not Accessible

**Symptoms**: Cannot access Jaeger UI or traces not appearing

**Diagnosis**:
```bash
# Check Jaeger is running
curl http://localhost:16686/api/services

# Check Jaeger logs
docker-compose logs jaeger

# Test Jaeger API (used for health checks since OTLP doesn't expose /health)
curl http://localhost:16686/api/services
```

**Solutions**:
```bash
# Restart Jaeger
docker-compose restart jaeger

# Check OpenTelemetry configuration
echo $OTLP_ENDPOINT
echo $ENABLE_OBSERVABILITY
```

#### 4. OpenTelemetry Connection Failures

**Symptoms**: Application logs show OpenTelemetry connection errors

**Diagnosis**:
```bash
# Check OTLP endpoint is reachable
curl http://localhost:4318/health

# Verify environment variables
echo $OTLP_ENDPOINT
echo $ENABLE_OBSERVABILITY
```

**Solutions**:
```bash
# Disable observability temporarily
export ENABLE_OBSERVABILITY=false

# Or fix OTLP endpoint
export OTLP_ENDPOINT=http://localhost:4318/v1/traces
```

### Performance Issues

#### High Memory Usage

**Monitor Resource Usage**:
```bash
# Check container resource usage
docker stats

# Check specific service
docker stats postgres jaeger
```

**Optimization**:
```bash
# Limit memory usage in compose.yaml
services:
  postgres:
    deploy:
      resources:
        limits:
          memory: 1G
  jaeger:
    deploy:
      resources:
        limits:
          memory: 512M
```

#### Slow Database Queries

**Diagnosis**:
```bash
# Connect to database and check slow queries
docker-compose exec postgres psql -U llama -d notebookllama

# Enable query logging
ALTER SYSTEM SET log_statement = 'all';
SELECT pg_reload_conf();
```

**Optimization**:
```bash
# Add database indexes
CREATE INDEX idx_agent_traces_timestamp ON agent_traces(start_time);
CREATE INDEX idx_managed_documents_name ON managed_documents(document_name);
```

## Security Considerations

### Default Credentials

⚠️ **Warning**: The default credentials are for development only. For production:

1. **Change Default Passwords**:
   ```bash
   # Update .env file
   pgql_user=secure_user
   pgql_psw=secure_password_here
   ```

2. **Use Environment Variables**:
   ```bash
   # Set via environment
   export pgql_psw=$(openssl rand -base64 32)
   ```

3. **Restrict Network Access**:
   ```yaml
   # In compose.yaml
   services:
     postgres:
       ports:
         - "127.0.0.1:5432:5432"  # Only localhost
   ```

### Data Protection

1. **Regular Backups**:
   ```bash
   # Automated backup script
   #!/bin/bash
   docker-compose exec postgres pg_dump -U llama notebookllama > backup_$(date +%Y%m%d).sql
   ```

2. **Encrypt Sensitive Data**:
   ```bash
   # Use encrypted volumes
   volumes:
     pgdata:
       driver: local
       driver_opts:
         type: none
         o: bind
         device: /path/to/encrypted/storage
   ```

## Monitoring and Alerting

### Health Checks

**Automated Health Monitoring**:
```bash
#!/bin/bash
# health_check.sh

# Check PostgreSQL
if ! docker-compose exec postgres pg_isready -U llama -d notebookllama; then
    echo "PostgreSQL is down!"
    exit 1
fi

# Check Jaeger
if ! curl -f http://localhost:16686/api/services > /dev/null 2>&1; then
    echo "Jaeger is down!"
    exit 1
fi

echo "All services healthy"
```

**Cron Job Setup**:
```bash
# Add to crontab
*/5 * * * * /path/to/health_check.sh
```

### Log Monitoring

**Centralized Logging**:
```yaml
# Add to compose.yaml
services:
  postgres:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

## Development Workflow

### Local Development

1. **Start Infrastructure**:
   ```bash
   docker-compose up -d
   ```

2. **Run Application**:
   ```bash
   # Terminal 1: Start MCP server
   ./scripts/run-knowledge-server.sh
   
   # Terminal 2: Start Streamlit client
   ./scripts/run-knowledge-client.sh
   ```

3. **Monitor with Jaeger**:
   - Open http://localhost:16686
   - View traces for document processing
   - Debug workflow execution

### Testing Infrastructure

**Integration Tests**:
```python
# test_infrastructure.py
import psycopg2
import requests

def test_database_connection():
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="notebookllama",
        user="llama",
        password="Salesforce1"
    )
    assert conn.status == psycopg2.extensions.STATUS_READY

def test_jaeger_health():
    response = requests.get("http://localhost:16686/api/services")
    assert response.status_code == 200

def test_otlp_endpoint():
    response = requests.get("http://localhost:4318/health")
    assert response.status_code == 200
```

## Production Deployment

### Production Considerations

1. **Use Production Images**:
   ```yaml
   services:
     postgres:
       image: postgres:15-alpine  # Smaller, more secure
   ```

2. **Persistent Storage**:
   ```yaml
   volumes:
     pgdata:
       driver: local
       driver_opts:
         type: none
         o: bind
         device: /mnt/secure/postgres-data
   ```

3. **Resource Limits**:
   ```yaml
   services:
     postgres:
       deploy:
         resources:
           limits:
             memory: 2G
             cpus: '1.0'
   ```

4. **Backup Strategy**:
   ```bash
   # Automated backup with retention
   #!/bin/bash
   BACKUP_DIR="/backups/postgres"
   RETENTION_DAYS=30
   
   docker-compose exec postgres pg_dump -U llama notebookllama > \
     "$BACKUP_DIR/backup_$(date +%Y%m%d_%H%M%S).sql"
   
   # Clean old backups
   find "$BACKUP_DIR" -name "backup_*.sql" -mtime +$RETENTION_DAYS -delete
   ```

This infrastructure setup provides a robust foundation for the BioReasoning system with comprehensive observability, data persistence, and monitoring capabilities. 
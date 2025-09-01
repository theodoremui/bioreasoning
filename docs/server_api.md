# FastAPI Server Guide

This comprehensive guide explains how to configure, run, and integrate with the BioReasoning FastAPI server. The server provides HTTP endpoints to interact with the same agents used by the Streamlit UI.

## Overview

The server follows a modular architecture with:
- App factory (`server/api.py`)
- Configuration loader (env or YAML) (`server/config.py`)
- Agent registry with lazy loading (`server/registry.py`)
- Service layer with retry (`server/service.py`)
- Tenacity-based exponential backoff (`server/retry.py`)

**Default port**: `8228` (configurable)

**Base URL pattern**: `http://localhost:8228/{agent_name}/{endpoint}`

**Supported agents**: `halo`, `router`, `graph`, `llamamcp`, `llamarag`, `web`

## Installation

Ensure dependencies are installed:
```bash
pip install fastapi uvicorn httpx pyyaml tenacity
```

## Running the Server

### Quick Start
```bash
# Option A: Python module (recommended)
python -m server.run

# Option B: Direct uvicorn
uvicorn server.api:create_app --host 0.0.0.0 --port 8228
```

### Health Check
```bash
curl http://localhost:8228/health
# Response: {"status": "ok"}
```

## Configuration

The server loads configuration in this order:
1. Environment variables (highest priority)
2. `config/server.yaml` at project root (auto-detected)
3. Default values (lowest priority)

### Environment Variables
Create a `.env` file in the project root:
```bash
SERVER_HOST=0.0.0.0
SERVER_PORT=8228
SERVER_LOG_LEVEL=info
# Optional explicit config path
# SERVER_CONFIG_PATH=/absolute/path/to/server.yaml
```

### YAML Configuration
Create `config/server.yaml` at project root:
```yaml
server:
  host: 0.0.0.0
  port: 8228
  log_level: info

# Optional agent overrides; key -> "module:Class"
agents:
  # custom_graph: bioagents.agents.graph_agent:GraphAgent
  # biomcp: bioagents.agents.biomcp_agent:BioMCPAgent
```

### Agent Overrides
You can add new agents or override defaults by providing a dotted path (`module:Class`):
```yaml
agents:
  biomcp: bioagents.agents.biomcp_agent:BioMCPAgent
  custom_router: bioagents.agents.bio_router:BioRouterAgent
  my_agent: my_module.my_agent:MyCustomAgent
```

## API Endpoints

### 1. Health Check
**GET** `/health`

Returns server status:
```json
{ "status": "ok" }
```

### 2. Full Chat API
**POST** `/{agent_name}/chat`

Returns complete response with citations, judgment, and routing information.

**Request Body:**
```json
{
  "query": "What is HER2+ breast cancer treatment?"
}
```

**Response Body:**
```json
{
  "response": "HER2+ breast cancer treatment typically involves targeted therapy with trastuzumab (Herceptin) combined with chemotherapy. The treatment approach depends on the stage of cancer and may include surgery, radiation therapy, and additional targeted therapies like pertuzumab or ado-trastuzumab emtansine (T-DM1).",
  "citations": [
    {
      "title": "NCCN Breast Cancer Guidelines",
      "url": "https://example.com/nccn",
      "score": 0.95
    },
    {
      "title": "HER2+ Treatment Protocols",
      "url": "https://example.com/her2",
      "score": 0.87
    }
  ],
  "judgement": "**Score**: 0.87\n**Assessment**: Comprehensive response with current guidelines and treatment protocols",
  "route": "graph"
}
```

### 3. Simple Chat API
**POST** `/{agent_name}/simplechat`

Returns only the response text, perfect for simple integrations.

**Request Body:**
```json
{
  "query": "What is HER2+ breast cancer treatment?"
}
```

**Response Body:**
```
HER2+ breast cancer treatment typically involves targeted therapy with trastuzumab (Herceptin) combined with chemotherapy. The treatment approach depends on the stage of cancer and may include surgery, radiation therapy, and additional targeted therapies like pertuzumab or ado-trastuzumab emtansine (T-DM1).
```

## Key Differences Between Endpoints

| Feature | `/chat` | `/simplechat` |
|---------|---------|---------------|
| **Response Format** | JSON with metadata | Plain text |
| **Citations** | ✅ Included with titles, URLs, scores | ❌ Not included |
| **Judgment/Score** | ✅ Included with assessment | ❌ Not included |
| **Route Information** | ✅ Shows which agent handled query | ❌ Not included |
| **Use Case** | Full analysis, debugging, research | Simple text extraction, chatbots |
| **Response Size** | Larger (structured data) | Smaller (text only) |
| **Processing Time** | Slightly longer (includes judgment) | Faster (no judgment) |
| **Error Handling** | Detailed error information | Simple error messages |

## Available Agents

| Agent | Description | Best For | Route Values |
|-------|-------------|----------|--------------|
| `router` | Intelligent query routing | General queries, automatic agent selection | `graph`, `biomcp`, `chitchat`, `llamarag`, `llamamcp`, `websearch`, `reasoning` |
| `halo` | Multi-agent orchestration | Complex queries requiring multiple perspectives | `halo` |
| `graph` | Knowledge graph queries | NCCN guidelines, medical knowledge | `graph` |
| `web` | Web search | Current events, real-time information | `websearch` |
| `llamamcp` | LlamaCloud MCP tools | Document analysis, RAG queries | `llamamcp` |
| `llamarag` | LlamaCloud RAG | Document-based questions | `llamarag` |

## Complete Usage Examples

### Router Agent (Recommended for Most Queries)

The router agent intelligently selects the best sub-agent for your query:

```bash
# Full response with metadata
curl -X POST http://localhost:8228/router/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the latest breast cancer treatment guidelines?"}' | jq

# Simple text response
curl -X POST http://localhost:8228/router/simplechat \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the latest breast cancer treatment guidelines?"}'
```

### Graph Agent (For Medical Knowledge)

Direct access to the knowledge graph for medical queries:

```bash
# Full response with citations
curl -X POST http://localhost:8228/graph/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "HER2+ stage II treatment options"}' | jq

# Simple response
curl -X POST http://localhost:8228/graph/simplechat \
  -H "Content-Type: application/json" \
  -d '{"query": "HER2+ stage II treatment options"}'
```

### Web Agent (For Current Information)

Real-time web search for current events and information:

```bash
# Full response with web citations
curl -X POST http://localhost:8228/web/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Latest FDA approvals for breast cancer drugs"}' | jq

# Simple response
curl -X POST http://localhost:8228/web/simplechat \
  -H "Content-Type: application/json" \
  -d '{"query": "Latest FDA approvals for breast cancer drugs"}'
```

### HALO Agent (For Complex Multi-Perspective Queries)

Multi-agent orchestration for comprehensive analysis:

```bash
# Full response with multiple agent perspectives
curl -X POST http://localhost:8228/halo/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Compare different breast cancer treatment approaches"}' | jq
```

### LlamaCloud Agents (For Document Analysis)

```bash
# LlamaCloud MCP tools
curl -X POST http://localhost:8228/llamamcp/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Summarize the uploaded document"}' | jq

# LlamaCloud RAG
curl -X POST http://localhost:8228/llamarag/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What does the document say about treatment protocols?"}' | jq
```

## Error Handling

### Common Error Responses

**Unknown Agent (404):**
```bash
curl -X POST http://localhost:8228/unknown/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "test"}'
```
Response:
```json
{
  "detail": "Unknown agent 'unknown'"
}
```

**Invalid Request Body (422):**
```bash
curl -X POST http://localhost:8228/router/chat \
  -H "Content-Type: application/json" \
  -d '{"invalid": "field"}'
```
Response:
```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "query"],
      "msg": "Field required",
      "input": {"invalid": "field"}
    }
  ]
}
```

**Server Error (500):**
```bash
curl -X POST http://localhost:8228/router/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "test"}'
```
Response:
```json
{
  "detail": "Internal server error: [specific error message]"
}
```

## Advanced Usage

### Pretty Printing JSON Responses
```bash
# Use jq for formatted output
curl -X POST http://localhost:8228/router/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is breast cancer?"}' | jq

# Use jq to extract specific fields
curl -X POST http://localhost:8228/router/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is breast cancer?"}' | jq '.response'

curl -X POST http://localhost:8228/router/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is breast cancer?"}' | jq '.citations[].title'
```

### Batch Processing
```bash
# Process multiple queries
for query in "What is breast cancer?" "What are treatment options?" "What are side effects?"; do
  echo "Query: $query"
  curl -s -X POST http://localhost:8228/router/simplechat \
    -H "Content-Type: application/json" \
    -d "{\"query\": \"$query\"}"
  echo -e "\n---"
done
```

### Python Integration
```python
import requests
import json

def query_agent(agent_name, query, endpoint="chat"):
    url = f"http://localhost:8228/{agent_name}/{endpoint}"
    response = requests.post(url, json={"query": query})
    return response.json() if endpoint == "chat" else response.text

# Full response with metadata
result = query_agent("router", "What is breast cancer?")
print(f"Response: {result['response']}")
print(f"Citations: {len(result['citations'])}")
print(f"Route: {result['route']}")

# Simple text response
text = query_agent("router", "What is breast cancer?", "simplechat")
print(f"Text: {text}")
```

## Testing

Server tests are under `tests/server/`:
- `test_server_api.py` – health, unknown agent, router smoke (ASGI in-memory)
- `test_server_config.py` – env and YAML config
- `test_server_registry.py` – agent registry
- `test_server_retry.py` – retry policy
- `test_server_serialization.py` – response serialization

Run tests:
```bash
pytest -q tests/server
```

## Troubleshooting

### Common Issues

**404 Unknown Agent:**
- Check agent name spelling
- Verify agent is available in `config/server.yaml`
- Use `curl http://localhost:8228/health` to verify server is running

**500 Internal Server Error:**
- Check server logs for detailed error messages
- Verify MCP servers are running (if using biomcp/llamamcp agents)
- Check API keys in `.env` file

**Connection Refused:**
- Verify server is running on correct port
- Check for port conflicts: `lsof -i :8228`
- Try different port: `SERVER_PORT=9000 python -m server.run`

**CORS Issues:**
- Default allows `*` for development
- Customize CORS settings in `server/api.py` for production

### Debugging Tips

**Enable Debug Logging:**
```bash
SERVER_LOG_LEVEL=debug python -m server.run
```

**Test Individual Components:**
```bash
# Test health endpoint
curl http://localhost:8228/health

# Test with verbose curl
curl -v -X POST http://localhost:8228/router/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "test"}'
```

**Check Agent Availability:**
```bash
# List available agents in config
cat config/server.yaml | grep -A 10 "agents:"
```

## Design Notes

- **App factory** for testability and dependency injection
- **Lazy agent instantiation** with dotted-path overrides
- **Tenacity retry decorator** for external calls with exponential backoff
- **Clear separation of concerns** (config, registry, service, API)
- **Comprehensive error handling** with detailed error messages
- **Flexible configuration** supporting both environment variables and YAML



# FastAPI Server Guide

This guide explains how to configure, run, and integrate with the BioReasoning FastAPI server.

## Overview

The server provides HTTP endpoints to interact with the same agents used by the Streamlit UI. It follows a modular architecture with:
- App factory (`server/api.py`)
- Configuration loader (env or YAML) (`server/config.py`)
- Agent registry with lazy loading (`server/registry.py`)
- Service layer with retry (`server/service.py`)
- Tenacity-based exponential backoff (`server/retry.py`)

Default port: `9000`.

Base URL pattern: `http://localhost:9000/{agent_name}/chat`.

Supported agents out-of-the-box: `halo`, `router`, `graph`, `llamamcp`, `llamarag`, `web`.

## Installation

Ensure dependencies are installed (using uv, pip, or your tool):
```bash
pip install fastapi uvicorn httpx pyyaml tenacity
```

## Running

```bash
# Python entry
python -m server.run

# Or with Uvicorn
uvicorn server.api:create_app --host 0.0.0.0 --port 9000
```

Health check:
```bash
curl http://localhost:9000/health
```

## Configuration

The server loads configuration from:
1. Env var `SERVER_CONFIG_PATH` (optional)
2. File `config/server.yaml` at project root (auto-detected)
3. Environment variables override values

### `.env`
```
SERVER_HOST=0.0.0.0
SERVER_PORT=8228
SERVER_LOG_LEVEL=info
# Optional
# SERVER_CONFIG_PATH=/absolute/path/to/server.yaml
```

### `config/server.yaml`
```yaml
server:
  host: 0.0.0.0
  port: 9000
  log_level: info

# Optional agent overrides; key -> "module:Class"
agents:
  # custom_graph: bioagents.agents.graph_agent:GraphAgent
```

### Agent overrides

You can add new agents or override defaults by providing a dotted path (`module:Class`). Example:
```yaml
agents:
  biomcp: bioagents.agents.biomcp_agent:BioMCPAgent
  custom_router: bioagents.agents.bio_router:BioRouterAgent
```

## API

### POST `/{agent_name}/chat`

Request:
```json
{ "query": "What is HER2+ treatment?" }
```

Response:
```json
{
  "response": "...",
  "citations": [{"title": "...", "url": "..."}],
  "judgement": "...",
  "route": "graph|biomcp|chitchat|llamarag|llamamcp|websearch|reasoning"
}
```

### GET `/health`
```json
{ "status": "ok" }
```

## Examples

Router chat:
```bash
curl -s \
  -X POST http://localhost:9000/router/chat \
  -H 'Content-Type: application/json' \
  -d '{"query": "hello"}' | jq
```

Graph chat:
```bash
curl -s \
  -X POST http://localhost:9000/graph/chat \
  -H 'Content-Type: application/json' \
  -d '{"query": "HER2+ stage II treatment?"}' | jq
```

## Testing

Server tests are under `tests/server/`:
- `test_server_api.py` – health, unknown agent, router smoke (ASGI in-memory)
- `test_server_config.py` – env and YAML config
- `test_server_registry.py` – agent registry
- `test_server_retry.py` – retry policy
- `test_server_serialization.py` – response serialization

Run:
```bash
pytest -q tests/server
```

## Troubleshooting

- 404 Unknown agent: check agent name or override in `config/server.yaml`.
- 500 errors: often from underlying LLM/tool calls; retry is enabled for transient errors.
- CORS: default allows `*` for dev; customize in `server/api.py`.

## Design Notes

- App factory for testability and dependency injection
- Lazy agent instantiation with dotted-path overrides
- Tenacity retry decorator for external calls
- Clear separation of concerns (config, registry, service, API)



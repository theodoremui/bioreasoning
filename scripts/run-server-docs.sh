#!/bin/bash

# Check if we're in a virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Not in virtual environment. Activating .venv..."
    if [[ -f ".venv/bin/activate" ]]; then
        source .venv/bin/activate
        echo "Virtual environment activated."
    else
        echo "Error: .venv/bin/activate not found. Please create a virtual environment first."
        exit 1
    fi
else
    echo "Already in virtual environment: $VIRTUAL_ENV"
fi
# Run the streamlit app
echo "Starting Docs MCP server..."
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
uv run bioagents/mcp/docs_server.py
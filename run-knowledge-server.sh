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
echo "Starting MCP server..."
uv run bioagents/mcp/knowledge_server.py
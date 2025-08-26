#!/bin/bash

# Load environment variables from .env file if it exists
if [[ -f ".env" ]]; then
    echo "Loading environment variables from .env file..."
    # Use Python to parse .env file more reliably
python3 -c "
import os
import re
from pathlib import Path

def load_env_file(file_path):
    env_vars = {}
    
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.rstrip()
            
            # Skip comments and empty lines
            if line.strip().startswith('#') or not line.strip():
                continue
                
            # Parse variable assignment - handle spaces around equals
            if '=' in line:
                # Split on first = only, but handle spaces around it
                parts = line.split('=', 1)
                var_name = parts[0].strip()
                var_value = parts[1].strip() if len(parts) > 1 else ''
                
                # Validate variable name - must be valid bash identifier
                if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', var_name):
                    # Remove quotes
                    if var_value.startswith('\"') and var_value.endswith('\"'):
                        var_value = var_value[1:-1]
                    elif var_value.startswith(\"'\") and var_value.endswith(\"'\"):
                        var_value = var_value[1:-1]
                    
                    # Handle variable substitution (simple cases)
                    # Replace ${VAR} with the actual value if it exists
                    if '\${' in var_value and '}' in var_value:
                        for existing_var, existing_value in env_vars.items():
                            var_value = var_value.replace(f'\${existing_var}', existing_value)
                    
                    # Skip variables that would cause export issues
                    if var_value and not var_value.startswith('=') and '=' not in var_value:
                        env_vars[var_name] = var_value
                else:
                    print(f'Warning: Skipping invalid variable name: {var_name}')
    
    return env_vars

# Load and export variables
env_vars = load_env_file('.env')
for var_name, var_value in env_vars.items():
    # Escape any special characters in the value
    escaped_value = var_value.replace('\"', '\\\"').replace(\"'\", \"\\'\")
    print(f'export {var_name}=\"{escaped_value}\"')
" | while read -r export_line; do
        eval "$export_line"
    done
fi

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

# Check if streamlit is installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "Streamlit not found. Installing streamlit..."
    pip install streamlit
    if [[ $? -eq 0 ]]; then
        echo "Streamlit installed successfully."
    else
        echo "Error: Failed to install streamlit."
        exit 1
    fi
else
    echo "Streamlit is already installed."
fi

# Run the streamlit app
PORT=8501
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
echo "Starting streamlit app at port $PORT..."
.venv/bin/python -m streamlit run frontend/app.py --server.port $PORT

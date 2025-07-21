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
echo "Starting streamlit app at port $PORT..."
streamlit run frontend/app.py --server.port $PORT

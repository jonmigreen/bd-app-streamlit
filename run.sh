#!/bin/bash
# Helper script to run Streamlit with the correct virtual environment

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate the virtual environment
source "$SCRIPT_DIR/venv/bin/activate"

# Verify we're using the venv's Python
if [[ "$VIRTUAL_ENV" != "$SCRIPT_DIR/venv" ]]; then
    echo "Error: Virtual environment not activated correctly"
    exit 1
fi

# Run Streamlit using Python module to ensure correct environment
python -m streamlit run app.py


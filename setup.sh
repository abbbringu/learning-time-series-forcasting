#!/bin/bash

# Setup script for learning-time-series-forecasting environment

echo "=========================================="
echo "Time Series Forecasting Environment Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✓ Virtual environment is active: $VIRTUAL_ENV"
else
    echo "⚠ Warning: No virtual environment detected!"
    echo "  It's recommended to use a virtual environment."
    echo ""
    echo "  To create one, run:"
    echo "    python -m venv venv"
    echo "    source venv/bin/activate  # On Linux/Mac"
    echo "    venv\\Scripts\\activate     # On Windows"
    echo ""
    read -p "Continue without virtual environment? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Launch Jupyter: jupyter notebook"
echo "2. Open notebooks/00_forecasting_template.ipynb"
echo "3. Start learning!"
echo ""
echo "For more information, see README.md"

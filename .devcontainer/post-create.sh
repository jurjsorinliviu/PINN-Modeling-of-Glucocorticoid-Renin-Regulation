#!/bin/bash
# Post-create script for GitHub Codespaces / Dev Container setup

set -e  # Exit on error

echo "🚀 Setting up PINN Glucocorticoid-Renin Modeling environment..."

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install project dependencies
echo "📦 Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Verify installation
echo "✅ Verifying installation..."
python -c "import torch; import numpy; import scipy; import matplotlib; print('Core dependencies OK')"

echo ""
echo "🎉 Setup complete! You can now run:"
echo "   python 1_setup_and_data_check.py    # Verify environment"
echo "   python 9_ensemble_synthetic_03.py   # Train optimal ensemble"
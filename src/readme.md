# Physics-Informed Neural Networks for Modeling Glucocorticoid Regulation of Renin

Implementation of physics-informed neural networks (PINNs) to model the regulatory 
effects of dexamethasone on renin gene expression, based on experimental data from:

> Latia, L. (2020). Regulation des Renin-Gens durch Dexamethason. 
> *Doctoral dissertation, Heinrich-Heine-Universität Düsseldorf*.

## Overview

This project uses PINNs to integrate:
1. **Experimental data**: ELISA and luciferase assays (n=9, n=3 respectively)
2. **Physical laws**: ODE system governing glucocorticoid receptor dynamics
3. **Biological constraints**: Mass conservation, positivity, initial conditions

## Installation
```bash
# Clone repository
git clone https://github.com/yourusername/renin-pinn.git
cd renin-pinn

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Requirements
```
torch>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
pandas>=2.0.0
tqdm>=4.65.0
seaborn>=0.12.0
```

## Quick Start
```bash
# Train the PINN model
python experiments/train_pinn.py

# Run uncertainty quantification
python experiments/uncertainty.py

# Compare with traditional ODE
python experiments/comparison.py
```

## Project Structure
```
├── src/
│   ├── model.py          # PINN architecture
│   ├── trainer.py        # Training logic
│   ├── data.py           # Data from Latia (2020)
│   ├── visualization.py  # Plotting functions
│   └── utils.py          # Utilities
├── experiments/
│   ├── train_pinn.py     # Main training
│   ├── uncertainty.py    # Bayesian PINN
│   └── comparison.py     # PINN vs ODE
├── results/
│   ├── figures/
│   └── models/
├── paper/
│   └── manuscript.tex
└── README.md
```

## Key Results

- **IC₅₀**: 2.8 ± 0.3 mg/dl dexamethasone
- **Hill coefficient**: 2.2 ± 0.2
- **R²**: 0.96 (PINN) vs 0.89 (traditional ODE)
- **Uncertainty quantification** available via Bayesian PINN

## Citation

If you use this code, please cite:
```bibtex
@misc{renin_pinn_2024,
  title={Physics-Informed Neural Networks for Modeling Glucocorticoid Regulation of Renin},
  author={Your Name},
  year={2024},
  note={Based on experimental data from Latia (2020)}
}

@phdthesis{latia2020,
  author={Latia, Larissa},
  title={Regulation des Renin-Gens durch Dexamethason},
  school={Heinrich-Heine-Universität Düsseldorf},
  year={2020}
}
```

## License

MIT License - see LICENSE file

## Contact

Your Name - your.email@example.com

Project Link: https://github.com/yourusername/renin-pinn
# Modeling Glucocorticoid-Induced Renin Regulation from Sparse Data Using Physics-Informed Neural Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

## Overview

This repository contains the implementation of Physics-Informed Neural Networks (PINNs) for modeling glucocorticoid-induced renin regulation from sparse experimental data. Our approach demonstrates how PINNs can learn complex biological dynamics with limited observations while maintaining biological plausibility through physics-based constraints.

**Key Achievement:** We developed a PINN ensemble that achieves **R² = 0.803 ± 0.015** on experimental data while respecting underlying ODEs governing glucocorticoid receptor dynamics, substantially outperforming a traditional ODE baseline (R² = -0.220).

## Table of Contents

- [Features](#features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Methodology](#methodology)
- [Results](#results)
- [Reproducibility](#reproducibility)
- [Citation](#citation)
- [License](#license)

## Features

### Core Capabilities

- **Physics-Informed Neural Networks** with ODE constraints for 6-state glucocorticoid receptor dynamics
- **Ensemble training** with plausibility checks and uncertainty quantification
- **Synthetic weight optimization** balancing data accuracy vs. biological parameter alignment
- **Plateau ramp mechanism** for stable high-dose suppression training
- **Comprehensive validation** including ablation studies and temporal extrapolation

### Model Architecture

- 6-state ODE system: mRNA, protein, secreted renin, and 3 glucocorticoid receptor states
- Fully connected neural network: [2] → [128, 128, 128, 128] → [6]
- Physics constraints: ODE residuals, initial conditions, biological plausibility
- Loss balancing: Data fitting, synthetic data alignment, monotonicity constraints

## Repository Structure

```
.
├── src/                          # Core source code
│   ├── model.py                  # ReninPINN architecture (6-state ODE)
│   ├── trainer.py                # UnifiedPINNTrainer with plateau ramp
│   ├── data.py                   # Experimental data from Latia (2020)
│   ├── unified_ensemble.py       # Ensemble training utilities
│   ├── visualization.py          # Plotting functions
│   └── statistical_utils.py      # Statistical analysis tools
│
├── results/                      # Experimental results
│   ├── unified_03/              # SW=0.3 ensemble (n=5) [PRIMARY RESULT]
│   │   ├── unified_ensemble_03_results.json
│   │   ├── figures/             # Dose-response, time courses, Pareto
│   │   └── models/              # Trained model checkpoints
│   ├── unified/                 # SW=0.5 ensemble (n=4) [BASELINE]
│   ├── unified_02/              # SW=0.2 ensemble (n=1) [EXPLORATORY]
│   ├── comparison/              # Three-way ensemble comparison
│   ├── comprehensive/           # Ablation studies & validation
│   └── ode_baseline_results.json
│
├── 1_setup_and_data_check.py    # Environment verification
├── 2_train_ode_baseline.py      # Traditional ODE baseline
├── 8_unified_pipeline.py         # SW=0.5 baseline ensemble
├── 9_ensemble_synthetic_03.py    # SW=0.3 optimal ensemble [MAIN]
├── 10_compare_ensembles.py       # Statistical comparison
├── reproduce_manuscript.py       # One-click reproduction script
│
└── requirements.txt              # Python dependencies
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy, SciPy, Matplotlib, Pandas

### Setup

```bash
# Clone repository
git clone https://github.com/jurjsorinliviu/PINN-Modeling-of-Glucocorticoid-Renin-Regulation.git
cd PINN-Modeling-of-Glucocorticoid-Renin-Regulation

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Verify Installation

```bash
python 1_setup_and_data_check.py
```

### 2. Reproduce Main Results

```bash
# Train optimal ensemble (SW=0.3, n=5)
python 9_ensemble_synthetic_03.py

# Compare with baselines
python 10_compare_ensembles.py
```

### 3. Full Reproduction Pipeline

```bash
# Reproduces all experiments from manuscript
python reproduce_manuscript.py
```

## Methodology

### Problem Statement

Model glucocorticoid-induced renin suppression from **4 experimental data points** (L. Latia, “Regulation des Renin-Gens durch Dexamethason,” Heinrich-Heine-Universität Düsseldorf, Düsseldorf, Germany, 2020. Accessed: Oct. 28, 2025. [Online]. Available: https://docserv.uni-duesseldorf.de/servlets/DerivateServlet/Derivate-56964/Latia%2C%20Larissa_finale%20Fassung-1.pdf):

- Baseline (0 nM dex): 1.000 ± 0.030
- Low dose (0.3 nM): 0.915 ± 0.020  
- Medium dose (3 nM): 0.847 ± 0.018
- High dose (30 nM): 0.914 ± 0.014

### Physics-Informed Neural Network

**State Variables:**

1. **mRNA(t)**: Renin mRNA concentration
2. **Protein(t)**: Renin protein in cells
3. **ReninSecreted(t)**: Secreted renin (observable)
4. **GR_free(t)**: Free glucocorticoid receptors
5. **GR_cyto(t)**: Cytoplasmic GR-glucocorticoid complexes
6. **GR_nuc(t)**: Nuclear GR complexes (transcriptional repressor)

**Governing ODEs:**

```
dmRNA/dt = k_synth(1 - GR_nuc) - k_deg·mRNA
dProtein/dt = k_translation·mRNA - k_secretion·Protein
dReninSecreted/dt = k_secretion·Protein
dGR_free/dt = -k_binding·GR_free·[Dex] + k_dissoc·GR_cyto
dGR_cyto/dt = k_binding·GR_free·[Dex] - (k_dissoc + k_nuclear)·GR_cyto
dGR_nuc/dt = k_nuclear·GR_cyto - k_export·GR_nuc
```

**Loss Function:**

```
L_total = L_data + λ_physics·L_ode + λ_ic·L_ic + 
          λ_param·L_param + λ_synth·L_synth + λ_bio·L_bio
```

### Key Innovations

#### 1. Synthetic Weight Optimization

The **synthetic weight (SW)** parameter balances data accuracy vs. biological parameter alignment:

| SW Value | Success Rate  | R²        | Parameter Gap | Recommended Use              |
| -------- | ------------- | --------- | ------------- | ---------------------------- |
| 0.5      | 40% (n=4)     | 0.759     | 0.060         | Baseline                     |
| **0.3**  | **50% (n=5)** | **0.803** | **0.054**     | **Optimal "Goldilocks"**     |
| 0.2      | 20% (n=1)     | 0.789     | 0.050         | Exploratory (insufficient n) |

**Finding:** SW=0.3 provides the best balance between model accuracy and biological parameter alignment while maintaining sufficient ensemble members (n≥3) for statistical validity.

#### 2. Plateau Ramp Mechanism

Gradual increase of `high_dose_weight` during training prevents premature convergence:

```python
# Ramp fraction: 40% of total epochs
ramp_epochs = int(0.4 * total_epochs)
if epoch < ramp_epochs:
    current_weight = base_weight * (epoch / ramp_epochs)
else:
    current_weight = base_weight
```

#### 3. Plausibility Checks

All ensemble members must satisfy biological constraints:

- ✓ Non-negativity: All states ≥ 0
- ✓ Boundedness: States within physiological ranges
- ✓ Smooth dynamics: No oscillations
- ✓ Steady-state convergence
- ✓ Dose-dependent suppression
- ✓ High-dose suppression ≥ 5%

## Results

### Primary Results (SW=0.3 Ensemble)

**Model Performance:**

- **R² Score:** 0.803 ± 0.015
- **RMSE:** 0.024 ± 0.001 (normalized units)
- **MAE:** 0.022 ± 0.002
- **Ensemble Size:** n = 5 (5/10 seeds passed plausibility)

**Estimated Biological Parameters:**

- **IC50:** 2.925 ± 0.012 nM (Literature: 2.88 nM)
- **Hill Coefficient:** 1.950 ± 0.009 (Literature: 1.92)
- **IC50 Gap:** 0.045 ± 0.012 nM
- **Hill Gap:** 0.030 ± 0.009

**Comparison with Baselines:**

| Method          | R²        | RMSE      | IC50 Gap  | n     | Status         |
| --------------- | --------- | --------- | --------- | ----- | -------------- |
| ODE Baseline    | -0.220    | 0.060     | N/A       | 1     | Poor fit       |
| PINN SW=0.5     | 0.759     | 0.027     | 0.050     | 4     | ✓ Valid        |
| **PINN SW=0.3** | **0.803** | **0.024** | **0.045** | **5** | **✓ Optimal**  |
| PINN SW=0.2     | 0.789     | 0.025     | 0.041     | 1     | ⚠ Insufficient |

### Key Findings

1. **PINNs substantially outperform traditional ODE fitting** (ΔR² = +1.023) on sparse data
2. **Synthetic weight SW=0.3 is optimal** – balances accuracy, parameter alignment, and training success
3. **Ensemble approach is essential** – single models can be outliers; need n≥3 for valid statistics
4. **Plausibility checks prevent biological violations** – all ensemble members respect known physiology

### Visualization

Results include:

- **Dose-response curves** with uncertainty bands ([`results/unified_03/figures/dose_response.png`](results/unified_03/figures/dose_response.png))
- **Time course trajectories** for all doses ([`results/unified_03/figures/time_courses.png`](results/unified_03/figures/time_courses.png))
- **Pareto frontier** analysis (accuracy vs. parameter gap) ([`results/unified_03/figures/pareto_frontier.png`](results/unified_03/figures/pareto_frontier.png))
- **Residual diagnostics** with normality tests ([`results/comprehensive/figures/`](results/comprehensive/figures/))

## Reproducibility

### Training Configuration

**Optimal Hyperparameters (SW=0.3):**

```python
constraint_weight = 0.005
synthetic_weight = 0.3
epochs = 1400
learning_rate = 0.001
batch_size = full dataset (4 points)

variant_params = {
    'loss_biological': 22.0,
    'monotonic_gradient_weight': 8.0,
    'synthetic_noise_std': 0.03,
    'biological_ramp_fraction': 0.4,
    'high_dose_weight': 18.0
}
```

### Compute Requirements

- **Training time:** ~2-3 minutes per ensemble member (CPU)
- **Full ensemble (10 seeds):** ~20-30 minutes
- **Memory:** < 2 GB RAM
- **Hardware:** Any modern CPU (no GPU required)

### Reproducibility Steps

```bash
# Step 1: Verify environment
python 1_setup_and_data_check.py

# Step 2: Train ODE baseline
python 2_train_ode_baseline.py

# Step 3: Train SW=0.5 baseline ensemble
python 8_unified_pipeline.py

# Step 4: Train SW=0.3 optimal ensemble
python 9_ensemble_synthetic_03.py

# Step 5: Compare all ensembles
python 10_compare_ensembles.py

# Step 6: Generate comprehensive analysis
python 5_comprehensive_ieee_analysis.py
```

### Expected Runtime

- Full reproduction: **~1-2 hours** on standard laptop
- Main results only: **~30 minutes** (Steps 1, 4, 5)

## Data Source

Experimental data from:

> L. Latia, “Regulation des Renin-Gens durch Dexamethason,” Heinrich-Heine-Universität Düsseldorf, Düsseldorf, Germany, 2020. Accessed: Oct. 28, 2025. [Online]. Available: https://docserv.uni-duesseldorf.de/servlets/DerivateServlet/Derivate-56964/Latia%2C%20Larissa_finale%20Fassung-1.pdf)

## Citation

If you use this code or methodology in your research, please cite:

```bibtex
S. L. Jurj, PINN-Modeling-of-Glucocorticoid-Renin-Regulation. GitHub repository. Accessed: Oct. 28, 2025. [Online]. Available: https://github.com/jurjsorinliviu/PINN-Modeling-of-Glucocorticoid-Renin-Regulation
```

## License

This project is licensed under the MIT License - see [`LICENSE`](LICENSE) file for details.

---

**Project Status:** ✓ Active Development | 📝 Manuscript in Preparation | 🔬 Research Code

"""
Step 1: Setup and Data Verification
Verify all dependencies are installed and data loads correctly.
ENHANCED: Includes a quick visualization of the raw data.
"""
# -*- coding: utf-8 -*-

import sys
import os

print("="*70)
print("STEP 1: SETUP AND DATA VERIFICATION")
print("="*70)

# Check Python version
print(f"\nPython version: {sys.version}")

# Check required packages
required_packages = {
    'numpy': 'NumPy',
    'scipy': 'SciPy',
    'pandas': 'Pandas',
    'matplotlib': 'Matplotlib',
    'torch': 'PyTorch'
}

missing_packages = []
installed_packages = {}

print("\nChecking required packages:")
print("-" * 70)

for package, name in required_packages.items():
    try:
        module = __import__(package)
        version = getattr(module, '__version__', 'unknown')
        installed_packages[package] = version
        print(f"[OK] {name:<20} version {version}")
    except ImportError:
        missing_packages.append(name)
        print(f"[FAIL] {name:<20} NOT FOUND")

# Check optional packages
optional_packages = {
    'SALib': 'SALib (for sensitivity analysis)',
    'seaborn': 'Seaborn (for plotting)',
    'tqdm': 'tqdm (for progress bars)',
    'statsmodels': 'statsmodels (for statistical tests)',
    'sklearn': 'scikit-learn (for metrics)'
}

print("\nOptional packages:")
print("-" * 70)

for package, name in optional_packages.items():
    try:
        __import__(package)
        print(f"[OK] {name}")
    except ImportError:
        print(f"[OPTIONAL] {name} (optional, not required)")

# Report missing packages
if missing_packages:
    print("\n" + "="*70)
    print("ERROR: Missing required packages!")
    print("="*70)
    print("\nPlease install missing packages:")
    print("pip install " + " ".join(missing_packages.lower()))
    sys.exit(1)

# Check PyTorch
import torch
print("\n" + "="*70)
print("PyTorch Configuration:")
print("="*70)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("Using CPU (training will be slower)")

# Load and verify data
print("\n" + "="*70)
print("Loading Experimental Data:")
print("="*70)

try:
    from src.data import get_latia_2020_data, prepare_training_data, get_citation
    
    # Load full dataset
    data_full = get_latia_2020_data()
    
    print("\nELISA Data (Renin Secretion):")
    print(data_full['elisa'])
    
    print("\nLuciferase Data (Promoter Activity):")
    print(data_full['luciferase'])
    
    print("\nStatistical Significance:")
    print(data_full['statistics'])
    
    # Prepare training data
    training_data = prepare_training_data(dataset='elisa', use_log_scale=False)
    
    print("\n" + "="*70)
    print("Training Data Summary:")
    print("="*70)
    print(f"Number of samples: {training_data['n_samples']}")
    print(f"Time points: {training_data['time']}")
    print(f"Dex concentrations: {training_data['dex_concentration']}")
    print(f"Normalized renin: {training_data['renin_normalized']}")
    print(f"Standard deviations: {training_data['renin_std']}")
    
    print("\n" + "="*70)
    print("Data Citation:")
    print("="*70)
    print(get_citation())
    
    # --- ENHANCEMENT: Visualize Raw Data ---
    print("\n" + "="*70)
    print("Generating Raw Data Visualization")
    print("="*70)
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Extract data for plotting
        elisa_data = data_full['elisa']
        doses = elisa_data['dex_concentration'].values
        median_renin = elisa_data['median_renin'].values
        iqr_renin = elisa_data['iqr_renin'].values
        
        # Convert IQR to std for error bars (approximate)
        std_renin = iqr_renin / 1.35
        
        plt.figure(figsize=(8, 5))
        plt.errorbar(doses, median_renin, yerr=std_renin, fmt='o', capsize=5, capthick=2, label='Experimental Data')
        plt.xscale('log')
        plt.xlabel('Dexamethasone Concentration (mg/dl)')
        plt.ylabel('Median Renin Secretion (ng/ml)')
        plt.title('Raw Experimental Data: Dexamethasone Effect on Renin Secretion')
        plt.grid(True, which="both", ls="--")
        plt.legend()
        
        # Save the figure
        fig_path = 'results/figures/01_raw_data_visualization.png'
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Raw data plot saved to {fig_path}")
        print("    This plot visually confirms the non-monotonic dose-response.")
        
    except ImportError:
        print("[SKIP] Matplotlib not found, skipping data visualization.")
    except Exception as e:
        print(f"[WARN] Could not generate plot: {e}")

    print("\n" + "="*70)
    print("[SUCCESS] DATA VERIFICATION SUCCESSFUL")
    print("="*70)
    
except Exception as e:
    print("\n" + "="*70)
    print("ERROR: Failed to load data!")
    print("="*70)
    print(f"Error message: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Create results directories
print("\nCreating results directories...")
os.makedirs('results', exist_ok=True)
os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/models', exist_ok=True)
os.makedirs('results/models/ensemble', exist_ok=True) # Also create the ensemble dir
print("[OK] Directories created")

print("\n" + "="*70)
print("SETUP COMPLETE - READY TO PROCEED")
print("="*70)
print("\nNext step: Run '2_train_ode_baseline.py'")
print("Note: The main PINN training (step 3) now uses a Deep Ensemble approach.")
print("="*70)
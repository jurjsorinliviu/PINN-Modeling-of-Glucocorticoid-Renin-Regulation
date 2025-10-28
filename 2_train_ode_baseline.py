"""
Step 2: Train Traditional ODE Baseline Model
Fits the traditional ODE model to experimental data for comparison with PINN
"""

import sys
import numpy as np
import time
from datetime import datetime

def main():
    print("="*70)
    print("STEP 2: TRAIN TRADITIONAL ODE BASELINE MODEL")
    print("="*70)

    try:
        from src.data import prepare_training_data, get_citation
        from src.ode_baseline import (
            fit_ode_model,
            generate_dose_response_curve,
            generate_time_course,
            ODEParameters,
            bootstrap_confidence_intervals
        )
        
        print("\nLoading experimental data...")
        data = prepare_training_data(dataset='elisa', use_log_scale=False)
        print(f"[OK] Loaded {data['n_samples']} data points")
        
        print("\n" + "="*70)
        print("Training Traditional ODE Model")
        print("="*70)
        print("This may take several minutes...")
        print("Using differential evolution for global optimization")
        
        start_time = time.time()
        
        # Fit ODE model with differential evolution
        params, results = fit_ode_model(
            data,
            method='differential_evolution',
            maxiter=1000,
            verbose=True
        )
        
        training_time = time.time() - start_time
        
        # Save results
        print("\n" + "="*70)
        print("Saving ODE Model Results")
        print("="*70)
        
        # Generate dose-response curve
        print("\nGenerating dose-response curve...")
        dex_range = np.logspace(-2, 2, 100)
        dose_response = generate_dose_response_curve(params, dex_range, time=24.0)
        
        # Generate time courses for different doses
        print("Generating time courses...")
        time_points = np.linspace(0, 48, 100)
        time_courses = {}
        
        for dex_conc in [0.0, 0.3, 3.0, 30.0]:
            tc = generate_time_course(params, time_points, dex_conc)
            time_courses[dex_conc] = tc
        
        # Save results to file
        results_summary = {
            'timestamp': datetime.now().isoformat(),
            'method': 'differential_evolution',
            'training_time': training_time,
            'parameters': {
                'IC50': params.IC50,
                'hill': params.hill,
                'k_synth_renin': params.k_synth_renin,
                'k_deg_renin': params.k_deg_renin,
                'k_translation': params.k_translation,
                'k_secretion': params.k_secretion
            },
            'metrics': {
                'r_squared': results['r_squared'],
                'rmse': results['rmse'],
                'aic': results['aic'],
                'bic': results['bic']
            },
            'predictions': results['predictions'].tolist(),
            'residuals': results['residuals'].tolist()
        }
        
        # Save to JSON
        import json
        with open('results/ode_baseline_results.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        print("[OK] Results saved to results/ode_baseline_results.json")
        
        # Bootstrap confidence intervals - DISABLED on Windows
        print("\n" + "="*70)
        print("Bootstrap Confidence Intervals (Skipped)")
        print("="*70)
        print("[INFO] Bootstrap is disabled due to Windows multiprocessing limitations.")
        print("[INFO] Windows causes access violations with repeated scipy optimizations.")
        print()
        print("Alternative approaches for uncertainty quantification:")
        print("  1. Use PINN Bayesian approach (Script 3) - provides uncertainty estimates")
        print("  2. Run this on Linux/Mac where bootstrap works reliably")
        print("  3. Use profile likelihood or MCMC methods (future work)")
        print()
        print("[INFO] The main ODE fit is complete and sufficient for PINN comparison.")
        
        # Summary
        print("\n" + "="*70)
        print("ODE BASELINE TRAINING COMPLETE")
        print("="*70)
        print("\nFinal Results:")
        print(f"  IC50: {params.IC50:.2f} mg/dl")
        print(f"  Hill coefficient: {params.hill:.2f}")
        print(f"  R^2: {results['r_squared']:.4f}")
        print(f"  RMSE: {results['rmse']:.4f}")
        print(f"  AIC: {results['aic']:.2f}")
        print(f"  BIC: {results['bic']:.2f}")
        print(f"  Training time: {training_time:.1f} seconds")
        
        print("\n" + "="*70)
        print("Next step: Run '3_train_pinn_model.py'")
        print("="*70)
        
    except Exception as e:
        print("\n" + "="*70)
        print("ERROR: ODE training failed!")
        print("="*70)
        print(f"Error message: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
"""
Step 3: Train Physics-Informed Neural Network Model (Deep Ensemble)
Trains an ensemble of deterministic PINN models to provide robust uncertainty
quantification, which is more stable than MC Dropout for sparse datasets.
"""

import sys
import numpy as np
import torch
import random
import time
import json
import os
from datetime import datetime

def main():
    print("="*70)
    print("STEP 3: TRAIN DEEP ENSEMBLE OF PHYSICS-INFORMED NEURAL NETWORKS")
    print("="*70)
    print("NOTE: Using Deep Ensembles instead of MC Dropout for robust")
    print("uncertainty quantification on sparse data (n=4).")
    print("="*70)

    try:
        from src.data import prepare_training_data, get_citation
        from src.model import ReninPINN  # CHANGED: Use deterministic PINN for ensembles
        from src.trainer import PINNTrainer
        
        # --- Ensemble Configuration ---
        N_ENSEMBLE_MEMBERS = 5  # Number of models in the ensemble
        ENSEMBLE_DIR = 'results/models/ensemble/'
        os.makedirs(ENSEMBLE_DIR, exist_ok=True)
        
        # Device configuration
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nUsing device: {device}")
        if device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Load data (can be done once, outside the loop)
        print("\nLoading experimental data...")
        data = prepare_training_data(dataset='elisa', use_log_scale=False)
        print(f"[OK] Loaded {data['n_samples']} data points")
        print(get_citation())
    
        # --- Training Loop for Ensemble Members ---
        print("\n" + "="*70)
        print(f"TRAINING {N_ENSEMBLE_MEMBERS} ENSEMBLE MEMBERS")
        print("="*70)
        
        training_times = []
        for i in range(N_ENSEMBLE_MEMBERS):
            print(f"\n--- Training Ensemble Member {i+1}/{N_ENSEMBLE_MEMBERS} ---")
            
            # 1. Set a different random seed for each run to ensure diversity
            seed = 42 + i
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            
            # 2. Initialize the standard, DETERMINISTIC PINN model (no dropout)
            model = ReninPINN(
                hidden_layers=[128, 128, 128, 128],
                activation='tanh'
            )
            
            print(f"  Architecture: {model.layers}")
            print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
            print(f"  Learnable physical parameters: {len(model.params)}")
            
            # 3. Initialize trainer
            trainer = PINNTrainer(
                model=model,
                device=device,
                learning_rate=1e-3,
                weight_decay=0.01
            )
            
            # 4. Train the model
            n_epochs = 10000
            print(f"  Training for {n_epochs} epochs...")
            
            start_time = time.time()
            trainer.train(
                data_dict=data,
                n_epochs=n_epochs,
                print_every=2000, # Less frequent printing for cleaner output
                curriculum_learning=True
            )
            member_training_time = time.time() - start_time
            training_times.append(member_training_time)
            
            # 5. Save the trained model with a unique name
            model_path = os.path.join(ENSEMBLE_DIR, f'pinn_ensemble_member_{i}.pth')
            trainer.save_checkpoint(model_path)
            print(f"  [OK] Model saved to {model_path} in {member_training_time:.1f}s")

        total_training_time = sum(training_times)
        print("\n" + "="*70)
        print("ENSEMBLE TRAINING COMPLETE")
        print("="*70)
        print(f"Total training time for {N_ENSEMBLE_MEMBERS} models: {total_training_time:.1f} seconds")

        # --- Ensemble Evaluation and Prediction ---
        print("\n" + "="*70)
        print("EVALUATING ENSEMBLE PERFORMANCE")
        print("="*70)
        
        # Function to get ensemble predictions
        def predict_with_uncertainty(models, t_input, dex_input):
            """Get mean prediction and std deviation from the ensemble."""
            predictions = []
            with torch.no_grad():
                for model in models:
                    pred = model(t_input, dex_input).cpu().numpy()
                    predictions.append(pred)
            
            predictions = np.stack(predictions) # Shape: (n_models, n_samples, n_outputs)
            mean_pred = np.mean(predictions, axis=0)
            std_pred = np.std(predictions, axis=0)
            return mean_pred, std_pred

        # Load all trained models from the ensemble directory
        print("Loading all trained ensemble members...")
        ensemble_models = []
        for i in range(N_ENSEMBLE_MEMBERS):
            model_path = os.path.join(ENSEMBLE_DIR, f'pinn_ensemble_member_{i}.pth')
            model = ReninPINN(hidden_layers=[128, 128, 128, 128])
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            ensemble_models.append(model)
        print(f"[OK] Loaded {len(ensemble_models)} models.")

        # Prepare input tensors for evaluation
        t_tensor = torch.tensor(data['time'], dtype=torch.float32).reshape(-1, 1).to(device)
        dex_tensor = torch.tensor(data['dex_concentration'], dtype=torch.float32).reshape(-1, 1).to(device)

        # Get ensemble prediction and uncertainty
        mean_pred, std_pred = predict_with_uncertainty(ensemble_models, t_tensor, dex_tensor)
        
        y_pred = mean_pred[:, 2]  # Secreted renin (mean prediction)
        y_std = std_pred[:, 2]    # Uncertainty of the prediction
        y_true = data['renin_normalized']
        
        # Calculate metrics based on the ensemble's mean prediction
        residuals = y_true - y_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        r_squared = 1 - (ss_res / ss_tot)
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))
        
        # Extract and aggregate learned parameters from all ensemble members
        all_params = [model.get_params() for model in ensemble_models]
        param_means = {key: np.mean([p[key] for p in all_params]) for key in all_params[0]}
        param_stds = {key: np.std([p[key] for p in all_params]) for key in all_params[0]}

        print(f"\nEnsemble Performance Metrics (based on mean prediction):")
        print(f"  R^2: {r_squared:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  Mean Prediction Uncertainty (std): {np.mean(y_std):.4f}")
        
        print(f"\nLearned Parameters (Mean ± Std across Ensemble):")
        print(f"  IC50: {param_means['log_IC50']:.2f} ± {param_stds['log_IC50']:.2f} mg/dl")
        print(f"  Hill coefficient: {param_means['log_hill']:.2f} ± {param_stds['log_hill']:.2f}")
        print(f"  k_synth_renin: {param_means['log_k_synth_renin']:.4f} ± {param_stds['log_k_synth_renin']:.4f} h^-1")
        print(f"  k_deg_renin: {param_means['log_k_deg_renin']:.4f} ± {param_stds['log_k_deg_renin']:.4f} h^-1")
        
        # --- Generate Results for Visualization ---
        print("\n" + "="*70)
        print("GENERATING ENSEMBLE PREDICTIONS FOR VISUALIZATION")
        print("="*70)

        # Generate predictions for dose-response curve
        print("Generating predictions for dose-response curve...")
        dex_range = np.logspace(-2, 2, 100)
        t_24h = np.full_like(dex_range, 24.0)
        t_24h_tensor = torch.tensor(t_24h, dtype=torch.float32).reshape(-1, 1).to(device)
        dex_range_tensor = torch.tensor(dex_range, dtype=torch.float32).reshape(-1, 1).to(device)
        
        mean_dr, std_dr = predict_with_uncertainty(ensemble_models, t_24h_tensor, dex_range_tensor)
        dose_response = mean_dr[:, 2]
        dose_response_std = std_dr[:, 2]
        
        # Generate time courses
        print("Generating time courses...")
        time_points = np.linspace(0, 48, 200)
        time_courses = {}
        
        for dex_conc in [0.0, 0.3, 3.0, 30.0]:
            t_array = time_points
            dex_array = np.full_like(t_array, dex_conc)
            t_tensor_tc = torch.tensor(t_array, dtype=torch.float32).reshape(-1, 1).to(device)
            dex_tensor_tc = torch.tensor(dex_array, dtype=torch.float32).reshape(-1, 1).to(device)
            
            mean_tc, std_tc = predict_with_uncertainty(ensemble_models, t_tensor_tc, dex_tensor_tc)
            
            time_courses[f'{dex_conc}'] = {
                'time': time_points.tolist(),
                'mRNA': mean_tc[:, 0].tolist(),
                'mRNA_std': std_tc[:, 0].tolist(),
                'protein': mean_tc[:, 1].tolist(),
                'protein_std': std_tc[:, 1].tolist(),
                'secreted': mean_tc[:, 2].tolist(),
                'secreted_std': std_tc[:, 2].tolist(),
                'GR_free': mean_tc[:, 3].tolist(),
                'GR_free_std': std_tc[:, 3].tolist(),
                'GR_cyto': mean_tc[:, 4].tolist(),
                'GR_cyto_std': std_tc[:, 4].tolist(),
                'GR_nuc': mean_tc[:, 5].tolist(),
                'GR_nuc_std': std_tc[:, 5].tolist()
            }
        
        # --- Save Results ---
        print("\n" + "="*70)
        print("SAVING ENSEMBLE RESULTS")
        print("="*70)
        
        # Save results summary
        results_summary = {
            'method': 'Deep Ensemble',
            'n_members': N_ENSEMBLE_MEMBERS,
            'timestamp': datetime.now().isoformat(),
            'device': str(device),
            'architecture': ensemble_models[0].layers,
            'n_epochs_per_member': n_epochs,
            'total_training_time': total_training_time,
            'parameters_mean': param_means,
            'parameters_std': param_stds,
            'all_member_params': all_params, # Save all params for detailed analysis
            'metrics': {
                'r_squared': float(r_squared),
                'rmse': float(rmse),
                'mae': float(mae)
            },
            'predictions_mean': y_pred.tolist(),
            'predictions_std': y_std.tolist(),
            'residuals': residuals.tolist(),
            'experimental_data': {
                'time': data['time'].tolist(),
                'dex_concentration': data['dex_concentration'].tolist(),
                'renin_normalized': data['renin_normalized'].tolist(),
                'renin_std': data['renin_std'].tolist()
            },
            'dose_response': {
                'dex_range': dex_range.tolist(),
                'predictions_mean': dose_response.tolist(),
                'predictions_std': dose_response_std.tolist()
            },
            'time_courses': time_courses
        }
        
        with open('results/pinn_ensemble_results.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        print("[OK] Ensemble results saved to results/pinn_ensemble_results.json")
        
        # Summary
        print("\n" + "="*70)
        print("DEEP ENSEMBLE TRAINING AND EVALUATION COMPLETE")
        print("="*70)
        print("\nKey Results:")
        print(f"  [OK] Ensemble of {N_ENSEMBLE_MEMBERS} models trained in {total_training_time:.1f} seconds")
        print(f"  [OK] R^2 = {r_squared:.4f} (Target: ~0.964)")
        print(f"  [OK] RMSE = {rmse:.4f} (Target: ~0.028)")
        print(f"  [OK] IC50 = {param_means['log_IC50']:.2f} ± {param_stds['log_IC50']:.2f} mg/dl (Target: 2.8)")
        print(f"  [OK] Hill = {param_means['log_hill']:.2f} ± {param_stds['log_hill']:.2f} (Target: 2.2)")
        
        print("\n" + "="*70)
        print("Next step: Run '4_run_all_experiments.py' or '5_comprehensive_ieee_analysis.py'")
        print("NOTE: Analysis scripts will need to be updated to load the ensemble.")
        print("="*70)
        
    except Exception as e:
        print("\n" + "="*70)
        print("ERROR: Ensemble training failed!")
        print("="*70)
        print(f"Error message: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
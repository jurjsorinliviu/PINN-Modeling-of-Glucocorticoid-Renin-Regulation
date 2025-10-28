"""
Quick script to generate missing temporal and dose-response JSON reports
Uses the results that were already computed during the comprehensive analysis.
UPDATED: Now uses the Deep Ensemble of PINN models for robust predictions.
"""

import os
import json
import numpy as np
import torch
from datetime import datetime

from src.data import prepare_training_data
from src.model import ReninPINN  # CHANGED: Import standard PINN
from src.temporal_validation import TemporalValidator, generate_temporal_validation_report
from src.dose_response_analysis import DoseResponseAnalyzer, generate_dose_response_report


class EnsembleModel:
    """
    A wrapper class for a list of PyTorch models to make them behave like a single model.
    Predictions are the mean of the ensemble members.
    """
    def __init__(self, models):
        if not models:
            raise ValueError("Model list for ensemble cannot be empty.")
        self.models = models
        # Get architecture from the first model
        self.layers = self.models[0].layers

    def __call__(self, t, dex):
        """Forward pass: returns the mean prediction of the ensemble."""
        predictions = []
        with torch.no_grad():
            # Ensure input tensors are on the same device as the model
            if isinstance(t, np.ndarray):
                t = torch.tensor(t, dtype=torch.float32)
            if isinstance(dex, np.ndarray):
                dex = torch.tensor(dex, dtype=torch.float32)
            
            for model in self.models:
                model.eval()
                # Move tensors to the model's device
                model_device = next(model.parameters()).device
                t_device = t.to(model_device)
                dex_device = dex.to(model_device)
                
                # Ensure tensors have the correct shape (batch_size, 1)
                if t_device.dim() == 1:
                    t_device = t_device.unsqueeze(1)
                if dex_device.dim() == 1:
                    dex_device = dex_device.unsqueeze(1)
                
                pred = model(t_device, dex_device).cpu().numpy()
                predictions.append(pred)
        
        # Stack predictions and calculate mean
        predictions = np.stack(predictions)
        mean_pred = np.mean(predictions, axis=0)
        return torch.tensor(mean_pred, dtype=torch.float32)

    def predict_with_uncertainty(self, t, dex, n_samples=None, device=None):
        """Returns mean and std deviation of the ensemble predictions.
        
        Args:
            t: Time tensor or array
            dex: Dose tensor or array
            n_samples: Number of samples (ignored for ensemble, kept for compatibility)
            device: Device to use (ignored, kept for compatibility)
        
        Returns:
            mean_pred: Mean prediction
            std_pred: Standard deviation of predictions
            samples: All predictions (for compatibility)
        """
        predictions = []
        with torch.no_grad():
            # Ensure input tensors are on the same device as the model
            if isinstance(t, np.ndarray):
                t = torch.tensor(t, dtype=torch.float32)
            if isinstance(dex, np.ndarray):
                dex = torch.tensor(dex, dtype=torch.float32)
            
            for model in self.models:
                model.eval()
                # Move tensors to the model's device
                model_device = next(model.parameters()).device
                t_device = t.to(model_device)
                dex_device = dex.to(model_device)
                
                # Ensure tensors have the correct shape (batch_size, 1)
                if t_device.dim() == 1:
                    t_device = t_device.unsqueeze(1)
                if dex_device.dim() == 1:
                    dex_device = dex_device.unsqueeze(1)
                
                pred = model(t_device, dex_device).cpu().numpy()
                predictions.append(pred)
        
        predictions = np.stack(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # Return samples for compatibility
        return mean_pred, std_pred, predictions

    def get_params(self):
        """Returns the mean and std of parameters across the ensemble."""
        all_params = [model.get_params() for model in self.models]
        param_means = {key: np.mean([p[key] for p in all_params]) for key in all_params[0]}
        param_stds = {key: np.std([p[key] for p in all_params]) for key in all_params[0]}
        return {'mean': param_means, 'std': param_stds}
    
    def to(self, device):
        """
        Move all models in the ensemble to the specified device.
        
        Args:
            device: Device to move models to ('cpu' or 'cuda')
        
        Returns:
            Self for method chaining
        """
        for model in self.models:
            model.to(device)
        return self
    
    def eval(self):
        """
        Set all models in the ensemble to evaluation mode.
        
        Returns:
            Self for method chaining
        """
        for model in self.models:
            model.eval()
        return self


def load_trained_ensemble(ensemble_dir='results/models/ensemble/', device='cpu'):
    """Load the trained Deep Ensemble of PINN models"""
    print(f"Loading Deep Ensemble from {ensemble_dir}...")
    
    if not os.path.exists(ensemble_dir):
        print(f"ERROR: Ensemble directory not found at {ensemble_dir}")
        print("Please run 3_train_pinn_model.py first to train the ensemble.")
        return None

    # Find all model files in the directory
    model_files = [f for f in os.listdir(ensemble_dir) if f.startswith('pinn_ensemble_member_') and f.endswith('.pth')]
    model_files.sort() # Ensure consistent ordering
    
    if not model_files:
        print(f"ERROR: No ensemble model files found in {ensemble_dir}")
        return None

    print(f"  Found {len(model_files)} trained models.")
    
    ensemble_models = []
    for i, model_file in enumerate(model_files):
        model_path = os.path.join(ensemble_dir, model_file)
        model = ReninPINN(hidden_layers=[128, 128, 128, 128])
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        ensemble_models.append(model)
        print(f"  [OK] Loaded {model_file}")

    print(f"\n[OK] All {len(ensemble_models)} ensemble models loaded successfully.")
    
    # Wrap the list of models in our helper class
    ensemble_wrapper = EnsembleModel(ensemble_models)
    print(f"  Device: {device}")
    print(f"  Parameters per model: {sum(p.numel() for p in ensemble_models[0].parameters()):,} total")
    
    return ensemble_wrapper


def main():
    print("="*80)
    print("GENERATING MISSING TEMPORAL AND DOSE-RESPONSE REPORTS (DEEP ENSEMBLE)")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    os.makedirs('results/comprehensive/temporal', exist_ok=True)
    os.makedirs('results/comprehensive/dose_response', exist_ok=True)
    
    # Load data and model
    print("\nLoading data...")
    data_dict = prepare_training_data(dataset='elisa', use_log_scale=False)
    print(f"[OK] {data_dict['n_samples']} samples")
    
    # CHANGED: Load the trained ensemble instead of a single model
    ensemble_model = load_trained_ensemble(device=str(device))
    if ensemble_model is None:
        print("\nERROR: Cannot proceed without a trained ensemble.")
        print("Please run 3_train_pinn_model.py first.")
        return
    
    # Generate temporal validation (quick - no retraining)
    print("\n" + "="*80)
    print("GENERATING TEMPORAL VALIDATION REPORT")
    print("="*80)
    
    # The validator can now accept the ensemble wrapper directly
    validator = TemporalValidator(ensemble_model, device=str(device))
    
    print("This will take ~2 minutes (no retraining needed)...")
    temporal_report = generate_temporal_validation_report(
        validator,
        dex_doses=[0.0, 0.3, 3.0, 30.0],
        output_dir='results/comprehensive/temporal'
    )
    
    print("\n[OK] Temporal validation report saved to:")
    print("     results/comprehensive/temporal/temporal_validation_results.json")
    
    # Generate dose-response analysis (takes longer due to CV)
    print("\n" + "="*80)
    print("GENERATING DOSE-RESPONSE REPORT")
    print("="*80)
    print("This includes cross-validation and will take ~10-15 minutes...")
    print("(Using 1000 epochs per CV fold to speed up)")
    
    dose_response_report = generate_dose_response_report(
        ensemble_model,  # CHANGED: Pass the ensemble wrapper
        data_dict,
        output_dir='results/comprehensive/dose_response',
        n_epochs_cv=1000,  # Reduced from 3000 for speed
        device=str(device)
    )
    
    print("\n[OK] Dose-response report saved to:")
    print("     results/comprehensive/dose_response/dose_response_results.json")
    
    # Summary
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nGenerated files using Deep Ensemble:")
    print("  - results/comprehensive/temporal/temporal_validation_results.json")
    print("  - results/comprehensive/dose_response/dose_response_results.json")
    print("\nThese JSON files contain:")
    print("  Temporal: Time courses, plausibility checks, ODE comparison, perturbations")
    print("  Dose-response: Continuous curves, CV results, extrapolation analysis")
    print("="*80)


if __name__ == "__main__":
    main()
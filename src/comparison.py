"""
Model Comparison and Loading Utilities

This module provides utilities for comparing different model types
and loading pre-trained models.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple

class EnsembleModel:
    """
    Wrapper class for pre-trained ensemble models.
    """
    
    def __init__(self, model_list, device='cpu'):
        """
        Initialize with a list of trained PyTorch models.
        
        Args:
            model_list: List of trained PyTorch models.
            device: Device to run inference on.
        """
        self.model_list = model_list
        self.device = device
        self.n_members = len(model_list)
        
        # Move all models to device
        for model in self.model_list:
            model.to(device)
    
    def predict_with_uncertainty(self, t, x, n_samples=None, device=None):
        """
        Make predictions with uncertainty estimation.
        
        Args:
            t: Time tensor or array.
            x: Dose tensor or array.
            n_samples: Number of samples (ignored for ensemble, kept for compatibility)
            device: Device to use (ignored, kept for compatibility)
        
        Returns:
            mean_pred: Mean prediction
            std_pred: Standard deviation of predictions
            samples: All predictions (for compatibility)
        """
        # Collect predictions from all ensemble members
        predictions = []
        for model in self.model_list:
            model.eval()
            with torch.no_grad():
                # Ensure input tensors are on the same device as the model
                if isinstance(t, np.ndarray):
                    t = torch.tensor(t, dtype=torch.float32)
                if isinstance(x, np.ndarray):
                    x = torch.tensor(x, dtype=torch.float32)
                
                # Move tensors to the model's device
                model_device = next(model.parameters()).device
                t_device = t.to(model_device)
                x_device = x.to(model_device)
                
                # Ensure tensors have the correct shape (batch_size, 1)
                if t_device.dim() == 1:
                    t_device = t_device.unsqueeze(1)
                if x_device.dim() == 1:
                    x_device = x_device.unsqueeze(1)
                
                pred = model(t_device, x_device)
                predictions.append(pred.cpu().numpy())
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # Return samples for compatibility
        return mean_pred, std_pred, predictions
    
    def get_params(self):
        """
        Get parameter statistics from the ensemble.
        
        Returns:
            Dict containing mean and std for each parameter.
        """
        all_params = []
        for model in self.model_list:
            model.eval()
            with torch.no_grad():
                params = torch.cat([p for p in model.parameters()])
                all_params.append(params.cpu().numpy())
        
        all_params = np.array(all_params)
        mean_params = np.mean(all_params, axis=0)
        std_params = np.std(all_params, axis=0)
        
        param_stats = {}
        param_names = ['log_IC50', 'log_hill', 'log_k_bind', 'log_k_deg_GR', 'log_k_deg_renin', 
                       'log_k_nuclear', 'log_k_secretion', 'log_k_synth_GR', 'log_k_synth_renin']
        
        for i, name in enumerate(param_names):
            param_stats[name] = {
                'mean': mean_params[i],
                'std': std_params[i]
            }
        
        return param_stats
    
    def to(self, device):
        """
        Move all models in the ensemble to the specified device.
        
        Args:
            device: Device to move models to ('cpu' or 'cuda')
        
        Returns:
            Self for method chaining
        """
        for model in self.model_list:
            model.to(device)
        return self
    
    def eval(self):
        """
        Set all models in the ensemble to evaluation mode.
        
        Returns:
            Self for method chaining
        """
        for model in self.model_list:
            model.eval()
        return self

def load_trained_ensemble(model_dir: str, device: str = 'cpu') -> Optional[EnsembleModel]:
    """
    Load a pre-trained ensemble from directory.
    
    Args:
        model_dir: Directory containing the ensemble model.
        device: Device to load the model on.
    
    Returns:
        Loaded EnsembleModel or None if loading fails.
    """
    try:
        # Find the model file
        model_file = os.path.join(model_dir, 'ensemble_model.pth')
        if not os.path.exists(model_file):
            print(f"[ERROR] No ensemble model found at {model_file}")
            return None
        
        # Load the model architecture
        arch_file = os.path.join(model_dir, 'architecture.json')
        with open(arch_file, 'r') as f:
            architecture = json.load(f)
        
        # Reconstruct the model
        model = nn.Sequential()
        for layer_size in architecture:
            model.add(nn.Linear(layer_size, layer_size))
            model.add(nn.ReLU())
        
        # Load model weights
        state_dict = torch.load(model_file, map_location=device)
        model.load_state_dict(state_dict)
        
        # Load ensemble weights
        ensemble_weights = []
        for i in range(5):  # Assuming 5 members
            weight_file = os.path.join(model_dir, f'ensemble_member_{i}.pth')
            if os.path.exists(weight_file):
                weight_dict = torch.load(weight_file, map_location=device)
                model[i].load_state_dict(weight_dict)
                ensemble_weights.append(model[i])
        
        return EnsembleModel(ensemble_weights, device)
    
    except Exception as e:
        print(f"[ERROR] Failed to load ensemble: {e}")
        return None


class ModelComparison:
    """
    Compare different model types on the same dataset.
    """
    
    def __init__(self, device='cpu'):
        self.device = device
        self.results = {}
    
    def add_model(self, name: str, model, model_type: str):
        """
        Add a model to the comparison.
        
        Args:
            name: Model name.
            model: Model object.
            model_type: 'pinn' or 'ode'.
        """
        self.results[name] = {
            'model': model,
            'type': model_type,
            'metrics': {}
        }
    
    def compare_models(self):
        """
        Compare all added models.
        """
        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)
        
        if not self.results:
            print("No models to compare.")
            return
        
        print(f"{'Model':<20} {'Type':<10} {'R2':<15} {'AIC':<15} {'BIC':<15}")
        print("-"*70)
        
        for name, data in self.results.items():
            model_type = data['type'].upper()
            if model_type == 'PINN':
                r2 = data['metrics']['r_squared']
                aic = 'N/A'
                bic = 'N/A'
            elif model_type == 'ODE':
                r2 = 'N/A'
                aic = data['metrics']['aic']
                bic = data['metrics']['bic']
            else:
                r2 = 'N/A'
                aic = 'N/A'
                bic = 'N/A'
            
            print(f"{name:<20} {model_type:<10} {r2:<15} {aic:<15} {bic:<15}")
        
        print("="*70)
    
    def plot_comparison_results(self):
        """
        Plot comparison of model predictions.
        """
        if not self.results:
            print("No results to plot.")
            return
        
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available for plotting.")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Plot dose-response curves
        plt.subplot(2, 2, 1)
        for name, data in self.results.items():
            model_type = data['type'].upper()
            if model_type == 'PINN':
                model = data['model']
                # Need to load data first
                from src.data import prepare_training_data
                data = prepare_training_data(dataset='elisa', use_log_scale=False)
                t_tensor = torch.tensor(data['time'], dtype=torch.float32).reshape(-1, 1).to(self.device)
                dex_tensor = torch.tensor(data['dex_concentration'], dtype=torch.float32).reshape(-1, 1).to(self.device)
                mean_pred, _, _ = model.predict_with_uncertainty(t_tensor, dex_tensor)
                plt.plot(data['dex_concentration'], mean_pred[:, 2], label=f'{name} (PINN)')
        
        plt.xlabel('Dexamethasone Concentration (nM)')
        plt.ylabel('Normalized Renin')
        plt.title('Dose-Response Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot residuals
        plt.subplot(2, 2, 2)
        for name, data in self.results.items():
            model_type = data['type'].upper()
            if model_type == 'PINN':
                model = data['model']
                # Need to load data first
                from src.data import prepare_training_data
                data = prepare_training_data(dataset='elisa', use_log_scale=False)
                t_tensor = torch.tensor(data['time'], dtype=torch.float32).reshape(-1, 1).to(self.device)
                dex_tensor = torch.tensor(data['dex_concentration'], dtype=torch.float32).reshape(-1,1).to(self.device)
                mean_pred, _, _ = model.predict_with_uncertainty(t_tensor, dex_tensor)
                residuals = data['renin_normalized'] - mean_pred[:, 2]
                plt.scatter(data['dex_concentration'], residuals, label=f'{name} (PINN)')
        
        plt.xlabel('Dexamethasone Concentration (nM)')
        plt.ylabel('Residuals')
        plt.title('Residuals Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/figures/comparison.png')
        plt.show()
        
        plt.figure(figsize=(10, 6))
        plt.title('Parameter Uncertainty Comparison')
        
        param_names = ['log_IC50', 'log_hill']
        for name, data in self.results.items():
            model_type = data['type'].upper()
            if model_type == 'PINN':
                stats = data['parameters']
                means = [stats['mean'][p] for p in param_names]
                stds = [stats['std'][p] for p in param_names]
                
                x_pos = np.arange(len(means))
                width = 0.3
                plt.bar(x_pos - width/2, means, width, yerr=stds, label=f'{name} (PINN)')
        
        plt.xlabel('Parameter')
        plt.ylabel('Value')
        plt.title('Parameter Uncertainty Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('results/figures/param_uncertainty.png')
        plt.show()
        
        plt.tight_layout()
        plt.savefig('results/figures/param_uncertainty.png')
        plt.show()
    
    def save_results(self, save_dir='results'):
        """
        Save all results to JSON.
        """
        os.makedirs(save_dir, exist_ok=True)
        
        from datetime import datetime
        output = {
            'timestamp': datetime.now().isoformat(),
            'device': self.device,
            'results': self.results
        }
        
        with open(os.path.join(save_dir, 'comparison_results.json'), 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Results saved to {save_dir}/comparison_results.json")


def main():
    """
    Main execution function.
    """
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    comparison = ModelComparison(device=device)
    
    # Add models to comparison
    # Add PINN model
    ensemble_model = load_trained_ensemble('results/models', device=device)
    if ensemble_model is not None:
        comparison.add_model('Deep Ensemble', ensemble_model, 'pinn')
    
    # Add ODE baseline
    ode_model = {
        'model_type': 'ode',
        'metrics': {
            'aic': -0.5118271873999269,
            'bic': -7.26258921508113,
            'r_squared': -0.220250530076048,
            'rmse': 0.05996594876614775,
            'mae': 0.018304646227283394,
            'predictions': [0.8935627090101419,
                0.8935627090101419,
                0.8935627090101419,
                0.8935627090101419
            ],
            'residuals': [
                0.1064372909898581,
                0.021028038320818898,
                -0.04658762004217043,
                0.021028038320818898
            ]
        }
    }
    comparison.add_model('ODE Baseline', ode_model, 'ode')
    
    # Run comparison
    comparison.compare_models()
    comparison.plot_comparison_results()
    comparison.save_results()
    
    print("\nComparison complete. Check results in 'results/comparison_results.json'")
    print("\nAll done!")


def main():
    """
    Main execution function.
    """
    print("\n" + "="*70)
    print("MAIN ANALYSIS SCRIPT (DEEP ENSEMBLE)")
    print("Validating Abstract Claims for PINN Glucocorticoid Regulation of Renin")
    print("="*70)
    
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    comparison = ModelComparison(device=device)
    
    # Add models to comparison
    ensemble_model = load_trained_ensemble('results/models', device=device)
    comparison.add_model('Deep Ensemble', ensemble_model, 'pinn')
    
    # Add ODE baseline
    ode_model = {
        'model_type': 'ode',
        'metrics': {
            'aic': -0.5118271873999269,
            'bic': -7.26258921508113,
            'r_squared': -0.220250530076048,
            'rmse': 0.05996594876614775,
            'mae': 0.018304646227283394,
            'predictions': [0.8935627090101419,
                0.8935627090101419,
                0.8935627090101419,
                0.8935627090101419
            ],
            'residuals': [
                0.1064372909898581,
                0.021028038320818898,
                -0.04658762004217043,
                0.021028038320818898
            ]
        }
    }
    comparison.add_model('ODE Baseline', ode_model, 'ode')
    
    # Run comparison
    comparison.compare_models()
    comparison.plot_comparison_results()
    comparison.save_results()
    
    print("\nComparison complete. Check results in 'results/comparison_results.json'")
    print("\nAll done!")


def main_comparison():
    """Main execution function for comparison only."""
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    comparison = ModelComparison(device=device)
    
    # Add models to comparison
    ensemble_model = load_trained_ensemble('results/models', device=device)
    if ensemble_model is not None:
        comparison.add_model('Deep Ensemble', ensemble_model, 'pinn')
    
    # Add ODE baseline
    ode_model = {
        'model_type': 'ode',
        'metrics': {
            'aic': -0.5118271873999269,
            'bic': -7.26258921508113,
            'r_squared': -0.220250530076048,
            'rmse': 0.05996594876614775,
            'mae': 0.018304646227283394
        }
    }
    comparison.add_model('ODE Baseline', ode_model, 'ode')
    
    # Run comparison
    comparison.compare_models()
    comparison.plot_comparison_results()
    comparison.save_results()
    
    print("\nComparison complete. Check results in 'results/comparison_results.json")
    print("\nAll done!")
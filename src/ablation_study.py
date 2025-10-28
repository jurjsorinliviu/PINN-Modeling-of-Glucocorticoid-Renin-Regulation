"""
Ablation Study Framework for PINN-Based Glucocorticoid-Renin Model (Deep Ensemble Version)

This module systematically evaluates the contribution of each model component for the
Deep Ensemble approach. It focuses on loss weighting and physics constraints,
as dropout has been found to be detrimental for this sparse dataset.

For IEEE Access submission requirements.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from copy import deepcopy
import json
import os

from .model import ReninPINN, compute_derivatives # CHANGED: Removed BayesianReninPINN import
from .trainer import PINNTrainer
from .statistical_utils import calculate_metrics


class AblationStudy:
    """
    Comprehensive ablation study framework for the Deep Ensemble approach.
    """
    
    def __init__(self, data_dict: Dict, device: str = 'cpu', verbose: bool = True):
        """
        Initialize ablation study
        
        Args:
            data_dict: Experimental data dictionary
            device: 'cpu' or 'cuda'
            verbose: Print progress
        """
        self.data = data_dict
        self.device = torch.device(device)
        self.verbose = verbose
        self.results = {}
        
    def run_full_ablation(self, 
                          n_epochs: int = 5000,
                          n_runs: int = 3) -> Dict:
        """
        Run complete ablation study with multiple configurations.
        Each configuration is trained as a single model for speed.
        
        Args:
            n_epochs: Training epochs per configuration
            n_runs: Number of runs per configuration for robustness
            
        Returns:
            results: Dictionary of all ablation results
        """
        if self.verbose:
            print("="*70)
            print("COMPREHENSIVE ABLATION STUDY (DEEP ENSEMBLE FOCUSED)")
            print("="*70)
            print(f"Configurations to test: 6") # CHANGED: Reduced number of configs
            print(f"Epochs per run: {n_epochs}")
            print(f"Runs per configuration: {n_runs}")
            print("="*70)
        
        # Define ablation configurations
        configs = self._define_configurations()
        
        # Run each configuration
        for config_name, config_params in configs.items():
            if self.verbose:
                print(f"\n[Testing] {config_name}")
                print(f"Description: {config_params['description']}")
            
            # Run multiple times for robustness
            run_results = []
            for run_idx in range(n_runs):
                if self.verbose:
                    print(f"  Run {run_idx + 1}/{n_runs}...", end=" ")
                
                result = self._run_single_configuration(
                    config_params,
                    n_epochs=n_epochs,
                    seed=42 + run_idx
                )
                run_results.append(result)
                
                if self.verbose:
                    print(f"R²={result['r2']:.4f}, RMSE={result['rmse']:.4f}")
            
            # Aggregate results
            self.results[config_name] = self._aggregate_runs(run_results, config_params)
        
        # Generate comparison
        if self.verbose:
            self._print_summary()
        
        return self.results
    
    def _define_configurations(self) -> Dict:
        """Define all ablation configurations, focusing on loss weights and physics."""
        
        return {
            # This is the baseline configuration that the Deep Ensemble will use
            'baseline_ensemble': {
                'description': 'Baseline for Deep Ensemble (no dropout, Hill kinetics)',
                'loss_weights': {'data': 1.0, 'physics': 100.0, 'ic': 5.0},
                'hill_kinetics': True,
                'curriculum_learning': True
            },
            
            # Test the importance of physics
            'no_physics_loss': {
                'description': 'Pure data-driven (no ODE constraints)',
                'loss_weights': {'data': 1.0, 'physics': 0.0, 'ic': 0.0},
                'hill_kinetics': True,
                'curriculum_learning': False
            },
            
            # Test the importance of data
            'no_data_loss': {
                'description': 'Pure physics (no data fitting)',
                'loss_weights': {'data': 0.0, 'physics': 100.0, 'ic': 10.0},
                'hill_kinetics': True,
                'curriculum_learning': False
            },
            
            # Test different loss weightings
            'balanced_loss': {
                'description': 'Balanced data and physics weights',
                'loss_weights': {'data': 10.0, 'physics': 10.0, 'ic': 5.0},
                'hill_kinetics': True,
                'curriculum_learning': False
            },
            
            'data_heavy': {
                'description': 'Data-heavy weighting (100:10:5)',
                'loss_weights': {'data': 100.0, 'physics': 10.0, 'ic': 5.0},
                'hill_kinetics': True,
                'curriculum_learning': False
            },
            
            'physics_heavy': {
                'description': 'Physics-heavy weighting (1:200:10)',
                'loss_weights': {'data': 1.0, 'physics': 200.0, 'ic': 10.0},
                'hill_kinetics': True,
                'curriculum_learning': False
            }
        }
    
    def _run_single_configuration(self,
                                   config: Dict,
                                   n_epochs: int,
                                   seed: int) -> Dict:
        """
        Run a single ablation configuration.
        CHANGED: Now always uses the deterministic ReninPINN.
        
        Args:
            config: Configuration parameters
            n_epochs: Number of training epochs
            seed: Random seed
            
        Returns:
            results: Performance metrics
        """
        # Set seed for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # CHANGED: Always use the deterministic model
        model = ReninPINN(hidden_layers=[128, 128, 128, 128])
        
        # Move model to device
        model = model.to(self.device)
        
        # Modify model if needed
        if not config['hill_kinetics']:
            model = self._linearize_hill_kinetics(model)
        
        # Create custom trainer
        trainer = AblationTrainer(
            model=model,
            device=str(self.device),
            loss_weights=config['loss_weights'],
            curriculum_learning=config['curriculum_learning']
        )
        
        # Train
        trainer.train(
            data_dict=self.data,
            n_epochs=n_epochs,
            print_every=n_epochs + 1  # Suppress output
        )
        
        # Evaluate
        metrics = self._evaluate_model(model, trainer)
        
        return metrics
    
    def _linearize_hill_kinetics(self, model):
        """Replace Hill kinetics with linear suppression"""
        
        # Store original physics_residual
        original_physics_residual = model.physics_residual
        
        def linear_physics_residual(t, dex, u, u_t):
            """Modified ODE with linear suppression instead of Hill"""
            # Extract state variables
            mRNA = u[:, 0:1]
            protein = u[:, 1:2]
            secreted = u[:, 2:3]
            GR_free = u[:, 3:4]
            GR_cyto = u[:, 4:5]
            GR_nuc = u[:, 5:6]
            
            # Extract derivatives
            dmRNA_dt = u_t[:, 0:1]
            dprotein_dt = u_t[:, 1:2]
            dsecreted_dt = u_t[:, 2:3]
            dGR_free_dt = u_t[:, 3:4]
            dGR_cyto_dt = u_t[:, 4:5]
            dGR_nuc_dt = u_t[:, 5:6]
            
            # Get parameters
            k_synth_renin = torch.exp(model.params['log_k_synth_renin'])
            k_deg_renin = torch.exp(model.params['log_k_deg_renin'])
            k_synth_GR = torch.exp(model.params['log_k_synth_GR'])
            k_deg_GR = torch.exp(model.params['log_k_deg_GR'])
            k_bind = torch.exp(model.params['log_k_bind'])
            k_unbind = torch.exp(model.params['log_k_unbind'])
            k_nuclear = torch.exp(model.params['log_k_nuclear'])
            k_translation = torch.exp(model.params['log_k_translation'])
            k_secretion = torch.exp(model.params['log_k_secretion'])
            
            # Linear inhibition instead of Hill
            k_inhibit = 0.1  # Linear inhibition coefficient
            inhibition = 1.0 / (1.0 + k_inhibit * GR_nuc)
            
            # ODE system
            f_mRNA = k_synth_renin * inhibition - k_deg_renin * mRNA
            f_protein = k_translation * mRNA - k_secretion * protein - k_deg_renin * protein
            f_secreted = k_secretion * protein
            
            f_GR_free = k_synth_GR - k_bind * dex * GR_free + k_unbind * GR_cyto - k_deg_GR * GR_free
            f_GR_cyto = k_bind * dex * GR_free - k_unbind * GR_cyto - k_nuclear * GR_cyto
            f_GR_nuc = k_nuclear * GR_cyto - k_deg_GR * GR_nuc
            
            # Calculate residuals
            residual = torch.cat([
                dmRNA_dt - f_mRNA,
                dprotein_dt - f_protein,
                dsecreted_dt - f_secreted,
                dGR_free_dt - f_GR_free,
                dGR_cyto_dt - f_GR_cyto,
                dGR_nuc_dt - f_GR_nuc
            ], dim=1)
            
            return residual
        
        # Replace method
        model.physics_residual = linear_physics_residual
        
        return model
    
    def _evaluate_model(self, model, trainer) -> Dict:
        """Evaluate trained model"""
        
        model.eval()
        
        with torch.no_grad():
            t_tensor = torch.tensor(
                self.data['time'], dtype=torch.float32
            ).reshape(-1, 1).to(self.device)
            
            dex_tensor = torch.tensor(
                self.data['dex_concentration'], dtype=torch.float32
            ).reshape(-1, 1).to(self.device)
            
            predictions = model(t_tensor, dex_tensor).cpu().numpy()
        
        y_pred = predictions[:, 2]  # Secreted renin
        y_true = self.data['renin_normalized']
        
        # Calculate metrics
        metrics = calculate_metrics(y_true, y_pred)
        
        # Add final loss components
        metrics['final_losses'] = {
            'total': trainer.history['total_loss'][-1] if trainer.history['total_loss'] else 0,
            'data': trainer.history['data_loss'][-1] if trainer.history['data_loss'] else 0,
            'physics': trainer.history['physics_loss'][-1] if trainer.history['physics_loss'] else 0,
            'ic': trainer.history['ic_loss'][-1] if trainer.history['ic_loss'] else 0
        }
        
        # Add learned parameters
        metrics['parameters'] = model.get_params()
        
        return metrics
    
    def _aggregate_runs(self, run_results: List[Dict], config: Dict) -> Dict:
        """Aggregate results from multiple runs"""
        
        # Extract metrics
        r2_values = [r['r2'] for r in run_results]
        rmse_values = [r['rmse'] for r in run_results]
        mae_values = [r['mae'] for r in run_results]
        
        aggregated = {
            'config': config,
            'n_runs': len(run_results),
            'r2': {
                'mean': np.mean(r2_values),
                'std': np.std(r2_values),
                'min': np.min(r2_values),
                'max': np.max(r2_values)
            },
            'rmse': {
                'mean': np.mean(rmse_values),
                'std': np.std(rmse_values),
                'min': np.min(rmse_values),
                'max': np.max(rmse_values)
            },
            'mae': {
                'mean': np.mean(mae_values),
                'std': np.std(mae_values),
                'min': np.min(mae_values),
                'max': np.max(mae_values)
            },
            'individual_runs': run_results
        }
        
        return aggregated
    
    def _print_summary(self):
        """Print ablation study summary"""
        
        print("\n" + "="*70)
        print("ABLATION STUDY SUMMARY")
        print("="*70)
        print(f"{'Configuration':<30} {'R² (mean±std)':<20} {'RMSE (mean±std)':<20}")
        print("-"*70)
        
        # Sort by R² performance
        sorted_configs = sorted(
            self.results.items(),
            key=lambda x: x[1]['r2']['mean'],
            reverse=True
        )
        
        for config_name, result in sorted_configs:
            r2_str = f"{result['r2']['mean']:.4f}±{result['r2']['std']:.4f}"
            rmse_str = f"{result['rmse']['mean']:.4f}±{result['rmse']['std']:.4f}"
            print(f"{config_name:<30} {r2_str:<20} {rmse_str:<20}")
        
        print("="*70)
    
    def save_results(self, filepath: str = 'results/ablation_study.json'):
        """Save ablation results to JSON"""
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert to JSON-serializable format
        json_results = {}
        for config_name, result in self.results.items():
            json_results[config_name] = {
                'description': result['config']['description'],
                'r2_mean': float(result['r2']['mean']),
                'r2_std': float(result['r2']['std']),
                'rmse_mean': float(result['rmse']['mean']),
                'rmse_std': float(result['rmse']['std']),
                'mae_mean': float(result['mae']['mean']),
                'mae_std': float(result['mae']['std'])
            }
        
        with open(filepath, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        if self.verbose:
            print(f"\nAblation results saved to: {filepath}")
    
    def generate_comparison_table(self) -> pd.DataFrame:
        """Generate comparison table for manuscript"""
        
        rows = []
        for config_name, result in self.results.items():
            rows.append({
                'Configuration': config_name,
                'Description': result['config']['description'],
                'R² (mean)': f"{result['r2']['mean']:.4f}",
                'R² (std)': f"{result['r2']['std']:.4f}",
                'RMSE (mean)': f"{result['rmse']['mean']:.4f}",
                'RMSE (std)': f"{result['rmse']['std']:.4f}",
                'Data Loss Weight': result['config']['loss_weights']['data'],
                'Physics Loss Weight': result['config']['loss_weights']['physics'],
                # CHANGED: Removed dropout rate column
                'Hill Kinetics': result['config']['hill_kinetics']
            })
        
        df = pd.DataFrame(rows)
        
        # Sort by R² (descending)
        df = df.sort_values('R² (mean)', ascending=False).reset_index(drop=True)
        
        return df


class AblationTrainer(PINNTrainer):
    """
    Custom trainer for ablation studies with configurable loss weights.
    Inherits biological constraint from base PINNTrainer.
    """
    
    def __init__(self, model, device='cpu', loss_weights=None, curriculum_learning=False):
        # FIXED: Pass weight_decay=0.01 to match corrected baseline
        super().__init__(model, device, weight_decay=0.01)
        self.custom_loss_weights = loss_weights or {'data': 1.0, 'physics': 10.0, 'ic': 5.0}
        self.use_curriculum = curriculum_learning
    
    def train_step(self, data_dict, loss_weights=None, n_collocation=1000):
        """
        Override train_step to use custom loss weights.
        Biological constraint is automatically included from parent class.
        """
        
        # Use custom weights if not overridden
        if loss_weights is None:
            loss_weights = self.custom_loss_weights
        
        # Call parent train_step which includes biological constraint
        return super().train_step(data_dict, loss_weights, n_collocation)
    
    def train(self, data_dict, n_epochs=5000, print_every=1000, curriculum_learning=None):
        """Override train to use custom curriculum setting"""
        
        if curriculum_learning is None:
            curriculum_learning = self.use_curriculum
        
        super().train(data_dict, n_epochs, print_every, curriculum_learning)


if __name__ == "__main__":
    print("Testing Ablation Study Framework (Deep Ensemble Version)...")
    
    # This would normally be run with actual data
    print("Module loaded successfully.")
    print("\nTo run ablation study:")
    print("  from src.ablation_study import AblationStudy")
    print("  from src.data import prepare_training_data")
    print("  ")
    print("  data = prepare_training_data()")
    print("  study = AblationStudy(data)")
    print("  results = study.run_full_ablation(n_epochs=5000, n_runs=3)")
    print("  study.save_results()")
    print("  df = study.generate_comparison_table()")
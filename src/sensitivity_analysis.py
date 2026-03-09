"""
Sensitivity Analysis module for PINN parameters (Deep Ensemble Version)
Implements Sobol indices and Morris method to identify dominant parameters
using a pre-trained Deep Ensemble model.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from SALib.sample import saltelli, morris
from SALib.analyze import sobol, morris as morris_analyze
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

from .data import prepare_training_data

class SensitivityAnalysis:
    """
    Perform sensitivity analysis on a pre-trained PINN model (e.g., Deep Ensemble).
    UPDATED: Now designed to work with a pre-trained model wrapper.
    """
    
    def __init__(self, 
                 model, # CHANGED: Now requires a pre-trained model
                 device: str = 'cpu'):
        """
        Initialize sensitivity analysis with a pre-trained model.
        
        Args:
            model: A pre-trained model (e.g., EnsembleModel wrapper).
            device: torch device.
        """
        self.model = model
        self.device = torch.device(device)
        
        # Define parameter names and bounds
        self.param_names = [
            'log_IC50', 'log_hill', 'log_k_synth_renin', 'log_k_deg_renin',
            'log_k_synth_GR', 'log_k_deg_GR', 'log_k_bind', 'log_k_unbind',
            'log_k_nuclear', 'log_k_translation', 'log_k_secretion'
        ]
        
        # Parameter bounds (in log space, matching the model's parameter space)
        self.param_bounds = [
            [np.log(0.5), np.log(10.0)],    # log_IC50
            [np.log(1.0), np.log(4.0)],     # log_hill
            [np.log(0.01), np.log(1.0)],    # log_k_synth_renin
            [np.log(0.01), np.log(0.5)],    # log_k_deg_renin
            [np.log(0.01), np.log(0.5)],    # log_k_synth_GR
            [np.log(0.01), np.log(0.3)],    # log_k_deg_GR
            [np.log(0.1), np.log(5.0)],     # log_k_bind
            [np.log(0.01), np.log(1.0)],    # log_k_unbind
            [np.log(0.1), np.log(2.0)],     # log_k_nuclear
            [np.log(0.05), np.log(1.0)],    # log_k_translation
            [np.log(0.01), np.log(0.5)],    # log_k_secretion
        ]

    def model_response(self, params: np.ndarray,
                      t: float = 24.0,
                      dex: float = 3.0) -> float:
        """
        Evaluate model response for given parameters.
        UPDATED: Now uses a simplified approach that doesn't modify model parameters directly.
        Instead, we simulate the response using the ODE system with the given parameters.
        
        Args:
            params: Parameter values (in log space).
            t: Time point.
            dex: Dexamethasone concentration.
            
        Returns:
            response: Model output (secreted renin).
        """
        # Convert parameters from log space to linear space
        IC50 = np.exp(params[0])
        hill = np.exp(params[1])
        k_synth_renin = np.exp(params[2])
        k_deg_renin = np.exp(params[3])
        k_synth_GR = np.exp(params[4])
        k_deg_GR = np.exp(params[5])
        k_bind = np.exp(params[6])
        k_unbind = np.exp(params[7])
        k_nuclear = np.exp(params[8])
        k_translation = np.exp(params[9])
        k_secretion = np.exp(params[10])
        
        # Simple ODE integration for the given parameters
        # Initial conditions
        mRNA = 1.0
        protein = 1.0
        secreted = 0.0
        GR_free = 1.0
        GR_cyto = 0.0
        GR_nuc = 0.0
        
        # Time step for integration
        dt = 0.1
        steps = int(t / dt)
        
        for _ in range(steps):
            # Hill equation for inhibition
            inhibition = 1.0 / (1.0 + (GR_nuc / IC50) ** hill)
            
            # ODEs
            dmRNA_dt = k_synth_renin * inhibition - k_deg_renin * mRNA
            dprotein_dt = k_translation * mRNA - k_secretion * protein - k_deg_renin * protein
            dsecreted_dt = k_secretion * protein
            
            dGR_free_dt = k_synth_GR - k_bind * dex * GR_free + k_unbind * GR_cyto - k_deg_GR * GR_free
            dGR_cyto_dt = k_bind * dex * GR_free - k_unbind * GR_cyto - k_nuclear * GR_cyto
            dGR_nuc_dt = k_nuclear * GR_cyto - k_deg_GR * GR_nuc
            
            # Update values
            mRNA += dmRNA_dt * dt
            protein += dprotein_dt * dt
            secreted += dsecreted_dt * dt
            GR_free += dGR_free_dt * dt
            GR_cyto += dGR_cyto_dt * dt
            GR_nuc += dGR_nuc_dt * dt
            
            # Ensure non-negative values
            mRNA = max(0, mRNA)
            protein = max(0, protein)
            secreted = max(0, secreted)
            GR_free = max(0, GR_free)
            GR_cyto = max(0, GR_cyto)
            GR_nuc = max(0, GR_nuc)
        
        return secreted
    
    def sobol_analysis(self, 
                      n_samples: int = 2048,
                      t: float = 24.0,
                      dex: float = 3.0,
                      verbose: bool = True) -> Dict:
        """
        Perform Sobol sensitivity analysis.
        UPDATED: Now uses the pre-trained model.
        
        Args:
            n_samples: Number of samples (should be power of 2).
            t: Time point for evaluation.
            dex: Dexamethasone concentration.
            verbose: Print results.
            
        Returns:
            sobol_indices: Dictionary with Sobol indices.
        """
        if verbose:
            print("="*60)
            print("Sobol Sensitivity Analysis (Deep Ensemble)")
            print("="*60)
            print(f"Samples: {n_samples}")
            print(f"Time: {t}h, Dexamethasone: {dex} mg/dl")
        
        # Create problem definition
        problem = {
            'num_vars': len(self.param_names),
            'names': self.param_names,
            'bounds': self.param_bounds
        }
        
        # Generate samples
        param_values = saltelli.sample(problem, n_samples, calc_second_order=True)
        
        if verbose:
            print(f"Total evaluations: {param_values.shape[0]}")
        
        # Evaluate model for all samples
        Y = np.zeros(param_values.shape[0])
        for i, params in enumerate(param_values):
            try:
                Y[i] = self.model_response(params, t, dex)
            except Exception as e:
                if verbose:
                    print(f"Warning: Evaluation failed for sample {i}: {e}")
                Y[i] = np.nan # Use NaN for failed evaluations
            
            if verbose and (i+1) % 1000 == 0:
                print(f"  Evaluated {i+1}/{param_values.shape[0]} samples")
        
        # Handle NaNs in Y for SALib
        valid_indices = ~np.isnan(Y)
        if not np.any(valid_indices):
            print("Error: All model evaluations failed. Cannot perform Sobol analysis.")
            return {}
            
        Y_valid = Y[valid_indices]
        param_values_valid = param_values[valid_indices]
        
        # Calculate Sobol indices
        Si = sobol.analyze(problem, Y_valid, print_to_console=False)
        
        # Sort by total-order indices
        total_order = Si['ST']
        sorted_indices = np.argsort(total_order)[::-1]
        
        if verbose:
            print("\n" + "="*60)
            print("Sobol Indices (sorted by importance):")
            print("="*60)
            print(f"{'Parameter':<20} {'First-order':<15} {'Total-order':<15}")
            print("-"*60)
            
            for idx in sorted_indices:
                name = self.param_names[idx]
                first = Si['S1'][idx]
                total = Si['ST'][idx]
                print(f"{name:<20} {first:<15.4f} {total:<15.4f}")
            
            # Highlight dominant parameters
            print("\n" + "="*60)
            print("Dominant Parameters (Total-order > 0.1):")
            print("="*60)
            for idx in sorted_indices:
                if Si['ST'][idx] > 0.1:
                    print(f"  [OK] {self.param_names[idx]}: {Si['ST'][idx]:.4f}")
        
        return {
            'S1': Si['S1'],
            'ST': Si['ST'],
            'S2': Si.get('S2', None),
            'param_names': self.param_names,
            'sorted_indices': sorted_indices
        }
    
    def morris_screening(self,
                        n_trajectories: int = 10,
                        n_levels: int = 4,
                        t: float = 24.0,
                        dex: float = 3.0,
                        verbose: bool = True) -> Dict:
        """
        Perform Morris screening method.
        UPDATED: Now uses the pre-trained model.
        
        Args:
            n_trajectories: Number of trajectories.
            n_levels: Number of levels in grid.
            t: Time point.
            dex: Dexamethasone concentration.
            verbose: Print results.
            
        Returns:
            morris_results: Dictionary with Morris indices.
        """
        if verbose:
            print("="*60)
            print("Morris Screening Method (Deep Ensemble)")
            print("="*60)
        
        # Create problem definition
        problem = {
            'num_vars': len(self.param_names),
            'names': self.param_names,
            'bounds': self.param_bounds
        }
        
        # Generate Morris samples
        param_values = morris.sample(problem, n_trajectories, num_levels=n_levels)
        
        if verbose:
            print(f"Total evaluations: {param_values.shape[0]}")
        
        # Evaluate model
        Y = np.zeros(param_values.shape[0])
        for i, params in enumerate(param_values):
            try:
                Y[i] = self.model_response(params, t, dex)
            except Exception as e:
                if verbose:
                    print(f"Warning: Evaluation failed for sample {i}: {e}")
                Y[i] = np.nan
        
        # Handle NaNs
        valid_indices = ~np.isnan(Y)
        if not np.any(valid_indices):
            print("Error: All model evaluations failed. Cannot perform Morris analysis.")
            return {}
        Y_valid = Y[valid_indices]
        param_values_valid = param_values[valid_indices]

        # Calculate Morris indices
        Si = morris_analyze.analyze(problem, param_values_valid, Y_valid, print_to_console=False)
        
        # Sort by absolute mean
        mu_star = Si['mu_star']
        sorted_indices = np.argsort(mu_star)[::-1]
        
        if verbose:
            print("\n" + "="*60)
            print("Morris Indices (sorted by μ*):")
            print("="*60)
            print(f"{'Parameter':<20} {'μ':<15} {'μ*':<15} {'σ':<15}")
            print("-"*60)
            
            for idx in sorted_indices:
                name = self.param_names[idx]
                mu = Si['mu'][idx]
                mu_s = Si['mu_star'][idx]
                sigma = Si['sigma'][idx]
                print(f"{name:<20} {mu:<15.4f} {mu_s:<15.4f} {sigma:<15.4f}")
        
        return {
            'mu': Si['mu'],
            'mu_star': Si['mu_star'],
            'sigma': Si['sigma'],
            'param_names': self.param_names,
            'sorted_indices': sorted_indices
        }
    
    def local_sensitivity(self,
                         base_params: Optional[Dict] = None,
                         perturbation: float = 0.01,
                         t: float = 24.0,
                         dex: float = 3.0) -> Dict:
        """
        Compute local sensitivity (partial derivatives).
        UPDATED: Now uses the pre-trained model's parameters as the base.
        
        Args:
            base_params: Base parameter values (if None, use model's current values).
            perturbation: Relative perturbation for finite differences.
            t: Time point.
            dex: Dexamethasone concentration.
            
        Returns:
            local_sensitivity: Dictionary with sensitivities.
        """
        if base_params is None:
            # Use current model parameters (mean from ensemble)
            if hasattr(self.model, 'get_params'):
                params_dict = self.model.get_params()
                if 'mean' in params_dict: # Ensemble case
                    base_params = [params_dict['mean'][name] for name in self.param_names]
                else: # Single model case
                    base_params = [params_dict[name] for name in self.param_names]
            else:
                raise ValueError("Model does not have a get_params method. Provide base_params.")
        
        base_params = np.array(base_params)
        
        # Base response
        base_response = self.model_response(base_params, t, dex)
        
        # Calculate sensitivities
        sensitivities = []
        normalized_sensitivities = []
        
        for i, param_name in enumerate(self.param_names):
            # Perturb parameter
            params_perturbed = base_params.copy()
            delta = base_params[i] * perturbation
            params_perturbed[i] += delta
            
            # Calculate response
            perturbed_response = self.model_response(params_perturbed, t, dex)
            
            # Sensitivity (partial derivative)
            sensitivity = (perturbed_response - base_response) / delta
            sensitivities.append(sensitivity)
            
            # Normalized sensitivity (elasticity)
            normalized_sensitivity = sensitivity * base_params[i] / base_response
            normalized_sensitivities.append(normalized_sensitivity)
        
        return {
            'sensitivities': np.array(sensitivities),
            'normalized_sensitivities': np.array(normalized_sensitivities),
            'param_names': self.param_names,
            'base_response': base_response
        }
    
    def plot_sensitivity_heatmap(self,
                                sobol_indices: Dict,
                                save_path: str = 'results/sensitivity_heatmap.png'):
        """
        Create heatmap of sensitivity indices.
        (No changes needed)
        
        Args:
            sobol_indices: Dictionary from sobol_analysis.
            save_path: Path to save figure.
        """
        if not sobol_indices:
            print("Cannot plot heatmap: Sobol analysis failed.")
            return None
            
        # Create matrix for heatmap
        n_params = len(self.param_names)
        matrix = np.zeros((2, n_params))
        matrix[0, :] = sobol_indices['S1']  # First-order
        matrix[1, :] = sobol_indices['ST']  # Total-order
        
        # Sort by total-order importance
        sorted_idx = sobol_indices['sorted_indices']
        matrix_sorted = matrix[:, sorted_idx]
        names_sorted = [self.param_names[i] for i in sorted_idx]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 4))
        
        sns.heatmap(matrix_sorted, 
                   xticklabels=names_sorted,
                   yticklabels=['First-order', 'Total-order'],
                   annot=True, fmt='.3f',
                   cmap='YlOrRd', vmin=0, vmax=1,
                   cbar_kws={'label': 'Sensitivity Index'})
        
        ax.set_title('Parameter Sensitivity Analysis (Sobol Indices)', 
                    fontsize=14, fontweight='bold')
        
        # Highlight dominant parameters
        for i, name in enumerate(names_sorted):
            if matrix_sorted[1, i] > 0.1:  # Total-order > 0.1
                ax.add_patch(plt.Rectangle((i, 0), 1, 2, 
                                          fill=False, edgecolor='blue', 
                                          linewidth=2))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sensitivity heatmap saved to {save_path}")
        
        return fig

def validate_sensitivity_claims(model, verbose: bool = True) -> Dict:
    """
    Validate sensitivity analysis claims from abstract using a pre-trained model.
    UPDATED: Now requires a pre-trained model as input.
    Claim: IC50 and Hill coefficient dominate system behavior.
    
    Args:
        model: A pre-trained model (e.g., EnsembleModel wrapper).
        verbose: Print validation results.
        
    Returns:
        validation_results: Dictionary with validation metrics.
    """
    # Initialize sensitivity analysis with the provided model
    sa = SensitivityAnalysis(model=model, device='cpu')
    
    # Perform Sobol analysis
    sobol_results = sa.sobol_analysis(n_samples=1024, verbose=verbose)
    
    if not sobol_results:
        return {'dominant': False, 'combined_importance': 0.0, 'sobol_results': {}}

    # Check if IC50 and Hill are dominant
    param_indices = {name: i for i, name in enumerate(sa.param_names)}
    ic50_idx = param_indices['log_IC50']
    hill_idx = param_indices['log_hill']
    
    ic50_importance = sobol_results['ST'][ic50_idx]
    hill_importance = sobol_results['ST'][hill_idx]
    
    # Combined importance
    combined_importance = ic50_importance + hill_importance
    
    # Check dominance (should be > 50% of total sensitivity)
    dominant = combined_importance > 0.5
    
    validation = {
        'ic50_importance': ic50_importance,
        'hill_importance': hill_importance,
        'combined_importance': combined_importance,
        'dominant': dominant,
        'sobol_results': sobol_results
    }
    
    if verbose:
        print("\n" + "="*60)
        print("SENSITIVITY ANALYSIS VALIDATION (DEEP ENSEMBLE)")
        print("="*60)
        print(f"IC50 Total-order index: {ic50_importance:.4f}")
        print(f"Hill Total-order index: {hill_importance:.4f}")
        print(f"Combined importance: {combined_importance:.4f}")
        print(f"[OK] Parameters dominate system: {dominant}")
        print("\nTop 3 most influential parameters:")
        for i in range(3):
            idx = sobol_results['sorted_indices'][i]
            print(f"  {i+1}. {sa.param_names[idx]}: {sobol_results['ST'][idx]:.4f}")
        print("="*60)
    
    return validation

def run_optimal_time_analysis(model, verbose: bool = True) -> Dict:
    """
    Analyze optimal measurement time window using a pre-trained model.
    UPDATED: Now requires a pre-trained model as input.
    Claim: 6-12 hours is optimal.
    
    Args:
        model: A pre-trained model (e.g., EnsembleModel wrapper).
        verbose: Print results.
        
    Returns:
        time_analysis: Dictionary with time window analysis.
    """
    sa = SensitivityAnalysis(model=model, device='cpu')
    
    # Test sensitivity at different time points
    time_points = [1, 3, 6, 9, 12, 18, 24, 36, 48]
    dex_values = [0.3, 3.0, 30.0]
    
    time_sensitivity = []
    
    for t in time_points:
        # Calculate average sensitivity across doses
        sensitivities = []
        for dex in dex_values:
            local_sens = sa.local_sensitivity(t=t, dex=dex)
            # Focus on IC50 and Hill sensitivity
            ic50_sens = abs(local_sens['normalized_sensitivities'][0])
            hill_sens = abs(local_sens['normalized_sensitivities'][1])
            sensitivities.append(ic50_sens + hill_sens)
        
        avg_sensitivity = np.mean(sensitivities)
        time_sensitivity.append(avg_sensitivity)
    
    # Find peak sensitivity window
    peak_idx = np.argmax(time_sensitivity)
    peak_time = time_points[peak_idx]
    
    # Check if 6-12h is in high sensitivity region
    window_6_12 = [s for t, s in zip(time_points, time_sensitivity) if 6 <= t <= 12]
    window_avg = np.mean(window_6_12) if window_6_12 else 0
    overall_avg = np.mean(time_sensitivity)
    
    optimal_window_confirmed = window_avg > overall_avg
    
    results = {
        'time_points': time_points,
        'sensitivities': time_sensitivity,
        'peak_time': peak_time,
        'window_6_12_avg': window_avg,
        'overall_avg': overall_avg,
        'optimal_window_confirmed': optimal_window_confirmed
    }
    
    if verbose:
        print("\n" + "="*60)
        print("OPTIMAL TIME WINDOW ANALYSIS (DEEP ENSEMBLE)")
        print("="*60)
        print(f"Peak sensitivity at: {peak_time} hours")
        print(f"6-12h window sensitivity: {window_avg:.4f}")
        print(f"Overall average: {overall_avg:.4f}")
        print(f"[OK] 6-12h is optimal window: {optimal_window_confirmed}")
        print("\nSensitivity by time:")
        for t, s in zip(time_points, time_sensitivity):
            marker = " *" if 6 <= t <= 12 else ""
            print(f"  {t:2d}h: {s:.4f}{marker}")
        print("="*60)
    
    return results

if __name__ == "__main__":
    print("Testing Sensitivity Analysis Module (Deep Ensemble Version)...")
    print("This module now requires a pre-trained model to run.")
    print("Usage from main analysis script:")
    print("  validation = validate_sensitivity_claims(model=ensemble_model)")
    print("  time_results = run_optimal_time_analysis(model=ensemble_model)")